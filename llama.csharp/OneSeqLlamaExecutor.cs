using Llama.csharp.Exceptions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public class OneSeqLlamaExecutor: IDisposable
    {
        /// <summary>
        /// The context used by the executor.
        /// </summary>
        public LLamaContext Context { get; }

        private Sequence _mainSeq;

        private readonly CancellationTokenSource _executorLifeToken = new CancellationTokenSource();

        private CancellationTokenSource _generationToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _seqStateSemaphore = new SemaphoreSlim(1);

        public OneSeqLlamaExecutor(LLamaContext context)
        {
            Context = context;
            _mainSeq = new Sequence(context)
            {
                Id = (LLamaSeqId)0,
                AntipromptProc = new AntipromptProcessor()
            };
        }

        /// <summary>
        /// Preprocess the inputs before the inference for given Sequence.
        /// </summary>
        /// <param name="text"></param>
        private List<LLamaToken> preprocessInputs(string text, bool addBos, bool special)
        {
            return Context.Tokenize(text, addBos, special).ToList();
        }

        /// <summary>
        /// Префил 
        /// </summary>
        /// <param name="seqIds"></param>
        /// <param name="texts"></param>
        /// <param name="addBos"></param>
        /// <param name="special"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public async Task ProcessPrompt(string text, bool addBos = false, bool special = true)
        {
            // простые проверки аргументов, без блокировки
            #region Checks 

            #endregion

            List<LLamaToken>? embeds;

            if (string.IsNullOrEmpty(text))
                embeds = null;
            else
                embeds = preprocessInputs(text, addBos, special);

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //блокировка для редактирования _prefillSeqs
            try
            {
                if (embeds != null)
                {
                    _mainSeq.InferState.State = SeqState.Prefill;
                    //заполнение Embeds последовательности
                    _mainSeq.TokensToPrefill.AddRange(embeds);

                    LLamaBatch batch = new LLamaBatch();

                    (DecodeResult, int, int) tuple = await Context.DecodeAsync(_mainSeq.TokensToPrefill, _mainSeq.Id, batch, _mainSeq.NextDecodedTokenPos);

                    if (tuple.Item1 != DecodeResult.Ok)
                    {
                        //tuple.Item2; хранит число неотдекоденных
                        throw new LLamaDecodeError(tuple.Item1);
                    }

                    _mainSeq.NextDecodedTokenPos = tuple.Item3;
                    _mainSeq.DecodedTokens.AddRange(_mainSeq.TokensToPrefill);
                    _mainSeq.TokensToPrefill.Clear();
                    _mainSeq.LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(batch.TokenCount - 1));
                    _mainSeq.InferState.State = SeqState.None;
                }
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        private string postProcessInfer() //проверка, что пора заканчивать
        {
            _mainSeq.InferState.TokenSampledAndDecoded = false;

            // Преобразуем токен в строку
            _mainSeq.Decoder.Add(_mainSeq.DecodedTokens.Last());
            var decoded = _mainSeq.Decoder.Read();

            // если сгенерирован антипромпт
            if (!string.IsNullOrEmpty(decoded) && _mainSeq.AntipromptProc.Add(decoded))
            {
                _mainSeq.InferState.State = SeqState.None;
            }

            if (_mainSeq.InferState.AutoStopFromEOG)
            {
                // если сгенерирован конец
                if (_mainSeq.DecodedTokens.Last().IsEndOfGeneration(Context.Vocab))
                {
                    _mainSeq.InferState.State = SeqState.None;
                }
            }

            // если кончились токены
            if (_mainSeq.InferState.RemainedTokens <= 0 && _mainSeq.InferParams.MaxTokens != -1)
            {
                _mainSeq.InferState.State = SeqState.None;
            }

            return decoded;
        }

        public async IAsyncEnumerable<string> Generate(IInferenceParams inferenceParams)
        {
            // простые проверки аргументов, без блокировки
            #region Checks 
            if (inferenceParams == null) throw new ArgumentNullException();
            #endregion

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                //проверки
                if (_mainSeq.InferParams.MaxTokens == 0) throw new Exception($"For prefill use ProcessPrompt");

                _mainSeq.InferState.State = SeqState.Generation;

                //инициализация для инференса
                _mainSeq.InferState = new InferStateArgs
                {
                    State = SeqState.Generation,
                    RemainedTokens = inferenceParams.MaxTokens,
                    AutoStopFromEOG = inferenceParams.AutoStopFromEOG
                };
                _mainSeq.InferParams = inferenceParams;
                _mainSeq.AntipromptProc.ClearString();
                _mainSeq.AntipromptProc.SetAntiprompts(inferenceParams.AntiPrompts ?? []);
                _mainSeq.Decoder.DecodeSpecialTokens = _mainSeq.InferParams.DecodeSpecialTokens;
                _mainSeq.TokensToPrefill.Clear(); //на всякий

                while (_mainSeq.InferState.State == SeqState.Generation)
                {
                    if (_generationToken.IsCancellationRequested) break;

                    #region Sampling
                    if (_mainSeq.LastLogits == null) throw new Exception($"sequence {_mainSeq.Id} is empty");

                    LLamaToken llamaToken;
                    using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)_mainSeq.LastLogits, out var native))
                    {
                        _mainSeq.InferParams.SamplingPipeline.Apply(ref native);
                        llamaToken = native.Data[(int)native.Selected].ID;
                        _mainSeq.InferParams.SamplingPipeline.Accept(llamaToken);
                    }

                    _mainSeq.TokensToPrefill.Add(llamaToken); // добавляем в контейнер для декода моделью только что выбранный токен
                    #endregion

                    #region Decode
                    LLamaBatch batch = new LLamaBatch();
                    batch.Add(llamaToken, _mainSeq.NextDecodedTokenPos, _mainSeq.Id, true);

                    DecodeResult res = await Context.DecodeAsync(batch);


                    if (res != DecodeResult.Ok)
                    {
                        //по сути состояние не сбивается, можно пробовать снова без восстановления
                        throw new LLamaDecodeError(res);
                    }

                    _mainSeq.NextDecodedTokenPos++;
                    _mainSeq.LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(batch.TokenCount - 1));
                    _mainSeq.DecodedTokens.Add(_mainSeq.TokensToPrefill.Last()); //записываем токены embeds как отдекодированные
                    _mainSeq.TokensToPrefill.Clear();
                    _mainSeq.InferState.TokenSampledAndDecoded = true; // надо делать постпроцессинг
                    _mainSeq.InferState.RemainedTokens--; // уменьшить колво оставшихся для генерации токенов
                    #endregion

                    string decoded = postProcessInfer();
                    yield return decoded;
                }
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        public void StopGeneration()
        {
            _generationToken?.Cancel();
            _generationToken?.Dispose();
            _generationToken = new CancellationTokenSource();
        }

        public void Dispose()
        {
            _generationToken?.Cancel();
            _executorLifeToken?.Cancel();
            _generationToken?.Dispose();
            _seqStateSemaphore?.Dispose();
            _executorLifeToken?.Dispose();
            Context?.Dispose();

        }


        /// <summary>
        /// Может пригодиться для клонирования контекста между последовательностями
        /// </summary>
        /// <param name="seqId"></param>
        /// <returns></returns>
        public async Task<int> GetNextDecodedTokenPos()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                return _mainSeq.NextDecodedTokenPos;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="seqId"></param>
        /// <returns></returns>
        public async Task<IReadOnlyList<LLamaToken>?> GetDecodedTokens()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                return _mainSeq.DecodedTokens;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="seqId"></param>
        /// <returns></returns>
        public async Task<LLamaTokenDataArray?> GetLastLogits()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                return _mainSeq.LastLogits;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        public string SampleOneTokenWithoutDecode(IInferenceParams inferenceParams)
        {
            if (_mainSeq.LastLogits == null) throw new Exception($"sequence {_mainSeq.Id} is empty");

            LLamaToken llamaToken;
            using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)_mainSeq.LastLogits, out var native))
            {
                _mainSeq.InferParams.SamplingPipeline.Apply(ref native);
                llamaToken = native.Data[(int)native.Selected].ID;
                //без Accept токена
            }
            string textPiece = "";
            _mainSeq.Decoder.Add(llamaToken);
            textPiece = _mainSeq.Decoder.Read();
            return textPiece;
        } 
        // полезно в случае, когда контекст имеет вид [данные]+[вопрос с требованием печати одного токена да/нет и т.п.] мы получаем ответ с помощью sample
        ////без его обработки декодом, далее удаляем блок с вопросом (с помощью функции удаления с изменением текущей позиции для новых токенов)
        ////и можем добавлять новые данные в [данные] для новых вопросов. Быстро так как большая часть контекста обработана

    }
}
