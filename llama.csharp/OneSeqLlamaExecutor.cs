using Llama.csharp.Exceptions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public class OneSeqLlamaExecutor : LlamaExecutorBase
    {
        public class LlamaExecutorState : LlamaBaseState
        {

            //     For preprocess all prompt in some count of batches (if prompt is long)
            public bool IsPromptRun { get; set; }
        }

        LLamaBatch _lastBatch = new LLamaBatch();

        //
        // Параметры:
        //   context:
        //
        //   logger:
        public OneSeqLlamaExecutor(LLamaContext context)//, ILogger? logger = null)
            : base(context)//, logger)
        {
        }

        public override LlamaBaseState GetStateData()
        {
            LlamaExecutorState state = new()
            {
                ConsumedSessionCount = _n_session_consumed,
                Embeds = _embeds.ToArray(),
                LastTokens = _last_n_tokens.ToArray(),
                MatchingSessionTokensCount = _n_matching_session_tokens,
                PastTokensCount = _pastTokensCount,
                SessionFilePath = _pathSession,
                SessionTokens = _session_tokens.ToArray(),
                LastTokensCapacity = _last_n_tokens.Capacity,
            };
            return state;
        }

        protected override void PreprocessInputs(string? text, bool addBos, bool special)
        {
            if (text != null)
            {
                LLamaToken[] array = Context.Tokenize(text, addBos, special);
                _embeds.AddRange(array);
            }
        }

        protected override async Task ProcessInputs(IInferenceParams inferenceParams) // заменить inferenceParams на ContextOverflowParams
        {
            for (int i = 0; i < _embeds.Count; i++)
            {
                _last_n_tokens.Enqueue(_embeds[i]);
            }

            if (_embeds.Count > 0)
            {
                //проверка выхода за пределы контекста
                if (_pastTokensCount + _embeds.Count > Context.ContextSize)
                {
                    //int tokensKeep = inferenceParams.TokensKeep;
                    
                    //HandleRunOutOfContext(tokensKeep); //обработка

                    // OR BREAK ЗДЕСЬ
                }
                
                _lastBatch = new LLamaBatch();

                (DecodeResult, int, int) tuple2 = await Context.DecodeAsync(_embeds, LLamaSeqId.Zero, _lastBatch, _pastTokensCount);
                _pastTokensCount = tuple2.Item3;
                if (tuple2.Item1 != 0)
                {
                    throw new LLamaDecodeError(tuple2.Item1);
                }
            }

            _embeds.Clear();
        }

        //
        // Сводка:
        //     Return whether to break the generation.
        //
        // Параметры:
        //   inferenceParams:
        //
        //   args:

        //TRUE здесь заканчивает генерацию
        protected override void PostProcess(IInferenceParams inferenceParams, InferStateArgs args) //проверка, что пора заканчивать
        {

            if (!string.IsNullOrEmpty(args.LastOutput) && AntipromptProcessor.Add(args.LastOutput))
            {
                args.StopGeneration = true;
                //return Task.FromResult<(bool, IReadOnlyList<string>)>((true, []));
            }

            if (args.AutoStopFromEOG)
            {
                if (_embeds.Last().IsEndOfGeneration(Context.Vocab))
                {
                    args.StopGeneration = true;
                }
            }

            if (args.RemainedTokens <= 0 && inferenceParams.MaxTokens != -1)
            {
                //args.RemainedTokens = inferenceParams.MaxTokens;
                args.StopGeneration = true;
            }
        }

        protected override async Task InferenceInternal(IInferenceParams inferenceParams, InferStateArgs args) // заменить на отдельно SAPMLE, отдельно DECODE, и decode поставить после возврата токена
                                                                                                               // для улучшения времени первого токена
        {
            if (_lastBatch.TokenCount == 0) { args.StopGeneration = true; } // последний обработанный батч был пустой, нечего сэмплировать
            else if (!args.StopGeneration)
            {
                LLamaToken llamaToken = inferenceParams.SamplingPipeline.Sample(Context.NativeHandle, _lastBatch.TokenCount - 1); //производим sample на основе итогового внутреннего состояния последнего декодированного токена

                _last_n_tokens.Enqueue(llamaToken); //добавляем токен в качестве отсемплированного для отслеживания в этой оболочке (антипромпт и тп)
                _embeds.Add(llamaToken); // добавляем в контейнер для декода моделью только что выбранный токен

                args.RemainedTokens--;
                args.ReturnValue = true; // токен отсемплирован, есть что возвращать (именно возврат текста, соответствующего токену)
            }

            _lastBatch = new LLamaBatch();

            if (_embeds.Count > 0)
            {
                if (_pastTokensCount + _embeds.Count > Context.ContextSize)
                {
                    //int tokensKeep = inferenceParams.TokensKeep;
                    
                    //HandleRunOutOfContext(tokensKeep);

                    // OR BREAK ЗДЕСЬ
                }

                (DecodeResult, int, int) tuple2 = await Context.DecodeAsync(_embeds, LLamaSeqId.Zero, _lastBatch, _pastTokensCount); // в прямом смысле проход декодера, расчет KV для батча
                _pastTokensCount = tuple2.Item3;
                if (tuple2.Item1 != 0)
                {
                    throw new LLamaDecodeError(tuple2.Item1);
                }
            }
        }
        public LLamaToken? SampleOneTokenWithoutDecode(IInferenceParams inferenceParams) // бесполезный метод, а может и нет. Дает новый токен за 2 мс, если запрос уже обработан. Тогда как InferenceInternal сразу его и обрабатывает, что ухудшает время преовго токена
        {
            LLamaToken? token = null;
            if (_lastBatch.TokenCount > 0)
            {
                token = inferenceParams.SamplingPipeline.Sample(Context.NativeHandle, _lastBatch.TokenCount - 1);
            }
            return token;
        } // полезно в случае, когда контекст имеет вид [данные]+[вопрос с требованием печати одного токена да/нет и т.п.] мы получаем ответ с помощью sample
        //без его обработки декодом, далее удаляем блок с вопросом (с помощью функции удаления с изменением текущей позиции для новых токенов)
        //и можем добавлять новые данные в [данные] для новых вопросов. Быстро так как большая часть контекста обработана
        public Span<float> GetLogits()
        {
            var logits = Context.NativeHandle.GetLogitsIth(_lastBatch.TokenCount - 1);
            return logits;
        }
    }
}
