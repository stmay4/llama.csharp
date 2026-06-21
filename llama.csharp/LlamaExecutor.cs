using Llama.csharp.Exceptions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System.Threading.Channels;

namespace Llama.csharp
{
    public class LlamaExecutor : IDisposable
    {
        /// <summary>
        /// The context used by the executor.
        /// </summary>
        public LLamaContext Context { get; }

        /// <summary>
        /// Package Composer
        /// Composes packages from tokens for batch processing of different context sequences
        /// Shared resource. Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private IBatchComposer _batchComposer;

        /// <summary>
        /// All currently active context sequences
        /// Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private Dictionary<LLamaSeqId, Sequence> _sequences = new Dictionary<LLamaSeqId, Sequence>();

        /// <summary>
        /// Container for sequences currently in the generation phase
        /// Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private Dictionary<Sequence, Channel<string>> _inferenceSeqs = new Dictionary<Sequence, Channel<string>>();

        /// <summary>
        /// Container for sequences in the prefill phase
        /// Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private Dictionary<Sequence, TaskCompletionSource> _prefillSeqs = new Dictionary<Sequence, TaskCompletionSource>();

        /// <summary>
        /// Counter for sequence IDs to assign unique identifiers
        /// Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private int _nextSequenceId = 0;

        /// <summary>
        /// Pool of freed (removed) sequences
        /// Data modification and retrieval only under _seqStateSemaphore
        /// </summary>
        private readonly Stack<LLamaSeqId> _freeIds = [];

        private readonly CancellationTokenSource _executorLifeToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _seqStateSemaphore = new SemaphoreSlim(1);

        /// <summary>
        /// Signal for the endless decode loop
        /// Set and Reset only under _seqStateSemaphore
        /// </summary>
        private readonly ManualResetEventSlim _workSignal = new ManualResetEventSlim(false);

        internal LlamaExecutor(LLamaContext context)
        {
            Context = context;
            _batchComposer = new SimpleBatchComposer();
            _ = DecodingLoop(); //start endless decode loop
        }

        /// <summary>
        /// Add new Sequence
        /// </summary>
        /// <returns> created sequence Id, -1 if out of SeqMax </returns>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<LLamaSeqId> CreateSequence()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                // If there are free IDs, use them
                if (_freeIds.TryPop(out var freeId))
                {
                    _sequences[freeId] = new Sequence(Context)
                    {
                        Id = freeId,
                        AntipromptProc = new AntipromptProcessor()
                    };
                    return freeId;
                }

                if (_nextSequenceId >= Context.Params.SeqMax) 
                    return (LLamaSeqId)(-1);

                LLamaSeqId sequenceId = (LLamaSeqId)_nextSequenceId;
                _sequences[sequenceId] = new Sequence(Context)
                {
                    Id = sequenceId,
                    AntipromptProc = new AntipromptProcessor()
                };
                _nextSequenceId++;
                return sequenceId;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Delete Sequence
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task DeleteSequence(LLamaSeqId id)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(id, out var seq))
                {
                    Context.NativeHandle.SeqMemoryRemoveAll(id);

                    _sequences.Remove(id);
                    _freeIds.Push(id); // Return ID to the pool of free IDs
                }
                else throw new IndexOutOfRangeException($"sequence {id} not exist");
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Tokenization of input before sending to sequence prefill
        /// </summary>
        /// <param name="text"></param>
        private List<LLamaToken> preprocessInputs(string text, bool addBos, bool special)
        {
            return Context.Tokenize(text, addBos, special).ToList();
        }

        /// <summary>
        /// Overloaded method for sending to prefill for convenience when appending text to a single sequence
        /// </summary>
        /// <param name="seqId"> ID of the sequence to append text to </param>
        /// <param name="text"> Text to add </param>
        /// <param name="addBos"> Whether to add the BOS token </param>
        /// <param name="special"> Whether to tokenize specialized tokens </param>
        /// <returns> Task for appending text to the sequence </returns>
        public async Task ProcessPrompt(LLamaSeqId seqId, string text, bool addBos = false, bool special = true)
        {
            Dictionary<LLamaSeqId, Task> completions = await ProcessPrompt([seqId], [text], [addBos], [special]);

            await completions[seqId];
        }

        /// <summary>
        /// Overloaded method for sending a batch of prefills with the same addBos and special settings
        /// </summary>
        /// <param name="seqIds"> List of sequences to append text to </param>
        /// <param name="texts"> List of texts to add, the first text to the first sequence, the second to the second, and so on </param>
        /// <param name="addBos"> Whether to add the BOS token </param>
        /// <param name="special"> Whether to tokenize specialized tokens </param>
        /// <returns> Dictionary with tasks for filling each sequence  </returns>
        public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, bool addBos = false, bool special = true)
        {
            List<bool> addBOS = new List<bool>();
            List<bool> useSpecialTokens = new List<bool>();

            for (int i = 0; i < seqIds.Count; i++)
            {
                addBOS.Add(addBos);
                useSpecialTokens.Add(special);
            }

            Dictionary<LLamaSeqId, Task> completions = await ProcessPrompt(seqIds, texts, addBOS, useSpecialTokens);
            return completions;
        }

        /// <summary>
        /// Main method for prefilling sequences
        /// </summary>
        /// <param name="seqIds"> List of sequences to append text to </param>
        /// <param name="texts"> List of texts to add, the first text to the first sequence, the second to the second, and so on </param>
        /// <param name="addBos"> List of addBos settings for each sequence being filled </param>
        /// <param name="special"> List of special settings for each sequence being filled </param>
        /// <returns> Dictionary with tasks for filling each sequence </returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="IndexOutOfRangeException"> The sequence {seqId} does not exist </exception>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, List<bool> addBos, List<bool> special)
        {
            // Simple argument checks, not under semaphore synchronization
            #region Checks 
            if (seqIds == null || texts == null || addBos == null || special == null) throw new ArgumentNullException();
            if (seqIds.Count == 0 || texts.Count == 0 || addBos.Count == 0 || special.Count == 0) throw new ArgumentException("Count is 0");
            if (texts.Count < seqIds.Count) throw new Exception("There are few texts");
            if (addBos.Count < seqIds.Count) throw new Exception("There are few addBos");
            if (special.Count < seqIds.Count) throw new Exception("There are few special");
            #endregion

            Dictionary<LLamaSeqId, Task> completions = new Dictionary<LLamaSeqId, Task>();
            Dictionary<LLamaSeqId, List<LLamaToken>?> tokenizedTexts = new Dictionary<LLamaSeqId, List<LLamaToken>?>();
            //tokenization
            for (int i = 0; i < seqIds.Count; i++)
            {
                if (string.IsNullOrEmpty(texts[i]))
                    tokenizedTexts[seqIds[i]] = null;
                else
                    tokenizedTexts[seqIds[i]] = preprocessInputs(texts[i], addBos[i], special[i]);
            }
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //semaphore synchronization
            try
            {
                foreach (LLamaSeqId seqId in seqIds)
                {
                    // Parameter checks inside the lock because resources are shared
                    if (!_sequences.TryGetValue(seqId, out var seq)) throw new IndexOutOfRangeException($"There is not {seqId} sequence");
                    if (seq.InferState.State != SeqState.None) throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
                    if (tokenizedTexts[seqId] != null)
                    {
                        TaskCompletionSource tcs = new TaskCompletionSource();
                        completions[seqId] = tcs.Task; // set prefill Task in completions for return from method

                        // Check for available space in the context cache
                        if (isOutOfContext(tokenizedTexts[seqId].Count))
                        {
                            tcs.SetException(new ContextFullException()); // Set an error in the task if the context is full
                        }
                        else
                        {
                            // Filling the TokensToPrefill sequence
                            seq.TokensToPrefill.AddRange(tokenizedTexts[seqId]);
                            seq.RealTokensCount += seq.TokensToPrefill.Count; // Adding prefill tokens count to real (not shared with other sequences) tokens count

                            seq.InferState.State = SeqState.Prefill; // Setting the sequence state to prefill

                            _workSignal.Set(); // Setting the work availability signal; the decode loop will wait for the exit from this method's lock

                            _prefillSeqs[seq] = tcs; // Adding to _prefillSeqs for processing in the decode loop
                        }
                    }
                    else // If there's nothing to process, return a completed task
                    {
                        TaskCompletionSource tcs = new TaskCompletionSource();
                        completions[seqId] = tcs.Task;
                        tcs.SetResult();
                    }
                }
            }
            finally
            {
                _seqStateSemaphore.Release();
            }

            return completions;
        }

        /// <summary>
        /// Overloaded method for generating one sequence
        /// </summary>
        /// <param name="seqId"></param>
        /// <param name="inferenceParam"></param>
        /// <returns> Channel<string> for sequence generation </returns>
        public async Task<Channel<string>> Generate(LLamaSeqId seqId, IInferenceParams inferenceParam)
        {
            Dictionary<LLamaSeqId, Channel<string>> channels = await Generate([seqId], [inferenceParam]);
            return channels[seqId];
        }

        /// <summary>
        /// Main method for generating sequences
        /// </summary>
        /// <param name="seqIds"> List of sequences to generate </param>
        /// <param name="inferenceParams"> List of generation parameters for each sequence </param>
        /// <returns> Dictionary of Channels<string> with sequence identifiers as keys </returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<Dictionary<LLamaSeqId, Channel<string>>> Generate(List<LLamaSeqId> seqIds, List<IInferenceParams> inferenceParams)
        {
            // Simple argument checks, not under semaphore synchronization
            #region Checks 
            if (seqIds == null || inferenceParams == null) throw new ArgumentNullException();
            if (seqIds.Count == 0 || inferenceParams.Count == 0) throw new ArgumentException("Count is 0");
            if (inferenceParams.Count < seqIds.Count) throw new Exception("There are few inferenceParams");
            #endregion

            Dictionary<LLamaSeqId, Channel<string>> channels = new Dictionary<LLamaSeqId, Channel<string>>();

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                int inferenceParamsCounter = 0;
                foreach (LLamaSeqId seqId in seqIds)
                {
                    _sequences.TryGetValue(seqId, out var seq);

                    // Parameter checks inside the lock because resources are shared
                    if (seq == null) throw new IndexOutOfRangeException($"There is not {seqId} sequence");
                    if (seq.InferState.State != SeqState.None) throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
                    if (seq.InferParams.MaxTokens == 0) throw new Exception($"For prefill use ProcessPrompt");

                    //sequence data generation init
                    seq.InferState = new InferStateArgs
                    {
                        State = SeqState.Generation,
                        RemainedTokens = inferenceParams[inferenceParamsCounter].MaxTokens,
                        AutoStopFromEOG = inferenceParams[inferenceParamsCounter].AutoStopFromEOG
                    };
                    seq.InferParams = inferenceParams[inferenceParamsCounter];
                    seq.AntipromptProc.ClearString();
                    seq.AntipromptProc.SetAntiprompts(inferenceParams[inferenceParamsCounter].AntiPrompts ?? []);
                    seq.Decoder.DecodeSpecialTokens = seq.InferParams.DecodeSpecialTokens;
                    seq.TokensToPrefill.Clear(); //на всякий

                    // create channel for return generation
                    var channel = Channel.CreateUnbounded<string>();
                    _inferenceSeqs[seq] = channel;
                    channels[seq.Id] = channel;

                    _workSignal.Set(); // Setting the work availability signal; the decode loop will wait for the exit from this method's lock

                    inferenceParamsCounter++; // Counter for iterating through inferenceParams
                }
            }
            finally
            {
                _seqStateSemaphore.Release();
            }

            return channels;
        }

        //TRUE здесь заканчивает генерацию
        private async Task postProcessInfer() //проверка, что пора заканчивать
        {
            List<(Sequence, Channel<string>)> seqsToRemove = new List<(Sequence, Channel<string>)>();
            foreach (var seq in _inferenceSeqs)
            {
                if (seq.Key.InferState.TokenSampledAndDecoded) //только обработанные в текущем батче
                {
                    seq.Key.InferState.TokenSampledAndDecoded = false;

                    // Преобразуем токен в строку
                    seq.Key.Decoder.Add(seq.Key.DecodedTokens.Last());
                    var decoded = seq.Key.Decoder.Read();

                    bool forRemoving = false;

                    try
                    {
                        await seq.Value.Writer.WriteAsync(decoded, _executorLifeToken.Token); //запись в канал последовательности
                    }
                    catch (Exception e)
                    {
                        //если чтото пошло не так - убираем, чтобы не мешало
                        forRemoving = true;
                    }


                    // если сгенерирован антипромпт
                    if (!string.IsNullOrEmpty(decoded) && seq.Key.AntipromptProc.Add(decoded))
                    {
                        forRemoving = true;
                    }

                    if (seq.Key.InferState.AutoStopFromEOG)
                    {
                        // если сгенерирован конец
                        if (seq.Key.DecodedTokens.Last().IsEndOfGeneration(Context.Vocab))
                        {
                            forRemoving = true;
                        }
                    }

                    // если кончились токены
                    if (seq.Key.InferState.RemainedTokens <= 0 && seq.Key.InferParams.MaxTokens != -1)
                    {
                        forRemoving = true;
                    }

                    if (forRemoving) seqsToRemove.Add((seq.Key, seq.Value));
                }
            }
            //удаляем те, что закончили выполнение из контейнера
            foreach ((Sequence, Channel<string>) seq in seqsToRemove)
            {
                endInference(seq);
            }
        }

        /// <summary>
        /// Освобождаем статус подпоследовательности от генерации и закрываем канал для записи новой инфы
        /// </summary>
        /// <param name="seq"></param>
        /// <param name="ch"></param>
        private void endInference((Sequence seq, Channel<string> ch) inferenceSeq)
        {
            _inferenceSeqs.Remove(inferenceSeq.seq);
            inferenceSeq.seq.InferState.State = SeqState.None;
            inferenceSeq.ch.Writer.Complete();
        }

        private void postProcessPrefill() //проверка, что пора заканчивать
        {
            List<(Sequence, TaskCompletionSource)> seqsToRemove = new List<(Sequence, TaskCompletionSource)>();
            foreach (var seq in _prefillSeqs)
            {
                if (seq.Key.TokensToPrefill.Count == 0)
                {
                    seqsToRemove.Add((seq.Key, seq.Value));
                }
            }
            //удаляем те, что закончили выполнение из контейнера
            foreach ((Sequence, TaskCompletionSource) seq in seqsToRemove)
            {
                endPrefill(seq);
            }
        }

        /// <summary>
        /// Освобождаем статус подпоследовательности от заполнения и отправляем сигнал о конце обработки
        /// </summary>
        /// <param name="seq"></param>
        /// <param name="ch"></param>
        private void endPrefill((Sequence seq, TaskCompletionSource tcs) prefillSeq)
        {
            _prefillSeqs.Remove(prefillSeq.seq);
            prefillSeq.seq.InferState.State = SeqState.None;
            prefillSeq.tcs.SetResult();
        }

        /// <summary>
        /// Логика выбывания из декода не здесь. Снаружи меняем тех, кто остается в sequences
        /// </summary>
        /// <param name="inferSeqs"></param>
        /// <param name="inferenceParams"></param>
        /// <returns></returns>
        /// <exception cref="LLamaDecodeError"></exception>
        private async Task decodeInternal(List<Sequence> inferSeqs, List<Sequence> prefillSeqs)
        {
            #region SamplingInferenceSeqs
            List<Sequence> seqsToRemove = new List<Sequence>();
            foreach (Sequence seq in inferSeqs)
            {
                //    const auto * logits = llama_get_logits_ith(ctx, idx);
                //    llama_token_data_array cur_p = { ... init from logits ... };
                //    llama_sampler_apply(smpl, cur_p);
                //    auto token = cur_p.data[cur_p.selected].id;
                //    llama_sampler_accept(smpl, token);
                //    return token;

                LLamaToken llamaToken;

                if (seq.TokensToPrefill.Count == 0 && seq.LastLogits != null) //все токены обработаны
                {
                    seq.EndDeleted = false;

                    using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)seq.LastLogits, out var native))
                    {
                        seq.InferParams.SamplingPipeline.Apply(ref native);
                        llamaToken = native.Data[(int)native.Selected].ID;
                        seq.InferParams.SamplingPipeline.Accept(llamaToken);
                    }
                }
                else if (seq.EndDeleted) //последних логитов нет, если конец удален, поэтому берем тот же токен, что модель выбирала до этого
                {
                    llamaToken = seq.DeletedNextToken;
                }// хранить все логиты не стоит, много весят, так что придется смириться с тем, что первый токен повторяется
                else throw new Exception($"sequence {seq.Id} LastLogits is empty. Generation impossible");

                if (isOutOfContext(1)) //так как llamatoken один, для MTP поменять в будущем
                {
                    seqsToRemove.Add(seq); // удаляем из последовательностей на сборку пакета, так как нет места в контексте
                }
                else
                {
                    seq.TokensToPrefill.Add(llamaToken); // добавляем в контейнер для декода моделью только что выбранный токен

                    seq.RealTokensCount += seq.TokensToPrefill.Count(); //Добавляем отсемплированное в реальные токены
                }   
            }
            foreach (Sequence seq in seqsToRemove)
            {
                Channel<string> ch = _inferenceSeqs[seq];

                inferSeqs.Remove(seq); //удаляем в том числе из текущего листа на сборку пакета

                endInference((seq,ch));
            }
            #endregion

            //Compose Batch
            LLamaBatch multiSeqBatch = _batchComposer.CreateBatch(inferSeqs, prefillSeqs, Context.BatchSize);

            // Decode
            DecodeResult res = await Context.DecodeAsync(multiSeqBatch); // Один батч состоит из токенов для инференса в начале и токенов для префилла в конце

            IReadOnlyDictionary<int, int> decodedSeqsListIds = _batchComposer.GetInferSeqsListIdsAndPos();
            IReadOnlyDictionary<int, (int count, int pos)> prefilledSeqsTokenCount = _batchComposer.GetPrefillSeqsTokenCount();

            // ERROR
            if (res != DecodeResult.Ok)
            {
                foreach (var item in decodedSeqsListIds)
                {
                    //ALL BAD
                }

                throw new LLamaDecodeError(res);
            }

            //Update Data
            foreach (var item in decodedSeqsListIds) //участвовавшие в пакете последовательности инференса
            {
                inferSeqs[item.Key].NextDecodedTokenPos++;
                inferSeqs[item.Key].LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(item.Value));
                inferSeqs[item.Key].DecodedTokens.Add(inferSeqs[item.Key].TokensToPrefill.Last()); //записываем токены embeds как отдекодированные
                inferSeqs[item.Key].TokensToPrefill.Clear();
                inferSeqs[item.Key].InferState.TokenSampledAndDecoded = true; // надо делать постпроцессинг (выводить в канал и проверять на конец)
                inferSeqs[item.Key].InferState.RemainedTokens--; // уменьшить колво оставшихся для генерации токенов
            }
            foreach (var item in prefilledSeqsTokenCount) //участвовавшие в пакете последовательности префила
            {
                prefillSeqs[item.Key].NextDecodedTokenPos += item.Value.count;
                prefillSeqs[item.Key].DecodedTokens.AddRange(prefillSeqs[item.Key].TokensToPrefill.GetRange(0, item.Value.count));
                prefillSeqs[item.Key].TokensToPrefill.RemoveRange(0, item.Value.count);
                if (prefillSeqs[item.Key].TokensToPrefill.Count == 0) // все из Embeds отдекодены?
                {
                    prefillSeqs[item.Key].LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(item.Value.pos));
                }
            }
        }

        /// <summary>
        /// Вечный цикл обработки батчей
        /// </summary>
        /// <returns></returns>
        private async Task DecodingLoop()
        {
            while (!_executorLifeToken.Token.IsCancellationRequested) // один проход включает генерацию одного токена где надо, если надо и префилл сколько поместится в батч до размера батча
            {
                await _workSignal.WaitAsync(_executorLifeToken.Token);//проверка isset с быстрым возвратом внутри waitasync

                await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //блокировка для работы с контейнерами _inferenceSeqs и _prefillSeqs
                try
                {
                    if (_inferenceSeqs.Count == 0 && _prefillSeqs.Count == 0) //если делать нечего
                    {
                        _workSignal.Reset(); //сбрасываем сигнал о наличии работы
                    }
                    else
                    {
                        await decodeInternal(_inferenceSeqs.Keys.ToList(), _prefillSeqs.Keys.ToList()); // в листы для удобства индексирования

                        await postProcessInfer();
                        postProcessPrefill();
                    }
                }
                finally
                {
                    _seqStateSemaphore.Release();
                }
            }
        }

        /// <summary>
        /// Последовательности в которые копируем будут полностью очищены
        /// </summary>
        /// <param name="srcId"></param>
        /// <param name="targetIds"></param>
        /// <param name="endPos"> не включительно</param>
        /// <returns></returns>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <exception cref="Exception"> seq State</exception>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task CopySeqPrefixTo(LLamaSeqId srcId, List<LLamaSeqId> targetIds, LLamaPos endPos)
        {
            if (targetIds == null) throw new ArgumentNullException();
            if (targetIds.Count == 0) throw new ArgumentException("Count is 0");
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(srcId, out var srcSeq))
                {
                    if (srcSeq.InferState.State != SeqState.None)
                        throw new Exception($"{srcId} sequence using in another place: {srcSeq.InferState.State}");
                    foreach (LLamaSeqId id in targetIds)
                    {
                        if (!_sequences.TryGetValue(id, out var seq))
                            throw new IndexOutOfRangeException($"sequence {id} not exist");
                        else
                        {
                            if (seq.InferState.State != SeqState.None)
                                throw new Exception($"{id} sequence using in another place: {seq.InferState.State}");
                        }
                    }

                    if ((int)endPos < 1) throw new ArgumentException("End position must be 1 or greater");

                    if ((int)endPos > srcSeq.NextDecodedTokenPos) throw new ArgumentException("End position must be lower or equal NextDecodedTokenPos");

                    //копирование
                    foreach (LLamaSeqId id in targetIds)
                    {
                        if (_sequences.TryGetValue(id, out var seq))
                        {
                            if (seq.NextDecodedTokenPos != 0)
                            {
                                Context.NativeHandle.SeqMemoryRemoveAll(id);
                                seq.ClearSequenceTokens();
                            }

                            Context.NativeHandle.SeqMemoryCopy(srcId, id, 0, endPos);
                            seq.CopyStateFrom(srcSeq);
                        }
                    }
                }
                else throw new IndexOutOfRangeException($"sequence {srcId} not exist");
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        public async Task DeleteSequenceEnd(LLamaSeqId seqId, LLamaPos startPos /*включительно*/)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(seqId, out var seq))
                {
                    if (seq.InferState.State != SeqState.None)
                        throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
                    if ((int)startPos >= seq.NextDecodedTokenPos) throw new ArgumentException("Start position must be lower then NextDecodedTokenPos");

                    if (Context.NativeHandle.SeqMemoryRemove(seqId, startPos, -1))
                    {
                        seq.SetNewEnd((int)startPos);
                    }
                    else
                    {
                        throw new Exception($"sequence is empty");
                    }
                }
                else throw new IndexOutOfRangeException($"sequence {seqId} not exist");
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Threade-Safe остановить генерацию указанной последовательности
        /// остановка будет произведена перед следующим шагом декода DecodingLoop
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        /// <exception cref="IndexOutOfRangeException"></exception>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task StopSeqGeneration(LLamaSeqId id)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(id, out var seq))
                {
                    if (_inferenceSeqs.TryGetValue(seq, out var channel))
                    {
                        endInference((seq, channel));
                    }
                    else throw new IndexOutOfRangeException($"sequence {id} not in generate");
                }
                else throw new IndexOutOfRangeException($"sequence {id} not exist");
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Может пригодиться для клонирования контекста между последовательностями
        /// </summary>
        /// <param name="seqId"></param>
        /// <returns></returns>
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<int> GetSequenceNextDecodedTokenPos(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(seqId, out var seq))
                    return seq.NextDecodedTokenPos;
                else throw new IndexOutOfRangeException($"sequence {seqId} not exist");
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
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<IReadOnlyList<LLamaToken>> GetSequenceDecodedTokens(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(seqId, out var seq))
                    return seq.DecodedTokens;
                else throw new IndexOutOfRangeException($"sequence {seqId} not exist");
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
        /// <exception cref="OperationCanceledException"></exception>
        /// <exception cref="ObjectDisposedException"></exception>
        public async Task<LLamaTokenDataArray?> GetSequenceLastLogits(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(seqId, out var seq))
                    return seq.LastLogits;
                else throw new IndexOutOfRangeException($"sequence {seqId} not exist");
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Метод для получения количества реальных токенов в контексте
        /// </summary>
        /// <returns></returns>
        public async Task<int> GetContextFilling()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                return getContextFilling();
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// подсчет количества использованных KV мест
        /// ВЫЗЫВАТЬ ТОЛЬКО ИЗ ПОД БЛОКИРОВКИ
        /// </summary>
        /// <returns></returns>
        private int getContextFilling() // надо понимать что проверка должна быть как при инициации префила, где нам уже ясно количество токенов префила,
                                        // так и при составлении батча в условно бесконечной генерации. Проверку в батче можно делать с условием не DecodedTokens, а планов на префилл
                                        // или пусть пользователь сам следит, а код просто стопается при конце места, но что делать с состоянием - оно сломается, если забить на конец
        {
            int count = 0;
            foreach (Sequence seq in _sequences.Values)
            {
                count += seq.RealTokensCount;
            }
            return count;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="count"> колво добавляемых токенов</param>
        /// <exception cref="ContextFullException"></exception>
        private bool isOutOfContext(int count)
        {
            if (getContextFilling() + count > Context.ContextSize)
            {
                return true;
            }
            return false;
        }

        public void Dispose()
        {
            _executorLifeToken?.Cancel();
            _workSignal?.Dispose();
            _seqStateSemaphore?.Dispose();
            _executorLifeToken?.Dispose();
            Context?.Dispose();
        }

        private interface IBatchComposer
        {
            /// <summary>
            /// Создать батч
            /// </summary>
            /// <param name="inferSeqs"></param>
            /// <param name="prefillSeqs"></param>
            /// <returns></returns>
            public LLamaBatch CreateBatch(IReadOnlyList<Sequence> inferSeqs, IReadOnlyList<Sequence> prefillSeqs, uint batchSize);
            /// <summary>
            /// Вернуть id в листе inferSeqs последовательностей, которые были в последнем батче для их постобработки
            /// </summary>
            /// <returns></returns>
            public IReadOnlyDictionary<int, int> GetInferSeqsListIdsAndPos();
            /// <summary>
            /// Количество обработанных батчем токенов префилла в последовательностях
            /// </summary>
            /// <returns></returns>
            public IReadOnlyDictionary<int, (int count, int pos)> GetPrefillSeqsTokenCount();
        }
        private class SimpleBatchComposer : IBatchComposer
        {
            private int _nextInferSeqBatchId = 0;
            private int _nextPrefillSeqBatchId = 0;
            private Dictionary<int, int> _lastInferSeqs = new Dictionary<int, int>();
            private Dictionary<int, (int count, int pos)> _lastPrefilledCount = new Dictionary<int, (int count, int pos)>();

            public LLamaBatch CreateBatch(IReadOnlyList<Sequence> inferSeqs, IReadOnlyList<Sequence> prefillSeqs, uint batchSize)
            {
                // очистка
                _lastInferSeqs.Clear();
                _lastPrefilledCount.Clear();

                LLamaBatch batch = new LLamaBatch(); // новый батч

                int batchPos = 0; //текущая позиция в батче
                int currentInferSeqId = 0; //id в inferSeqs

                for (int i = 0; i < inferSeqs.Count; i++)
                {
                    if (batchPos < batchSize)
                    {
                        currentInferSeqId = _nextInferSeqBatchId + i;
                        currentInferSeqId = currentInferSeqId % inferSeqs.Count;

                        batch.Add(inferSeqs[currentInferSeqId].TokensToPrefill.Last(), inferSeqs[currentInferSeqId].NextDecodedTokenPos, inferSeqs[currentInferSeqId].Id, true);
                        _lastInferSeqs[currentInferSeqId] = batchPos;
                        batchPos++; //увеличиваем значение позиции в батче
                    }
                    else
                    {
                        _nextInferSeqBatchId = currentInferSeqId + 1; // сохраняем, чтобы в след батче начать с этой позиции, если последовательностей больше чем размер батча
                        break;
                    }
                }


                Dictionary<int, int> currentRequiredLogitsIds = new Dictionary<int, int>(); // позиции на которых надо ставить Logits в true
                for (int k = 0; k < prefillSeqs.Count; k++)
                {
                    currentRequiredLogitsIds[k] = prefillSeqs[k].NextDecodedTokenPos + prefillSeqs[k].TokensToPrefill.Count - 1;
                }

                int currentPrefillSeqId = 0; // id текушей обрабатываемой последовательности из prefillSeqs

                bool TokenAdded = true;

                while (batchPos < batchSize && TokenAdded)
                {
                    TokenAdded = false;
                    for (int i = 0; i < prefillSeqs.Count; i++)
                    {
                        if (batchPos < batchSize)
                        {
                            currentPrefillSeqId = _nextPrefillSeqBatchId + i;
                            currentPrefillSeqId = currentPrefillSeqId % prefillSeqs.Count;

                            if (!_lastPrefilledCount.ContainsKey(currentPrefillSeqId)) _lastPrefilledCount[currentPrefillSeqId] = (0, -1); // инициализация, если последовательность еще не обрабатывалась

                            if (_lastPrefilledCount[currentPrefillSeqId].count < prefillSeqs[currentPrefillSeqId].TokensToPrefill.Count)
                            {

                                int count = _lastPrefilledCount[currentPrefillSeqId].count; //колво загруженных в батч в данном цикле
                                int pos = prefillSeqs[currentPrefillSeqId].NextDecodedTokenPos + count; // позиция текущего загружаемого токена в последовательности, расчитанная из последней + колво

                                batch.Add(
                                    prefillSeqs[currentPrefillSeqId].TokensToPrefill[count],
                                    pos,
                                    prefillSeqs[currentPrefillSeqId].Id,
                                    pos == currentRequiredLogitsIds[currentPrefillSeqId]
                                    ); //добавляем токен последовательности
                                TokenAdded = true;
                                _lastPrefilledCount[currentPrefillSeqId] = (count + 1, batchPos); //обновляем колво загруженных в батч в данном цикле

                                batchPos++; //увеличиваем значение позиции в батче
                            }
                        }
                        else
                        {
                            _nextPrefillSeqBatchId = currentPrefillSeqId + 1;
                            break; //покинуть если место закончилось
                        }
                    }
                }


                return batch;
            }

            public IReadOnlyDictionary<int, int> GetInferSeqsListIdsAndPos()
            {
                return _lastInferSeqs;
            }

            public IReadOnlyDictionary<int, (int count, int pos)> GetPrefillSeqsTokenCount()
            {
                return _lastPrefilledCount;
            }
        }
    }
}
