using Llama.csharp.Exceptions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class LlamaExecutor : IDisposable
    {
        /// <summary>
        /// The context used by the executor.
        /// </summary>
        public LLamaContext Context { get; }

        /// <summary>
        /// Составитель батчей
        /// Изменение и получение данных только под _seqStateSemaphore
        /// </summary>
        private IBatchComposer batchComposer;

        /// <summary>
        /// all life sequences
        /// Изменение и получение данных только под _seqStateSemaphore
        /// </summary>
        private Dictionary<LLamaSeqId, Sequence> _sequences = new Dictionary<LLamaSeqId, Sequence>();
        
        /// <summary>
        /// контейнер последовательностей в стадии генерации
        /// Изменение и получение данных только под _seqStateSemaphore
        /// </summary>
        private Dictionary<Sequence, Channel<string>> _inferenceSeqs = new Dictionary<Sequence, Channel<string>>();

        /// <summary>
        /// контейнер последовательностей в стадии префила
        /// Изменение и получение данных только под _seqStateSemaphore
        /// </summary>
        private Dictionary<Sequence, TaskCompletionSource> _prefillSeqs = new Dictionary<Sequence, TaskCompletionSource>();
        
        /// <summary>
        /// счетчик id последовательностей
        /// Изменение и получение данных только под _seqStateSemaphore
        /// </summary>
        private int _nextSequenceId = 0;

        private readonly CancellationTokenSource _executorLifeToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _seqStateSemaphore = new SemaphoreSlim(1);

        /// <summary>
        /// Сигнал для вечного цикла декода
        /// Set и Reset только под _seqStateSemaphore
        /// </summary>
        private readonly ManualResetEventSlim _workSignal = new ManualResetEventSlim(false);

        internal LlamaExecutor(LLamaContext context)
        {
            Context = context;
            batchComposer = new SimpleBatchComposer();
            _ = DecodingLoop(); //запуск бесконечного цикла обработки
        }

        /// <summary>
        /// Thread-Safe add new Sequence
        /// </summary>
        /// <returns> created sequence Id, -1 if out of SeqMax </returns>
        public async Task<LLamaSeqId> CreateSequence()
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //блокировка для редактирования _prefillSeqs
            try
            {
                if (_nextSequenceId >= Context.Params.SeqMax) return (LLamaSeqId)(-1);

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

        public async Task DeleteSequence(LLamaSeqId id)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //блокировка для возможного редактирования _inferenceSeqs или _prefillSeqs
            try
            {
                // удаление KV и удаление из _inferenceSeqs или _prefillSeqs если там, дефрагментация

                _sequences.Remove(id);
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Preprocess the inputs before the inference for given Sequence.
        /// </summary>
        /// <param name="text"></param>
        private List<LLamaToken> preprocessInputs(string text, bool addBos, bool special)
        {
            return Context.Tokenize(text, addBos, special).ToList();
        }

        public async Task<Task> ProcessPrompt(LLamaSeqId seqId, string text, bool addBos = false, bool special = true)
        {
            Dictionary<LLamaSeqId, Task> completions = await ProcessPrompt([seqId], [text], [addBos], [special]);
            return completions[seqId];
        }
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
        /// Префил подпоследовательностей
        /// </summary>
        /// <param name="seqIds"></param>
        /// <param name="texts"></param>
        /// <param name="addBos"></param>
        /// <param name="special"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, List<bool> addBos, List<bool> special)
        {
            // простые проверки аргументов, без блокировки
            #region Checks 
            if (seqIds == null || texts == null || addBos == null || special == null) throw new ArgumentNullException();
            if (seqIds.Count == 0 || texts.Count == 0 || addBos.Count == 0 || special.Count == 0) throw new ArgumentException("Count is 0");
            if (texts.Count < seqIds.Count) throw new Exception("There are few texts");
            if (addBos.Count < seqIds.Count) throw new Exception("There are few addBos");
            if (special.Count < seqIds.Count) throw new Exception("There are few special");
            #endregion

            Dictionary<LLamaSeqId, Task> completions = new Dictionary<LLamaSeqId, Task>();
            Dictionary<LLamaSeqId, List<LLamaToken>?> embeds = new Dictionary<LLamaSeqId, List<LLamaToken>?>();
            //токенизация
            for (int i = 0; i < seqIds.Count; i++)
            {
                if (string.IsNullOrEmpty(texts[i])) 
                    embeds[seqIds[i]] = null;
                else
                    embeds[seqIds[i]] = preprocessInputs(texts[i], addBos[i], special[i]);
            }

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); //блокировка для редактирования _prefillSeqs
            try
            {
                foreach (LLamaSeqId seqId in seqIds)
                {
                    //проверки параметров внутри блокировки, так как ресурсы общие
                    if (!_sequences.TryGetValue(seqId, out var seq)) throw new IndexOutOfRangeException($"There is not {seqId} sequence");
                    if (seq.InferState.State != SeqState.None) throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
                    if (embeds[seqId] != null)
                    {
                        //заполнение Embeds последовательности
                        seq.Embeds.AddRange(embeds[seqId]);

                        seq.InferState.State = SeqState.Prefill; //устанавливаем состояние в префил
                        TaskCompletionSource tcs = new TaskCompletionSource();
                        completions[seqId] = tcs.Task; //ждем завершения именно нашего процесса

                        _prefillSeqs[seq] = tcs; //сохраняем для работы над префиллом в общем батче

                        _workSignal.Set(); // устанавливаем сигнал о работе, при этом цикл декода будет ждать выхода из блокировки данного метода
                    }
                    else //если обрабатывать нечего
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


        public async Task<Channel<string>> Generate(LLamaSeqId seqId, IInferenceParams inferenceParam)
        {
            Dictionary<LLamaSeqId, Channel<string>> channels = await Generate([seqId], [inferenceParam]);
            return channels[seqId];
        }
        
        /// <summary>
        /// Генерация подпоследовательностей
        /// </summary>
        /// <param name="seqIds"></param>
        /// <param name="inferenceParams"></param>
        /// <returns></returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public async Task<Dictionary<LLamaSeqId, Channel<string>>> Generate(List<LLamaSeqId> seqIds, List<IInferenceParams> inferenceParams)
        {
            // простые проверки аргументов, без блокировки
            #region Checks 
            if (seqIds == null || inferenceParams == null) throw new ArgumentNullException();
            if (seqIds.Count == 0 || inferenceParams.Count == 0) throw new ArgumentException("Count is 0");
            if (inferenceParams.Count < seqIds.Count) throw new Exception("There are few inferenceParams");
            #endregion

            Dictionary<LLamaSeqId, Channel<string>> channels = new Dictionary<LLamaSeqId, Channel<string>>();

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (inferenceParams.Count < seqIds.Count) throw new Exception("There are few inferenceParams");

                int inferenceParamsCounter = 0;
                foreach (LLamaSeqId seqId in seqIds)
                {
                    _sequences.TryGetValue(seqId, out var seq);

                    //проверки
                    if (seq == null) throw new IndexOutOfRangeException($"There is not {seqId} sequence");
                    if (seq.InferState.State != SeqState.None) throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
                    if (seq.InferParams.MaxTokens == 0) throw new Exception($"For prefill use ProcessPrompt");

                    //инициализация для инференса
                    seq.InferState = new InferStateArgs
                    {
                        State = SeqState.Generation,
                        RemainedTokens = inferenceParams[inferenceParamsCounter].MaxTokens,
                        AutoStopFromEOG = inferenceParams[inferenceParamsCounter].AutoStopFromEOG
                    };
                    seq.InferParams = inferenceParams[inferenceParamsCounter];
                    seq.AntipromptProc.ClearString();
                    seq.AntipromptProc.SetAntiprompts(inferenceParams[inferenceParamsCounter].AntiPrompts ?? []);
                    seq.Embeds.Clear(); //на всякий

                    //создание канала куда возвращаем генерацию данной последовательности
                    var channel = Channel.CreateUnbounded<string>();
                    _inferenceSeqs[seq] = channel;
                    channels[seq.Id] = channel;

                    _workSignal.Set(); // устанавливаем сигнал о работе, при этом цикл декода будет ждать выхода из блокировки данного метода

                    inferenceParamsCounter++; //счетчик для перебора inferenceParams
                }
            }
            finally
            {
                _seqStateSemaphore.Release();
            }

            return channels; //словарь каналов с индексами id
        }

        //TRUE здесь заканчивает генерацию
        private async Task postProcessInfer() //проверка, что пора заканчивать
        {
            List<(Sequence, Channel<string>)> seqsToRemove = new List<(Sequence, Channel<string>)>();
            foreach (var seq in _inferenceSeqs) //chanell.Complete
            {
                if (seq.Key.InferState.TokenSampledAndDecoded) //только обработанные в текущем батче
                {
                    seq.Key.InferState.TokenSampledAndDecoded = false;

                    // Преобразуем токен в строку
                    seq.Key.Decoder.DecodeSpecialTokens = seq.Key.InferParams.DecodeSpecialTokens;
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
                _inferenceSeqs.Remove(seq.Item1);
                endInferenceChannel(seq);
            }
        }

        /// <summary>
        /// Освобождаем статус подпоследовательности от генерации и закрываем канал для записи новой инфы
        /// </summary>
        /// <param name="seq"></param>
        /// <param name="ch"></param>
        private void endInferenceChannel((Sequence seq, Channel<string> ch) inferenceSeq)
        {
            inferenceSeq.seq.InferState.State = SeqState.None;
            inferenceSeq.ch.Writer.Complete();
        }

        private void postProcessPrefill() //проверка, что пора заканчивать
        {
            List<(Sequence, TaskCompletionSource)> seqsToRemove = new List<(Sequence, TaskCompletionSource)>();
            foreach (var seq in _prefillSeqs) //chanell.Complete
            {
                if (seq.Key.Embeds.Count == 0)
                {
                    seqsToRemove.Add((seq.Key, seq.Value));
                }
            }
            //удаляем те, что закончили выполнение из контейнера
            foreach ((Sequence, TaskCompletionSource) seq in seqsToRemove)
            {
                _prefillSeqs.Remove(seq.Item1);
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
            foreach (Sequence seq in inferSeqs)
            {
                //    const auto * logits = llama_get_logits_ith(ctx, idx);
                //    llama_token_data_array cur_p = { ... init from logits ... };
                //    llama_sampler_apply(smpl, cur_p);
                //    auto token = cur_p.data[cur_p.selected].id;
                //    llama_sampler_accept(smpl, token);
                //    return token;

                if (seq.Embeds.Count == 0) //все токены обработаны
                {
                    if (seq.LastLogits == null) throw new Exception($"sequence {seq.Id} is empty");

                    LLamaToken llamaToken;
                    using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)seq.LastLogits, out var native))
                    {
                        seq.InferParams.SamplingPipeline.Apply(ref native);
                        llamaToken = native.Data[(int)native.Selected].ID;
                        seq.InferParams.SamplingPipeline.Accept(llamaToken);
                    }

                    seq.Embeds.Add(llamaToken); // добавляем в контейнер для декода моделью только что выбранный токен
                }
            }
            #endregion

            //Compose Batch
            LLamaBatch multiSeqBatch = batchComposer.CreateBatch(inferSeqs, prefillSeqs, Context.BatchSize);

            // Decode
            DecodeResult res = await Context.DecodeAsync(multiSeqBatch); // Один батч состоит из токенов для инференса в начале и токенов для префилла в конце

            IReadOnlyDictionary<int, int> decodedSeqsListIds = batchComposer.GetInferSeqsListIdsAndPos();
            IReadOnlyDictionary<int, (int count, int pos)> prefilledSeqsTokenCount = batchComposer.GetPrefillSeqsTokenCount();

            // ERROR
            if (res != 0)
            {
                foreach (var item in decodedSeqsListIds)
                {
                    //ALL BAD
                }

                throw new LLamaDecodeError(res);
            }

            //Update Data
            foreach (var item in decodedSeqsListIds)
            {
                inferSeqs[item.Key].NextDecodedTokenPos++;
                inferSeqs[item.Key].LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(item.Value));
                inferSeqs[item.Key].DecodedTokens.Add(inferSeqs[item.Key].Embeds.Last()); //записываем токены embeds как отдекодированные
                inferSeqs[item.Key].Embeds.Clear();
                inferSeqs[item.Key].InferState.TokenSampledAndDecoded = true; // надо делать постпроцессинг
                inferSeqs[item.Key].InferState.RemainedTokens--; // уменьшить колво оставшихся для генерации токенов
            }
            foreach (var item in prefilledSeqsTokenCount)
            {
                prefillSeqs[item.Key].NextDecodedTokenPos += item.Value.count;
                prefillSeqs[item.Key].DecodedTokens.AddRange(prefillSeqs[item.Key].Embeds.GetRange(0, item.Value.count));
                prefillSeqs[item.Key].Embeds.RemoveRange(0, item.Value.count);
                if (prefillSeqs[item.Key].Embeds.Count == 0)
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
                        await decodeInternal(_inferenceSeqs.Keys.ToList(), _prefillSeqs.Keys.ToList());

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
        /// Threade-Safe остановить генерацию указанной последовательности
        /// остановка будет произведена перед следующим шагом декода DecodingLoop
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        public async Task StopSeqGeneration(LLamaSeqId id)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (_sequences.TryGetValue(id, out var seq))
                {
                    if (_inferenceSeqs.TryGetValue(seq, out var channel))
                    {
                        channel.Writer.Complete();
                        _inferenceSeqs.Remove(seq);
                    }
                }
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
        public async Task<int?> GetSequenceNextDecodedTokenPos(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                _sequences.TryGetValue(seqId, out var seq);
                return seq?.NextDecodedTokenPos;
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
        public async Task<IReadOnlyList<LLamaToken>?> GetSequenceDecodedTokens(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                _sequences.TryGetValue(seqId, out var seq);
                return seq?.DecodedTokens;
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
        public async Task<LLamaTokenDataArray?> GetSequenceLastLogits(LLamaSeqId seqId)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                _sequences.TryGetValue(seqId, out var seq);
                return seq?.LastLogits;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// подсчет количества использованных KV мест
        /// </summary>
        /// <returns></returns>
        private int usedBySeqsContextKVsCount() // надо понимать что проверка должна быть как при инициации префила, где нам уже ясно количество токенов префила,
                                                // так и при составлении батча в условно бесконечной генерации. Проверку в батче можно делать с условием не DecodedTokens, а планов на префилл
                                                // или пусть пользователь сам следит, а код просто стопается при конце места, но что делать с состоянием - оно сломается, если забить на конец
        {
            int count = 0;
            foreach (Sequence seq in _sequences.Values)
            {
                count += seq.DecodedTokens.Count();
            }
            return count;
        }


        /// <summary>
        /// After running out of the context, take some tokens from the original prompt and recompute the logits in batches.
        /// </summary>
        /// <param name="tokensToKeep"></param>
        private void handleRunOutOfContext(int tokensToKeep)
        {
            //// if we run out of context:
            //// - take the tokensToKeep first tokens from the original prompt (via n_past)
            //// - take half of the last (n_ctx - tokensToKeep) tokens and recompute the logits in batches
            //int n_left = _pastTokensCount - tokensToKeep;
            //int n_discard = n_left / 2;

            //NativeApi.llama_kv_self_seq_rm(Context.NativeHandle, LLamaSeqId.Zero, tokensToKeep, tokensToKeep + n_discard);
            //NativeApi.llama_kv_self_seq_add(Context.NativeHandle, LLamaSeqId.Zero, tokensToKeep + n_discard, _pastTokensCount, -n_discard);

            //_pastTokensCount -= n_discard;
            //// stop saving session if we run out of context
            //_pathSession = string.Empty;
        }

        public void Dispose()
        {
            _executorLifeToken.Cancel();
            _executorLifeToken?.Dispose();
            _seqStateSemaphore.Dispose();
            Context?.Dispose();
        }

        /// <summary>
        /// State arguments that are used in single inference
        /// </summary>
        private class InferStateArgs
        {
            /// <summary>
            /// Lock help
            /// </summary>
            public SeqState State { get; set; } = SeqState.None;
            /// <summary>
            /// Sequence must be postprocessed
            /// </summary>
            public bool TokenSampledAndDecoded { get; set; } = false;
            /// <summary>
            /// Tokens count remained to be used. (n_remain)
            /// </summary>
            public int RemainedTokens { get; set; }
            /// <summary>
            /// 
            /// </summary>
            public bool AutoStopFromEOG { get; set; }
        }

        private class Sequence(LLamaContext context)
        {
            public required LLamaSeqId Id { get; init; }
            public int NextDecodedTokenPos { get; set; } = 0;
            public LLamaTokenDataArray? LastLogits { get; set; } = null;
            public List<LLamaToken> DecodedTokens { get; set; } = new List<LLamaToken>();
            /// <summary>
            /// токены для обработки декодом
            /// </summary>
            public List<LLamaToken> Embeds = new();

            /// <summary>
            /// Tracks anti-prompts across streamed output.
            /// </summary>
            public required AntipromptProcessor AntipromptProc { get; init; }

            public InferStateArgs InferState { get; set; } = new InferStateArgs();

            public IInferenceParams InferParams { get; set; } = new InferenceParams();

            public readonly StreamingTokenDecoder Decoder = new StreamingTokenDecoder(context);
        }

        private enum SeqState
        {
            None,
            Generation,
            Prefill
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

                        batch.Add(inferSeqs[currentInferSeqId].Embeds.Last(), inferSeqs[currentInferSeqId].NextDecodedTokenPos, inferSeqs[currentInferSeqId].Id, true);
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
                    currentRequiredLogitsIds[k] = prefillSeqs[k].NextDecodedTokenPos + prefillSeqs[k].Embeds.Count - 1;
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

                            if (_lastPrefilledCount[currentPrefillSeqId].count < prefillSeqs[currentPrefillSeqId].Embeds.Count)
                            {

                                int count = _lastPrefilledCount[currentPrefillSeqId].count; //колво загруженных в батч в данном цикле
                                int pos = prefillSeqs[currentPrefillSeqId].NextDecodedTokenPos + count; // позиция текущего загружаемого токена в последовательности, расчитанная из последней + колво

                                batch.Add(
                                    prefillSeqs[currentPrefillSeqId].Embeds[count],
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
