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

        /// <summary>
        /// Post-processes sequences that have been sampled and had their sampled token decoded.
        /// Determines which sequences should be terminated as a result of this step.
        /// </summary>
        private async Task postProcessInfer()
        {
            List<(Sequence, Channel<string>)> seqsToRemove = new List<(Sequence, Channel<string>)>();
            foreach (var seq in _inferenceSeqs)
            {
                if (seq.Key.InferState.TokenSampledAndDecoded) // Only for sequences processed in the current batch
                {
                    seq.Key.InferState.TokenSampledAndDecoded = false;

                    // Convert the token to a string
                    seq.Key.Decoder.Add(seq.Key.DecodedTokens.Last());
                    var decoded = seq.Key.Decoder.Read();

                    bool forRemoving = false;

                    try
                    {
                        await seq.Value.Writer.WriteAsync(decoded, _executorLifeToken.Token); // Write to the sequence's channel
                    }
                    catch (Exception e)
                    {
                        // If something went wrong, remove it
                        forRemoving = true;
                    }


                    // If an anti-prompt was generated
                    if (!string.IsNullOrEmpty(decoded) && seq.Key.AntipromptProc.Add(decoded))
                    {
                        forRemoving = true;
                    }

                    if (seq.Key.InferState.AutoStopFromEOG)
                    {
                        // If an EOG token was generated
                        if (seq.Key.DecodedTokens.Last().IsEndOfGeneration(Context.Vocab))
                        {
                            forRemoving = true;
                        }
                    }

                    // If the number of generated tokens has reached MaxTokens
                    if (seq.Key.InferState.RemainedTokens <= 0 && seq.Key.InferParams.MaxTokens != -1)
                    {
                        forRemoving = true;
                    }

                    if (forRemoving) seqsToRemove.Add((seq.Key, seq.Value));
                }
            }
            // Remove the ones that have finished execution from the _inferenceSeqs container
            foreach ((Sequence, Channel<string>) seq in seqsToRemove)
            {
                endInference(seq);
            }
        }

        /// <summary>
        /// Releases the sequence from generation and closes its output channel.
        /// </summary>
        /// <param name="inferenceSeq">The sequence and its associated channel to finalize.</param>
        private void endInference((Sequence seq, Channel<string> ch) inferenceSeq)
        {
            _inferenceSeqs.Remove(inferenceSeq.seq);
            inferenceSeq.seq.InferState.State = SeqState.None;
            inferenceSeq.ch.Writer.Complete();
        }

        /// <summary>
        /// Post-processes the sequences that are in the prefill stage.
        /// </summary>
        private void postProcessPrefill()
        {
            List<(Sequence, TaskCompletionSource)> seqsToRemove = new List<(Sequence, TaskCompletionSource)>();
            foreach (var seq in _prefillSeqs)
            {
                // If there are no more tokens to prefill
                if (seq.Key.TokensToPrefill.Count == 0)
                {
                    seqsToRemove.Add((seq.Key, seq.Value));
                }
            }
            // Remove the ones that have finished execution from the container
            foreach ((Sequence, TaskCompletionSource) seq in seqsToRemove)
            {
                endPrefill(seq);
            }
        }

        /// <summary>
        /// Releases the sequence from prefill and signals completion.
        /// </summary>
        /// <param name="prefillSeq">The sequence and its associated <see cref="TaskCompletionSource"/> to finalize.</param>
        private void endPrefill((Sequence seq, TaskCompletionSource tcs) prefillSeq)
        {
            _prefillSeqs.Remove(prefillSeq.seq);
            prefillSeq.seq.InferState.State = SeqState.None;
            prefillSeq.tcs.SetResult();
        }

        /// <summary>
        /// Performs one decoding step:
        /// <list type="number">
        /// <item>Samples the next token for each inference sequence.</item>
        /// <item>Builds a batch that contains the newly sampled tokens (for inference) and the pending prefill tokens.</item>
        /// <item>Runs the model decoder on that batch.</item>
        /// <item>Updates the state and logits of all participating sequences.</item>
        /// </list>
        /// </summary>
        /// <param name="inferSeqs">Sequences currently in the inference (generation) stage.</param>
        /// <param name="prefillSeqs">Sequences currently in the prefill stage.</param>
        /// <returns>A task representing the asynchronous decode operation.</returns>
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

                if (seq.TokensToPrefill.Count == 0 && seq.LastLogits != null) // All tokens have been processed by model
                {
                    seq.EndDeleted = false;

                    using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)seq.LastLogits, out var native))
                    {
                        seq.InferParams.SamplingPipeline.Apply(ref native);
                        llamaToken = native.Data[(int)native.Selected].ID;
                        seq.InferParams.SamplingPipeline.Accept(llamaToken);
                    }
                }
                else if (seq.EndDeleted) // No last logits available because the end was deleted, so reuse the token the model previously selected
                {
                    llamaToken = seq.DeletedNextToken;
                }// Storing all logits is too expensive, so we accept that the first token will be repeated
                else throw new Exception($"sequence {seq.Id} LastLogits is empty. Generation impossible");

                if (isOutOfContext(1)) // Since there is only one LLamaToken; for MTP change in the future
                {
                    seqsToRemove.Add(seq); // Remove from sequences for batch assembly because there is no space in the context
                }
                else
                {
                    seq.TokensToPrefill.Add(llamaToken); // Add the newly sampled token to the container for model decoding

                    seq.RealTokensCount += seq.TokensToPrefill.Count(); // Add the sampled token to the real token count
                }   
            }
            foreach (Sequence seq in seqsToRemove)
            {
                Channel<string> ch = _inferenceSeqs[seq];

                inferSeqs.Remove(seq); // Also remove from the current list used for batch assembly

                endInference((seq,ch));
            }
            #endregion

            // Compose Batch
            LLamaBatch multiSeqBatch = _batchComposer.CreateBatch(inferSeqs, prefillSeqs, Context.BatchSize);

            // Decode
            DecodeResult res = await Context.DecodeAsync(multiSeqBatch); // A single batch consists of inference tokens at the beginning and prefill tokens at the end

            IReadOnlyDictionary<int, int> decodedSeqsListIds = _batchComposer.GetInferSeqsListIdsAndPos();
            IReadOnlyDictionary<int, (int count, int pos)> prefilledSeqsTokenCount = _batchComposer.GetPrefillSeqsTokenCount();

            // Error handling
            if (res != DecodeResult.Ok)
            {
                foreach (var item in decodedSeqsListIds)
                {
                    //ALL BAD
                }

                throw new LLamaDecodeError(res);
            }

            // Update Data
            foreach (var item in decodedSeqsListIds) // Inference sequences that participated in the batch
            {
                inferSeqs[item.Key].NextDecodedTokenPos++;
                inferSeqs[item.Key].LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(item.Value));
                inferSeqs[item.Key].DecodedTokens.Add(inferSeqs[item.Key].TokensToPrefill.Last()); // Record the tokens from TokensToPrefill as decoded
                inferSeqs[item.Key].TokensToPrefill.Clear();
                inferSeqs[item.Key].InferState.TokenSampledAndDecoded = true; // Postprocessing is needed
                inferSeqs[item.Key].InferState.RemainedTokens--; // Decrease the count of remaining tokens to generate
            }
            foreach (var item in prefilledSeqsTokenCount) // Prefill sequences that participated in the batch
            {
                prefillSeqs[item.Key].NextDecodedTokenPos += item.Value.count;
                prefillSeqs[item.Key].DecodedTokens.AddRange(prefillSeqs[item.Key].TokensToPrefill.GetRange(0, item.Value.count));
                prefillSeqs[item.Key].TokensToPrefill.RemoveRange(0, item.Value.count);
                if (prefillSeqs[item.Key].TokensToPrefill.Count == 0) // Have all TokensToPrefill been decoded?
                {
                    prefillSeqs[item.Key].LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(item.Value.pos));
                }
            }
        }

        /// <summary>
        /// Infinite loop that processes batches: each iteration generates one token for sequences that need it
        /// and prefills as many tokens as fit into the batch, up to the batch size.
        /// </summary>
        /// <returns>A task representing the asynchronous decoding loop.</returns>
        private async Task DecodingLoop()
        {
            while (!_executorLifeToken.Token.IsCancellationRequested) // One pass generates one token where needed and prefills as much as fits
            {
                await _workSignal.WaitAsync(_executorLifeToken.Token); // Checks if set with fast return inside WaitAsync

                await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); // sync Lock 
                try
                {
                    if (_inferenceSeqs.Count == 0 && _prefillSeqs.Count == 0) // Nothing to do
                    {
                        _workSignal.Reset(); // Reset the work signal
                    }
                    else
                    {
                        await decodeInternal(_inferenceSeqs.Keys.ToList(), _prefillSeqs.Keys.ToList()); // Convert to lists for easier indexing

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
        /// Copies a prefix from the source sequence to each target sequence.
        /// Target sequences are completely cleared before the copy.
        /// </summary>
        /// <param name="srcId">Identifier of the source sequence.</param>
        /// <param name="targetIds">List of target sequence identifiers that will receive the copied prefix.</param>
        /// <param name="endPos">
        /// The end position (exclusive) of the prefix to copy.
        /// Must be ≥ 1 and ≤ the source's <c>NextDecodedTokenPos</c>.
        /// </param>
        /// <returns>A task representing the asynchronous copy operation.</returns>
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

                    //copy
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

        /// <summary>
        /// Deletes tokens from the end of a sequence starting from a given position (inclusive).
        /// The sequence must not be currently in use.
        /// </summary>
        /// <param name="seqId">The identifier of the sequence to modify.</param>
        /// <param name="startPos">The position (inclusive) from which to delete tokens to the end.</param>
        /// <returns>A task representing the asynchronous delete operation.</returns>
        /// <exception cref="Exception"></exception>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="IndexOutOfRangeException"></exception>
        public async Task DeleteSequenceEnd(LLamaSeqId seqId, LLamaPos startPos /* inclusive */)
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
        /// Requests that generation for the specified sequence be stopped.
        /// The stop takes effect before the next decoding step in <see cref="DecodingLoop"/>.
        /// </summary>
        /// <param name="id">The identifier of the sequence to stop.</param>
        /// <returns>
        /// A task representing the asynchronous stop request.
        /// </returns>
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
        /// Retrieves the current decoding position (<see cref="Sequence.NextDecodedTokenPos"/>) for the specified sequence.
        /// Useful for context cloning between sequences.
        /// </summary>
        /// <param name="seqId">The identifier of the sequence.</param>
        /// <returns>The value of <see cref="Sequence.NextDecodedTokenPos"/>.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the sequence does not exist.</exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is canceled via the executor's life token.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the semaphore or executor has been disposed.</exception>
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
        /// Retrieves the list of tokens that have been decoded so far for the specified sequence.
        /// </summary>
        /// <param name="seqId">The identifier of the sequence.</param>
        /// <returns>
        /// A read‑only list of <see cref="LLamaToken"/> values that have been decoded for the sequence.
        /// </returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the sequence does not exist.</exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is canceled via the executor's life token.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the semaphore or executor has been disposed.</exception>
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
        /// Retrieves the last logits (<see cref="LLamaTokenDataArray"/>) for the specified sequence, if available.
        /// </summary>
        /// <param name="seqId">The identifier of the sequence.</param>
        /// <returns>
        /// The last <see cref="LLamaTokenDataArray"/> produced for the sequence, or <c>null</c> if none have been computed yet.
        /// </returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the sequence does not exist.</exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is canceled via the executor's life token.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the semaphore or executor has been disposed.</exception>
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
        /// Samples a single token for the specified sequence without decoding it (the model state is not advanced).
        /// Useful for quick probing, e.g., when the context is [data] + [yes/no question]
        /// and you want to peek at the answer without committing it.
        /// </summary>
        /// <param name="seqId">The identifier of the sequence to sample from.</param>
        /// <param name="inferenceParams"> The inference parameters to use for sampling.</param>
        /// <returns>The decoded string piece corresponding to the sampled token.</returns>
        /// <exception cref="IndexOutOfRangeException">Thrown if the sequence does not exist.</exception>
        /// <exception cref="Exception">
        /// Thrown if the sequence has no logits, is currently in use by another operation,
        /// or no inference parameters are available.
        /// </exception>
        /// <exception cref="OperationCanceledException">Thrown if the operation is canceled via the executor's life token.</exception>
        /// <exception cref="ObjectDisposedException">Thrown if the semaphore or executor has been disposed.</exception>
        public async Task<string> SampleOneTokenWithoutDecode(LLamaSeqId seqId, IInferenceParams inferenceParams)
        {
            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                if (!_sequences.TryGetValue(seqId, out var seq))
                    throw new IndexOutOfRangeException($"Sequence {seqId} does not exist");
                if (seq.InferState.State != SeqState.None)
                    throw new Exception($"{seqId} sequence is being used in another place: {seq.InferState.State}");
                if (seq.LastLogits == null)
                    throw new Exception($"Sequence {seqId} has no logits");
                if (inferenceParams == null)
                    throw new Exception($"No inference parameters");

                LLamaToken llamaToken;
                using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)seq.LastLogits, out var native))
                {
                    inferenceParams.SamplingPipeline.Apply(ref native);
                    llamaToken = native.Data[(int)native.Selected].ID;
                    // Do not call Accept – the token is not committed
                }

                seq.Decoder.Add(llamaToken);
                string textPiece = seq.Decoder.Read();
                return textPiece;
            }
            finally
            {
                _seqStateSemaphore.Release();
            }
        }

        /// <summary>
        /// Returns the total number of real tokens currently stored in the context.
        /// </summary>
        /// <returns>The sum of <see cref="Sequence.RealTokensCount"/> across all sequences.</returns>
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
        /// Calculates the total number of real tokens currently occupying KV-cache slots.
        /// <b>MUST BE CALLED ONLY UNDER LOCK</b> (inside a <see cref="_seqStateSemaphore"/> critical section).
        /// </summary>
        /// <returns>The sum of <see cref="Sequence.RealTokensCount"/> across all sequences.</returns>
        private int getContextFilling()
        {
            int count = 0;
            foreach (Sequence seq in _sequences.Values)
            {
                count += seq.RealTokensCount;
            }
            return count;
        }

        /// <summary>
        /// Checks whether adding the specified number of tokens would exceed the context size.
        /// </summary>
        /// <param name="count">Number of tokens to add.</param>
        /// <returns><c>true</c> if there is not enough space; otherwise, <c>false</c>.</returns>
        /// <remarks>
        /// <b>MUST BE CALLED ONLY UNDER LOCK</b> (inside a <see cref="_seqStateSemaphore"/> critical section).
        /// </remarks>
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
            /// Composes a batch from the given inference and prefill sequences.
            /// </summary>
            /// <param name="inferSeqs">Sequences in the inference (generation) stage.</param>
            /// <param name="prefillSeqs">Sequences in the prefill stage.</param>
            /// <param name="batchSize">Maximum number of tokens allowed in the batch.</param>
            /// <returns>A new <see cref="LLamaBatch"/> containing the selected tokens.</returns>
            public LLamaBatch CreateBatch(IReadOnlyList<Sequence> inferSeqs, IReadOnlyList<Sequence> prefillSeqs, uint batchSize);

            /// <summary>
            /// Returns the indices (in the <paramref name="inferSeqs"/> list) of the inference sequences
            /// that participated in the last composed batch, along with their corresponding logit positions.
            /// </summary>
            /// <returns>
            /// A dictionary where the key is the index into the original <paramref name="inferSeqs"/> list,
            /// and the value is the logit position assigned to that sequence in the batch.
            /// </returns>
            public IReadOnlyDictionary<int, int> GetInferSeqsListIdsAndPos();

            /// <summary>
            /// Returns the number of prefill tokens that were processed for each prefill sequence in the last batch,
            /// together with the position of the last token of that sequence in the logits.
            /// </summary>
            /// <returns>
            /// A dictionary where the key is the index into the original <paramref name="prefillSeqs"/> list,
            /// and the value is a tuple containing <c>count</c> (number of tokens processed) and
            /// <c>pos</c> (logit position of the last token).
            /// </returns>
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
                // Clear
                _lastInferSeqs.Clear();
                _lastPrefilledCount.Clear();

                LLamaBatch batch = new LLamaBatch(); // New batch

                int batchPos = 0; // Current position in the batch
                int currentInferSeqId = 0; // Index in inferSeqs

                for (int i = 0; i < inferSeqs.Count; i++)
                {
                    if (batchPos < batchSize)
                    {
                        currentInferSeqId = _nextInferSeqBatchId + i;
                        currentInferSeqId = currentInferSeqId % inferSeqs.Count;

                        batch.Add(inferSeqs[currentInferSeqId].TokensToPrefill.Last(), inferSeqs[currentInferSeqId].NextDecodedTokenPos, inferSeqs[currentInferSeqId].Id, true);
                        _lastInferSeqs[currentInferSeqId] = batchPos;
                        batchPos++; // Increment batch position
                    }
                    else
                    {
                        // Save to start from this position in the next batch if there are more sequences than batch size
                        _nextInferSeqBatchId = currentInferSeqId + 1;
                        break;
                    }
                }


                Dictionary<int, int> currentRequiredLogitsIds = new Dictionary<int, int>(); // Positions at which Logits must be set to true
                for (int k = 0; k < prefillSeqs.Count; k++)
                {
                    currentRequiredLogitsIds[k] = prefillSeqs[k].NextDecodedTokenPos + prefillSeqs[k].TokensToPrefill.Count - 1;
                }

                int currentPrefillSeqId = 0; // Index of the current sequence being processed from prefillSeqs

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

                            if (!_lastPrefilledCount.ContainsKey(currentPrefillSeqId)) _lastPrefilledCount[currentPrefillSeqId] = (0, -1); // Initialize if the sequence has not been processed yet

                            if (_lastPrefilledCount[currentPrefillSeqId].count < prefillSeqs[currentPrefillSeqId].TokensToPrefill.Count)
                            {

                                int count = _lastPrefilledCount[currentPrefillSeqId].count; // count of added to batch in this loop
                                int pos = prefillSeqs[currentPrefillSeqId].NextDecodedTokenPos + count; // current pos in sequence of token for adding

                                batch.Add(
                                    prefillSeqs[currentPrefillSeqId].TokensToPrefill[count],
                                    pos,
                                    prefillSeqs[currentPrefillSeqId].Id,
                                    pos == currentRequiredLogitsIds[currentPrefillSeqId]
                                    ); //add token to batch
                                TokenAdded = true;
                                _lastPrefilledCount[currentPrefillSeqId] = (count + 1, batchPos); // Increment the number of tokens added to the batch and store the batch position (used later when PrefillCount reaches zero)

                                batchPos++; // Increment batch position
                            }
                        }
                        else
                        {
                            _nextPrefillSeqBatchId = currentPrefillSeqId + 1;
                            break; // Exit if space is exhausted
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
