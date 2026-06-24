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
        /// Tokenizes the input text according to the specified flags.
        /// </summary>
        /// <param name="text">The text to tokenize.</param>
        /// <param name="addBos">Whether to prepend the BOS token.</param>
        /// <param name="special">Whether to treat special tokens as such.</param>
        private List<LLamaToken> preprocessInputs(string text, bool addBos, bool special)
        {
            return Context.Tokenize(text, addBos, special).ToList();
        }

        /// <summary>
        /// Processes a single prompt by tokenizing the text and running the prefill phase.
        /// </summary>
        /// <param name="text">The text to process.</param>
        /// <param name="addBos">Whether to add the BOS token.</param>
        /// <param name="special">Whether to tokenize special tokens.</param>
        /// <returns>A task representing the asynchronous prefill operation.</returns>
        /// <exception cref="LLamaDecodeError">Thrown if the underlying decode fails.</exception>
        public async Task ProcessPrompt(string text, bool addBos = false, bool special = true)
        {
            // Simple argument checks, no locking needed
            #region Checks 

            #endregion

            List<LLamaToken>? embeds;

            if (string.IsNullOrEmpty(text))
                embeds = null;
            else
                embeds = preprocessInputs(text, addBos, special);

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token); // Lock for modifying the sequence state
            try
            {
                if (embeds != null)
                {
                    _mainSeq.InferState.State = SeqState.Prefill;
                    // Fill the sequence's TokensToPrefill
                    _mainSeq.TokensToPrefill.AddRange(embeds);

                    LLamaBatch batch = new LLamaBatch();

                    (DecodeResult, int, int) tuple = await Context.DecodeAsync(_mainSeq.TokensToPrefill, _mainSeq.Id, batch, _mainSeq.NextDecodedTokenPos);

                    if (tuple.Item1 != DecodeResult.Ok)
                    {
                        //tuple.Item2 stores the number of tokens that were not decoded
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

        /// <summary>
        /// Post-processes the inference step: decodes the sampled token and checks stopping conditions.
        /// </summary>
        /// <returns>The decoded string piece, which may be empty.</returns>
        private string postProcessInfer()
        {
            _mainSeq.InferState.TokenSampledAndDecoded = false;

            // Convert the token to a string
            _mainSeq.Decoder.Add(_mainSeq.DecodedTokens.Last());
            var decoded = _mainSeq.Decoder.Read();

            // If an anti-prompt was generated
            if (!string.IsNullOrEmpty(decoded) && _mainSeq.AntipromptProc.Add(decoded))
            {
                _mainSeq.InferState.State = SeqState.None;
            }

            if (_mainSeq.InferState.AutoStopFromEOG)
            {
                // If an EOG token was generated
                if (_mainSeq.DecodedTokens.Last().IsEndOfGeneration(Context.Vocab))
                {
                    _mainSeq.InferState.State = SeqState.None;
                }
            }

            // If the number of generated tokens has reached MaxTokens
            if (_mainSeq.InferState.RemainedTokens <= 0 && _mainSeq.InferParams.MaxTokens != -1)
            {
                _mainSeq.InferState.State = SeqState.None;
            }

            return decoded;
        }

        /// <summary>
        /// Generates tokens based on the specified inference parameters and streams the results as strings.
        /// </summary>
        /// <param name="inferenceParams">The inference parameters (samplers, max tokens, anti-prompts, etc.).</param>
        /// <returns>An async stream of decoded string pieces.</returns>
        /// <exception cref="ArgumentNullException">Thrown if <paramref name="inferenceParams"/> is null.</exception>
        /// <exception cref="Exception">Thrown if the sequence has not been prefilled or is already generating.</exception>
        /// <exception cref="LLamaDecodeError">Thrown if a decode step fails.</exception>
        public async IAsyncEnumerable<string> Generate(IInferenceParams inferenceParams)
        {
            // Simple argument checks, no locking needed
            #region Checks 
            if (inferenceParams == null) throw new ArgumentNullException();
            #endregion

            await _seqStateSemaphore.WaitAsync(_executorLifeToken.Token);
            try
            {
                // Verification
                if (_mainSeq.InferParams.MaxTokens == 0) throw new Exception($"For prefill use ProcessPrompt");

                _mainSeq.InferState.State = SeqState.Generation;

                // Initialize inference state
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
                _mainSeq.TokensToPrefill.Clear(); // Just in case

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

                    _mainSeq.TokensToPrefill.Add(llamaToken); // Add the newly sampled token to the container for decoding
                    #endregion

                    #region Decode
                    LLamaBatch batch = new LLamaBatch();
                    batch.Add(llamaToken, _mainSeq.NextDecodedTokenPos, _mainSeq.Id, true);

                    DecodeResult res = await Context.DecodeAsync(batch);


                    if (res != DecodeResult.Ok)
                    {
                        // The state remains consistent, so retrying without recovery is possible
                        throw new LLamaDecodeError(res);
                    }

                    _mainSeq.NextDecodedTokenPos++;
                    _mainSeq.LastLogits = LLamaTokenDataArray.Create(Context.NativeHandle.GetLogitsIth(batch.TokenCount - 1));
                    _mainSeq.DecodedTokens.Add(_mainSeq.TokensToPrefill.Last()); // Record the embeds tokens as decoded
                    _mainSeq.TokensToPrefill.Clear();
                    _mainSeq.InferState.TokenSampledAndDecoded = true; // Post-processing is required
                    _mainSeq.InferState.RemainedTokens--; // Decrease the number of remaining tokens to generate
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

        /// <summary>
        /// Requests cancellation of the current generation. The stop takes effect before the next token is sampled.
        /// </summary>
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
        /// Retrieves the current decoding position (<see cref="Sequence.NextDecodedTokenPos"/>).
        /// Useful for context cloning between sequences.
        /// </summary>
        /// <returns>The current value of <see cref="Sequence.NextDecodedTokenPos"/>.</returns>
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
        /// Retrieves the list of tokens that have been decoded so far.
        /// </summary>
        /// <returns>A read‑only list of <see cref="LLamaToken"/> values, or null if none.</returns>
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
        /// Retrieves the last computed logits, if available.
        /// </summary>
        /// <returns>The last <see cref="LLamaTokenDataArray"/>, or null if not yet computed.</returns>
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

        /// <summary>
        /// Samples a single token without decoding it (the model state is not advanced).
        /// Useful for quick probing, e.g., when the context is [data] + [yes/no question]
        /// and you want to peek at the answer without committing it.
        /// </summary>
        /// <param name="inferenceParams">The inference parameters to use for sampling.</param>
        /// <returns>The decoded string piece corresponding to the sampled token.</returns>
        public string SampleOneTokenWithoutDecode(IInferenceParams inferenceParams)
        {
            if (_mainSeq.LastLogits == null) throw new Exception($"sequence {_mainSeq.Id} is empty");

            LLamaToken llamaToken;
            using (var handle = LLamaTokenDataArrayNative.Create((LLamaTokenDataArray)_mainSeq.LastLogits, out var native))
            {
                inferenceParams.SamplingPipeline.Apply(ref native);
                llamaToken = native.Data[(int)native.Selected].ID;
                // Do not call Accept – the token is not committed
            }
            string textPiece = "";
            _mainSeq.Decoder.Add(llamaToken);
            textPiece = _mainSeq.Decoder.Read();
            return textPiece;
        } 
    }
}
