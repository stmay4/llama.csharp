using Llama.csharp.Exceptions;
using System.Diagnostics;
using System.Text;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace Llama.csharp.Native
{
    public sealed class SafeLLamaContextHandle : SafeLLamaHandleBase
    {
        #region properties and fields
        /// <summary>
        /// Total number of tokens in the context
        /// </summary>
        public uint ContextSize => LlamaCpp.Llama_ContextNCtx(this);

        /// <summary>
        /// Dimension of embedding vectors
        /// </summary>
        public int EmbeddingSize => ThrowIfDisposed().EmbeddingSize;

        /// <summary>
        /// Get the maximum batch size for this context
        /// </summary>
        public uint BatchSize => LlamaCpp.Llama_ContextNBatch(this);

        /// <summary>
        /// Get the physical maximum batch size for this context
        /// </summary>
        public uint UBatchSize => LlamaCpp.Llama_ContextNUbatch(this);

        /// <summary>
        /// Get or set the number of threads used for generation of a single token.
        /// </summary>
        public int GenerationThreads
        {
            get => LlamaCpp.Llama_ContextNThreads(this);
            set => LlamaCpp.Llama_ContextSetNThreads(this, value, BatchThreads);
        }

        /// <summary>
        /// Get or set the number of threads used for prompt and batch processing (multiple token).
        /// </summary>
        public int BatchThreads
        {
            get => LlamaCpp.Llama_ContextNThreadsBatch(this);
            set => LlamaCpp.Llama_ContextSetNThreads(this, GenerationThreads, value);
        }

        /// <summary>
        /// Get the pooling type for this context
        /// </summary>
        public LLamaPoolingType PoolingType => LlamaCpp.Llama_ContextPoolingType(this);

        /// <summary>
        /// Get the model which this context is using
        /// </summary>
        public SafeLlamaModelHandle ModelHandle => ThrowIfDisposed();

        /// <summary>
        /// Get the vocabulary for the model this context is using
        /// </summary>
        public SafeLlamaModelHandle.Vocabulary Vocab => ThrowIfDisposed().Vocab;

        private SafeLlamaModelHandle? _model;
        #endregion

        #region construction/destruction
        static SafeLLamaContextHandle() { }

        /// переопределенный метод класса SafeHandle из System.Runtime.InteropServices, который выполняется при Dispose
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Llama_ContextFree(handle);
            SetHandle(nint.Zero);

            // Decrement refcount on model
            _model?.DangerousRelease();
            _model = null!;

            return true;
        }

        private SafeLlamaModelHandle ThrowIfDisposed()
        {
            if (IsClosed)
                throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - it has been disposed");
            if (_model == null || _model.IsClosed)
                throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");

            return _model!;
        }

        /// <summary>
        /// Create a new llama_state for the given model
        /// </summary>
        /// <param name="model"></param>
        /// <param name="lparams"></param>
        /// <returns></returns>
        /// <exception cref="RuntimeError"></exception>
        public static SafeLLamaContextHandle Create(SafeLlamaModelHandle model, LLamaContextParams lparams)
        {
            var ctx = LlamaCpp.Llama_ContextInitFromModel(model, lparams);
            if (ctx == null)
                throw new RuntimeError("Failed to create context from model");

            // Increment the model reference count while this context exists.
            // DangerousAddRef throws if it fails, so there is no need to check "success"
            ctx._model = model;
            var success = false;
            ctx._model.DangerousAddRef(ref success);

            return ctx;
        }
        #endregion

        #region LoRA
        /// <summary>
        /// Add a LoRA adapter to this context
        /// </summary>
        /// <param name="lora"></param>
        /// <param name="scale"></param>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="RuntimeError"></exception>
        //public void AddLoraAdapter(LoraAdapter lora, float scale)
        //{
        //    if (lora.Model != ModelHandle)
        //        throw new ArgumentException("Cannot add LoRA adapter which was loaded for a different model");
        //    if (!lora.Loaded)
        //        throw new ArgumentException("Cannot add LoRA adapter which has been unloaded");

        //    var err = llama_set_adapter_lora(this, lora.Pointer, scale);
        //    if (err != 0)
        //        throw new RuntimeError("Failed to set lora adapter");
        //}

        /// <summary>
        /// Remove a LoRA adapter from this context
        /// </summary>
        /// <param name="lora"></param>
        /// <returns>Indicates if the lora was in this context and was remove</returns>
        //public bool RemoveLoraAdapter(LoraAdapter lora)
        //{
        //    if (lora.Model != ModelHandle)
        //        return false;

        //    var err = llama_rm_adapter_lora(this, lora.Pointer);
        //    return err == 0;
        //}

        /// <summary>
        /// Remove all LoRA adapters from this context
        /// </summary>
        public void ClearLoraAdapters()
        {
            LlamaCpp.Llama_ContextClearAdapterLora(this);
        }
        #endregion

        #region GetLogits
        /// <summary>
        /// Token logits obtained from the last call to llama_decode.
        /// The logits for the last token are stored in the last row.
        /// Only tokens with `logits = true` requested are present.<br/>
        /// Can be mutated in order to change the probabilities of the next token.<br />
        /// Rows: n_tokens<br />
        /// Cols: n_vocab
        /// </summary>
        /// <param name="numTokens">
        /// The amount of tokens whose logits should be retrieved, in <b>[numTokens X n_vocab]</b> format.<br/>
        /// Tokens' order is based on their order in the LlamaBatch (so, first tokens are first, etc).<br/>
        /// This is helpful when requesting logits for many tokens in a sequence, or want to decode multiple sequences in one go.
        /// </param>
        /// <returns></returns>
        public Span<float> GetLogits(int numTokens = 1)
        {
            var model = ThrowIfDisposed();

            unsafe
            {
                var logits = LlamaCpp.Llama_ContextGetLogits(this);
                return new Span<float>(logits, model.Vocab.Count * numTokens);
            }
        }

        /// <summary>
        /// Logits for the ith token. Equivalent to: llama_get_logits(ctx) + i*n_vocab
        /// </summary>
        /// <param name="i"></param>
        /// <returns></returns>
        public Span<float> GetLogitsIth(int i)
        {
            var model = ThrowIfDisposed();

            unsafe
            {
                var logits = LlamaCpp.Llama_ContextGetLogitsIth(this, i);
                if (logits == null)
                    throw new GetLogitsInvalidIndexException(i);

                return new Span<float>(logits, model.Vocab.Count);
            }
        }
        #endregion

        #region GetEmbeddings()
        /// <summary>
        /// Get the embeddings for the ith sequence.
        /// Equivalent to: llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
        /// </summary>
        /// <returns>A pointer to the first float in an embedding, length = ctx.EmbeddingSize</returns>
        public Span<float> GetEmbeddingsIth(LLamaPos pos)
        {
            var model = ThrowIfDisposed();

            unsafe
            {
                var embd = LlamaCpp.Llama_ContextGetEmbeddingsIth(this, pos.Value);
                Debug.Assert(embd != null);
                return new Span<float>(embd, model.EmbeddingSize);
            }
        }

        /// <summary>
        /// Get the embeddings for the a specific sequence.
        /// Equivalent to: llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
        /// </summary>
        /// <returns>A pointer to the first float in an embedding, length = ctx.EmbeddingSize</returns>
        public Span<float> GetEmbeddingsSeq(LLamaSeqId seq)
        {
            var model = ThrowIfDisposed();

            unsafe
            {
                var embd = LlamaCpp.Llama_ContextGetEmbeddingsSeq(this, seq);
                Debug.Assert(embd != null);
                return new Span<float>(embd, model.EmbeddingSize);
            }
        }
        #endregion

        #region tokens
        /// <summary>
        /// Convert the given text into tokens
        /// </summary>
        /// <param name="text">The text to tokenize</param>
        /// <param name="add_bos">Whether the "BOS" token should be added</param>
        /// <param name="encoding">Encoding to use for the text</param>
        /// <param name="special">Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.</param>
        /// <returns></returns>
        /// <exception cref="RuntimeError"></exception>
        public LLamaToken[] Tokenize(string text, bool add_bos, bool special, Encoding encoding)
        {
            return ThrowIfDisposed().Vocab.Tokenize(text, add_bos, special, encoding);
        }

        /// <summary>
        /// Convert a single llama token into bytes
        /// </summary>
        /// <param name="token">Token to decode</param>
        /// <param name="dest">A span to attempt to write into. If this is too small nothing will be written</param>
        /// <returns>The size of this token. **nothing will be written** if this is larger than `dest`</returns>
        public uint TokenToSpan(LLamaToken token, Span<byte> dest)
        {
            return ThrowIfDisposed().Vocab.TokenToSpan(token, dest);
        }
        #endregion

        #region infer
        /// <summary>
        /// This object exists to ensure there is only ever 1 inference running at a time. This is a workaround for thread safety issues in llama.cpp itself.
        /// Most notably CUDA, which seems to use some global singleton resources and will crash if multiple inferences are run (even against different models).
        /// 
        /// For more information see these issues:
        ///  - https://github.com/SciSharp/LLamaSharp/issues/596
        ///  - https://github.com/ggerganov/llama.cpp/issues/3960
        ///
        /// If these are ever resolved this lock can probably be removed.
        /// </summary>
        private static readonly object GlobalInferenceLock = new();

        /// <summary>
        /// Wait until all computations are finished. This is automatically done when using any of the functions to obtain computation results
        /// and is not necessary to call it explicitly in most cases.
        /// </summary>
        public void Synchronize()
        {
            lock (GlobalInferenceLock)
                LlamaCpp.Llama_Synchronize(this);
        }

        /// <summary>
        /// Processes a batch of tokens with the encoder part of the encoder-decoder model. Stores the encoder output
        /// internally for later use by the decoder cross-attention layers.
        /// </summary>
        /// <param name="batch"></param>
        /// <returns>0 = success <br />&lt; 0 = error (the KV cache state is restored to the state before this call)</returns>
        public DecodeResult Encode(LLamaBatch batch)
        {
            if (batch.TokenCount == 0)
                return DecodeResult.Ok;

            lock (GlobalInferenceLock)
                using (batch.ToNativeBatch(out var nb))
                    return (DecodeResult)LlamaCpp.Llama_ContextEncode(this, nb);
        }

        /// <summary>
        /// </summary>
        /// <param name="batch"></param>
        /// <returns>Positive return values does not mean a fatal error, but rather a warning:<br />
        ///  - 0: success<br />
        ///  - 1: could not find a KV slot for the batch (try reducing the size of the batch or increase the context)<br />
        ///  - &lt; 0: error (the KV cache state is restored to the state before this call)<br />
        /// </returns>
        public DecodeResult Decode(LLamaBatch batch)
        {
            if (batch.TokenCount == 0)
                return DecodeResult.Ok;

            lock (GlobalInferenceLock)
                using (batch.ToNativeBatch(out var nb))
                    return (DecodeResult)LlamaCpp.Llama_ContextDecode(this, nb);
        }

        /// <summary>
        /// Decode a set of tokens in batch-size chunks.
        /// </summary>
        /// <param name="tokens"></param>
        /// <param name="id"></param>
        /// <param name="batch"></param>
        /// <param name="n_past"></param>
        /// <returns>A tuple, containing the decode result and the number of tokens that have <b>not</b> been decoded yet.</returns>
        internal (DecodeResult, int) Decode(List<LLamaToken> tokens, LLamaSeqId id, LLamaBatch batch, ref int n_past)
        {
            if (tokens.Count == 0)
                return (DecodeResult.Ok, 0);

            var batchSize = checked((int)BatchSize);

            for (var i = 0; i < tokens.Count; i += batchSize)
            {
                // Вычисляем размер текущего куска: либо полный batchSize, либо остаток
                var n_eval = Math.Min(batchSize, tokens.Count - i);

                batch.Clear();

                // Добавляем токены текущего куска
                for (var j = 0; j < n_eval; j++)
                {
                    // Флаг logits=true только для самого последнего токена ВСЕГО списка
                    bool isLastToken = (i + j) == tokens.Count - 1;
                    batch.Add(tokens[i + j], n_past++, id, isLastToken);
                }

                var returnCode = Decode(batch);
                if (returnCode != DecodeResult.Ok)
                    return (returnCode, tokens.Count - i); // Возвращаем количество НЕобработанных токенов
            }

            return (DecodeResult.Ok, 0);
        }

        /// <summary>
        /// </summary>
        /// <param name="batch"></param>
        /// <returns>Positive return values does not mean a fatal error, but rather a warning:<br />
        ///  - 0: success<br />
        ///  - 1: could not find a KV slot for the batch (try reducing the size of the batch or increase the context)<br />
        ///  - &lt; 0: error<br />
        /// </returns>
        public DecodeResult Decode(LLamaBatchEmbeddings batch)
        {
            if (batch.EmbeddingsCount == 0)
                return DecodeResult.Ok;

            lock (GlobalInferenceLock)
                using (batch.ToNativeBatch(out var nb))
                    return (DecodeResult)LlamaCpp.Llama_ContextDecode(this, nb);
        }
        #endregion

        #region timing
        /// <summary>
        /// Get performance information
        /// </summary>
        /// <returns></returns>
        public LLamaPerfContextTimings GetTimings()
        {
            return LlamaCpp.Llama_PerfContext(this);
        }

        /// <summary>
        /// Reset all performance information for this context
        /// </summary>
        public void ResetTimings()
        {
            LlamaCpp.Llama_PerfContextReset(this);
        }
        #endregion

        #region memory

        // Clear the memory contents
        // If data == true, the data buffers will also be cleared together with the metadata
        internal void ClearMemory(bool data)
        {
            LlamaCpp.Llama_ContextMemoryClear(this, data);
        }

        // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
        // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
        // seq_id < 0 : match any sequence
        // p0 < 0     : [0,  p1]
        // p1 < 0     : [p0, inf)
        internal (bool,int) SeqMemoryRemove(LLamaSeqId seq, LLamaPos p0, LLamaPos p1)
        {
            LLamaPos minPos = LlamaCpp.Llama_ContextMemorySeqPosMin(this, seq);
            LLamaPos maxPos = LlamaCpp.Llama_ContextMemorySeqPosMax(this, seq);

            if (minPos == -1 || maxPos == -1) return (false,-1); //sequence is empty
            if ((int)p1 > (int)maxPos) return (false,1);

            return (LlamaCpp.Llama_ContextMemorySeqRemove(this, seq, p0, p1),0);
        }

        internal void SeqMemoryRemoveAll(LLamaSeqId seq)
        {
            LLamaPos minPos = LlamaCpp.Llama_ContextMemorySeqPosMin(this, seq);

            if (minPos == -1) return; //sequence is empty

            LlamaCpp.Llama_ContextMemorySeqRemove(this, seq, minPos, -1);
        }

        // Copy all tokens that belong to the specified sequence to another sequence
        // p0 < 0 : [0,  p1]
        // p1 < 0 : [p0, inf)
        internal void SeqMemoryCopy(LLamaSeqId src, LLamaSeqId dest, LLamaPos p0, LLamaPos p1)
        {
            LlamaCpp.Llama_ContextMemorySeqCopy(this, src, dest, p0, p1);
        }

        // Removes all tokens that do not belong to the specified sequence
        internal void SeqMemoryKeep(LLamaSeqId seq)
        {
            LlamaCpp.Llama_ContextMemorySeqKeep(this, seq);
        }

        #endregion

    }
}
