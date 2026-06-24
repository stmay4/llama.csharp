using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// PARTIAL context functions
    /// </summary>
    public static partial class LlamaCpp
    {
        #region LLAMA API functions

        #region delegates

        /// <summary>
        /// Create a new llama_context with the given model.
        /// </summary>
        /// <param name="model"></param>
        /// <param name="params"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeLLamaContextHandle llama_init_from_model(SafeLlamaModelHandle model, LLamaContextParams @params);

        /// <summary>
        /// Frees all allocated memory in the given llama_context
        /// </summary>
        /// <param name="ctx"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_free(IntPtr ctx);

        /// <summary>
        /// Set a callback which can abort computation
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="abort_callback"></param>
        /// <param name="abort_callback_data"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate void llama_set_abort_callback(SafeLLamaContextHandle ctx, GgmlAbortCallback abort_callback, void* abort_callback_data);

        /// <summary>
        /// If this returns true computation is cancelled
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public unsafe delegate bool GgmlAbortCallback(void* data);

        /// <summary>
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="batch"></param>
        /// <returns>Positive return values does not mean a fatal error, but rather a warning:<br />
        ///  - 0: success<br />
        ///  - 1: could not find a KV slot for the batch (try reducing the size of the batch or increase the context)<br />
        ///  - &lt; 0: error<br />
        /// </returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_decode(SafeLLamaContextHandle ctx, LLamaNativeBatch batch);

        /// <summary>
        /// Processes a batch of tokens with the encoder part of the encoder-decoder model. Stores the encoder output
        /// internally for later use by the decoder cross-attention layers.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="batch"></param>
        /// <returns>0 = success <br />&lt; 0 = error</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_encode(SafeLLamaContextHandle ctx, LLamaNativeBatch batch);

        /// <summary>
        /// Set the number of threads used for decoding
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="n_threads">n_threads is the number of threads used for generation (single token)</param>
        /// <param name="n_threads_batch">n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)</param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_set_n_threads(SafeLLamaContextHandle ctx, int n_threads, int n_threads_batch);

        /// <summary>
        /// Get the number of threads used for generation of a single token.
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_n_threads(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get the number of threads used for prompt and batch processing (multiple token).
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_n_threads_batch(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Token logits obtained from the last call to llama_decode
        /// The logits for the last token are stored in the last row
        /// Can be mutated in order to change the probabilities of the next token.<br />
        /// Rows: n_tokens<br />
        /// Cols: n_vocab
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* llama_get_logits(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Logits for the ith token. Equivalent to: llama_get_logits(ctx) + i*n_vocab
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="i"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* llama_get_logits_ith(SafeLLamaContextHandle ctx, int i);

        /// <summary>
        /// Get the size of the context window for the model for this context
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint llama_n_ctx(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get the batch size for this context
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint llama_n_batch(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get the ubatch size for this context
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint llama_n_ubatch(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get the n_seq_max for this context
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint llama_n_seq_max(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Returns the **actual** size in bytes of the state (logits, embedding and kv_cache).
        /// Only use when saving the state, not when restoring it, otherwise the size may be too small.
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint llama_state_get_size(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Copies the state to the specified destination address.
        /// Destination needs to have allocated enough memory.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="dest"></param>
        /// <param name="size"></param>
        /// <returns>the number of bytes copied</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate nuint llama_state_get_data(SafeLLamaContextHandle ctx, byte* dest, nuint size);

        /// <summary>
        /// Set the state reading from the specified address
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="src"></param>
        /// <param name="size"></param>
        /// <returns>the number of bytes read</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate nuint llama_state_set_data(SafeLLamaContextHandle ctx, byte* src, nuint size);

        /// <summary>
        /// Get the exact size needed to copy the KV cache of a single sequence
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint llama_state_seq_get_size(SafeLLamaContextHandle ctx, LLamaSeqId seqId);

        /// <summary>
        /// Copy the KV cache of a single sequence into the specified buffer
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="dst"></param>
        /// <param name="size"></param>
        /// <param name="seqId"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate nuint llama_state_seq_get_data(SafeLLamaContextHandle ctx, byte* dst, nuint size, LLamaSeqId seqId);

        /// <summary>
        /// Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="src"></param>
        /// <param name="size"></param>
        /// <param name="destSeqId"></param>
        /// <returns>
        ///  - Positive: Ok
        ///  - Zero: Failed to load
        /// </returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate nuint llama_state_seq_set_data(SafeLLamaContextHandle ctx, byte* src, nuint size, LLamaSeqId destSeqId);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaPerfContextTimings llama_perf_context(SafeLLamaContextHandle ctx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_perf_context_print(SafeLLamaContextHandle ctx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_perf_context_reset(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Wait until all computations are finished. This is automatically done when using any of the functions to obtain computation results
        /// and is not necessary to call it explicitly in most cases.
        /// </summary>
        /// <param name="ctx"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_synchronize(SafeLLamaContextHandle ctx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_set_adapter_lora(SafeLLamaContextHandle context, IntPtr adapter, float scale);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_rm_adapter_lora(SafeLLamaContextHandle context, IntPtr adapter);

        // Remove all LoRA adapters from given context

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_clear_adapter_lora(SafeLLamaContextHandle context);

        /// <summary>
        /// Get the pooling type for this context
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaPoolingType llama_pooling_type(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get all output token embeddings.
        /// When pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model, the embeddings for which
        /// llama_batch.logits[i] != 0 are stored contiguously in the order they have appeared in the batch.
        /// shape: [n_outputs*n_embd]
        /// Otherwise, returns an empty span.
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* llama_get_embeddings(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Get the embeddings for a sequence id.
        /// Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
        /// when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[1] with the rank of the sequence
        /// otherwise: float[n_embd] (1-dimensional)
        /// </summary>
        /// <returns>A pointer to the first float in an embedding, length = ctx.EmbeddingSize</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* llama_get_embeddings_seq(SafeLLamaContextHandle ctx, LLamaSeqId id);

        /// <summary>
        /// Get the embeddings for the ith sequence.
        /// Equivalent to: llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
        /// </summary>
        /// <returns>A pointer to the first float in an embedding, length = ctx.EmbeddingSize</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* llama_get_embeddings_ith(SafeLLamaContextHandle ctx, int i);

        /// <summary>
        /// Set whether the model is in warmup mode or not
        /// If true, all model tensors are activated during llama_decode() to load and cache their weights.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="warmup"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_set_warmup(SafeLLamaContextHandle ctx, [MarshalAs(UnmanagedType.U1)] bool warmup);

        /// <summary>
        /// Возвращает базовые параметры для создания контекста
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaContextParams llama_context_default_params();

        /// <summary>
        /// Apply a loaded control vector to a llama_context, or if data is NULL, clear
        /// the currently loaded vector.
        /// n_embd should be the size of a single layer's control, and data should point
        /// to an n_embd x n_layers buffer starting from layer 1.
        /// il_start and il_end are the layer range the vector should apply to (both inclusive)
        /// See llama_control_vector_load in common to load a control vector.
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="data"></param>
        /// <param name="len"></param>
        /// <param name="n_embd"></param>
        /// <param name="il_start"></param>
        /// <param name="il_end"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_apply_adapter_cvec(SafeLLamaContextHandle ctx, float* data, nuint len, int n_embd, int il_start, int il_end);

        /// <summary>
        ///   LLAMA_API  llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
        /// </summary>
        /// <param name="ctx"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_get_memory(SafeLLamaContextHandle ctx);

        /// <summary>
        /// Clear the memory contents. If data == true, the data buffers will also be cleared together with the metadata
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="data"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_memory_clear(IntPtr /* llama_memory_t */ mem, [MarshalAs(UnmanagedType.U1)] bool data);

        /// <summary>
        /// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        /// <param name="p0"></param>
        /// <param name="p1"></param>
        /// <returns>Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_memory_seq_rm(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq, LLamaPos p0, LLamaPos p1);

        /// <summary>
        /// Copy all tokens that belong to the specified sequence to another sequence
        /// Note that this does not allocate extra KV cache memory - it simply assigns the tokens to the new sequence
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="src"></param>
        /// <param name="dest"></param>
        /// <param name="p0">p0 &lt; 0 : [0,  p1]</param>
        /// <param name="p1">p1 &lt; 0 : [p0, inf)</param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_memory_seq_cp(IntPtr /* llama_memory_t */ mem, LLamaSeqId src, LLamaSeqId dest, LLamaPos p0, LLamaPos p1);

        /// <summary>
        /// Removes all tokens that do not belong to the specified sequence
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_memory_seq_keep(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq);

        /// <summary>
        /// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        /// <param name="p0">p0 &lt; 0 : [0,  p1]</param>
        /// <param name="p1">p1 &lt; 0 : [p0, inf)</param>
        /// <param name="delta"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_memory_seq_add(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq, LLamaPos p0, LLamaPos p1, LLamaPos delta);

        /// <summary>
        /// Integer division of the positions by factor of `d > 1`
        /// <br />
        /// p0 &lt; 0 : [0,  p1]
        /// <br />
        /// p1 &lt; 0 : [p0, inf)
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        /// <param name="p0">p0 &lt; 0 : [0,  p1]</param>
        /// <param name="p1">p1 &lt; 0 : [p0, inf)</param>
        /// <param name="d"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_memory_seq_div(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq, LLamaPos p0, LLamaPos p1, int d);

        /// <summary>
        /// Returns the smallest position present in the memory for the specified sequence.
        /// This is typically non-zero only for SWA caches.
        /// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory.
        /// Return -1 if the sequence is empty.
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaPos llama_memory_seq_pos_min(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq);

        /// <summary>
        /// Returns the largest position present in the memory for the specified sequence.
        /// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory.
        /// Return -1 if the sequence is empty.
        /// </summary>
        /// <param name="mem"></param>
        /// <param name="seq"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaPos llama_memory_seq_pos_max(IntPtr /* llama_memory_t */ mem, LLamaSeqId seq);

        /// <summary>
        /// Check if the memory supports shifting
        /// </summary>
        /// <param name="mem"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_memory_can_shift(IntPtr /* llama_memory_t */ mem);


        #endregion
        #region functions

        private static llama_init_from_model _llama_init_from_model;
        private static llama_free _llama_free;
        private static llama_set_abort_callback _llama_set_abort_callback;
        private static llama_decode _llama_decode;
        private static llama_encode _llama_encode;
        private static llama_set_n_threads _llama_set_n_threads;
        private static llama_n_threads _llama_n_threads;
        private static llama_n_threads_batch _llama_n_threads_batch;
        private static llama_get_logits _llama_get_logits;
        private static llama_get_logits_ith _llama_get_logits_ith;
        private static llama_n_ctx _llama_n_ctx;
        private static llama_n_batch _llama_n_batch;
        private static llama_n_ubatch _llama_n_ubatch;
        private static llama_n_seq_max _llama_n_seq_max;
        private static llama_state_get_size _llama_state_get_size;
        private static llama_state_get_data _llama_state_get_data;
        private static llama_state_set_data _llama_state_set_data;
        private static llama_state_seq_get_size _llama_state_seq_get_size;
        private static llama_state_seq_get_data _llama_state_seq_get_data;
        private static llama_state_seq_set_data _llama_state_seq_set_data;
        private static llama_perf_context _llama_perf_context;
        private static llama_perf_context_print _llama_perf_context_print;
        private static llama_perf_context_reset _llama_perf_context_reset;
        private static llama_synchronize _llama_synchronize;
        private static llama_set_adapter_lora _llama_set_adapter_lora;
        private static llama_rm_adapter_lora _llama_rm_adapter_lora;
        private static llama_clear_adapter_lora _llama_clear_adapter_lora;
        private static llama_pooling_type _llama_pooling_type;
        private static llama_get_embeddings _llama_get_embeddings;
        private static llama_get_embeddings_seq _llama_get_embeddings_seq;
        private static llama_get_embeddings_ith _llama_get_embeddings_ith;
        private static llama_set_warmup _llama_set_warmup;
        private static llama_context_default_params _llama_context_default_params;
        private static llama_apply_adapter_cvec _llama_apply_adapter_cvec;
        private static llama_get_memory _llama_get_memory;
        private static llama_memory_clear       _llama_memory_clear;
        private static llama_memory_seq_rm      _llama_memory_seq_rm;
        private static llama_memory_seq_cp      _llama_memory_seq_cp;
        private static llama_memory_seq_keep    _llama_memory_seq_keep;
        private static llama_memory_seq_pos_min _llama_memory_seq_pos_min;
        private static llama_memory_seq_pos_max _llama_memory_seq_pos_max;

        #endregion
        #endregion

        #region GGML API functions

        #region delegates
        #endregion
        #region functions
        #endregion
        #endregion

        /// <summary>
        /// <purpose> load functions for SafeLlamaContextHandle </purpose>
        /// </summary>
        private static void LoadContextFunctions()
        {
            _llama_context_default_params = GetLibFunction<llama_context_default_params>(_llamaHandle, "llama_context_default_params");
            _llama_init_from_model = GetLibFunction<llama_init_from_model>(_llamaHandle, "llama_init_from_model");
            _llama_free = GetLibFunction<llama_free>(_llamaHandle, "llama_free");
            _llama_set_abort_callback = GetLibFunction<llama_set_abort_callback>(_llamaHandle, "llama_set_abort_callback");
            _llama_decode = GetLibFunction<llama_decode>(_llamaHandle, "llama_decode");
            _llama_encode = GetLibFunction<llama_encode>(_llamaHandle, "llama_encode");
            _llama_set_n_threads = GetLibFunction<llama_set_n_threads>(_llamaHandle, "llama_set_n_threads");
            _llama_n_threads = GetLibFunction<llama_n_threads>(_llamaHandle, "llama_n_threads");
            _llama_n_threads_batch = GetLibFunction<llama_n_threads_batch>(_llamaHandle, "llama_n_threads_batch");
            _llama_get_logits = GetLibFunction<llama_get_logits>(_llamaHandle, "llama_get_logits");
            _llama_get_logits_ith = GetLibFunction<llama_get_logits_ith>(_llamaHandle, "llama_get_logits_ith");
            _llama_n_ctx = GetLibFunction<llama_n_ctx>(_llamaHandle, "llama_n_ctx");
            _llama_n_batch = GetLibFunction<llama_n_batch>(_llamaHandle, "llama_n_batch");
            _llama_n_ubatch = GetLibFunction<llama_n_ubatch>(_llamaHandle, "llama_n_ubatch");
            _llama_n_seq_max = GetLibFunction<llama_n_seq_max>(_llamaHandle, "llama_n_seq_max");
            _llama_state_get_size = GetLibFunction<llama_state_get_size>(_llamaHandle, "llama_state_get_size");
            _llama_state_get_data = GetLibFunction<llama_state_get_data>(_llamaHandle, "llama_state_get_data");
            _llama_state_set_data = GetLibFunction<llama_state_set_data>(_llamaHandle, "llama_state_set_data");
            _llama_state_seq_get_size = GetLibFunction<llama_state_seq_get_size>(_llamaHandle, "llama_state_seq_get_size");
            _llama_state_seq_get_data = GetLibFunction<llama_state_seq_get_data>(_llamaHandle, "llama_state_seq_get_data");
            _llama_state_seq_set_data = GetLibFunction<llama_state_seq_set_data>(_llamaHandle, "llama_state_seq_set_data");
            _llama_perf_context = GetLibFunction<llama_perf_context>(_llamaHandle, "llama_perf_context");
            _llama_perf_context_print = GetLibFunction<llama_perf_context_print>(_llamaHandle, "llama_perf_context_print");
            _llama_perf_context_reset = GetLibFunction<llama_perf_context_reset>(_llamaHandle, "llama_perf_context_reset");
            _llama_synchronize = GetLibFunction<llama_synchronize>(_llamaHandle, "llama_synchronize");
            //_llama_set_adapter_lora = GetLibFunction<llama_set_adapter_lora>(_llamaHandle, "llama_set_adapter_lora");
            //_llama_rm_adapter_lora = GetLibFunction<llama_rm_adapter_lora>(_llamaHandle, "llama_rm_adapter_lora");
            //_llama_clear_adapter_lora = GetLibFunction<llama_clear_adapter_lora>(_llamaHandle, "llama_clear_adapter_lora");
            _llama_pooling_type = GetLibFunction<llama_pooling_type>(_llamaHandle, "llama_pooling_type");
            _llama_get_embeddings = GetLibFunction<llama_get_embeddings>(_llamaHandle, "llama_get_embeddings");
            _llama_get_embeddings_seq = GetLibFunction<llama_get_embeddings_seq>(_llamaHandle, "llama_get_embeddings_seq");
            _llama_get_embeddings_ith = GetLibFunction<llama_get_embeddings_ith>(_llamaHandle, "llama_get_embeddings_ith");
            _llama_set_warmup = GetLibFunction<llama_set_warmup>(_llamaHandle, "llama_set_warmup");
            //_llama_apply_adapter_cvec = GetLibFunction<llama_apply_adapter_cvec>(_llamaHandle, "llama_apply_adapter_cvec");
            _llama_get_memory       = GetLibFunction<llama_get_memory>(_llamaHandle, "llama_get_memory");
            _llama_memory_clear     = GetLibFunction<llama_memory_clear>(_llamaHandle, "llama_memory_clear");
            _llama_memory_seq_rm    = GetLibFunction<llama_memory_seq_rm>(_llamaHandle, "llama_memory_seq_rm");    
            _llama_memory_seq_cp    = GetLibFunction<llama_memory_seq_cp>(_llamaHandle, "llama_memory_seq_cp");
            _llama_memory_seq_keep  = GetLibFunction<llama_memory_seq_keep>(_llamaHandle, "llama_memory_seq_keep");
            _llama_memory_seq_pos_min = GetLibFunction<llama_memory_seq_pos_min>(_llamaHandle, "llama_memory_seq_pos_min");
            _llama_memory_seq_pos_max = GetLibFunction<llama_memory_seq_pos_max>(_llamaHandle, "llama_memory_seq_pos_max");
        }

        public static LLamaContextParams Llama_ContextDefaultParams()
        {
            EnsureInitialized();
            return _llama_context_default_params();
        }

        /// <summary> 
        /// Создаёт новый контекст на основе модели. 
        /// </summary>
        public static SafeLLamaContextHandle Llama_ContextInitFromModel(SafeLlamaModelHandle model, LLamaContextParams @params)
        {
            EnsureInitialized();
            return _llama_init_from_model(model, @params);
        }

        /// <summary> Освобождает контекст. </summary>
        public static void Llama_ContextFree(IntPtr ctx)
        {
            EnsureInitialized();
            _llama_free(ctx);
        }

        /// <summary> Устанавливает callback для прерывания вычислений. </summary>
        /// <param name="ctx">Контекст.</param>
        /// <param name="abortCallback">Делегат, который вызывается нативно библиотекой для проверки необходимости
        /// прервать вычисление</param>
        /// <param name="data">Пользовательские данные, передаваемые в callback.</param>
        public static unsafe void Llama_ContextSetAbortCallback(SafeLLamaContextHandle ctx, GgmlAbortCallback abortCallback, void* data)
        {
            EnsureInitialized();
            _llama_set_abort_callback(ctx, abortCallback, data);
        }

        /// <summary> Декодирует пакет токенов. </summary>
        /// <returns>0 - успех, 1 - нет KV слота, отрицательное - ошибка.</returns>
        public static int Llama_ContextDecode(SafeLLamaContextHandle ctx, LLamaNativeBatch batch)
        {
            EnsureInitialized();
            return _llama_decode(ctx, batch);
        }

        /// <summary> Кодирует пакет токенов (для encoder‑decoder моделей). </summary>
        public static int Llama_ContextEncode(SafeLLamaContextHandle ctx, LLamaNativeBatch batch)
        {
            EnsureInitialized();
            return _llama_encode(ctx, batch);
        }

        /// <summary> Устанавливает количество потоков для декодирования. </summary>
        public static void Llama_ContextSetNThreads(SafeLLamaContextHandle ctx, int nThreads, int nThreadsBatch)
        {
            EnsureInitialized();
            _llama_set_n_threads(ctx, nThreads, nThreadsBatch);
        }

        /// <summary> Возвращает количество потоков для генерации одного токена. </summary>
        public static int Llama_ContextNThreads(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_n_threads(ctx);
        }

        /// <summary> Возвращает количество потоков для пакетной обработки. </summary>
        public static int Llama_ContextNThreadsBatch(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_n_threads_batch(ctx);
        }

        /// <summary> Возвращает указатель на логиты последнего вызова llama_decode. </summary>
        public static unsafe float* Llama_ContextGetLogits(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_get_logits(ctx);
        }

        /// <summary> Возвращает указатель на логиты i-го токена. </summary>
        public static unsafe float* Llama_ContextGetLogitsIth(SafeLLamaContextHandle ctx, int i)
        {
            EnsureInitialized();
            return _llama_get_logits_ith(ctx, i);
        }

        /// <summary> Возвращает размер контекстного окна для данного контекста. </summary>
        public static uint Llama_ContextNCtx(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_n_ctx(ctx);
        }

        /// <summary> Возвращает размер пакета для этого контекста. </summary>
        public static uint Llama_ContextNBatch(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_n_batch(ctx);
        }

        /// <summary> Возвращает размер ubatch для этого контекста. </summary>
        public static uint Llama_ContextNUbatch(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_n_ubatch(ctx);
        }

        /// <summary> Возвращает точный размер состояния (логиты, эмбеддинги, KV‑кэш). </summary>
        public static nuint Llama_ContextStateGetSize(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_state_get_size(ctx);
        }

        /// <summary> Копирует состояние в указанный буфер (unsafe). </summary>
        public static unsafe nuint Llama_ContextStateGetData(SafeLLamaContextHandle ctx, byte* dest, nuint size)
        {
            EnsureInitialized();
            return _llama_state_get_data(ctx, dest, size);
        }

        /// <summary> Копирует состояние в буфер Span. </summary>
        public static unsafe nuint Llama_ContextStateGetData(SafeLLamaContextHandle ctx, Span<byte> dest)
        {
            EnsureInitialized();
            fixed (byte* p = dest)
            {
                return _llama_state_get_data(ctx, p, (nuint)dest.Length);
            }
        }

        /// <summary> Восстанавливает состояние из буфера (unsafe). </summary>
        public static unsafe nuint Llama_ContextStateSetData(SafeLLamaContextHandle ctx, byte* src, nuint size)
        {
            EnsureInitialized();
            return _llama_state_set_data(ctx, src, size);
        }

        /// <summary> Восстанавливает состояние из буфера Span. </summary>
        public static unsafe nuint Llama_ContextStateSetData(SafeLLamaContextHandle ctx, ReadOnlySpan<byte> src)
        {
            EnsureInitialized();
            fixed (byte* p = src)
            {
                return _llama_state_set_data(ctx, p, (nuint)src.Length);
            }
        }

        /// <summary> Возвращает точный размер KV‑кэша для указанной последовательности. </summary>
        public static nuint Llama_ContextStateSeqGetSize(SafeLLamaContextHandle ctx, LLamaSeqId seqId)
        {
            EnsureInitialized();
            return _llama_state_seq_get_size(ctx, seqId);
        }

        /// <summary> Копирует KV‑кэш последовательности в буфер (unsafe). </summary>
        public static unsafe nuint Llama_ContextStateSeqGetData(SafeLLamaContextHandle ctx, byte* dst, nuint size, LLamaSeqId seqId)
        {
            EnsureInitialized();
            return _llama_state_seq_get_data(ctx, dst, size, seqId);
        }

        /// <summary> Копирует KV‑кэш последовательности в буфер Span. </summary>
        public static unsafe nuint Llama_ContextStateSeqGetData(SafeLLamaContextHandle ctx, Span<byte> dst, LLamaSeqId seqId)
        {
            EnsureInitialized();
            fixed (byte* p = dst)
            {
                return _llama_state_seq_get_data(ctx, p, (nuint)dst.Length, seqId);
            }
        }

        /// <summary> Восстанавливает KV‑кэш последовательности из буфера (unsafe). </summary>
        public static unsafe nuint Llama_ContextStateSeqSetData(SafeLLamaContextHandle ctx, byte* src, nuint size, LLamaSeqId destSeqId)
        {
            EnsureInitialized();
            return _llama_state_seq_set_data(ctx, src, size, destSeqId);
        }

        /// <summary> Восстанавливает KV‑кэш последовательности из буфера Span. </summary>
        public static unsafe nuint Llama_ContextStateSeqSetData(SafeLLamaContextHandle ctx, ReadOnlySpan<byte> src, LLamaSeqId destSeqId)
        {
            EnsureInitialized();
            fixed (byte* p = src)
            {
                return _llama_state_seq_set_data(ctx, p, (nuint)src.Length, destSeqId);
            }
        }

        /// <summary> Возвращает структуру с замерами производительности контекста. </summary>
        public static LLamaPerfContextTimings Llama_PerfContext(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_perf_context(ctx);
        }

        /// <summary> Выводит замеры производительности в stderr. </summary>
        public static void Llama_PerfContextPrint(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            _llama_perf_context_print(ctx);
        }

        /// <summary> Сбрасывает замеры производительности. </summary>
        public static void Llama_PerfContextReset(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            _llama_perf_context_reset(ctx);
        }

        /// <summary> Ожидает завершения всех вычислений. Обычно вызывается автоматически. </summary>
        public static void Llama_Synchronize(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            _llama_synchronize(ctx);
        }

        /// <summary> Устанавливает LoRA‑адаптер для контекста. </summary>
        public static int Llama_ContextSetAdapterLora(SafeLLamaContextHandle ctx, IntPtr adapter, float scale)
        {
            EnsureInitialized();
            return _llama_set_adapter_lora(ctx, adapter, scale);
        }

        /// <summary> Удаляет указанный LoRA‑адаптер из контекста. </summary>
        public static int Llama_ContextRemoveAdapterLora(SafeLLamaContextHandle ctx, IntPtr adapter)
        {
            EnsureInitialized();
            return _llama_rm_adapter_lora(ctx, adapter);
        }

        /// <summary> Удаляет все LoRA‑адаптеры из контекста. </summary>
        public static int Llama_ContextClearAdapterLora(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_clear_adapter_lora(ctx);
        }

        /// <summary> Возвращает тип пулинга для этого контекста. </summary>
        public static LLamaPoolingType Llama_ContextPoolingType(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_pooling_type(ctx);
        }

        /// <summary> Возвращает указатель на эмбеддинги. </summary>
        public static unsafe float* Llama_ContextGetEmbeddings(SafeLLamaContextHandle ctx)
        {
            EnsureInitialized();
            return _llama_get_embeddings(ctx);
        }

        /// <summary> Возвращает указатель на эмбеддинги для заданной последовательности. </summary>
        public static unsafe float* Llama_ContextGetEmbeddingsSeq(SafeLLamaContextHandle ctx, LLamaSeqId id)
        {
            EnsureInitialized();
            return _llama_get_embeddings_seq(ctx, id);
        }

        /// <summary> Возвращает указатель на эмбеддинги для i-го выхода. </summary>
        public static unsafe float* Llama_ContextGetEmbeddingsIth(SafeLLamaContextHandle ctx, int i)
        {
            EnsureInitialized();
            return _llama_get_embeddings_ith(ctx, i);
        }

        /// <summary> Включает/выключает режим прогрева модели. </summary>
        public static void Llama_ContextSetWarmup(SafeLLamaContextHandle ctx, bool warmup)
        {
            EnsureInitialized();
            _llama_set_warmup(ctx, warmup);
        }

        public static unsafe int Llama_ContextApplyControlVector(SafeLLamaContextHandle ctx, float* data, nuint len, int n_embd, int il_start, int il_end)
        {
            EnsureInitialized();
            return _llama_apply_adapter_cvec(ctx, data, len, n_embd, il_start, il_end);
        }

        public static void Llama_ContextMemoryClear(SafeLLamaContextHandle ctx, bool data)
        {
            EnsureInitialized();
            _llama_memory_clear(_llama_get_memory(ctx), data);
        }

        public static bool Llama_ContextMemorySeqRemove(SafeLLamaContextHandle ctx, LLamaSeqId seq, LLamaPos p0, LLamaPos p1)
        {
            EnsureInitialized();
            return _llama_memory_seq_rm(_llama_get_memory(ctx), seq, p0, p1);
        }

        public static void Llama_ContextMemorySeqCopy(SafeLLamaContextHandle ctx, LLamaSeqId src, LLamaSeqId dest, LLamaPos p0, LLamaPos p1)
        {
            EnsureInitialized();
            _llama_memory_seq_cp(_llama_get_memory(ctx), src, dest, p0, p1);
        }

        public static void Llama_ContextMemorySeqKeep(SafeLLamaContextHandle ctx, LLamaSeqId seq)
        {
            EnsureInitialized();
            _llama_memory_seq_keep(_llama_get_memory(ctx), seq);
        }

        public static LLamaPos Llama_ContextMemorySeqPosMin(SafeLLamaContextHandle ctx, LLamaSeqId seq)
        {
            EnsureInitialized();
            return _llama_memory_seq_pos_min(_llama_get_memory(ctx), seq);
        }

        public static LLamaPos Llama_ContextMemorySeqPosMax(SafeLLamaContextHandle ctx, LLamaSeqId seq)
        {
            EnsureInitialized();
            return _llama_memory_seq_pos_max(_llama_get_memory(ctx), seq);
        }
    }
}
