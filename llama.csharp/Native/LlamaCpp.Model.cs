using CommunityToolkit.HighPerformance.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace Llama.csharp.Native
{
    public static partial class LlamaCpp
    {
        #region LLAMA API functions

        #region delegates

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaRopeType llama_model_rope_type(SafeLlamaModelHandle model);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate byte* llama_model_chat_template(SafeLlamaModelHandle model, string? name);

        /// <summary>
        /// Load the model from a file
        /// If the file is split into multiple parts, the file name must follow this pattern: {name}-%05d-of-%05d.gguf
        /// If the split file name does not follow this pattern, use llama_model_load_from_splits
        /// </summary>
        /// <param name="path"></param>
        /// <param name="params"></param>
        /// <returns>The loaded model, or null on failure.</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeLlamaModelHandle llama_model_load_from_file(string path, LLamaModelParams @params);

        /// <summary>
        /// Load the model from multiple splits (support custom naming scheme)
        /// The paths must be in the correct order
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        //todo: support llama_model_load_from_splits
        private unsafe delegate SafeLlamaModelHandle llama_model_load_from_splits(char** paths, nuint nPaths, LLamaModelParams @params);

        /// <summary>
        /// Frees all allocated memory associated with a model
        /// </summary>
        /// <param name="model"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_model_free(IntPtr model);

        /// <summary>
        /// Get the number of metadata key/value pairs
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_meta_count(SafeLlamaModelHandle model);

        /// <summary>
        /// Get metadata key name by index
        /// </summary>
        /// <param name="model"></param>
        /// <param name="index"></param>
        /// <param name="buf">буфер с результатом</param>
        /// <param name="bufSize"></param>
        /// <returns>The length of the string on success (even if the buffer is too small). -1 is the key does not exist.</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_model_meta_key_by_index(SafeLlamaModelHandle model, int index, byte* buf, nint bufSize);

        /// <summary>
        /// Get metadata value as a string by index
        /// </summary>
        /// <param name="model"></param>
        /// <param name="index"></param>
        /// <param name="buf"></param>
        /// <param name="bufSize"></param>
        /// <returns>The length of the string on success (even if the buffer is too small). -1 is the key does not exist.</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_model_meta_val_str_by_index(SafeLlamaModelHandle model, int index, byte* buf, nint bufSize);

        /// <summary>
        /// Get metadata value as a string by key name
        /// </summary>
        /// <param name="model"></param>
        /// <param name="key"></param>
        /// <param name="buf"></param>
        /// <param name="bufSize"></param>
        /// <returns>The length of the string on success, or -1 on failure</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_model_meta_val_str(SafeLlamaModelHandle model, byte* key, byte* buf, nint bufSize);


        /// <summary>
        /// Get the number of tokens in the model vocabulary
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_n_vocab(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the size of the context window for the model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_n_ctx_train(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the dimension of embedding vectors from this model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_n_embd(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the number of layers in this model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_n_layer(SafeLlamaModelHandle model);


        /// <summary>
        /// Get the number of heads in this model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_n_head(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the number of KV heads in this model
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_n_head_kv(SafeLlamaModelHandle model);

        /// <summary>
        /// Get a string describing the model type
        /// </summary>
        /// <param name="model"></param>
        /// <param name="buf"></param>
        /// <param name="bufSize"></param>
        /// <returns>The length of the string on success (even if the buffer is too small)., or -1 on failure</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_model_desc(SafeLlamaModelHandle model, byte* buf, nint bufSize);

        /// <summary>
        /// Get the size of the model in bytes
        /// </summary>
        /// <param name="model"></param>
        /// <returns>The size of the model</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate ulong llama_model_size(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the number of parameters in this model
        /// </summary>
        /// <param name="model"></param>
        /// <returns>The functions return the length of the string on success, or -1 on failure</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate ulong llama_model_n_params(SafeLlamaModelHandle model);

        /// <summary>
        /// Get the model's RoPE frequency scaling factor
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate float llama_model_rope_freq_scale_train(SafeLlamaModelHandle model);

        /// <summary>
        /// For encoder-decoder models, this function returns id of the token that must be provided
        /// to the decoder to start generating output sequence. For other models, it returns -1.
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_model_decoder_start_token(SafeLlamaModelHandle model);

        /// <summary>
        /// Returns true if the model contains an encoder that requires llama_encode() call
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_model_has_encoder(SafeLlamaModelHandle model);

        /// <summary>
        /// Returns true if the model contains a decoder that requires llama_decode() call
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_model_has_decoder(SafeLlamaModelHandle model);


        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_adapter_lora_init(SafeLlamaModelHandle model, string path);

        /// <summary>
        /// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_model_is_recurrent(SafeLlamaModelHandle model);

        /// <summary>
        /// Returns true if the model is hybrid (like Jamba, Granite, etc.)
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_model_is_hybrid(SafeLlamaModelHandle model);

        /// <summary>
        /// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
        /// </summary>
        /// <param name="model"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_model_is_diffusion(SafeLlamaModelHandle model);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaVocabNative* llama_model_get_vocab(SafeLlamaModelHandle model);

        /// <summary>
        /// Возвращает базовые параметры для загрузки модели
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaModelParams llama_model_default_params();

        #endregion

        #region functions

        private static llama_model_rope_type             _llama_model_rope_type;
        private static llama_model_chat_template         _llama_model_chat_template;
        private static llama_model_load_from_file        _llama_model_load_from_file;
        private static llama_model_free                  _llama_model_free;
        private static llama_model_meta_count            _llama_model_meta_count;
        private static llama_model_meta_key_by_index     _llama_model_meta_key_by_index;
        private static llama_model_meta_val_str_by_index _llama_model_meta_val_str_by_index;
        private static llama_model_meta_val_str          _llama_model_meta_val_str;
        private static llama_n_vocab                     _llama_n_vocab;
        private static llama_model_n_ctx_train           _llama_model_n_ctx_train;
        private static llama_model_n_embd                _llama_model_n_embd;
        private static llama_model_n_layer               _llama_model_n_layer;
        private static llama_model_n_head                _llama_model_n_head;
        private static llama_model_n_head_kv             _llama_model_n_head_kv;
        private static llama_model_desc                  _llama_model_desc;
        private static llama_model_size                  _llama_model_size;
        private static llama_model_n_params              _llama_model_n_params;
        private static llama_model_rope_freq_scale_train _llama_model_rope_freq_scale_train;
        private static llama_model_decoder_start_token   _llama_model_decoder_start_token;
        private static llama_model_has_encoder           _llama_model_has_encoder;
        private static llama_model_has_decoder           _llama_model_has_decoder;
        private static llama_adapter_lora_init           _llama_adapter_lora_init;
        private static llama_model_is_recurrent          _llama_model_is_recurrent;
        private static llama_model_is_hybrid             _llama_model_is_hybrid;
        private static llama_model_is_diffusion          _llama_model_is_diffusion;
        private static llama_model_get_vocab             _llama_model_get_vocab;
        private static llama_model_default_params        _llama_model_default_params;

        #endregion

        #endregion

        #region GGML API functions

        // Делегаты для функций

        // Экземпляры делегатов

        #endregion


        /// <summary>
        /// <purpose> Задает указатели на функции llama cpp, необходимые SafeLlamaModelHandle </purpose>
        /// </summary>
        private static void LoadModelFunctions()
        {
            _llama_model_default_params = GetLibFunction<llama_model_default_params>(_llamaHandle, "llama_model_default_params");
            _llama_model_rope_type = GetLibFunction<llama_model_rope_type>(_llamaHandle, "llama_model_rope_type");
            _llama_model_chat_template = GetLibFunction<llama_model_chat_template>(_llamaHandle, "llama_model_chat_template");
            _llama_model_load_from_file = GetLibFunction<llama_model_load_from_file>(_llamaHandle, "llama_model_load_from_file");
            _llama_model_free = GetLibFunction<llama_model_free>(_llamaHandle, "llama_model_free");
            _llama_model_meta_count = GetLibFunction<llama_model_meta_count>(_llamaHandle, "llama_model_meta_count");
            _llama_model_meta_key_by_index = GetLibFunction<llama_model_meta_key_by_index>(_llamaHandle, "llama_model_meta_key_by_index");
            _llama_model_meta_val_str_by_index = GetLibFunction<llama_model_meta_val_str_by_index>(_llamaHandle, "llama_model_meta_val_str_by_index");
            _llama_model_meta_val_str = GetLibFunction<llama_model_meta_val_str>(_llamaHandle, "llama_model_meta_val_str");
            _llama_n_vocab = GetLibFunction<llama_n_vocab>(_llamaHandle, "llama_n_vocab");
            _llama_model_n_ctx_train = GetLibFunction<llama_model_n_ctx_train>(_llamaHandle, "llama_model_n_ctx_train");
            _llama_model_n_embd = GetLibFunction<llama_model_n_embd>(_llamaHandle, "llama_model_n_embd");
            _llama_model_n_layer = GetLibFunction<llama_model_n_layer>(_llamaHandle, "llama_model_n_layer");
            _llama_model_n_head = GetLibFunction<llama_model_n_head>(_llamaHandle, "llama_model_n_head");
            _llama_model_n_head_kv = GetLibFunction<llama_model_n_head_kv>(_llamaHandle, "llama_model_n_head_kv");
            _llama_model_n_params = GetLibFunction<llama_model_n_params>(_llamaHandle, "llama_model_n_params");
            _llama_model_rope_freq_scale_train = GetLibFunction<llama_model_rope_freq_scale_train>(_llamaHandle, "llama_model_rope_freq_scale_train");
            _llama_model_decoder_start_token = GetLibFunction<llama_model_decoder_start_token>(_llamaHandle, "llama_model_decoder_start_token");
            _llama_model_has_encoder = GetLibFunction<llama_model_has_encoder>(_llamaHandle, "llama_model_has_encoder");
            _llama_model_has_decoder = GetLibFunction<llama_model_has_decoder>(_llamaHandle, "llama_model_has_decoder");
            _llama_adapter_lora_init = GetLibFunction<llama_adapter_lora_init>(_llamaHandle, "llama_adapter_lora_init");
            _llama_model_is_recurrent = GetLibFunction<llama_model_is_recurrent>(_llamaHandle, "llama_model_is_recurrent");
            _llama_model_is_hybrid = GetLibFunction<llama_model_is_hybrid>(_llamaHandle, "llama_model_is_hybrid");
            _llama_model_is_diffusion = GetLibFunction<llama_model_is_diffusion>(_llamaHandle, "llama_model_is_diffusion");
            _llama_model_get_vocab = GetLibFunction<llama_model_get_vocab>(_llamaHandle, "llama_model_get_vocab");
        }

        public static LLamaModelParams Llama_ModelDefaultParams()
        {
            EnsureInitialized();
            return _llama_model_default_params();
        }
        public static LLamaRopeType Llama_ModelRopeType(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_rope_type(model);
        }

        public static int Llama_ModelNContextTrain(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_ctx_train(model);
        }
        public static float Llama_ModelRopeFreqScaleTrain(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_rope_freq_scale_train(model);
        }

        public static int Llama_ModelNEmbd(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_embd(model);
        }

        public static ulong Llama_ModelSize(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_size(model);
        }

        public static ulong Llama_ModelNParams(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_params(model);
        }

        public static int Llama_ModelNLayer(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_layer(model);
        }

        public static int Llama_ModelNHead(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_head(model);
        }

        public static int Llama_ModelNHeadKV(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_n_head_kv(model);
        }

        public static bool Llama_ModelHasEncoder(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_has_encoder(model);
        }

        public static bool Llama_ModelHasDecoder(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_has_decoder(model);
        }

        public static bool Llama_ModelIsRecurrent(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_is_recurrent(model);
        }

        public static bool Llama_ModelIsHybrid(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_is_hybrid(model);
        }

        public static bool Llama_ModelIsDiffusion(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_is_diffusion(model);
        }

        public static unsafe int Llama_ModelDesc(SafeLlamaModelHandle model, byte* buf, nint bufSize)
        {
            EnsureInitialized();
            return _llama_model_desc(model, buf, bufSize);
        }

        public static int Llama_ModelMetaCount(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_meta_count(model);
        }

        public static void Llama_ModelFree(IntPtr model)
        {
            EnsureInitialized();
            _llama_model_free(model);
        }

        public static SafeLlamaModelHandle Llama_ModelLoadFromFile(string path, LLamaModelParams @params)
        {
            EnsureInitialized();
            return _llama_model_load_from_file(path, @params);
        }

        public static unsafe LLamaVocabNative* Llama_ModelGetVocab(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_get_vocab(model);
        }

        public static unsafe byte* Llama_ModelChatTemplate(SafeLlamaModelHandle model, string? name)
        {
            EnsureInitialized();
            return _llama_model_chat_template(model, name);
        }

        public static int Llama_ModelDecoderStartToken(SafeLlamaModelHandle model)
        {
            EnsureInitialized();
            return _llama_model_decoder_start_token(model);
        }


        /// <summary>
        /// Get metadata key name by index
        /// </summary>
        /// <param name="model">Model to fetch from</param>
        /// <param name="index">Index of key to fetch</param>
        /// <param name="dest">buffer to write result into</param>
        /// <returns>The length of the string on success (even if the buffer is too small). -1 is the key does not exist.</returns>
        public static int Llama_ModelMetaKeyByIndex(SafeLlamaModelHandle model, int index, Span<byte> dest)
        {
            EnsureInitialized();
            unsafe
            {
                fixed (byte* destPtr = dest)
                {
                    return _llama_model_meta_key_by_index(model, index, destPtr, dest.Length);
                }
            }
        }

        /// <summary>
        /// Get metadata value as a string by index
        /// </summary>
        /// <param name="model">Model to fetch from</param>
        /// <param name="index">Index of val to fetch</param>
        /// <param name="dest">Buffer to write result into</param>
        /// <returns>The length of the string on success (even if the buffer is too small). -1 is the key does not exist.</returns>
        public static int Llama_ModelMetaValStrByIndex(SafeLlamaModelHandle model, int index, Span<byte> dest)
        {
            EnsureInitialized();
            unsafe
            {
                fixed (byte* destPtr = dest)
                {
                    return _llama_model_meta_val_str_by_index(model, index, destPtr, dest.Length);
                }
            }
        }

        /// <summary>
        /// Get metadata value as a string by key name
        /// </summary>
        /// <param name="model"></param>
        /// <param name="key"></param>
        /// <param name="dest"></param>
        /// <returns>The length of the string on success, or -1 on failure</returns>
        public static int Llama_ModelMetaValStr(SafeLlamaModelHandle model, string key, Span<byte> dest)
        {
            EnsureInitialized();

            var bytesCount = Encoding.UTF8.GetByteCount(key);
            using var bytes = SpanOwner<byte>.Allocate(bytesCount);

            unsafe
            {
                fixed (char* keyPtr = key)
                fixed (byte* bytesPtr = bytes.Span)
                fixed (byte* destPtr = dest)
                {
                    // Convert text into bytes
                    Encoding.UTF8.GetBytes(keyPtr, key.Length, bytesPtr, bytesCount);

                    return _llama_model_meta_val_str(model, bytesPtr, destPtr, dest.Length);
                }
            }
        }
    }
}
