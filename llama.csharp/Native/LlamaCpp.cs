using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// <tips> Библиотеки явно не выгружаются с помощью NativeLibrary.Free во время работы приложения, 
    /// так как llama cpp может создавать свои потоки, которые таким образом не остановить - требует проверки (backend_free и подобное) </tips>
    /// </summary>
    public static partial class LlamaCpp
    {
        private static bool _initialized = false;
        private static readonly object _initLock = new();

        // Дескрипторы загруженных библиотек
        private static IntPtr _llamaHandle = IntPtr.Zero;
        private static IntPtr _ggmlHandle = IntPtr.Zero;
        private static IntPtr _ggmlBaseHandle = IntPtr.Zero;

        #region LLAMA API functions

        // Делегаты для функций
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate long llama_max_devices();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_backend_init();

        /// <summary>
        /// Print system information
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_print_system_info();


        /// <summary>
        /// Check if memory mapping is supported
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_supports_mmap();

        /// <summary>
        /// Check if memory locking is supported
        /// </summary>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool llama_supports_mlock();


        // Экземпляры делегатов
        private static llama_max_devices _llama_max_devices;
        private static llama_backend_init _llama_backend_init;
        private static llama_print_system_info _llama_print_system_info;
        private static llama_supports_mmap _llama_supports_mmap;
        private static llama_supports_mlock _llama_supports_mlock;

        #endregion

        #region GGML API functions

        #region delegates
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void ggml_backend_load([MarshalAs(UnmanagedType.LPStr)] string backendPath);

        /// <summary>
        /// Get the number of available backend devices
        /// </summary>
        /// <returns>Count of available backend devices</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint ggml_backend_dev_count();

        /// <summary>
        /// Get a backend device by index
        /// </summary>
        /// <param name="i">Device index</param>
        /// <returns>Pointer to the backend device</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr ggml_backend_dev_get(nuint i);

        

        #endregion

        #region functions

        private static ggml_backend_load _ggml_backend_load;
        private static ggml_backend_dev_count _ggml_backend_dev_count;
        private static ggml_backend_dev_get _ggml_backend_dev_get;

        #endregion

        #endregion

        #region GGMLBase API functions

        #region delegates

        /// <summary>
        /// Get the buffer type for a backend device
        /// </summary>
        /// <param name="dev">Backend device pointer</param>
        /// <returns>Pointer to the buffer type</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr ggmlbase_backend_dev_buffer_type(IntPtr dev);

        /// <summary>
        /// Get the name of a buffer type
        /// </summary>
        /// <param name="buft">Buffer type pointer</param>
        /// <returns>Name of the buffer type</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr ggmlbase_backend_buft_name(IntPtr buft);

        #endregion

        #region functions

        private static ggmlbase_backend_dev_buffer_type _ggmlbase_backend_dev_buffer_type;
        private static ggmlbase_backend_buft_name _ggmlbase_backend_buft_name;

        #endregion

        #endregion

        static LlamaCpp() { }

        /// <summary>
        /// <purpose> Выполняет инициализацию: загрузка библиотек llama cpp, получение функций библиотек,
        /// инициализация и загрузка бекендов. </purpose>
        /// </summary>
        /// <exception cref="DirectoryNotFoundException"></exception>
        /// <exception cref="DllNotFoundException"></exception>
        public static void Initialize(string llamaPath, string ggmlPath, string ggmlBasePath, List<string> backendPaths)
        {
            if (_initialized) return;

            lock (_initLock)
            {
                if (_initialized) return;

                ValidateStringParameter(llamaPath, nameof(llamaPath));
                ValidateStringParameter(ggmlPath, nameof(ggmlPath));
                ValidateStringParameter(ggmlBasePath, nameof(ggmlBasePath));

                ValidateListParameter(backendPaths, nameof(backendPaths));

                // Загружаем основные библиотеки
                _llamaHandle = LoadNativeLibrary(llamaPath, "llama.cpp");
                _ggmlHandle = LoadNativeLibrary(ggmlPath, "GGML");
                _ggmlBaseHandle = LoadNativeLibrary(ggmlBasePath, "GGML Base");

                // Получаем указатели на функции
                _llama_max_devices = GetLibFunction<llama_max_devices>(_llamaHandle, "llama_max_devices");
                _llama_backend_init = GetLibFunction<llama_backend_init>(_llamaHandle, "llama_backend_init");
                _llama_print_system_info = GetLibFunction<llama_print_system_info>(_llamaHandle, "llama_print_system_info");
                _llama_supports_mmap = GetLibFunction<llama_supports_mmap>(_llamaHandle, "llama_supports_mmap");
                _llama_supports_mlock = GetLibFunction<llama_supports_mlock>(_llamaHandle, "llama_supports_mlock");

                _ggml_backend_load = GetLibFunction<ggml_backend_load>(_ggmlHandle, "ggml_backend_load");
                _ggml_backend_dev_count = GetLibFunction<ggml_backend_dev_count>(_ggmlHandle, "ggml_backend_dev_count");
                _ggml_backend_dev_get = GetLibFunction<ggml_backend_dev_get>(_ggmlHandle, "ggml_backend_dev_get");

                _ggmlbase_backend_dev_buffer_type = GetLibFunction<ggmlbase_backend_dev_buffer_type>(_ggmlBaseHandle, "ggml_backend_dev_buffer_type");
                _ggmlbase_backend_buft_name = GetLibFunction<ggmlbase_backend_buft_name>(_ggmlBaseHandle, "ggml_backend_buft_name");

                LoadModelFunctions();
                LoadVocabFunctions();
                LoadContextFunctions();
                LoadSamplerFunctions();

                // Выполняем инициализацию
                _llama_max_devices();
                _llama_backend_init();

                // Загружаем бэкенды
                foreach (string backend in backendPaths)
                {
                    _ggml_backend_load(backend);
                }

                _initialized = true;
            }
        }

        private static IntPtr LoadNativeLibrary(string path, string libraryDisplayName)
        {
            if (!NativeLibrary.TryLoad(path, out IntPtr handle) || handle == IntPtr.Zero)
            {
                throw new DllNotFoundException(
                    $"Library load fail: '{libraryDisplayName}'. " +
                    $"Path: {path}. "
                );
            }
            return handle;
        }

        private static void ValidateStringParameter(string value, string parameterName)
        {
            if (string.IsNullOrEmpty(value))
            {
                throw new ArgumentException($"Parameter '{parameterName}' cannot be null or empty.", parameterName);
            }
        }

        private static void ValidateListParameter(List<string> list, string parameterName)
        {
            if (list == null)
            {
                throw new ArgumentNullException(parameterName, $"Parameter '{parameterName}' cannot be null.");
            }

            if (list.Count == 0)
            {
                throw new ArgumentException($"Parameter '{parameterName}' cannot be empty.", parameterName);
            }

            for (int i = 0; i < list.Count; i++)
            {
                if (string.IsNullOrEmpty(list[i]))
                {
                    throw new ArgumentException($"Element at index {i} in '{parameterName}' cannot be null or empty.", parameterName);
                }
            }
        }

        /// <summary>
        /// <purpose> Предназначен для возврата ссылки на функцию по ее имени </purpose>
        /// </summary>
        /// <typeparam name="Delegate"></typeparam>
        /// <param name="libraryHandle"> Ссылка на библиотеку </param>
        /// <param name="functionName"> Имя функции, ссылку на которую надо получить </param>
        /// <returns> Возвращает ссылку на функцию библиотеки </returns>
        /// <exception cref="EntryPointNotFoundException"> функция не найдена </exception>
        private static Delegate GetLibFunction<Delegate>(IntPtr libraryHandle, string functionName)
        {
            if (libraryHandle == IntPtr.Zero) throw new ArgumentNullException("Library handle is null");

            // Получаем указатель на функцию
            if (NativeLibrary.TryGetExport(libraryHandle, functionName, out IntPtr functionPtr))
            {
                // Преобразуем указатель в делегат
                return Marshal.GetDelegateForFunctionPointer<Delegate>(functionPtr);
            }

            throw new EntryPointNotFoundException($"Function {functionName} not exist in library");
        }

        /// <summary>
        /// <purpose> Проверяет инициализацию библиотеки </purpose>
        /// </summary>
        /// <exception cref="InvalidOperationException"> Библиотеки не загружены </exception>
        private static void EnsureInitialized()
        {
            if (!_initialized)
                throw new InvalidOperationException("First of all call LlamaCpp.Initialize()");
        }

        /// <summary>
        /// <purpose> Выполняет функцию API llama.h - llama_max_devices </purpose>
        /// </summary>
        /// <returns> Возвращает количество устройств для вывода </returns>
        public static long Llama_GetMaxDevices()
        {
            EnsureInitialized();
            return _llama_max_devices();
        }

        //#pragma warning disable CS0626 // Method, operator, or accessor is marked external and has no attributes on it
        //        /// <summary>
        //        /// Call once at the end of the program - currently only used for MPI (MPI (Message Passing Interface) — это стандарт интерфейса для передачи сообщений между процессами в параллельных вычислениях. кластера ПК)
        //        /// </summary>
        //        public static extern void llama_backend_free();
        //#pragma warning restore CS0626 // Method, operator, or accessor is marked external and has no attributes on it

        public static nuint GGML_BackendDevCount()
        {
            EnsureInitialized();
            return _ggml_backend_dev_count();
        }

        public static IntPtr GGML_BackendDevGet(nuint i)
        {
            EnsureInitialized();
            return _ggml_backend_dev_get(i);
        }

        public static IntPtr GGMLBase_BackendDevBufferType(IntPtr dev)
        {
            EnsureInitialized();
            return _ggmlbase_backend_dev_buffer_type(dev);
        }

        public static IntPtr GGMLBase_BackendBuftName(IntPtr buft)
        {
            EnsureInitialized();
            return _ggmlbase_backend_buft_name(buft);
        }

        public static bool Llama_SupportsMmap()
        {
            EnsureInitialized();
            return _llama_supports_mmap();
        }

        public static bool Llama_SupportsMlock()
        {
            EnsureInitialized();
            return _llama_supports_mlock();
        }

        ///// <summary>
        ///// Check if GPU offload is supported
        ///// </summary>
        ///// <returns></returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //[return: MarshalAs(UnmanagedType.U1)]
        //public static extern bool llama_supports_gpu_offload();

        ///// <summary>
        ///// Check if RPC offload is supported
        ///// </summary>
        ///// <returns></returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //[return: MarshalAs(UnmanagedType.U1)]
        //public static extern bool llama_supports_rpc();

        // Note: this is not implemented because we don't have a definition for `ggml_numa_strategy` in C#. That definition doesn't
        //       exist because it's not in llama.h, it's in ggml.h which we don't currently build a wrapper for. If there's demand
        //       for better NUMA support that will need adding.
        ///// <summary>
        ///// Optional, enable NUMA optimisations
        ///// </summary>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern void llama_numa_init(ggml_numa_strategy numa);

        ///// <summary>
        ///// Load session file
        ///// </summary>
        ///// <param name="ctx"></param>
        ///// <param name="path_session"></param>
        ///// <param name="tokens_out"></param>
        ///// <param name="n_token_capacity"></param>
        ///// <param name="n_token_count_out"></param>
        ///// <returns></returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //[return: MarshalAs(UnmanagedType.U1)]
        //public static extern bool llama_state_load_file(SafeLLamaContextHandle ctx, string path_session, LLamaToken[] tokens_out, ulong n_token_capacity, out ulong n_token_count_out);

        ///// <summary>
        ///// Save session file
        ///// </summary>
        ///// <param name="ctx"></param>
        ///// <param name="path_session"></param>
        ///// <param name="tokens"></param>
        ///// <param name="n_token_count"></param>
        ///// <returns></returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //[return: MarshalAs(UnmanagedType.U1)]
        //public static extern bool llama_state_save_file(SafeLLamaContextHandle ctx, string path_session, LLamaToken[] tokens, ulong n_token_count);

        ///// <summary>
        ///// Saves the specified sequence as a file on specified filepath. Can later be loaded via <see cref="llama_state_load_file(SafeLLamaContextHandle, string, LLamaToken[], ulong, out ulong)"/>
        ///// </summary>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern unsafe nuint llama_state_seq_save_file(SafeLLamaContextHandle ctx, string filepath, LLamaSeqId seq_id, LLamaToken* tokens, nuint n_token_count);

        ///// <summary>
        ///// Loads a sequence saved as a file via <see cref="llama_state_save_file(SafeLLamaContextHandle, string, LLamaToken[], ulong)"/> into the specified sequence
        ///// </summary>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern unsafe nuint llama_state_seq_load_file(SafeLLamaContextHandle ctx, string filepath, LLamaSeqId dest_seq_id, LLamaToken* tokens_out, nuint n_token_capacity, out nuint n_token_count_out);

        ///// <summary>
        ///// Set whether to use causal attention or not. If set to true, the model will only attend to the past tokens
        ///// </summary>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern void llama_set_causal_attn(SafeLLamaContextHandle ctx, [MarshalAs(UnmanagedType.U1)] bool causalAttn);

        ///// <summary>
        ///// Set whether the model is in embeddings mode or not. 
        ///// </summary>
        ///// <param name="ctx"></param>
        ///// <param name="embeddings">If true, embeddings will be returned but logits will not</param>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern void llama_set_embeddings(SafeLLamaContextHandle ctx, [MarshalAs(UnmanagedType.U1)] bool embeddings);




        
        ///// <summary>
        ///// Apply chat template. Inspired by hf apply_chat_template() on python.
        ///// </summary>
        ///// <param name="tmpl">A Jinja template to use for this chat. If this is nullptr, the model’s default chat template will be used instead.</param>
        ///// <param name="chat">Pointer to a list of multiple llama_chat_message</param>
        ///// <param name="n_msg">Number of llama_chat_message in this chat</param>
        ///// <param name="add_ass">Whether to end the prompt with the token(s) that indicate the start of an assistant message.</param>
        ///// <param name="buf">A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)</param>
        ///// <param name="length">The size of the allocated buffer</param>
        ///// <returns>The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.</returns>
        //public static unsafe int llama_chat_apply_template(byte* tmpl, LLamaChatMessage* chat, nuint n_msg, [MarshalAs(UnmanagedType.U1)] bool add_ass, byte* buf, int length)
        //{
        //    return internal_llama_chat_apply_template(tmpl, chat, n_msg, add_ass, buf, length);

        //    [DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "llama_chat_apply_template")]
        //    static extern int internal_llama_chat_apply_template(byte* tmpl, LLamaChatMessage* chat, nuint n_msg, [MarshalAs(UnmanagedType.U1)] bool add_ass, byte* buf, int length);
        //}

        ///// <summary>
        ///// Get list of built-in chat templates
        ///// </summary>
        ///// <param name="output"></param>
        ///// <param name="len"></param>
        ///// <returns></returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern unsafe int llama_chat_builtin_templates(char** output, nuint len);





        ///// <summary>
        ///// Build a split GGUF final path for this chunk.
        ///// llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
        ///// </summary>
        ///// <param name="split_path"></param>
        ///// <param name="maxlen"></param>
        ///// <param name="path_prefix"></param>
        ///// <param name="split_no"></param>
        ///// <param name="split_count"></param>
        ///// <returns>Returns the split_path length.</returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern int llama_split_path(string split_path, nuint maxlen, string path_prefix, int split_no, int split_count);

        ///// <summary>
        ///// Extract the path prefix from the split_path if and only if the split_no and split_count match.
        ///// llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
        ///// </summary>
        ///// <param name="split_prefix"></param>
        ///// <param name="maxlen"></param>
        ///// <param name="split_path"></param>
        ///// <param name="split_no"></param>
        ///// <param name="split_count"></param>
        ///// <returns>Returns the split_prefix length.</returns>
        //[DllImport(llamaDll, CallingConvention = CallingConvention.Cdecl)]
        //public static extern int llama_split_prefix(string split_prefix, nuint maxlen, string split_path, int split_no, int split_count);

    }
}
