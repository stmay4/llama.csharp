using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// A C# representation of the mtmd.h `mtmd_context_params` struct
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct MtmdContextParams
    {
        /// <summary>
        /// if true, use loaded gpu backend
        /// </summary>
        public bool use_gpu
        {
            readonly get => Convert.ToBoolean(_use_gpu);
            set => _use_gpu = Convert.ToSByte(value);
        }
        private sbyte _use_gpu;

        /// <summary>
        /// if true, print timings
        /// </summary>
        public bool print_timings
        {
            readonly get => Convert.ToBoolean(_print_timings);
            set => _print_timings = Convert.ToSByte(value);
        }
        private sbyte _print_timings;


        /// <summary>
        /// number of threads to use
        /// </summary>
        public int n_threads;

        /// <summary>
        /// deprecated, use media_marker instead
        /// </summary>
        public IntPtr image_marker;

        /// <summary>
        /// media marker string (pointer to null-terminated UTF-8)
        /// </summary>
        public IntPtr media_marker;

        /// <summary>
        /// flash attention type (see llama_flash_attn_type enum)
        /// </summary>
        public LlamaFlashAttentionType flash_attn_type;

        /// <summary>
        /// whether to run a warmup encode pass after initialization
        /// </summary>
        public bool warmup
        {
            readonly get => Convert.ToBoolean(_warmup);
            set => _warmup = Convert.ToSByte(value);
        }
        private sbyte _warmup;

        /// <summary>
        /// minimum number of tokens for image input (default: read from metadata)
        /// </summary>
        public int image_min_tokens;

        /// <summary>
        /// maximum number of tokens for image input (default: read from metadata)
        /// </summary>
        public int image_max_tokens;

        /// <summary>
        /// callback function passed over to mtmd proper (function pointer)
        /// </summary>
        public IntPtr cb_eval;

        /// <summary>
        /// user data for cb_eval
        /// </summary>
        public IntPtr cb_eval_user_data;

        /// <summary>
        /// maximum number of output tokens in a batch (default: 1024)
        /// Note: this is not a hard-limit; the first image will always be added even if it exceeds this limit.
        /// </summary>
        public int batch_max_tokens;

        /// <summary>
        /// progress callback (function pointer). Pass IntPtr.Zero to disable.
        /// </summary>
        public IntPtr progress_callback;

        /// <summary>
        /// user data for progress_callback
        /// </summary>
        public IntPtr progress_callback_user_data;

        /// <summary>
        /// Get the default LLamaMtmdParams
        /// </summary>
        /// <returns></returns>
        public static MtmdContextParams Default()
        {
            return LlamaCpp.Mtmd_DefaultContextParams();
        }
    }
}
