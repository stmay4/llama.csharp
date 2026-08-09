using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Interfaces
{
    /// <summary>
    /// The parameters for initializing a mmproj.
    /// </summary>
    public interface IMtmdParams
    {
        /// <summary>
        /// if true, use loaded gpu backend
        /// </summary>
        bool UseGpu { get; }

        /// <summary>
        /// if true, print timings
        /// </summary>
        bool PrintTimings { get; }


        /// <summary>
        /// number of threads to use
        /// </summary>
        int? Threads { get; }

        /// <summary>
        /// media marker string (pointer to null-terminated UTF-8)
        /// </summary>
        //IntPtr? MediaMarker { get; }

        /// <summary>
        /// flash attention type (see llama_flash_attn_type enum)
        /// </summary>
        LlamaFlashAttentionType? FlashAttention { get; }

        /// <summary>
        /// whether to run a warmup encode pass after initialization
        /// </summary>
        bool? Warmup { get; }

        /// <summary>
        /// minimum number of tokens for image input (default: read from metadata)
        /// </summary>
        int? ImageMinTokens { get; }

        /// <summary>
        /// maximum number of tokens for image input (default: read from metadata)
        /// </summary>
        int? ImageMaxTokens { get; }

        /// <summary>
        /// maximum number of output tokens in a batch (default: 1024)
        /// Note: this is not a hard-limit; the first image will always be added even if it exceeds this limit.
        /// </summary>
        int? BatchSize { get; }
    }
}
