using System;
using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// Represents a mapping between a tensor name pattern and a backend buffer type
    /// Original C++ type: llama_model_tensor_buft_override
    /// (for MOE load)
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct LLamaModelTensorBufferOverride
    {
        /// <summary>
        /// Tensor name pattern to match
        /// </summary>
        public byte* Pattern;

        /// <summary>
        /// Backend buffer type to use for matching tensors, as obtained via ggml_backend_dev_buffer_type
        /// </summary>
        public nint BufferType;
    }
}
