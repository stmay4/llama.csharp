using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// </summary>
    /// <remarks>
    /// struct llama_sampler {
    ///    const struct llama_sampler_i * iface;
    ///    llama_sampler_context_t ctx;
    ///};
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct LLamaSampler
    {
        /// <summary>
        /// Holds the function pointers which make up the actual sampler
        /// </summary>
        public LLamaSamplerI* Interface;

        /// <summary>
        /// Any additional context this sampler needs, may be anything. We will use it
        /// to hold a GCHandle.
        /// </summary>
        public nint Context;
    }
}
