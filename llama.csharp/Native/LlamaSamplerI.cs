using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// // user code can implement the interface below in order to create custom llama_sampler
    ///struct llama_sampler_i
    ///{
    ///    const char*              (*name)  (const struct llama_sampler * smpl);                                   // can be NULL
    ///    void                     (* accept) (      struct llama_sampler * smpl, llama_token token);              // can be NULL
    ///    void                     (* apply) (      struct llama_sampler * smpl, llama_token_data_array* cur_p);   // required
    ///    void                     (* reset) (      struct llama_sampler * smpl);                                  // can be NULL
    ///    struct llama_sampler *   (* clone) (const struct llama_sampler * smpl);                                  // can be NULL if ctx is NULL
    ///    void                     (* free) (      struct llama_sampler * smpl);                                   // can be NULL if ctx is NULL

    ///    // TODO: API for internal libllama usage for appending the sampling to an existing ggml_cgraph
    ///    //void (*apply_ggml) (struct llama_sampler * smpl, ...);
    ///};
    /// <remarks>C++ llama_sampler_i</remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLamaSamplerI
    {
        /// <summary>
        /// Get the name of this sampler
        /// </summary>
        /// <param name="smpl"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public unsafe delegate byte* NameDelegate(ref LLamaSampler smpl);

        /// <summary>
        /// Update internal sampler state after a token has been chosen
        /// </summary>
        /// <param name="smpl"></param>
        /// <param name="token"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void AcceptDelegate(ref LLamaSampler smpl, LLamaToken token);

        /// <summary>
        /// Apply this sampler to a set of logits
        /// </summary>
        /// <param name="smpl"></param>
        /// <param name="logits"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ApplyDelegate(ref LLamaSampler smpl, ref LLamaTokenDataArrayNative logits);

        /// <summary>
        /// Reset the internal state of this sampler
        /// </summary>
        /// <param name="smpl"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void ResetDelegate(ref LLamaSampler smpl);

        /// <summary>
        /// Create a clone of this sampler
        /// </summary>
        /// <param name="smpl"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public unsafe delegate LLamaSampler* CloneDelegate(ref LLamaSampler smpl);

        /// <summary>
        /// Free all resources held by this sampler
        /// </summary>
        /// <param name="smpl"></param>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void FreeDelegate(ref LLamaSampler smpl);

        public unsafe delegate*<byte*> Name;
        public unsafe delegate*<LLamaSampler*, LLamaToken, void> Accept;
        public unsafe delegate*<LLamaSampler*, LLamaTokenDataArrayNative*, void> Apply;
        public unsafe delegate*<LLamaSampler*, void> Reset;
        public unsafe delegate*<LLamaSampler*, nint> Clone;
        public unsafe delegate*<LLamaSampler*, void> Free;
    }
}
