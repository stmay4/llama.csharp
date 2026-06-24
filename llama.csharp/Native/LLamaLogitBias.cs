using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// A bias to apply directly to a logit. Equivalent llama_logit_bias
    /// For samplers
    /// </summary>
    /// <remarks>
    /// 
    /// typedef struct llama_logit_bias {
    ///     lama_token token;
    ///     float bias;
    /// }
    /// llama_logit_bias;
    /// 
    /// </remarks>
    [StructLayout(LayoutKind.Sequential)]
    public record struct LLamaLogitBias
    {
        /// <summary>
        /// The token to apply the bias to
        /// </summary>
        public LLamaToken Token;

        /// <summary>
        /// The bias to add
        /// </summary>
        public float Bias;
    }
}
