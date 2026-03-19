using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// 
    /// </summary>
    /// <remarks>llama_sampler_chain_params</remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLamaSamplerChainParams
    {
        /// <summary>
        /// whether to measure performance timings
        /// </summary>
        public bool NoPerf
        {
            readonly get => Convert.ToBoolean(_no_perf);
            set => _no_perf = Convert.ToSByte(value);
        }
        private sbyte _no_perf;

        /// <summary>
        /// Get the default LLamaSamplerChainParams
        /// </summary>
        /// <returns></returns>
        public static LLamaSamplerChainParams Default()
        {
            return LlamaCpp.Llama_SamplerChainDefaultParams();
        }
    }
}
