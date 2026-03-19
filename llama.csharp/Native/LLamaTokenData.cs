using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// Contains a pointer to an array of LLamaTokenData which is pinned in memory.
    /// </summary>
    /// <remarks>C# equivalent of llama_token_data_array</remarks>
    [StructLayout(LayoutKind.Sequential)]
    public struct LLamaTokenData
    {
        /// <summary>
        /// token id
        /// </summary>
        public LLamaToken ID;

        /// <summary>
        /// log-odds of the token
        /// </summary>
        public float Logit;

        /// <summary>
        /// probability of the token
        /// </summary>
        public float Probability;

        /// <summary>
        /// Create a new LLamaTokenData
        /// </summary>
        /// <param name="id"></param>
        /// <param name="logit"></param>
        /// <param name="probability"></param>
        public LLamaTokenData(LLamaToken id, float logit, float probability)
        {
            ID = id;
            Logit = logit;
            Probability = probability;
        }
    }
}
