using Llama.csharp.Native;

namespace Llama.csharp.Interfaces
{
    /// <summary>
    /// Convert a span of logits into a single sampled token. This interface can be implemented to completely customise the sampling process.
    /// </summary>
    public interface ISamplingPipeline
        : IDisposable
    {
        /// <summary>
        /// Sample a single token from the given context at the given position
        /// </summary>
        /// <param name="ctx">The context being sampled from</param>
        /// <param name="index">Position of logits in last batch </param>
        /// <returns></returns>
        LLamaToken Sample(SafeLLamaContextHandle ctx, int index);

        /// <summary>
        /// Apply this pipeline to a set of token data
        /// </summary>
        /// <param name="ctx"></param>
        /// <param name="data"></param>
        public void Apply(LLamaTokenDataArray data);

        public void Apply(ref LLamaTokenDataArrayNative data);

        /// <summary>
        /// Reset all internal state of the sampling pipeline
        /// </summary>
        void Reset();

        /// <summary>
        /// Update the pipeline, with knowledge that a particular token was just accepted
        /// </summary>
        /// <param name="token"></param>
        void Accept(LLamaToken token);
    }
}
