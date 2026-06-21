using Llama.csharp.Native;

namespace Llama.csharp
{
    /// <summary>
    /// Sampling pipeline configured via <see cref="TunableSamplerPipelineSettings"/>.
    /// </summary>
    public class TunableSamplerPipeline : BaseSamplingPipeline
    {
        /// <summary>
        /// Default pipeline settings. Uses an empty sampler list and a <see cref="GreedySampler"/> as finalizer.
        /// </summary>
        private TunableSamplerPipelineSettings _settings = new TunableSamplerPipelineSettings([],new GreedySampler());

        /// <summary>
        /// Initializes a new instance of <see cref="TunableSamplerPipeline"/> with the specified settings.
        /// </summary>
        /// <param name="settings">The settings that define the sampler chain.</param>
        public TunableSamplerPipeline(TunableSamplerPipelineSettings settings)
        {
            _settings = settings;
        }

        protected override SafeLLamaSamplerChainHandle CreateChain()
        {
            var chain = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());

            tuneChainFromSettings(chain);

            return chain;
        }
        /// <summary>
        /// Applies the settings stored in <see cref="_settings"/> to the sampler chain being built.
        /// </summary>
        /// <param name="chain">The sampler chain to configure.</param>
        private void tuneChainFromSettings(SafeLLamaSamplerChainHandle chain)
        {
            foreach (ISampler sampler in _settings.Samplers)
            {
                sampler.AddToChain(chain);
            }
            _settings.FinalizeSampler.AddToChain(chain);
        }
    }
}
