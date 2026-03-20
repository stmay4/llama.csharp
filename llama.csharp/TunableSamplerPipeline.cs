using Llama.csharp.Native;

namespace Llama.csharp
{
    public class TunableSamplerPipeline : BaseSamplingPipeline
    {

        private TunableSamplerPipelineSettings _settings = new TunableSamplerPipelineSettings([],new GreedySampler());

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
