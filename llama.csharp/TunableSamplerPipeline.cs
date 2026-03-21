using Llama.csharp.Native;

namespace Llama.csharp
{
    /// <summary>
    /// Настраиваемый с помощью TunableSamplerPipelineSettings конвеер семплинга
    /// </summary>
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
        /// <summary>
        /// Применяет настройки заданные в _settings (TunableSamplerPipelineSettings) к создаваемому конвееру chain
        /// </summary>
        /// <param name="chain"></param>
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
