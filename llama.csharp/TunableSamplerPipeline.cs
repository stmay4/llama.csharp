using Llama.csharp.Native;

namespace Llama.csharp
{
    internal class TunableSamplerPipeline : BaseSamplingPipeline
    {

        private TunableSamplerPipelineSettings? _settings = null;

        public TunableSamplerPipeline(TunableSamplerPipelineSettings? settings)
        {
            _settings = settings;
        }

        protected override SafeLLamaSamplerChainHandle CreateChain(SafeLLamaContextHandle context)
        {
            var chain = SafeLLamaSamplerChainHandle.Create(LLamaSamplerChainParams.Default());

            tuneChainFromSettings(chain);

            return chain;
        }

        private SafeLLamaSamplerChainHandle tuneChainFromSettings(SafeLLamaSamplerChainHandle chain)
        {
            //здесь настройка
            return chain;
        }
    }
}
