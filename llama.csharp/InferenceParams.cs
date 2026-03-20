using Llama.csharp.Interfaces;

namespace Llama.csharp
{
    public record InferenceParams
        : IInferenceParams
    {
        public int TokensKeep { get; set; } = 0;
        public int MaxTokens { get; set; } = -1;
        public bool DecodeSpecialTokens { get; set; } = true;
        public bool AutoStopFromEOG { get; set; } = true;
        public IReadOnlyList<string> AntiPrompts { get; set; } = [];
        public ISamplingPipeline SamplingPipeline { get; set; } 
            = new TunableSamplerPipeline(new TunableSamplerPipelineSettings(
                [new TopKSampler()],
                new Mirostat2Sampler()));
    }
}
