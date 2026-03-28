using Llama.csharp.Interfaces;

namespace Llama.csharp
{
    public record InferenceParams: IInferenceParams
    {
        public int TokensKeep { get; init; } = 0;
        public int MaxTokens { get; init; } = -1; //-1 - Бесконечность
        public bool DecodeSpecialTokens { get; init; } = true;
        public bool AutoStopFromEOG { get; init; } = true;
        public IReadOnlyList<string> AntiPrompts { get; init; } = [];
        public ISamplingPipeline SamplingPipeline { get; init; } 
            = new TunableSamplerPipeline(new TunableSamplerPipelineSettings(
                [new TopKSampler()],
                new Mirostat2Sampler()));
    }
}
