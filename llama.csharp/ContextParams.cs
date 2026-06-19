using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System.Text;

namespace Llama.csharp
{
    public class ContextParams : IContextParams
    {
        public uint? ContextSize { get; init; } = 0;

        public uint BatchSize { get; init; } = 512;

        public uint UBatchSize { get; init; } = 512;

        public uint SeqMax { get; init; } = 1; //max is 64 maybe, maybe 256. I haven't checked yet.

        public bool Embeddings { get; init; } = false;

        public float? RopeFrequencyBase { get; init; }

        public float? RopeFrequencyScale { get; init; }


        /// <summary>
        /// This parameter is not part of LlamaContextParams, it is stored here for convenience.
        /// 
        /// Specifies the decoding for the StreamingDecoder, which is used during inference
        /// to convert tokens into bytes of the specified encoding.
        /// </summary>
        private string EncodingName { get; init; } = Encoding.UTF8.WebName;
        public Encoding Encoding
        {
            get => Encoding.GetEncoding(EncodingName);
            init => EncodingName = value.WebName;
        }

        public int? Threads { get; init; }

        public int? BatchThreads { get; init; }

        public float? YarnExtrapolationFactor { get; init; }

        public float? YarnAttentionFactor { get; init; }

        public float? YarnBetaFast { get; init; }

        public float? YarnBetaSlow { get; init; }

        public uint? YarnOriginalContext { get; init; }

        public RopeScalingType? YarnScalingType { get; init; }

        public GGMLType? TypeK { get; init; }

        public GGMLType? TypeV { get; init; }

        public bool NoKqvOffload { get; init; } = true;

        public LlamaFlashAttentionType FlashAttention { get; init; } = LlamaFlashAttentionType.Auto;

        public float? DefragThreshold { get; init; }

        public LLamaPoolingType PoolingType { get; init; } = LLamaPoolingType.Unspecified;

        public LLamaAttentionType AttentionType { get; init; } = LLamaAttentionType.Unspecified;
        public bool? KVunified { get; init; } = true;
        public bool? OPoffload { get; init; } = null;
        public bool? NoPerf { get; init; } = null;
    }
}
