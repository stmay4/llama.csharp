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

        public uint SeqMax { get; init; } = 1;

        public bool Embeddings { get; init; } = false;

        public float? RopeFrequencyBase { get; init; }

        public float? RopeFrequencyScale { get; init; }


        /// <summary>
        /// Параметр не находится в LlamaContextParams, просто здесь хранится для удобства
        /// 
        /// Задает раскодировку для StreamingDecoder, который используется при инференсе 
        /// для превращения токенов в байты заданной кодировки
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

        public bool NoKqvOffload { get; init; }

        public LlamaFlashAttentionType FlashAttention { get; init; } = LlamaFlashAttentionType.Auto;

        public float? DefragThreshold { get; init; }

        public LLamaPoolingType PoolingType { get; init; } = LLamaPoolingType.Unspecified;

        public LLamaAttentionType AttentionType { get; init; } = LLamaAttentionType.Unspecified;
        public bool KVunified { get; init; } = true;
    }
}
