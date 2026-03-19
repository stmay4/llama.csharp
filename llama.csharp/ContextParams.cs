using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System.Text;

namespace Llama.csharp
{
    public class ContextParams : IContextParams
    {
        public uint? ContextSize { get; set; } = 0;

        public uint BatchSize { get; set; } = 512;

        public uint UBatchSize { get; set; } = 512;

        public uint SeqMax { get; set; } = 1;

        public bool Embeddings { get; set; } = false;

        public float? RopeFrequencyBase { get; set; }

        public float? RopeFrequencyScale { get; set; }


        /// <summary>
        /// Параметр не находится в LlamaContextParams, просто здесь хранится для удобства
        /// 
        /// Задает раскодировку для StreamingDecoder, который используется при инференсе 
        /// для превращения токенов в байты заданной кодировки
        /// </summary>
        private string EncodingName { get; set; } = Encoding.UTF8.WebName;
        public Encoding Encoding
        {
            get => Encoding.GetEncoding(EncodingName);
            set => EncodingName = value.WebName;
        }

        public int? Threads { get; set; }

        public int? BatchThreads { get; set; }

        public float? YarnExtrapolationFactor { get; set; }

        public float? YarnAttentionFactor { get; set; }

        public float? YarnBetaFast { get; set; }

        public float? YarnBetaSlow { get; set; }

        public uint? YarnOriginalContext { get; set; }

        public RopeScalingType? YarnScalingType { get; set; }

        public GGMLType? TypeK { get; set; }

        public GGMLType? TypeV { get; set; }

        public bool NoKqvOffload { get; set; }

        public LlamaFlashAttentionType FlashAttention { get; set; } = LlamaFlashAttentionType.Auto;

        public float? DefragThreshold { get; set; }

        public LLamaPoolingType PoolingType { get; set; } = LLamaPoolingType.Unspecified;

        public LLamaAttentionType AttentionType { get; set; } = LLamaAttentionType.Unspecified;
    }
}
