using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public readonly record struct LlamaEmbedding
    {
        public Memory<float> Data { get; }
        public LlamaEmbeddingType Type { get; }

        public LlamaEmbedding(Memory<float> data, LlamaEmbeddingType type)
        {
            Data = data;
            Type = type;
        }
    }
}
