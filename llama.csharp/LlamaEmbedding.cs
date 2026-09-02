using Llama.csharp.Native;
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
        
        // установлено если MROPE модель. позиция относительная - относительно первого эмбеддинга картинки, перед занесением в llama_decode надо добавить колво обрабтанных токенов
        internal MtmdDecoderPosNative? Pos { get; }

        internal bool UseNonCausal { get; }
        internal LlamaEmbedding(Memory<float> data, LlamaEmbeddingType type, bool useNonCausal, MtmdDecoderPosNative? pos = null)
        {
            Data = data;
            Type = type;
            UseNonCausal = useNonCausal;
            Pos = pos;
        }
    }
}
