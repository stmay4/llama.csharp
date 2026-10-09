using Llama.csharp.Native;

namespace Llama.csharp
{
    public readonly record struct LlamaEmbedding
    {
        public Memory<float> Data { get; }
        public LlamaEmbeddingType Type { get; }

        // set if MROPE model. The position here is relative - relative to the first embedding for an image/audio; before passing to llama_decode, the number of processed tokens must be added to it
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
