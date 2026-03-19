#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

namespace Llama.csharp.Native
{
    /// <summary>
    /// llama_rope_type
    /// </summary>
    public enum LLamaRopeType
    {
        None = -1,
        Norm = 0,
        //todo:NEOX = GGML_ROPE_TYPE_NEOX,
        //todo:MROPE = LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_MROPE,
        //todo:IMROPE = LLAMA_ROPE_TYPE_MROPE = GGML_ROPE_TYPE_IMROPE,
        //todo:VISION = LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    }
}
