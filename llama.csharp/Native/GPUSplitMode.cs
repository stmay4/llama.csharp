namespace Llama.csharp.Native
{
    /// <remarks>C++ llama_split_mode</remarks>
    public enum GPUSplitMode
    {
        /// <summary>
        /// Single GPU
        /// </summary>
        None = 0,

        /// <summary>
        /// Split layers and KV across GPUs
        /// </summary>
        Layer = 1,

        /// <summary>
        /// split layers and KV across GPUs, use tensor parallelism if supported
        /// </summary>
        Row = 2,

        Tensor = 3,
    }
}
