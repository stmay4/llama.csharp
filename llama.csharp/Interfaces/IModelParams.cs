using Llama.csharp.Abstractions;
using Llama.csharp.Native;
using System.Buffers;
using System.Collections;

namespace Llama.csharp.Interfaces
{
    /// <summary>
    /// The parameters for initializing a LLama model.
    /// </summary>
    public interface IModelParams
    {

        /// <summary>
        /// Buffer type overrides for specific tensor patterns, allowing you to specify hardware devices to use for individual tensors or sets of tensors.
        /// Equivalent to --override-tensor or -ot on the llama.cpp command line or tensor_buft_overrides internally.
        /// </summary>
        List<TensorBufferOverride> TensorBufferOverrides { get; }

        /// <summary>
        /// Number of layers to run in VRAM / GPU memory (n_gpu_layers)
        /// </summary>
        int GpuLayerCount { get; }

        /// <summary>
        /// How to split the model across multiple GPUs
        /// </summary>
        GPUSplitMode? SplitMode { get; }

        /// <summary>
        /// main_gpu interpretation depends on split_mode:
        /// <list type="bullet">
        ///     <item>
        ///         <term>None</term>
        ///         <description>The GPU that is used for the entire mode.</description>
        ///     </item>
        ///     <item>
        ///         <term>Row</term>
        ///         <description>The GPU that is used for small tensors and intermediate results.</description>
        ///     </item>
        ///     <item>
        ///         <term>Layer</term>
        ///         <description>Ignored.</description>
        ///     </item>
        /// </list>
        /// </summary>
        int MainGpu { get; set; }

        /// <summary>
        /// how split tensors should be distributed across GPUs
        /// </summary>
        TensorSplitsCollection TensorSplits { get; }

        /// <summary>
        /// Override specific metadata items in the model
        /// </summary>
        //List<MetadataOverride> MetadataOverrides { get; }

        /// <summary>
        /// Load vocab only (no weights)
        /// </summary>
        bool VocabOnly { get; }

        /// <summary>
        /// Use mmap for faster loads (use_mmap)
        /// </summary>
        bool UseMemorymap { get; }

        /// <summary>
        /// Use mlock to keep model in memory (use_mlock)
        /// </summary>
        bool UseMemoryLock { get; }

        /// <summary>
        /// Validate model tensor data before loading
        /// </summary>
        bool CheckTensors { get; }

        /// <summary>
        /// 
        /// </summary>
        bool UseExtraBufs { get; }

        /// <summary>
        /// 
        /// </summary>
        bool NoHost { get; }

        /// <summary>
        /// No allocate memory for weights (not work?)
        /// </summary>
        bool NoAlloc { get; }

        /// <summary>
        /// Path to gguf model
        /// </summary>
        string ModelPath { get; }
    }

    public sealed class TensorSplitsCollection
    : IEnumerable<float>
    {
        internal readonly float[] Splits = new float[LlamaCpp.Llama_GetMaxDevices()];

        /// <summary>
        /// The size of this array
        /// </summary>
        public int Length => Splits.Length;

        /// <summary>
        /// Get or set the proportion of work to do on the given device.
        /// </summary>
        /// <remarks>"[ 3, 2 ]" will assign 60% of the data to GPU 0 and 40% to GPU 1.</remarks>
        /// <param name="index"></param>
        /// <returns></returns>
        public float this[int index]
        {
            get => Splits[index];
            set => Splits[index] = value;
        }

        /// <summary>
        /// Create a new tensor splits collection, copying the given values
        /// </summary>
        /// <param name="splits"></param>
        /// <exception cref="ArgumentException"></exception>
        public TensorSplitsCollection(float[] splits)
        {
            if (splits.Length > Splits.Length)
                throw new ArgumentException($"Must supply at most {Splits.Length} tensor splits", nameof(splits));

            splits.CopyTo(Splits.AsSpan());
        }

        /// <summary>
        /// Create a new tensor splits collection with all values initialised to the default
        /// </summary>
        public TensorSplitsCollection()
        {
        }

        /// <summary>
        /// Set all values to zero
        /// </summary>
        public void Clear()
        {
            Array.Clear(Splits, 0, Splits.Length);
        }

        internal MemoryHandle Pin()
        {
            return Splits.AsMemory().Pin();
        }

        #region IEnumerator
        /// <inheritdoc />
        public IEnumerator<float> GetEnumerator()
        {
            return ((IEnumerable<float>)Splits).GetEnumerator();
        }

        /// <inheritdoc />
        IEnumerator IEnumerable.GetEnumerator()
        {
            return Splits.GetEnumerator();
        }
        #endregion
    }
}
