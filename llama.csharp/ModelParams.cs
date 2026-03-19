using Llama.csharp.Abstractions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public record ModelParams : IModelParams
    {
        public List<TensorBufferOverride> TensorBufferOverrides { get; set; } = new();

        public int GpuLayerCount { get; set; } = 0;

        public GPUSplitMode? SplitMode { get; set; }

        public int MainGpu { get; set; } = 0;

        public TensorSplitsCollection TensorSplits { get; set; } = new();

        public bool VocabOnly { get; set; }

        public bool UseMemorymap { get; set; } = true;

        public bool UseMemoryLock { get; set; }

        public bool CheckTensors { get; set; }

        public bool UseExtraBufs { get; set; }

        public bool NoHost { get; set; }

        public bool NoAlloc { get; set; }

        public string ModelPath { get; set; }

        public ModelParams(string modelPath)
        {
            ModelPath = modelPath;
        }
    }
}
