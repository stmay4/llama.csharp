using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public sealed class LLamaWeights
        : IDisposable
    {
        /// <summary>
        /// The native handle, which is used in the native APIs
        /// </summary>
        /// <remarks>Be careful how you use this!</remarks>
        public SafeLlamaModelHandle NativeHandle { get; }

        /// <summary>
        /// Total number of tokens in the context
        /// </summary>
        public int ContextSize => NativeHandle.ContextSize;

        /// <summary>
        /// Get the size of this model in bytes
        /// </summary>
        public ulong SizeInBytes => NativeHandle.SizeInBytes;

        /// <summary>
        /// Get the number of parameters in this model
        /// </summary>
        public ulong ParameterCount => NativeHandle.ParameterCount;

        /// <summary>
        /// Dimension of embedding vectors
        /// </summary>
        public int EmbeddingSize => NativeHandle.EmbeddingSize;

        /// <summary>
        /// Get the special tokens of this model
        /// </summary>
        public SafeLlamaModelHandle.Vocabulary Vocab => NativeHandle.Vocab;

        /// <summary>
        /// All metadata keys in this model
        /// </summary>
        public IReadOnlyDictionary<string, string> Metadata { get; set; }

        private LLamaWeights(SafeLlamaModelHandle weights)
        {
            NativeHandle = weights;
            Metadata = weights.ReadMetadata();
        }


        /// <summary>
        /// Load weights into memory
        /// </summary>
        /// <param name="params"></param>
        /// <returns></returns>
        public static LLamaWeights LoadFromFile(IModelParams @params)
        {
            using var pin = @params.ToLlamaModelParams(out var lparams);
            var weights = SafeLlamaModelHandle.LoadFromFile(@params.ModelPath, lparams);
            return new LLamaWeights(weights);
        }

        public static NoAllocModelInfo LoadInfoNoAlloc(string modelPath)
        {
            ModelParams @params = new ModelParams(modelPath)
            {
                NoAlloc = true
            };
            using var pin = @params.ToLlamaModelParams(out var lparams);
            var weights = SafeLlamaModelHandle.LoadFromFile(@params.ModelPath, lparams);

            NoAllocModelInfo info = new NoAllocModelInfo()
            {
                Metadata = weights.ReadMetadata(),
                ContextSize = weights.ContextSize,
                SizeInBytes = weights.SizeInBytes,
                ParameterCount = weights.ParameterCount,
                EmbeddingSize = weights.EmbeddingSize
            };

            weights.Dispose();

            return info;
        }

        public void Dispose()
        {
            NativeHandle.Dispose();
        }

        /// <summary>
        /// Create a llama_context using this model
        /// </summary>
        /// <param name="params"></param>
        /// <param name="logger"></param>
        /// <returns></returns>
        public LLamaContext CreateContext(IContextParams @params)//, ILogger? logger = null)
        {
            return new LLamaContext(this, @params);//, logger);
        }
    }
}
