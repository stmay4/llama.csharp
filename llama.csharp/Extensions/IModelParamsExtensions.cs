using Llama.csharp.Abstractions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System.Runtime.InteropServices;
using System.Text;

namespace Llama.csharp.Extensions
{
    /// <summary>
    /// Extension methods to the IModelParams interface
    /// </summary>
    public static class IModelParamsExtensions
    {
        /// <summary>
        /// Convert the given `IModelParams` into a `LLamaModelParams`
        /// </summary>
        /// <param name="params"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        /// <exception cref="FileNotFoundException"></exception>
        /// <exception cref="ArgumentException"></exception>
        public static IDisposable ToLlamaModelParams(this IModelParams @params, out LLamaModelParams result)
        {
            if (@params.UseMemoryLock && !LlamaCpp.Llama_SupportsMlock())
                throw new NotSupportedException("'UseMemoryLock' is not supported (llama_supports_mlock() == false)");
            if (@params.UseMemorymap && !LlamaCpp.Llama_SupportsMmap())
                throw new NotSupportedException("'UseMemorymap' is not supported (llama_supports_mmap() == false)");

            var disposer = new GroupDisposable();

            result = LLamaModelParams.Default();

            result.main_gpu = @params.MainGpu;
            result.n_gpu_layers = @params.GpuLayerCount < 0 ? int.MaxValue : @params.GpuLayerCount;
            if (@params.SplitMode.HasValue)
                result.split_mode = @params.SplitMode.Value;

            result.use_mlock = @params.UseMemoryLock;
            result.use_mmap = @params.UseMemorymap;
            result.vocab_only = @params.VocabOnly;
            result.check_tensors = @params.CheckTensors;

            unsafe
            {
                result.tensor_split = (float*)disposer.Add(@params.TensorSplits.Pin()).Pointer;
            }

            // Add tensor buffer overrides
            unsafe
            {
                result.tensor_buft_overrides = ConvertOverrides(@params.TensorBufferOverrides, disposer);
            }

            unsafe
            {
                result.kv_overrides = (LLamaModelMetadataOverride*)nint.Zero;
            }

            result.no_alloc = @params.NoAlloc;


            return disposer;
        }

        /// <summary>
        /// Get a map from name of device (`ggml_backend_buft_name`) to the device type (`ggml_backend_dev_buffer_type`)
        /// </summary>
        /// <returns>Dictionary mapping buffer type names to their handles</returns>
        private static IReadOnlyDictionary<string, nint> GetAvailableBufferTypes()
        {
            var result = new Dictionary<string, nint>();

            var count = LlamaCpp.GGML_BackendDevCount();
            for (nuint i = 0; i < count; i++)
            {
                var dev = LlamaCpp.GGML_BackendDevGet(i);
                var buft = LlamaCpp.GGMLBase_BackendDevBufferType(dev);

                var name = Marshal.PtrToStringAnsi(LlamaCpp.GGMLBase_BackendBuftName(buft));
                if (string.IsNullOrEmpty(name))
                    continue;

                result[name] = buft;
            }

            return result;
        }

        private static unsafe LLamaModelTensorBufferOverride* ConvertOverrides(List<TensorBufferOverride> overrides, GroupDisposable disposer)
        {
            // Early out if there are no overrides
            if (overrides.Count == 0)
                return null;

            var bufferTypes = GetAvailableBufferTypes();

            var overridesCount = 0;
            var overridesArray = new LLamaModelTensorBufferOverride[overrides.Count + 1];

            foreach (var @override in overrides)
            {
                // Check if we have this buffer type
                if (!bufferTypes.TryGetValue(@override.BufferType, out var bufferType))
                    continue;

                // Create null terminated string and pin this memory so it can be passed to native code
                var patternBytes = Encoding.UTF8.GetBytes(@override.Pattern + "\0");
                var patternPin = patternBytes.AsMemory().Pin();
                disposer.Add(patternPin);

                // Add the item to the overridesArray
                overridesArray[overridesCount++] = new()
                {
                    Pattern = (byte*)patternPin.Pointer,
                    BufferType = bufferType
                };
            }

            // Early out if there were no valid overrides
            if (overridesCount == 0)
                return null;

            // Pin it so it can be safely passed across to native code
            var overrideArrayPin = overridesArray.AsMemory().Pin();
            disposer.Add(overrideArrayPin);

            return (LLamaModelTensorBufferOverride*)overrideArrayPin.Pointer;
        }
    }
}
