Конечно, вот перевод вашего документа на английский. Я постарался сохранить структуру, стиль и все технические детали.

# Public API Documentation

In addition to this documentation, you can use the following sources:

- **Built-in documentation** — all public methods of `LlamaExecutor` are equipped with `<summary>` comments in English, accessible via IntelliSense in the IDE.
- **Usage examples** — the `test_program` subproject (methods `SimpleChat`, `BatchGenerator`) and the `IntegrationTest` project contain up-to-date examples covering the main usage scenarios.

## Library Initialization

### ■ Library Initialization Method

```csharp
public static void Initialize(string llamaPath, string ggmlPath, string ggmlBasePath, List<string> backendPaths)
```
**What does it do?**<br>
Loads the engine from the specified files.

**Parameters:**<br>
- `string llamaPath` - path to `llama.dll`/`.so`
- `string ggmlPath` - path to `ggml.dll`/`.so`
- `string ggmlBasePath` - path to `ggml-base.dll`/`.so`
- `List<string> backendPaths` - paths to backends like CPU, Vulkan, CUDA, etc. Can be specified in any order.

**Returns:**<br>
`void`

**Exceptions:**<br>

Checking parameters for `null` and `empty` throws:<br>
```csharp
throw new ArgumentException($"Parameter '{parameterName}' cannot be null or empty.", parameterName);
```

For a non-existent file or similar issue with the main libraries:<br>
```csharp
throw new DllNotFoundException(
    $"Library load fail: '{libraryDisplayName}'. " +
    $"Path: {path}. "
);
```
For backends:<br>
```csharp
throw new Exception($"Loading backend {backend} fail");
```
with the name of the file that caused the error.

If a function is missing in the loaded library:<br>
```csharp
throw new EntryPointNotFoundException($"Function {functionName} not exist in library");
```

**Usage:**<br>

```csharp
LlamaCpp.Initialize(
    "./llama/llama.dll", 
    "./llama/ggml.dll", 
    "./llama/ggml-base.dll",
    [ // In this case, backends are loaded: CPU and Vulkan GPU
        "./llama/ggml-cpu-alderlake.dll",
        "./llama/ggml-vulkan.dll"
    ]
);
```

It must be called **before any other methods that use llama.cpp**, otherwise such methods will throw an exception:  

```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize()");
```

**Additional info:**<br>
The engine files can be downloaded from [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases). Different builds are available for different architectures.<br>
For Windows OS, the files end with `.dll`, for Linux — `.so`.
The choice of the CPU backend version for the x64 architecture is based on the PC's processor generation. The `ggml-cpu-x64` version is the slowest as it contains no optimizations.

**What if:**<br>

- **I load several CPU backends?**

the first CPU backend in the list will be applied.

- **I load only a CPU backend, but then specify parameters related to GPU in the model and executor parameters (number of layers, etc.)?**

no error will occur, the parameters will be ignored on the llama.cpp side.

## Model

### ■ Model Loading Parameters

```csharp
ModelParams parametres = new ModelParams(_modelPath) {...};
```
**What does it do?**<br>
An instance defining the loading settings for a model at the specified path `_modelPath`.

**Parameters:**<br>

**Only the path to the file is mandatory. If other values are not filled in, they are set either at the wrapper level or at the llama.cpp level using the `_llama_model_default_params()` function.**

- `public string ModelPath { get; set; }` - path to the model, mandatory, set in the constructor.
- `public int GpuLayerCount { get; set; } = 0;` - number of layers to offload to GPU.
- `public List<TensorBufferOverride> TensorBufferOverrides { get; set; } = new();` - overrides layer placement, applied after `GpuLayerCount`. Useful for MOE models (usage is in the examples).
- `public GPUSplitMode? SplitMode { get; set; }` - how to split between multiple GPUs (not tested) (default from the `default layer` — by layers).
- `public int MainGpu { get; set; } = 0;` - which GPU to use.
- `public TensorSplitsCollection TensorSplits { get; set; } = new();` - overrides tensors between multiple GPUs (not tested).
- `public bool VocabOnly { get; set; }` - load only the vocabulary (without weights). Currently used for loading metadata without allocating memory, instead of `noAlloc` (default from `default false`).
- `public bool UseMemorymap { get; set; } = true;` - use mmap if possible.
- `public bool UseMemoryLock { get; set; }` - if `true`, cannot be unloaded from memory by other programs (default from `default false`).
- `public bool CheckTensors { get; set; }` - check tensors during loading (default from `default false`).
- `public bool UseExtraBufs { get; set; }` - (default from `default true`).
- `public bool NoHost { get; set; }` - (default from `default false`).
- `public bool NoAlloc { get; set; }` - load only metadata (seems not to be working, so the `LoadInfoNoAlloc` method currently works via `VocabOnly`) (default from `default false`).

**Returns:**<br>
An instance of `ModelParams`.

**Usage:**<br>

```csharp
ModelParams parametres = new ModelParams(_modelPath)
{
    GpuLayerCount = 999
};
```

**Additional info:**<br>
Available layer values are 0-999 (for the used version of llama.cpp; in newer versions, `-1` may also be available to load all). The specified number can be greater than the total number of layers.

### ■ Synchronous Model Loading Method

```csharp
public static LLamaWeights LoadFromFile(IModelParams @params)
```
**What does it do?**<br>
Loads the model with the specified loading parameters, blocking the thread while loading. Returns the `LLamaWeights` model object.

**Parameters:**<br>
- `IModelParams @params` - the parameters described above.

**Returns:**<br>
An instance of `LLamaWeights`.

**Exceptions:**<br>

if an unreadable file is specified:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
if the file is readable, but the engine failed to load it:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Usage:**<br>

```csharp
LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
```

- **Not enough memory, and loading has started?**

depends on where the model is being loaded. If it's RAM, then most likely part of the model will be swapped to disk. If it's GPU - it depends on the backend: Vulkan will throw an exception `throw new LoadWeightsFailedException(modelPath);`, and CUDA reportedly uses a swap to RAM.

### ■ Asynchronous Model Loading Method

```csharp
public static async Task<LLamaWeights> LoadFromFileAsync(IModelParams @params, CancellationToken cancellationToken = default)
```
**What does it do?**<br>
Loads the model asynchronously with the specified loading parameters, with the option to cancel the task via a token. Returns the `LLamaWeights` model object.

**Parameters:**<br>
- `IModelParams @params` - the parameters described above.
- `CancellationToken cancellationToken = default` - cancellation token.

**Returns:**<br>
An instance of `LLamaWeights`.

**Exceptions:**<br>

if an unreadable file is specified:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
if the file is readable, but the engine failed to load it:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Usage:**<br>

```csharp
LLamaWeights model = await LLamaWeights.LoadFromFileAsync(parameters);
```
or
```csharp
LLamaWeights model = await LLamaWeights.LoadFromFileAsync(parameters, _modelLifeTokenSource.Token);
```

**Additional info:**<br>
The model loading method does not support cancellation, so the `cancellationToken` will not be able to stop the loading itself, but it will stop the return of the object after the model has finished loading.

**What if:**<br>
Same as for the synchronous method.

<h3 id="LLamaWeights-fields">■ LLamaWeights Instance Fields</h3>

After creating a model from a file using `LoadFromFile` or `LoadFromFileAsync`, an `LLamaWeights` instance is returned with the following public fields:
```csharp
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
```

Through `NativeHandle`, fields for determining the model type (decoder only, decoder-encoder, recurrent, hybrid, diffusion) and many other things are also accessible.

Through `Vocab`, tokenization is available via 
```csharp
public LLamaToken[] Tokenize(string text, bool addBos, bool special, Encoding encoding)
```
and detokenization via 
```csharp
public string? LLamaTokenToString(LLamaToken? token, bool isSpecialToken)
```

You can also get the IDs of specialized tokens (BOS, EOS, and the like) and (if needed) convert them to string values using `LLamaTokenToString`, and then prefill them in the executor with `decodeSpecialToken=true` so they are tokenized again as specialized tokens. *This may be needed to compose a prefill string, for example to add the BOS and EOS tokens on which the model was trained.*

### ■ Model Metadata Loading Method

```csharp
public static NoAllocModelInfo LoadInfoNoAlloc(string modelPath)
```

**What does it do?**<br>
A separate method that can be used to load only metadata (without the model weights, and therefore without allocating memory for the model). It can be useful for getting model information for display somewhere or for planning usage without needing to load the entire model.
With a standard full model load, the list of available data is larger (see [LLamaWeights Fields](#LLamaWeights-fields)).

**Parameters:**<br>
- `string modelPath` - path to the model.

**Returns:**<br>
An instance of `NoAllocModelInfo`.

```csharp
public class NoAllocModelInfo
{
    public required IReadOnlyDictionary<string, string> Metadata { get; init; }
    public required int ContextSize { get; init; }
}
```
I am currently only getting `ContextSize` here, as the metadata key for it is more or less the same for everyone. The methods for getting the context size or the model itself without a loaded model do not work at the moment.

**Exceptions:**<br>

if an unreadable file is specified:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
if the file is readable, but the engine failed to load it:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Usage:**<br>

```csharp
NoAllocModelInfo modelinfo = LLamaWeights.LoadInfoNoAlloc(modelPath);
```

### ■ Model Resource Release Method

```csharp
public void Dispose()
```

**What does it do?**<br>
Releases all model data in the wrapper and in llama.cpp, including weights, etc. This frees all the memory occupied by this model.

**Usage:**<br>

```csharp
model.Dispose();
```

**Additional info:**<br>
Must be called at the end of working with the model.

**What if:**<br>
- **I call it, but I still have executors attached to the model?**

you will get an exception:
```csharp
throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");
```

## Creating Executors from an LLamaWeights Model Instance

### ■ Context Creation Parameters

```csharp
ContextParams ctxParams = new ContextParams() {...};
```

**What does it do?**<br>
Defines an instance that sets the context settings for the executor (each executor is a wrapper over one model context for working with it).

**Parameters:**<br>

**There are no mandatory fields; all have a default value. However, if the context size is not specified, the maximum possible size will be allocated by default, and it may not fit in memory (probably swapping will occur).**

- `public uint? ContextSize { get; init; } = 0;` - if 0, the maximum allowable context size of the model for which it will be created is allocated.
- `public uint BatchSize { get; init; } = 512;` - batch size (if the prefill is 1500 tokens, it will be processed in 3 batches with the default value of 512).
- `public uint UBatchSize { get; init; } = 512;` - the actual batch size during computation.
- `public uint SeqMax { get; init; } = 1;` - the number of sequences that can be created when working with the context (the maximum value allowed by llama.cpp is **256**).
- `public bool Embeddings { get; init; } = false;` - not used yet, for future embeddings retrieval.
- `public float? RopeFrequencyBase { get; init; }`
- `public float? RopeFrequencyScale { get; init; }`
- `private string EncodingName { get; init; } = Encoding.UTF8.WebName;` - the encoding to which generated tokens are converted.
- `public int? Threads { get; init; }` - number of threads for model decoding within this context.
- `public int? BatchThreads { get; init; }` - number of threads for batch decoding within this context (I don't know how it differs from the previous one; usually, if necessary, I only set the `Threads` field).
- `public float? YarnExtrapolationFactor { get; init; }`
- `public float? YarnAttentionFactor { get; init; }`
- `public float? YarnBetaFast { get; init; }`
- `public float? YarnBetaSlow { get; init; }`
- `public uint? YarnOriginalContext { get; init; }`
- `public RopeScalingType? YarnScalingType { get; init; }`
- `public GGMLType? TypeK { get; init; }` - sets the data type for keys in the KV cache of the context (default `f16`, changing it seems to affect quality more than quantizing the model itself).
- `public GGMLType? TypeV { get; init; }` - sets the data type for values in the KV cache of the context (default `f16`, changing it seems to affect quality more than quantizing the model itself; also, key accuracy is reportedly more important than value accuracy).
- `public bool NoKqvOffload { get; init; } = true;` - offload the context to GPU (`false` - offload, `true` - do not offload).
- `public LlamaFlashAttentionType FlashAttention { get; init; } = LlamaFlashAttentionType.Auto;`
- `public float? DefragThreshold { get; init; }`
- `public LLamaPoolingType PoolingType { get; init; } = LLamaPoolingType.Unspecified;`
- `public LLamaAttentionType AttentionType { get; init; } = LLamaAttentionType.Unspecified;`
- `public bool? KVunified { get; init; } = true;` - when working with multiple sequences, it's better to set it to `true` (if `false`, the context is divided into blocks of size `context_length/seqMax` and works with them separately; the cache cannot be shared between sequences, and copying duplicates it. If `true`, different sequences fill a common cache without dividing it into blocks, with the ability to share the common cache (hence "unified"), based on adding information about which sequences are using a cache cell).
- `public bool? OPoffload { get; init; } = null;` - offload the operation tensor to GPU? `true` - if the loaded model for which the executor is being created is fully or partially offloaded to the GPU. But if the GPU backend is loaded, but the model is loaded entirely on CPU (RAM), it's better to set it to `false`, otherwise the video card is not used for model computation, but approximately 1GB of VRAM is still occupied.
- `public bool? NoPerf { get; init; } = null;` - whether or not to let llama.cpp collect performance metrics (how to retrieve them is described later in the executor section).

**Returns:**<br>
An instance of `ContextParams`.

**Usage:**<br>

```csharp
ContextParams ctxParams = new ContextParams()
{
    ContextSize = 16000,
    //SeqMax = 1, // number of sequences available to create, default is already one
    //NoKqvOffload = false // If using GPU and enough VRAM, you can offload the context to GPU
    //...
};
```

### ■ Creating an Executor

```csharp
public LlamaExecutor CreateExecutor(IContextParams @params)
```

**What does it do?**<br>
Creates an executor instance designed to work with one or more sequences and their batch processing.

**Parameters:**<br>
- `IContextParams @params` - the context parameters mentioned above.

**Returns:**<br>
An instance of `LlamaExecutor`.

**Exceptions:**<br>

if the model has already been disposed:
```csharp
throw new ObjectDisposedException("Cannot create context, model weights have been disposed");
```
if something went wrong on the llama.cpp side:
```csharp
throw new RuntimeError("Failed to create context from model");
```

**Usage:**<br>

```csharp
LlamaExecutor executor = model.CreateExecutor(ctxParams);
```

### ■ Creating an Executor for a Single Sequence

```csharp
public OneSeqLlamaExecutor CreateOneSeqExecutor(IContextParams @params)
```

**What does it do?**<br>
Creates an executor instance designed to work with only one sequence (therefore, there is no batch processing). It is intended for working with small models, as it has less overhead compared to `LlamaExecutor`.

**Parameters:**<br>
- `IContextParams @params` - the context parameters mentioned above.

**Returns:**<br>
An instance of `LlamaExecutor`.

**Exceptions:**<br>

if the model has already been disposed:
```csharp
throw new ObjectDisposedException("Cannot create context, model weights have been disposed");
```
if something went wrong on the llama.cpp side:
```csharp
throw new RuntimeError("Failed to create context from model");
```

**Usage:**<br>

```csharp
OneSeqLlamaExecutor executor = model.CreateOneSeqExecutor(ctxParams);
```

**Additional info:**<br>
You can create several `OneSeqLlamaExecutor` instances and work with them separately, as if they were different sequences of a single `LlamaExecutor`, but this is very inefficient.
Example:
If 1 sequence takes `x` time, processing two parallel `OneSeqLlamaExecutor` instances will take more than `2x` time.
Batch processing of two sequences inside a single `LlamaExecutor` takes approximately `1.2x` time (for my hardware).
Also, it is not possible to share a common cache between different `OneSeqLlamaExecutor` instances, which is possible for two sequences inside a single `LlamaExecutor`.

FURTHER IN PROGRESS...