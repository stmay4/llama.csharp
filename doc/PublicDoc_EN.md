# Public API Documentation

In addition to this documentation, you can use the following sources:

- **Built-in documentation** — all public methods of `LlamaExecutor` are equipped with `<summary>` comments in English, accessible via IntelliSense in the IDE.
- **Usage examples** — the `test_program` subproject (methods `SimpleChat`, `BatchGenerator`) and the `IntegrationTest` project contain up-to-date examples covering the main usage scenarios.

## Library Initialization

### ■ Library Initialization Method

```csharp
public static void Initialize(string llamaPath, string ggmlPath, string ggmlBasePath, List<string> backendPaths, string? mtmdPath = null)
```
**What does it do?**<br>
Loads the engine from the specified files.

**Parameters:**<br>
- `string llamaPath` - path to `llama.dll`/`.so`
- `string ggmlPath` - path to `ggml.dll`/`.so`
- `string ggmlBasePath` - path to `ggml-base.dll`/`.so`
- `List<string> backendPaths` - paths to backends like CPU, Vulkan, CUDA, etc. Can be specified in any order.
- `string? mtmdPath = null` - optional path to `mtmd.dll`/`.so`. Enables multimodal support (images and audio). Without it, every mtmd call throws `InvalidOperationException`. See [Multimodality (mtmd)](#multimodality-mtmd).

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
    ],
    "./llama/mtmd.dll" // optional: enables multimodality (mtmd)
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
public LlamaExecutor CreateExecutor(IContextParams @params, MtmdSpec? mtmdSpec = null)
```

**What does it do?**<br>
Creates an executor instance designed to work with one or more sequences and their batch processing.

**Parameters:**<br>
- `IContextParams @params` - the context parameters mentioned above.
- `MtmdSpec? mtmdSpec = null` - optional multimodal specification (see `MtmdContext.GetSpecification()`). Passing it puts the executor into multimodal mode right away; alternatively it can be enabled later via `LlamaExecutor.InitializeMtmdAsync`. See [Multimodality (mtmd)](#multimodality-mtmd).

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

## Working with the LlamaExecutor

### ■ General Information About the Executor

**What does it do?**<br>
`LlamaExecutor` is a wrapper around a single model context, designed for parallel work with several sequences (`LLamaSeqId`). Prefill, generation and multimodal prefill are performed asynchronously and are merged into common batches: a background decoding loop runs inside the executor; in one pass it samples one token for each generating sequence and tops up prefill tokens up to the batch size (`ContextParams.BatchSize`).

To work with multimodal content, the executor must be created with an mtmd specification (`MtmdSpec`) or switched into multimodal mode later via `InitializeMtmdAsync` (see [Multimodality (mtmd)](#multimodality-mtmd)).

**Fields:**<br>
- `public LLamaContext Context { get; }` - the executor's context. Through it you can access tokenization (`Context.Tokenize`), the native handle (`Context.NativeHandle`), the vocabulary (`Context.Vocab`) and the context parameters (`Context.Params`).

**Additional info:**<br>
The executor owns the context: `Dispose()` on the executor also disposes the context. The model (`LLamaWeights`) is not disposed and must outlive the executor.

### ■ Sequence Creation Method

```csharp
public async Task<LLamaSeqId> CreateSequence()
```
**What does it do?**<br>
Creates a new sequence inside the executor and returns its id. Ids previously released by `DeleteSequence` are reused first, then new ones are handed out until `ContextParams.SeqMax` is reached.

**Parameters:**<br>
None.

**Returns:**<br>
`Task<LLamaSeqId>` - the id of the created sequence; `(LLamaSeqId)(-1)` if the `SeqMax` limit is exhausted.

**Exceptions:**<br>
- `OperationCanceledException` - the executor lifetime token was cancelled.
- `ObjectDisposedException` - the executor (or its semaphore) has already been disposed.

**Usage:**<br>

```csharp
LLamaSeqId seq1 = await executor.CreateSequence();
```

**What if:**<br>
- **I create a second sequence with SeqMax = 1 (the default)?**

the method returns `-1`, no exception is thrown. To work with several sequences, set `SeqMax` in `ContextParams` when creating the executor.

- **I call the method concurrently from several threads?**

it is safe: all accesses are synchronized by a shared semaphore and every sequence gets a unique id.

### ■ Sequence Deletion Method

```csharp
public async Task DeleteSequence(LLamaSeqId id)
```
**What does it do?**<br>
Deletes a sequence completely: it frees the KV-cache cells it occupied, interrupts its prefill or generation if needed (the prefill task and the generation channel finish) and returns the id to the pool of free ids.

**Parameters:**<br>
- `LLamaSeqId id` - the id of the sequence to delete.

**Returns:**<br>
`Task`

**Exceptions:**<br>
- `IndexOutOfRangeException` - no sequence with such an id exists:
```csharp
throw new IndexOutOfRangeException($"sequence {id} not exist");
```
- `OperationCanceledException` / `ObjectDisposedException` - the executor is stopped or disposed.

**Usage:**<br>

```csharp
await executor.DeleteSequence(seq1);
```

### ■ Text Prefill Method (ProcessPrompt)

```csharp
public async Task ProcessPrompt(LLamaSeqId seqId, string text, bool addBos = false, bool special = true)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, bool addBos = false, bool special = true)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, List<bool> addBos, List<bool> special)
```
**What does it do?**<br>
Tokenizes the text and queues the tokens for prefill in the specified sequences. The background loop of the executor decodes those tokens: tokens of different sequences are merged into common batches (round-robin, up to `ContextParams.BatchSize` tokens per pass).

**Parameters:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - the sequences the text is added to. In the batch call the n-th text is added to the n-th sequence.
- `string text` / `List<string> texts` - the text to add. An empty string is allowed: for that sequence an already completed task is returned and nothing is added.
- `bool addBos = false` (or a list, one per sequence) - whether to add a BOS token before the text. Examples usually pass `model.Vocab.ShouldAddBOS`.
- `bool special = true` (or a list, one per sequence) - whether to tokenize special tokens written directly in the text (for example `BOS`/`EOS`, chat-template markers and the mtmd `BOM`/`EOM` strings). If `false`, such tokens are treated as ordinary text.

**Returns:**<br>
- the single variant - a `Task` that completes when the sequence prefill is fully decoded;
- the batch variant - a `Dictionary<LLamaSeqId, Task>` with one task per sequence. The method returns immediately, the tasks must be awaited (`await`/`Task.WhenAll`).

**Exceptions:**<br>
- `ArgumentNullException` - `null` was passed instead of a list.
- `ArgumentException` - an empty list was passed:
```csharp
throw new ArgumentException("Count is 0");
```
- `Exception` - fewer texts/flags than sequences:
```csharp
throw new Exception("There are few texts");
throw new Exception("There are few addBos");
throw new Exception("There are few special");
```
- `IndexOutOfRangeException` - no sequence with such an id exists:
```csharp
throw new IndexOutOfRangeException($"There is not {seqId} sequence");
```
- `Exception` - the sequence is already busy with another operation:
```csharp
throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
```
- `ContextFullException` - there is not enough room in the context. The exception is **not thrown by the method directly**; it is placed into the returned task, so it surfaces when the task is awaited.

**Usage:**<br>

```csharp
LLamaSeqId seq1 = await executor.CreateSequence();

// Single prefill: waits for the text to be fully decoded
await executor.ProcessPrompt(seq1, "<system>\n You are a helpful assistant\n</system>\n<user>\n Hello! </user>\n<assistant>\n", model.Vocab.ShouldAddBOS);
```
```csharp
// Batch prefill of several sequences
Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([seq1, seq2, seq3], [text1, text2, text3]);
await Task.WhenAll(prefills.Values);
```
```csharp
// Custom addBos/special per sequence
Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt(
    [seq1, seq2],
    [text1, text2],
    [model.Vocab.ShouldAddBOS, false],
    [true, true]);
await Task.WhenAll(prefills.Values);
```

**What if:**<br>
- **I prefill a sequence that is already busy?** you get a `sequence using in another place` exception - wait for the current operation to finish.
- **the text does not fit into the context?** the prefill task completes with a `ContextFullException`. You can check the occupied space in advance with [`GetContextFilling`](#sequence-state-read-methods).
- **I need to append text to an already filled sequence?** just call `ProcessPrompt` again: the tokens are appended and the sequence position keeps growing.

### ■ Generation Method (Generate)

```csharp
public async Task<Channel<string>> Generate(LLamaSeqId seqId, IInferenceParams inferenceParam)
public async Task<Dictionary<LLamaSeqId, Channel<string>>> Generate(List<LLamaSeqId> seqIds, List<IInferenceParams> inferenceParams)
```
**What does it do?**<br>
Starts generation for the specified sequences and returns a channel with the text stream. The executor samples the next token, decodes it with the model and writes the resulting text fragment into the `Channel<string>` of the corresponding sequence.

**Parameters:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - the sequences to generate for. The n-th parameters are applied to the n-th sequence.
- `IInferenceParams inferenceParam` / `List<IInferenceParams> inferenceParams` - generation parameters (see [Generation parameters](#generation-parameters-inferenceparams)).

**Returns:**<br>
- the single variant - `Task<Channel<string>>`;
- the batch variant - `Task<Dictionary<LLamaSeqId, Channel<string>>>`.

The channel is read until it completes:

```csharp
Channel<string> ch = await executor.Generate(seq1, inferenceParams);

await foreach (string text in ch.Reader.ReadAllAsync())
{
    Console.Write(text);
}
```

**Generation finishes and the channel closes when:**<br>
1. `MaxTokens` tokens have been generated (`MaxTokens != -1`);
2. any of the `AntiPrompts` strings appears in the output (substring search in the accumulated output; the fragment containing the antiprompt is still delivered to the channel);
3. the model emits an EOG token and `AutoStopFromEOG = true`;
4. the context runs out of room;
5. `StopSeqGeneration`, `DeleteSequence`, `ClearSequence` or the executor's `Dispose` is called.

**Exceptions:**<br>
- `ArgumentNullException` / `ArgumentException("Count is 0")` - invalid argument lists.
- `Exception("There are few inferenceParams")` - fewer parameters than sequences.
- `IndexOutOfRangeException` - no sequence with such an id exists:
```csharp
throw new IndexOutOfRangeException($"There is not {seqId} sequence");
```
- `Exception` - the sequence is busy with another operation: `sequence using in another place`.
- `Exception("For prefill use ProcessPrompt")` - the sequence was given parameters with `MaxTokens = 0`.
- `InvalidOperationException` - an attempt to generate on an empty sequence (prefill is required first):
```csharp
throw new InvalidOperationException($"Cannot generate on an empty sequence (id={seqId}). Please prefill some text first.");
```
- `LLamaDecodeError` - `llama_decode` returned an error; in the batched executor the error surfaces inside the background decoding loop.

**Usage:**<br>

```csharp
InferenceParams inferenceParams = new InferenceParams()
{
    MaxTokens = 200,
    AutoStopFromEOG = true,
    DecodeSpecialTokens = true,
    AntiPrompts = ["</assistant>"]
};

Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
```
```csharp
// Batch generation: sequences generate in parallel within common batches
Dictionary<LLamaSeqId, Channel<string>> channels = await executor.Generate([seq1, seq2], [inferenceParams, inferenceParams]);

await foreach (string text in channels[seq1].Reader.ReadAllAsync()) { /* ... */ }
await foreach (string text in channels[seq2].Reader.ReadAllAsync()) { /* ... */ }
```

**Additional info:**<br>
When generation finishes, the sequence state returns to `None` while the context is preserved - you can call `Generate` again (without repeating the prefill) or append text via `ProcessPrompt`.

**What if:**<br>
- **`MaxTokens = -1`?** generation is not limited by token count and stops only on an antiprompt, EOG or a `StopSeqGeneration` call.
- **I call `Generate` for several sequences at once?** they will generate in batches, sharing the idle compute time.
- **the same `InferenceParams` instance is passed to two sequences?** this is allowed, but the sampling pipeline will be shared between them (see the sampling section).

### ■ Generation Parameters (InferenceParams)

```csharp
InferenceParams inferenceParams = new InferenceParams() {...};
```
**What does it do?**<br>
Sets the generation parameters of a single sequence. It implements the `IInferenceParams` interface, so you can write your own implementation if needed.

**Parameters:**<br>
- `public int TokensKeep { get; init; } = 0;` - how many tokens of the original prefill to keep. It is currently not applied in the generation loop (reserved).
- `public int MaxTokens { get; init; } = -1;` - how many new tokens to generate; `-1` means no limit. The value `0` means "generate nothing": `Generate` with such parameters throws.
- `public bool DecodeSpecialTokens { get; init; } = true;` - whether to convert special tokens (BOS, EOS, etc.) into their text representation in the output. If `false`, they do not appear in the text.
- `public bool AutoStopFromEOG { get; init; } = true;` - automatically stop generation when an EOG token is produced (the check is done in post-processing).
- `public IReadOnlyList<string> AntiPrompts { get; init; } = [];` - strings that stop generation as soon as they appear in the output.
- `public ISamplingPipeline SamplingPipeline { get; init; }` - the sampling pipeline. The default value is:
```csharp
new TunableSamplerPipeline(new TunableSamplerPipelineSettings(
    [new TopKSampler()],
    new Mirostat2Sampler()));
```

**Returns:**<br>
An `InferenceParams` instance (it is a `record` with `init` properties, so it is configured with an object initializer).

**Usage:**<br>

```csharp
InferenceParams inferenceParams = new InferenceParams()
{
    MaxTokens = 200,
    AutoStopFromEOG = true,
    DecodeSpecialTokens = true,
    AntiPrompts = ["</assistant>"],
    SamplingPipeline = new TunableSamplerPipeline(
        new TunableSamplerPipelineSettings(
            [new TopKSampler() { K = 30 }],
            new Mirostat2Sampler() { Seed = 256 }
        )
    )
};
```

### ■ Sampling Pipeline (SamplingPipeline)

**What does it do?**<br>
The sampling pipeline turns model logits into one selected token. The executor calls its methods on every generation step, so in most cases it is enough to configure `TunableSamplerPipeline` through `InferenceParams.SamplingPipeline`.

**The `ISamplingPipeline` contract:**<br>

```csharp
public interface ISamplingPipeline : IDisposable
{
    LLamaToken Sample(SafeLLamaContextHandle ctx, int index);
    void Apply(LLamaTokenDataArray data);
    void Apply(ref LLamaTokenDataArrayNative data);
    void Reset();
    void Accept(LLamaToken token);
}
```
- `Apply` - applies the sampler chain to the logits and picks a candidate token;
- `Accept` - tells the pipeline that the token was accepted (updates internal state: penalties, Mirostat, grammar);
- `Reset` - resets the pipeline state;
- `Sample` - samples a token directly from the context.

A custom `ISamplingPipeline` implementation gives full control over sampling; the `BaseSamplingPipeline` base class takes care of creating and disposing the native sampler chain.

**Configuring `TunableSamplerPipeline`:**<br>

```csharp
public class TunableSamplerPipeline : BaseSamplingPipeline
public TunableSamplerPipeline(TunableSamplerPipelineSettings settings)
```
```csharp
public class TunableSamplerPipelineSettings
{
    public List<ISampler> Samplers { get; set; }
    public IFinalizeSampler FinalizeSampler { get; set; }
    public TunableSamplerPipelineSettings(List<ISampler> samplers, IFinalizeSampler finalizeSampler);
}
```
The samplers from `Samplers` are applied to the logits in order, then `FinalizeSampler` picks the final token.

**Samplers (`ISampler`):**<br>

| Sampler | What it does | Default values |
|---|---|---|
| `TopKSampler` | keeps the K most likely tokens | `K = 40`, `0` disables it |
| `TopNSigmaSampler` | threshold `mean + N * std` | `N = 1` |
| `TopPSampler` | nucleus: cumulative probability up to P | `P = 0.95`, `MinKeep = 1` |
| `MinPSampler` | keeps tokens with probability at least `P * maxProb` | `P = 0.05`, `MinKeep = 1` |
| `TypicalSampler` | typical sampling | `P = 0.2`, `MinKeep = 1` |
| `TemperatureSampler` | divides logits by the temperature before softmax | `T = 0.8` |
| `AdapTSampler` | adaptive temperature based on distribution entropy | `T = 0.8`, `Delta = 0.40`, `Exponent = 1` |
| `XTCSampler` | with probability P removes the most likely tokens | `P = 0.5`, `T = 0.1`, `MinKeep = 1`, `Seed = 42` |
| `GrammarSampler` | constrains the output with a GBNF grammar | `Grammar`, requires `Vocab = model.Vocab` |
| `PenaltiesSampler` | repetition, frequency and presence penalties | `PenaltyCount = 64` (`0` disables it, `-1` means the whole context), `RepeatPenalty = 1`, `FrequencyPenalty = 0`, `PresencePenalty = 0` |
| `LogitBiasSampler` | shifts the logits of the given tokens | a `LogitBias` dictionary, requires `Vocab = model.Vocab` |

Values are validated when the sampler is created: for example, `TopPSampler` with `P > 1` or `TemperatureSampler` with `T <= 0` throws `ArgumentOutOfRangeException`.

**Finalize samplers (`IFinalizeSampler`):**<br>

| Sampler | What it does | Default values |
|---|---|---|
| `GreedySampler` | always picks the most likely token | - |
| `DistributionSampler` | random pick from the distribution | `Seed = 42` |
| `Mirostat2Sampler` | Mirostat v2 with target perplexity | `Seed = 42`, `Tau = 3`, `Eta = 0.1` |

**Grammar:**<br>

```csharp
public record Grammar(string Gbnf, string Root);
```
`Gbnf` is the grammar text in GBNF format, `Root` is the name of the start rule.

**Usage:**<br>

```csharp
SamplingPipeline = new TunableSamplerPipeline(
    new TunableSamplerPipelineSettings(
        [
            new PenaltiesSampler() { RepeatPenalty = 1.1f },
            new TemperatureSampler() { T = 0.7f },
            new TopKSampler() { K = 30 },
            new TopPSampler() { P = 0.9f, MinKeep = 1 }
        ],
        new DistributionSampler() { Seed = 256 }
    )
)
```
```csharp
// Strictly grammar-constrained generation (for example, JSON)
SamplingPipeline = new TunableSamplerPipeline(
    new TunableSamplerPipelineSettings(
        [new GrammarSampler { Vocab = model.Vocab, Grammar = new Grammar(jsonGbnf, "root") }],
        new GreedySampler()
    )
)
```

**What if:**<br>
- **the same `InferenceParams` is used by two sequences?**

they will share a single pipeline instance along with its state (penalties, Mirostat, grammar position). For independent generation, create a separate `InferenceParams` (and its own pipeline) per sequence.

- **the pipeline is no longer needed?**

it owns native resources, so call `Dispose()` (for example `inferenceParams.SamplingPipeline.Dispose()`).

### ■ Generation and Prefill Stop Methods

```csharp
public async Task StopSeqGeneration(LLamaSeqId id)
public async Task StopSeqPrefill(LLamaSeqId id, LLamaPos startPos /* inclusive */)
```
**What does it do?**<br>
- `StopSeqGeneration` - requests that the sequence's generation be stopped; the stop is applied before the next decoding step and the generation channel closes.
- `StopSeqPrefill` - interrupts a running prefill and removes all tokens added by it, starting from `startPos` (inclusive), returning the sequence state back to that position.

**Parameters:**<br>
- `LLamaSeqId id` - the sequence id.
- `LLamaPos startPos` - the position (inclusive) from which tokens are removed; it must be less than the current `NextDecodedTokenPos` of the sequence.

**Returns:**<br>
`Task`

**Exceptions:**<br>
- `IndexOutOfRangeException` - the sequence does not exist (`sequence {id} not exist`), is not generating (`sequence {id} not in generate`) or is not in prefill (`sequence {id} not in prefill`).
- `ArgumentException` - `startPos` is greater than or equal to `NextDecodedTokenPos`:
```csharp
throw new ArgumentException("Start position must be lower then NextDecodedTokenPos");
```
- `Exception` - the sequence is empty (`sequence is empty`).
- `Exception` - the operation is unavailable for recurrent and hybrid models:
```csharp
throw new Exception("for recurrent and hybrid models, use state checkpoints(not yet implemented)");
```

**Usage:**<br>

```csharp
await executor.StopSeqGeneration(seq1);
```

### ■ Sequence State Read Methods

```csharp
public async Task<int> GetSequenceNextDecodedTokenPos(LLamaSeqId seqId)
public async Task<IReadOnlyList<LLamaToken>> GetSequenceDecodedTokens(LLamaSeqId seqId)
public async Task<LLamaTokenDataArray?> GetSequenceLastLogits(LLamaSeqId seqId)
public async Task<int> GetContextFilling()
```
**What does it do?**<br>
- `GetSequenceNextDecodedTokenPos` - returns the position of the sequence's next token (in effect the number of already processed tokens). Useful, for example, for cloning a prefix with `CopySeqPrefixTo`.
- `GetSequenceDecodedTokens` - returns the list of all decoded tokens of the sequence. In multimodal sessions, placeholder tokens stand in place of media positions (see [Placeholder tokens](#placeholder-tokens)).
- `GetSequenceLastLogits` - returns the sequence's last logits, or `null` if they have not been computed yet (for instance when the sequence is empty).
- `GetContextFilling` - the total number of real tokens occupying context cells across all executor sequences (compare it with `Context.ContextSize`).

**Parameters:**<br>
- `LLamaSeqId seqId` - the sequence id (except `GetContextFilling`, which takes no parameters).

**Returns:**<br>
`Task<int>`, `Task<IReadOnlyList<LLamaToken>>` or `Task<LLamaTokenDataArray?>` depending on the method.

**Exceptions:**<br>
- `IndexOutOfRangeException` - no sequence with such an id exists.
- `OperationCanceledException` / `ObjectDisposedException` - the executor is stopped or disposed.

**Usage:**<br>

```csharp
int pos = await executor.GetSequenceNextDecodedTokenPos(seq1);
IReadOnlyList<LLamaToken> tokens = await executor.GetSequenceDecodedTokens(seq1);
LLamaTokenDataArray? logits = await executor.GetSequenceLastLogits(seq1);
int filling = await executor.GetContextFilling();
```

### ■ Trial Sampling Method (SampleOneTokenWithoutDecode)

```csharp
public async Task<string> SampleOneTokenWithoutDecode(LLamaSeqId seqId, IInferenceParams inferenceParams)
```
**What does it do?**<br>
Samples one token from the sequence's last logits, but **does not decode it and does not commit it**: the model state and the sequence position remain unchanged. Handy for a quick probe, for example when a yes/no question is in the context and you want to peek at the answer without affecting it.

**Parameters:**<br>
- `LLamaSeqId seqId` - the sequence id.
- `IInferenceParams inferenceParams` - the sampling parameters (only the sampling pipeline is used).

**Returns:**<br>
`Task<string>` - the text fragment corresponding to the sampled token.

**Exceptions:**<br>
- `IndexOutOfRangeException` - the sequence does not exist (`Sequence {seqId} does not exist`).
- `Exception` - the sequence is busy with another operation, has no logits, or no parameters were passed:
```csharp
throw new Exception($"{seqId} sequence is being used in another place: {seq.InferState.State}");
throw new Exception($"Sequence {seqId} has no logits");
throw new Exception($"No inference parameters");
```
The method calls `Apply` but **not** `Accept`, so the state accepted with the token (penalties, grammar) is not updated.

**Usage:**<br>

```csharp
string probe = await executor.SampleOneTokenWithoutDecode(seq1, inferenceParams);
```

### ■ Copy Sequence Prefix Method (CopySeqPrefixTo)

```csharp
public async Task CopySeqPrefixTo(LLamaSeqId srcId, List<LLamaSeqId> targetIds, LLamaPos endPos)
```
**What does it do?**<br>
Copies the prefix of the source sequence into every target sequence. The target sequences are fully cleared before copying. If `endPos` equals the source's current position, the whole state is copied; otherwise only a part is copied.

**Parameters:**<br>
- `LLamaSeqId srcId` - the prefix source.
- `List<LLamaSeqId> targetIds` - the receiving sequences.
- `LLamaPos endPos` - the end position of the copied prefix (exclusive). It must be at least 1 and must not exceed the source's `NextDecodedTokenPos`.

**Returns:**<br>
`Task`

**Exceptions:**<br>
- `ArgumentNullException` - `targetIds` is `null`.
- `ArgumentException` - an empty target list, `endPos < 1` or `endPos > NextDecodedTokenPos`:
```csharp
throw new ArgumentException("Count is 0");
throw new ArgumentException("End position must be 1 or greater");
throw new ArgumentException("End position must be lower or equal NextDecodedTokenPos");
```
- `IndexOutOfRangeException` - the source or one of the targets does not exist.
- `Exception` - a sequence is busy; partial copying is unavailable for recurrent models:
```csharp
throw new Exception("for recurrent models, use state checkpoints(not yet implemented)");
```

**Additional info:**<br>
With a partial copy, the first token generated by a target sequence repeats the token that stood at position `endPos` of the source (the logits of that position are not copied).

**Usage:**<br>

```csharp
// A shared prefill for two dialogue branches
await executor.ProcessPrompt(seq1, prompt, model.Vocab.ShouldAddBOS);

LLamaPos endPos = await executor.GetSequenceNextDecodedTokenPos(seq1);
await executor.CopySeqPrefixTo(seq1, [seq2], endPos);

Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
```

### ■ Sequence Editing and Clearing Methods

```csharp
public async Task DeleteSequenceEnd(LLamaSeqId seqId, LLamaPos startPos /* inclusive */)
public async Task ClearSequence(LLamaSeqId id)
```
**What does it do?**<br>
- `DeleteSequenceEnd` - removes tokens from `startPos` (inclusive) to the end of the sequence. With `startPos = 0` it clears the whole sequence. Used to roll back generation and reuse the prefix.
- `ClearSequence` - clears the sequence's KV-cache and state, interrupting any running prefill or generation for it. It works for all model types (unlike `DeleteSequenceEnd`).

**Parameters:**<br>
- `LLamaSeqId seqId`/`id` - the sequence id.
- `LLamaPos startPos` - the position (inclusive) from which tokens are removed; it must be less than `NextDecodedTokenPos`.

**Returns:**<br>
`Task`

**Exceptions:**<br>
- `IndexOutOfRangeException` - no sequence with such an id exists.
- `ArgumentException` - `startPos` is greater than or equal to `NextDecodedTokenPos`.
- `Exception` - the sequence is busy (`sequence using in another place`) or empty (`sequence is empty`).
- `Exception` - the operation is unavailable for recurrent and hybrid models:
```csharp
throw new Exception("for recurrent and hybrid models, use state checkpoints(not yet implemented)");
```

**Usage:**<br>

```csharp
// Generated an answer, rolled it back and generated again
LLamaPos startPos = await executor.GetSequenceNextDecodedTokenPos(seq1);
Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
await foreach (string _ in ch1.Reader.ReadAllAsync()) { }

await executor.DeleteSequenceEnd(seq1, startPos);

Channel<string> ch2 = await executor.Generate(seq1, inferenceParams);
```
```csharp
await executor.ClearSequence(seq1);
```

### ■ Executor Disposal Method

```csharp
public void Dispose()
```
**What does it do?**<br>
Stops the background decoding loop and disposes the executor's synchronization and its context (`Context`). The `LLamaWeights` model is not disposed.

**Usage:**<br>

```csharp
executor.Dispose();
model.Dispose();
```

**Additional info:**<br>
Dispose executors **before** the model. If the model is disposed while the executor remains, accessing the context throws:
```csharp
throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");
```
Before disposing, active generations should be stopped (`StopSeqGeneration`) or awaited.

### ■ Typical LlamaExecutor Exceptions

| Exception | When it occurs |
|---|---|
| `IndexOutOfRangeException` | there is no sequence with the given id, or it is not in the required state (`not in prefill` / `not in generate`) |
| `Exception` `sequence using in another place` | the sequence is already busy with another operation |
| `ContextFullException` | there is not enough room in the context (delivered through the prefill or mtmd-prefill task) |
| `InvalidOperationException` | an attempt to generate on an empty sequence |
| `ArgumentException` / `ArgumentNullException` | invalid or incomplete argument lists, wrong positions |
| `Exception` `use state checkpoints(not yet implemented)` | state-editing operations on recurrent and hybrid models |
| `LLamaDecodeError` | a `llama_decode` error inside the background decoding loop |
| `OperationCanceledException` / `ObjectDisposedException` | the executor is stopped or disposed |

## Multimodality (mtmd)

Multimodality (mtmd) support lets you feed not only text but also images and audio into the model. Three components are involved:

1. the `mtmd` library (`mtmd.dll`/`mtmd.so`), loaded during initialization;
2. the `mmproj` file - multimodal projector weights matching the loaded model;
3. the `MtmdContext` object, which encodes media into embeddings, and the `LlamaExecutor`, which inserts those embeddings into a sequence.

### ■ Loading mtmd During Initialization

The mtmd library is loaded with the optional fifth parameter of `LlamaCpp.Initialize` (see [Library Initialization](#library-initialization) for details):

```csharp
LlamaCpp.Initialize(
    "./llama/llama.dll",
    "./llama/ggml.dll",
    "./llama/ggml-base.dll",
    ["./llama/ggml-cpu-alderlake.dll"],
    "./llama/mtmd.dll" // enables multimodality
);
```

**What if:**<br>
- **mtmd was not loaded?** every call to the multimodal API throws:
```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize() with MTMD dll/so");
```
For example, `MtmdContextParams.Default()` throws it, and that method is called when creating an mtmd context.

**Additional info:**<br>
The `llama.dll`, `ggml*.dll` and `mtmd.dll` files come from the same [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) release. The `mmproj` file is downloaded separately - it must match the model (on a Hugging Face model page these are the `mmproj-*.gguf` files). The model itself must also be multimodal (Qwen3-VL, Qwen3-ASR, Gemma and similar).

### ■ Mtmd Context Creation Parameters

```csharp
MtmdParams @params = new MtmdParams() {...};
```
**What does it do?**<br>
Defines the settings of the multimodal projector (`mmproj`). Every field except `UseGpu` and `PrintTimings` is nullable: the value `null` means "use the native default" (`MtmdContextParams.Default()`).

**Parameters:**<br>
- `public bool UseGpu { get; init; } = false;` - whether to use the loaded GPU backend for the media encoder.
- `public bool PrintTimings { get; init; } = false;` - whether to print encoding timings.
- `public int? Threads { get; init; } = null;` - the number of encoder threads.
- `public LlamaFlashAttentionType? FlashAttention { get; init; } = null;` - the flash attention type for mtmd.
- `public bool? Warmup { get; init; } = null;` - whether to run a warmup pass after initialization (the native default is `true`).
- `public int? ImageMinTokens { get; init; } = null;` - the minimum number of tokens per image (by default taken from the `mmproj` metadata).
- `public int? ImageMaxTokens { get; init; } = null;` - the maximum number of tokens per image (by default from the metadata). Lowering it speeds up image encoding and saves context.
- `public int? BatchSize { get; init; } = null;` - the maximum number of encoder output tokens per batch (1024 by default). This is not a hard limit: the first image is always added even if it exceeds it.

**Returns:**<br>
An `MtmdParams` instance.

**Usage:**<br>

```csharp
MtmdParams mtmdParams = new MtmdParams
{
    UseGpu = false,
    Threads = 8,
    ImageMaxTokens = 1000,
    BatchSize = 1024
};
```

### ■ Creating an Mtmd Context

```csharp
public static MtmdContext CreateFromFile(string mmprojFile, LLamaWeights llamaModel, IMtmdParams @params)
```
**What does it do?**<br>
Loads the multimodal projector weights from the given file, binds them to the model and creates an `MtmdContext`, which starts its own background media encoding loop.

**Parameters:**<br>
- `string mmprojFile` - the path to the `mmproj` file (the multimodal projector matching the model).
- `LLamaWeights llamaModel` - the already loaded model.
- `IMtmdParams @params` - the parameters described above.

**Returns:**<br>
An `MtmdContext` instance.

**Exceptions:**<br>
- `InvalidOperationException` - mtmd was not loaded during initialization:
```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize() with MTMD dll/so");
```
- `InvalidOperationException` - the `mmproj` file cannot be read:
```csharp
throw new InvalidOperationException($"Model file '{mmprojPath}' is not readable");
```
- `LoadWeightsFailedException` - the engine failed to load `mmproj`:
```csharp
throw new LoadWeightsFailedException(mmprojPath);
```

**Usage:**<br>

```csharp
LLamaWeights model = LLamaWeights.LoadFromFile(new ModelParams(modelPath) { GpuLayerCount = 99 });
MtmdContext ctx = MtmdContext.CreateFromFile(mmprojPath, model, mtmdParams);
```

**Additional info:**<br>
The `mmproj` must match the model: a foreign projector leads to a load failure or to incorrect embeddings. `MtmdContext` does not own the model - disposing the mtmd context does not dispose the model.

### ■ MtmdContext Properties

```csharp
public bool NonCasualDecode { get; }  // full (non-causal) attention when decoding images
public bool MropeDecode { get; }      // whether MROPE positioning is used
public bool SupportVision { get; }    // whether the projector supports images
public bool SupportAudio { get; }     // whether the projector supports audio
public int AudioSampleRate { get; }   // the sample rate expected by the model
```
**What does it do?**<br>
The properties are read-only and reflect the capabilities of the loaded `mmproj`. `AudioSampleRate` is used by `DecodeWavToMonoFloat` for audio resampling.

**Usage:**<br>

```csharp
Console.WriteLine($"Vision: {ctx.SupportVision}, Audio: {ctx.SupportAudio}, SampleRate: {ctx.AudioSampleRate}");
```

**What if:**<br>
- **the projector does not support images?** `EncodeImage*` throws `throw new Exception("Vision dont support");`
- **the projector does not support audio?** `EncodeAudio*` and `DecodeWavToMonoFloat` throw `throw new Exception("Audio dont support");`

### ■ Mtmd Specification and Attaching It to the Executor

```csharp
public MtmdSpec GetSpecification()
public async Task InitializeMtmdAsync(MtmdSpec mtmdSpec)
```
```csharp
public readonly record struct MtmdSpec (bool UseMrope);
```
**What does it do?**<br>
- `GetSpecification` returns a compact specification of the mtmd context (the MROPE usage flag) that the executor needs in order to compute embedding positions correctly.
- `InitializeMtmdAsync` switches an already created executor into multimodal mode - used when the executor was created without a specification.

**Usage:**<br>

```csharp
// Option 1: at executor creation time
LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());
```
```csharp
// Option 2: attach later
LlamaExecutor executor = model.CreateExecutor(ctxParams);
await executor.InitializeMtmdAsync(ctx.GetSpecification());
```

**What if:**<br>
- **the executor was created without mtmd and I call `ProcessMtmdEmbeds`?**

you get an exception:
```csharp
throw new Exception("Mtmd was not initialized");
```

- **`UseMrope = true`?** embedding positions are encoded with four coordinates (t, y, x, z) instead of one - the executor takes this into account automatically.

### ■ Encoding Images

```csharp
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromPath(string filePath)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromRGB(uint nx, uint ny, Memory<byte> image)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImageFromPaths(List<string> filePaths)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
```
**What does it do?**<br>
Tokenizes media through mtmd (obtaining the begin-of-media and end-of-media markers plus the image chunks) and queues the embeddings for background encoding. The single methods wait until the embeddings are ready, while the batch methods return a `Task` per image (encoding runs in the background and is merged into common encoder batches).

**Parameters:**<br>
- `string filePath` / `List<string> filePaths` - the path(s) to image files. The formats read by the SixLabors.ImageSharp library are supported (png, jpeg, bmp, webp and others).
- `uint nx, uint ny` - the image width and height in pixels.
- `Memory<byte> image` - a top-down RGB24 buffer: `nx * ny * 3` bytes, row by row starting from the top.
- `List<uint> nxs, List<uint> nys, List<Memory<byte>> images` - the same for batch processing: the n-th element of each list describes the n-th image.

**Returns:**<br>
A tuple `(string BOM, string EOM, ...)` where:
- `BOM` - the string of special begin-of-media tokens; it must be inserted into the prompt before the embeddings;
- `EOM` - the string of special end-of-media tokens; it is inserted after the embeddings;
- `embeds` - for the single methods a ready `LlamaEmbedding[]` array, for the batch methods a `Task<LlamaEmbedding[]>` that must be awaited.

**Exceptions:**<br>
- `Exception` - the projector does not support images: `"Vision dont support"` (see [MtmdContext Properties](#mtmdcontext-properties)).
- `ArgumentNullException` / `ArgumentException` - invalid input buffers (the `RegisterEncode` checks).
- `Exception` - the chunk structure returned by mtmd is broken (for example `start and end chunks must be TEXT`).

**Usage:**<br>

```csharp
// A single image: the embeddings are ready immediately
var result = await ctx.EncodeImageFromPath("./photo.png");

// Several images: encoding is batched
var results = await ctx.EncodeImageFromPaths(["./1.png", "./2.png", "./3.png"]);
foreach (var item in results)
{
    LlamaEmbedding[] embeds = await item.embeds; // await each result
}
```

**Additional info:**<br>
The number of tokens an image occupies is controlled by the `ImageMinTokens`/`ImageMaxTokens` parameters of `MtmdParams`; where necessary mtmd rescales the image.

### ■ Encoding Audio

```csharp
public float[] DecodeWavToMonoFloat(string path)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudioFromWav(string wavFilePath)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudio(Memory<float> audio)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudiosFromWav(List<string> wavFilePaths)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudios(List<Memory<float>> audio)
```
**What does it do?**<br>
- `DecodeWavToMonoFloat` - reads a WAV file, converts it to mono (stereo is averaged), resamples to `AudioSampleRate` when needed and returns PCM float32 in the [-1, 1] range.
- The other methods work exactly like the image encoding methods: the single ones wait for the embeddings, the batch ones return tasks.

**Parameters:**<br>
- `string path` / `string wavFilePath` / `List<string> wavFilePaths` - the path(s) to WAV files.
- `Memory<float> audio` / `List<Memory<float>> audio` - already prepared mono PCM float32.

**Returns:**<br>
- `float[]` - mono PCM float32;
- `(BOM, EOM, embeds)` tuples - the same as for images.

**Exceptions:**<br>
- `Exception` - the projector does not support audio: `Audio dont support`.
- `NotSupportedException` - more than two channels in the file:
```csharp
throw new NotSupportedException($"Unsupported channel count: {channels}");
```

**Usage:**<br>

```csharp
var result = await ctx.EncodeAudioFromWav("./audio.wav");
```

### ■ Inserting Embeddings Into a Sequence (ProcessMtmdEmbeds)

```csharp
public async Task ProcessMtmdEmbeds(LLamaSeqId seqId, LlamaEmbedding[] embeds)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessMtmdEmbeds(List<LLamaSeqId> seqIds, List<LlamaEmbedding[]> embeds)
```
**What does it do?**<br>
Queues media embeddings for multimodal prefill in the specified sequences. The background loop of the executor decodes the embeddings, merging them into batches (up to `ContextParams.BatchSize` per pass); for non-causal chunks attention is toggled automatically for the duration of the batch. Once finished, the sequence position advances by the number of embeddings, placeholder tokens are added to the history and the logits of the last position become available - after that you can continue with an ordinary text prefill (`EOM` + question) or start generation.

**Parameters:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - the sequences.
- `LlamaEmbedding[] embeds` / `List<LlamaEmbedding[]> embeds` - the embeddings: the n-th array corresponds to the n-th sequence. Each array must hold **one media unit** (a single image or a single audio), because text tokens must stand between different media items.

**Returns:**<br>
- the single variant - a `Task` that completes when the sequence embeddings are decoded;
- the batch variant - a `Dictionary<LLamaSeqId, Task>`; the tasks must be awaited.

**Exceptions:**<br>
- `Exception("Mtmd was not initialized")` - the executor is not in multimodal mode.
- `ArgumentNullException` / `ArgumentException("Count is 0")` / `Exception("There are few embeds")` - invalid lists.
- `IndexOutOfRangeException` - no sequence with such an id exists.
- `Exception` - the sequence is busy with another operation: `sequence using in another place`.
- `ContextFullException` - there is not enough room in the context; the exception is placed into the returned task.

**Usage:**<br>

```csharp
// Order matters: text with BOM -> embeddings -> text with EOM
await executor.ProcessPrompt(seq1, "<|im_start|>user\n " + result.BOM);
await executor.ProcessMtmdEmbeds(seq1, result.embeds);
await executor.ProcessPrompt(seq1, result.EOM + "What is shown in the image?\n<|im_end|>\n<|im_start|>assistant\n");
```
```csharp
// Batch insertion into several sequences
Dictionary<LLamaSeqId, Task> embedsTasks = await executor.ProcessMtmdEmbeds([seq1, seq2, seq3], [emb1, emb2, emb3]);
await Task.WhenAll(embedsTasks.Values);
```

**What if:**<br>
- **an empty embeddings array is passed?**

for that sequence an already completed task is returned - handy when a particular request has no media.

- **the returned task is not awaited?**

the embeddings may not be decoded yet and the subsequent prefill/generation will see an incomplete sequence. Always await the tasks.

### ■ Mtmd Embeddings (LlamaEmbedding and LlamaEmbeddingType)

```csharp
public readonly record struct LlamaEmbedding
{
    public Memory<float> Data { get; }      // the embedding vector
    public LlamaEmbeddingType Type { get; } // the modality type
}
```
```csharp
public enum LlamaEmbeddingType { Text, Image, Audio, HiddenState, Video }
```
**What does it do?**<br>
Describes a single media embedding vector. `Data` holds the vector itself (its length matches the model's input embedding dimension) and `Type` is the modality. The MROPE position and the non-causal attention flag are stored internally and filled in by mtmd automatically.

**Additional info:**<br>
You do not need to create `LlamaEmbedding` values manually - they come from the `Encode*` methods and are passed to `ProcessMtmdEmbeds`.

### ■ Placeholder Tokens

```csharp
public static bool IsMtmdPlaceholder(this LLamaToken token)
public static LLamaToken CreateImagePlaceholder()  // -100
public static LLamaToken CreateVideoPlaceholder()  // -101
public static LLamaToken CreateAudioPlaceholder()  // -102
```
**What does it do?**<br>
After embeddings are decoded, the executor records one placeholder token per media position into the sequence history. These are internal tokens: they do not exist in the model vocabulary, they are never sampled or decoded, and they only exist to keep sequence positions and history consistent.

**Additional info:**<br>
The base constants live in `MtmdTokenConstants` (`MtmdImagePlaceholderBase = -100`, `MtmdVideoPlaceholderBase = -101`, `MtmdAudioPlaceholderBase = -102`); the range from -200 to -100 is reserved for placeholders so that they never collide with `InvalidToken` (-1) or model special tokens (usually >= 0).

**What if:**<br>
- **a placeholder token shows up in `GetSequenceDecodedTokens`?**

it is a media position rather than a real vocabulary token. You can check it with the `token.IsMtmdPlaceholder()` extension; there is no need to pass such tokens to `ProcessPrompt` - the `BOM`/`EOM` strings are used for text.

### ■ Example: Asking About an Image

```csharp
// 1. Initialize the libraries with mtmd
LlamaCpp.Initialize(
    Path.Combine(baseDllPath, "llama.dll"),
    Path.Combine(baseDllPath, "ggml.dll"),
    Path.Combine(baseDllPath, "ggml-base.dll"),
    [Path.Combine(baseDllPath, "ggml-cpu-alderlake.dll")],
    Path.Combine(baseDllPath, "mtmd.dll"));

// 2. Model
LLamaWeights model = LLamaWeights.LoadFromFile(new ModelParams(modelPath));

// 3. Multimodal context
MtmdParams mtmdParams = new MtmdParams
{
    UseGpu = false,
    Threads = 8,
    ImageMaxTokens = 1000
};
MtmdContext ctx = MtmdContext.CreateFromFile(mmprojPath, model, mtmdParams);

// 4. Encode the image
var result = await ctx.EncodeImageFromPath("./test.png");

// 5. Executor with the multimodal specification
LlamaExecutor executor = model.CreateExecutor(
    new ContextParams() { ContextSize = 8000 },
    ctx.GetSpecification());

LLamaSeqId seq1 = await executor.CreateSequence();

// 6. Prefill: text with BOM, then the embeddings, then EOM and the question
await executor.ProcessPrompt(seq1, "<|im_start|>system\n you are a helpfull assistant\n<|im_end|>\n<|im_start|>user\n " + result.BOM);
await executor.ProcessMtmdEmbeds(seq1, result.embeds);
await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image?\n<|im_end|>\n<|im_start|>assistant\n");

// 7. Generation
InferenceParams inferenceParams = new InferenceParams()
{
    MaxTokens = 200,
    AutoStopFromEOG = true,
    DecodeSpecialTokens = true,
    AntiPrompts = []
};

Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
await foreach (string text in ch1.Reader.ReadAllAsync())
{
    Console.Write(text);
}

// 8. Disposal
ctx.Dispose();
executor.Dispose();
model.Dispose();
```

The batch variant (several images across several sequences) follows the same scheme: a shared text prefill in the first sequence, `CopySeqPrefixTo` for the others, then `ProcessMtmdEmbeds` with lists and parallel generation (an example lives in `Llama.csharp.IntegrationTest/TestMtmd.cs`).

### ■ Disposing mtmd Resources

```csharp
public void Dispose()
```
**What does it do?**<br>
Stops the background encoding loop, disposes the native mtmd context and cancels unfinished encoding tasks (awaiting such tasks completes with a cancellation). The model is not disposed.

**Usage:**<br>

```csharp
ctx.Dispose();
executor.Dispose();
model.Dispose();
```

**Additional info:**<br>
Temporary buffers (bitmaps) are released automatically right after encoding is registered, so there is nothing extra to clean up. Dispose the model last: if it goes first, accesses to the `mtmd` and language contexts throw `ObjectDisposedException`.