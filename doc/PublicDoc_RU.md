# Документация публичного интерфейса

Помимо документации вы можете использовать следующие источники:

- **Встроенная документация** — все публичные методы `LlamaExecutor` снабжены `<summary>`-комментариями на английском языке, доступными через IntelliSense в IDE.
- **Примеры использования** — в подпроекте `test_program` (методы `SimpleChat`, `BatchGenerator`) и в проекте `IntegrationTest` есть актуальные примеры, покрывающие основные сценарии работы.

## Инициализация библиотеки

### ■ Метод инициализации библиотеки

```csharp
public static void Initialize(string llamaPath, string ggmlPath, string ggmlBasePath, List<string> backendPaths)
```
**Что делает?**<br>
Загружает движок из указанных файлов

**Параметры:**<br>
- string llamaPath - путь к llama.dll/.so
- string ggmlPath - путь к ggml.dll/.so
- string ggmlBasePath - путь к ggml-base.dll/.so
- List<string> backendPaths - пути к бэкендам CPU, Vulkan, CUDA и тп. Могут быть указаны в любом порядке

**Возврат:**<br>
void

**Исключения:**<br>

Проверка параметров на null и empty вызывает:<br>
```csharp
throw new ArgumentException($"Parameter '{parameterName}' cannot be null or empty.", parameterName);
```

При вводе несуществующего файла и подобное для основных библиотек:<br>
```csharp
throw new DllNotFoundException(
    $"Library load fail: '{libraryDisplayName}'. " +
    $"Path: {path}. "
);
```
для бэкендов:<br>
```csharp
throw new Exception($"Loading backend {backend} fail");
``` 
с указанием файла, который вызвал ошибку

Если в загруженной библиотеке не хватает функции:<br>
```csharp
throw new EntryPointNotFoundException($"Function {functionName} not exist in library");
```

**Использование:**<br>

```csharp
LlamaCpp.Initialize(
    "./llama/llama.dll", 
    "./llama/ggml.dll", 
    "./llama/ggml-base.dll",
    [ // В данном случае загружаются бэкенды: CPU и Vulkan GPU
        "./llama/ggml-cpu-alderlake.dll",
        "./llama/ggml-vulkan.dll"
    ]
);
```

Вызывается **перед любыми другими методами, использующими llama.cpp**, иначе такие методы вызовут исключение:  

```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize()");
```

**Доп. информация:**<br>
Файлы движка скачиваются с [релизов llama.cpp](https://github.com/ggml-org/llama.cpp/releases) для разных архитектур разные сборки.<br>
Для ОС Windows файлы заканчиваются на .dll, для linux - .so.
Выбор версии бэкенда CPU для архитектуры x64 осуществляется на основе поколения процессора ПК, версия ggml-cpu-x64 самая медленная, так как не содержит оптимизаций

**Что если:**<br>

- **я загружу несколько CPU бэкендов?**

применятся первый в списке CPU бэкенд

- **я загружу только CPU бэкенд, а далее буду указывать в параметрах модели и исполнителя параметры, связанные с GPU (колво слоев и т.п.)?**

ошибки не будет, параметры проигнорируются на стороне llama.cpp

## Модель

### ■ Параметры загрузки модели

```csharp
ModelParams parametres = new ModelParams(_modelPath) {...};
```
**Что делает?**<br>
Экземпляр, определяющий настройки загрузки для модели по указанному пути _modelPath

**Параметры:**<br>

**Обязательно указывать только путь к файлу, остальные значения при отсутствии заполнения либо задаются на уровне обертки, либо на уровне llama.cpp с помощью функции _llama_model_default_params()**

- public string ModelPath { get; set; } - путь к модели, обязателен, устанавливается в конструкторе
- public int GpuLayerCount { get; set; } = 0; - кол-во слоев, выгружаемых на GPU
- public List<TensorBufferOverride> TensorBufferOverrides { get; set; } = new(); - переопределение расположения слоев, применяется после GpuLayerCount, полезно для MOE (использование есть в примерах)
- public GPUSplitMode? SplitMode { get; set; } - как делить между несколькими GPU (не проверял) (по умолчанию из default layer - по слоям)
- public int MainGpu { get; set; } = 0; - какой GPU использовать
- public TensorSplitsCollection TensorSplits { get; set; } = new(); - переопределить тензоры между несколькими GPU (не проверял)
- public bool VocabOnly { get; set; } - загрузить только словарь (без весов), сейчас используется для загрузки метаданных без выделения памяти вместо noAlloc (по умолчанию из default false)
- public bool UseMemorymap { get; set; } = true; - использовать mmap, если возможно
- public bool UseMemoryLock { get; set; } - если true, нельзя выгрузить из памяти другими программами (по умолчанию из default false)
- public bool CheckTensors { get; set; } - проверка тензоров при загрузке (по умолчанию из default false)
- public bool UseExtraBufs { get; set; } - (по умолчанию из default true)
- public bool NoHost { get; set; } - (по умолчанию из default false)
- public bool NoAlloc { get; set; } - загрузить только метаданные (вроде не работает, поэтому пока метод LoadInfoNoAlloc работает через VocabOnly) (по умолчанию из default false)

**Возврат:**<br>
Экземпляр ModelParams

**Использование:**<br>

```csharp
ModelParams parametres = new ModelParams(_modelPath)
{
    GpuLayerCount = 999
};
```

**Доп. информация:**<br>
Доступные значения слоев 0-999 (для используемой версии llama.cpp, в новых версиях также буде доступно указать -1 для загрузки всех), указанное число может быть больше общего кол-ва слоев.

### ■ Метод синхронной загрузки модели

```csharp
public static LLamaWeights LoadFromFile(IModelParams @params)
```
**Что делает?**<br>
Загружает модель по указанным параметрам загрузки, блокируя поток на время загрузки. Возвращает объект модели LLamaWeights

**Параметры:**<br>
- IModelParams @params - параметры, описанные выше

**Возврат:**<br>
Экземпляр LLamaWeights

**Исключения:**<br>

если указан нечитаемый файл:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
а если читаемый, но движок не смог его загрузить:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Использование:**<br>

```csharp
LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
```

- **не хватает места в памяти, а загрузка начата?**

зависит от места, куда загружается модель. Если это RAM, то скорее всего будет своп части модели на диск. Если GPU - то зависит от бэкенда: Vulkan вызовет исключение throw new LoadWeightsFailedException(modelPath); , а с CUDA вроде используется своп в RAM

### ■ Метод асинхронной загрузки модели

```csharp
public static async Task<LLamaWeights> LoadFromFileAsync(IModelParams @params, CancellationToken cancellationToken = default)
```
**Что делает?**<br>
Загружает модель по указанным параметрам загрузки асинхронно с возможностью отменить загрузку через токен. Возвращает объект модели LLamaWeights.

**Параметры:**<br>
- IModelParams @params - параметры, описанные выше
- CancellationToken cancellationToken = default - токен для отмены

**Возврат:**<br>
Экземпляр LLamaWeights

**Исключения:**<br>

если указан нечитаемый файл:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
а если читаемый, но движок не смог его загрузить:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Использование:**<br>

```csharp
LLamaWeights model = await LLamaWeights.LoadFromFileAsync(parameters);
```
или
```csharp
LLamaWeights model = await LLamaWeights.LoadFromFileAsync(parameters, _modelLifeTokenSource.Token);
```

**Доп. информация:**<br>
Метод загрузки модели не поддерживает отмену, поэтому cancellationToken не сможет остановить саму загрузку, но он остановит возврат объекта после завершения загрузки модели

**Что если:**<br>
Такие же как и для синхронного метода

<h3 id="LLamaWeights-fields">■ Поля экземпляра LLamaWeights</h3>

После создания модели из файла с помощью LoadFromFile или LoadFromFileAsync возвращается экземпляр LLamaWeights с следующими публичными полями:
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

Через NativeHandle также доступны поля для определения типа модели (decoder only, decoder-encoder, recurrent, hybrid, diffusion) и куча чего еще

Через Vocab доступны токенизация через 
```csharp
public LLamaToken[] Tokenize(string text, bool addBos, bool special, Encoding encoding)
```
и детокенизация через 
```csharp
public string? LLamaTokenToString(LLamaToken? token, bool isSpecialToken)
```

А также можно получить номера специализированных токенов (BOS, EOS и подобное) и (если надо) преобразовать их в строковые значения через LLamaTokenToString, и потом запрефилить в исполнителе с decodeSpecialToken=true, чтобы они токенизировались снова как специализированные. *может понадобится для сборки строки префила, чтобы добавлять токены BOS EOS на которых была обучена модель:*

### ■ Метод загрузки метаданных модели

```csharp
public static NoAllocModelInfo LoadInfoNoAlloc(string modelPath)
```

**Что делает?**<br>
Отдельный метод, который может быть использован для загрузки только метаданных (без весов модели и соответственно без выделения памяти под модель). Может быть полезен для получения данных о модели для отображения гденибудь или планирования использования без необходимости грузить всю модель.
При стандартной полной загрузке модели список доступных данных больше (см. [Поля LlamaWeights](#LLamaWeights-fields)).

**Параметры:**<br>
- string modelPath - путь к модели

**Возврат:**<br>
Экземпляр NoAllocModelInfo

```csharp
public class NoAllocModelInfo
{
    public required IReadOnlyDictionary<string, string> Metadata { get; init; }
    public required int ContextSize { get; init; }
}
```
Получаю здесь только ContextSize на данный момент, так как ключ в метаданных к нему более менее у всех одинаковый, а методы для получения размера контекста или самой модели без загруженной модели не работают.

**Исключения:**<br>

если указан нечитаемый файл:
```csharp
throw new InvalidOperationException($"Model file '{modelPath}' is not readable");
```
а если читаемый, но движок не смог его загрузить:
```csharp
throw new LoadWeightsFailedException(modelPath);
```

**Использование:**<br>

```csharp
NoAllocModelInfo modelinfo = LLamaWeights.LoadInfoNoAlloc(modelPath);
```

### ■ Метод освобождения ресурсов модели

```csharp
public void Dispose()
```

**Что делает?**<br>
Освобождает все данные модели в обертке и в llama.cpp, включая веса и прочее. Это освобождает всю память занятую данной моделью

**Использование:**<br>

```csharp
model.Dispose();
```

**Доп. информация:**<br>
Должен быть вызван в конце работы с моделью

**Что если:**<br>
- **я вызову, а у меня остались привязанные к модели исполнители?**

получите исключение:
```csharp
throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");
```

## Создание исполнителей из экземпляра модели LLamaWeights

### ■ Параметры создания контекста

```csharp
ContextParams ctxParams = new ContextParams() {...};
```

**Что делает?**<br>
Задает экземпляр, определяющий настройки контекста исполнителя (каждый исполнитель это обертка над одним контекстом модели для работы с ним)

**Параметры:**<br>

**Обязательных полей нет, для всех есть дефолтное значение, но если не указать размер контекста, то по умолчанию будет выделен максимально возможный и он может не поместится в памяти (наверно своп будет)**

- public uint? ContextSize { get; init; } = 0; - если 0, то выделяется макс допустимый размер контекста модели, для которой он будет создан
- public uint BatchSize { get; init; } = 512; - размер пакета (если префил на 1500 токенов, то будет обработан в 3 пакета для дефолтного значения в 512)
- public uint UBatchSize { get; init; } = 512; - реальный размер пакета при вычислениях
- public uint SeqMax { get; init; } = 1; - количество последовательностей, которые можно будет создать при работе с контекстом (максимальное значение разрешенное на стороне llama.cpp **256**)
- public bool Embeddings { get; init; } = false; - пока не используется, в будущем для получения эмбеддингов
- public float? RopeFrequencyBase { get; init; }
- public float? RopeFrequencyScale { get; init; }
- private string EncodingName { get; init; } = Encoding.UTF8.WebName; - кодировка, в которую преобразовыать генерируемые токены
- public int? Threads { get; init; } - кол-во потоков для декода моделью в пределах этого контекста
- public int? BatchThreads { get; init; } - кол-во потоков для декода батча моделью в пределах этого контекста (не знаю чем отличается от прошлого, обычно, елси необходимо, ставлю только поле Threads)
- public float? YarnExtrapolationFactor { get; init; }
- public float? YarnAttentionFactor { get; init; }
- public float? YarnBetaFast { get; init; }
- public float? YarnBetaSlow { get; init; }
- public uint? YarnOriginalContext { get; init; }
- public RopeScalingType? YarnScalingType { get; init; }
- public GGMLType? TypeK { get; init; } - задает тип данных для ключей в KV кэше контекста (по умолчанию f16, изменение вроде бы сильнее влияет на качество, чем квантованию самой модели)
- public GGMLType? TypeV { get; init; } - задает тип данных для значений в KV кэше контекста (по умолчанию f16, изменение вроде бы сильнее влияет на качество, чем квантованию самой модели, также точность ключей более важна, чем точность значений вроде)
- public bool NoKqvOffload { get; init; } = true; - выгрузка контекста на GPU (false - выгружать, true - не выгружать)
- public LlamaFlashAttentionType FlashAttention { get; init; } = LlamaFlashAttentionType.Auto;
- public float? DefragThreshold { get; init; }
- public LLamaPoolingType PoolingType { get; init; } = LLamaPoolingType.Unspecified;
- public LLamaAttentionType AttentionType { get; init; } = LLamaAttentionType.Unspecified;
- public bool? KVunified { get; init; } = true; - при работе с несколькими последовательностями лучше ставить в true (если false, то делит контекст на блоки длины context_length/seqMax и работает с ними отдельно, кэш между последовательностями не разделить, при копировании дублируется, если true - разные последовательности заполняют общий кэш без его деления на блоки с возможностью разделять общий кэш (поэтому unified), основано на добавлении к ячейкам кэша информации о том, какие последовательности его используют)
- public bool? OPoffload { get; init; } = null; -  выгружать тензор операций на GPU? true - если загруженная модель для которой создается исполнитель выгружена полностью или частично на GPU, а если GPU бэкенд загружен, но модель загружена полностью на CPU (RAM), то лучше установить в false, а иначе видеокарта не используется для вычислений модели, но память VRAM приблизительно в 1гб занята.
- public bool? NoPerf { get; init; } = null; - производить или нет на стороне llama.cpp замеры производительности (как их получить далее в описании исполнителя)

**Возврат:**<br>
Экземпляр ContextParams

**Использование:**<br>

```csharp
ContextParams ctxParams = new ContextParams()
{
    ContextSize = 16000,
    //SeqMax = 1, // number of sequences available to create, default is already one
    //NoKqvOffload = false // If using GPU and enough VRAM, you can offload the context to GPU
    //...
};
```

### ■ Создание исполнителя

```csharp
public LlamaExecutor CreateExecutor(IContextParams @params)
```

**Что делает?**<br>
Создает экземпляр исполнителя, предназначенного для работы с одной или несколькими последовательностями и их пакетной обработки

**Параметры:**<br>
- IContextParams @params - параметры контекста, о которых сказано выше

**Возврат:**<br>
Экземпляр LlamaExecutor

**Исключения:**<br>

если модель уже освобождена:
```csharp
throw new ObjectDisposedException("Cannot create context, model weights have been disposed");
```
если что-то пошло не так на стороне llama.cpp:
```csharp
throw new RuntimeError("Failed to create context from model");
```

**Использование:**<br>

```csharp
LlamaExecutor executor = model.CreateExecutor(ctxParams);
```


### ■ Создание исполнителя, работающего только с одной последовательностью

```csharp
public OneSeqLlamaExecutor CreateOneSeqExecutor(IContextParams @params)
```

**Что делает?**<br>
Создает экземпляр исполнителя, предназначенного для работы только с одной последовательностью (соответственно и пакетной обработки нет). Предназначен для работы с малыми моделями, так как дает меньше оверхеда по сравнению с LlamaExecutor

**Параметры:**<br>
- IContextParams @params - параметры контекста, о которых сказано выше


**Возврат:**<br>
Экземпляр LlamaExecutor

**Исключения:**<br>

если модель уже освобождена:
```csharp
throw new ObjectDisposedException("Cannot create context, model weights have been disposed");
```
если что-то пошло не так на стороне llama.cpp:
```csharp
throw new RuntimeError("Failed to create context from model");
```

**Использование:**<br>

```csharp
OneSeqLlamaExecutor executor = model.CreateOneSeqExecutor(ctxParams);
```

**Доп. информация:**<br>
Можно создать несколько OneSeqLlamaExecutor и работать с ними раздельно как будто с разными последовательностями одного LlamaExecutor, но это очень не эффективно. 
Пример:
если 1 последовательность обрабатывается x времени то на обработку двух параллельных OneSeqLlamaExecutor уйдет более 2x времени
А для пакетной обработки двух последовательностей внутри одного LlamaExecutor понадобится примерно 1,2x времени (для моего оборудования)
Также между разными OneSeqLlamaExecutor нельзя разделить общий кэш, что возможно для двух последовательностей внутри одного LlamaExecutor.

ДАЛЕЕ В ПРОЦЕССЕ...