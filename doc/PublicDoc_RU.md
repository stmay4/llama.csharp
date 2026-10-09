# Документация публичного интерфейса

Помимо документации вы можете использовать следующие источники:

- **Встроенная документация** — все публичные методы `LlamaExecutor` снабжены `<summary>`-комментариями на английском языке, доступными через IntelliSense в IDE.
- **Примеры использования** — в подпроекте `test_program` (методы `SimpleChat`, `BatchGenerator`) и в проекте `IntegrationTest` есть актуальные примеры, покрывающие основные сценарии работы.

## Инициализация библиотеки

### ■ Метод инициализации библиотеки

```csharp
public static void Initialize(string llamaPath, string ggmlPath, string ggmlBasePath, List<string> backendPaths, string? mtmdPath = null)
```
**Что делает?**<br>
Загружает движок из указанных файлов

**Параметры:**<br>
- string llamaPath - путь к llama.dll/.so
- string ggmlPath - путь к ggml.dll/.so
- string ggmlBasePath - путь к ggml-base.dll/.so
- List<string> backendPaths - пути к бэкендам CPU, Vulkan, CUDA и тп. Могут быть указаны в любом порядке
- string? mtmdPath = null - необязательный путь к mtmd.dll/.so (библиотека мультимодальности). Если указан, загружаются функции mtmd, и становятся доступны `MtmdContext` и мультимодальный префил (см. раздел [Мультимодальность](#мультимодальность-mtmd)). Если не указан - mtmd недоступен

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
    ],
    "./llama/mtmd.dll" // необязательно: подключает мультимодальность (mtmd)
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
public LlamaExecutor CreateExecutor(IContextParams @params, MtmdSpec? mtmdSpec = null)
```

**Что делает?**<br>
Создает экземпляр исполнителя, предназначенного для работы с одной или несколькими последовательностями и их пакетной обработки

**Параметры:**<br>
- IContextParams @params - параметры контекста, о которых сказано выше
- MtmdSpec? mtmdSpec = null - спецификация мультимодального контекста (`MtmdContext.GetSpecification()`), если планируется мультимодальный префил. Можно не указывать и подключить позже через `executor.InitializeMtmdAsync()` (см. раздел [Мультимодальность](#мультимодальность-mtmd))

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

## Работа с исполнителем LlamaExecutor

### ■ Общая информация об исполнителе

**Что делает?**<br>
`LlamaExecutor` — обёртка над одним контекстом модели, предназначенная для параллельной работы с несколькими последовательностями (`LLamaSeqId`). Префил, генерация и мультимодальный префил выполняются асинхронно и объединяются в общие батчи: внутри исполнителя работает бесконечный фоновый цикл декодирования, который за один проход сэмплирует по одному токену для каждой генерирующейся последовательности и добирает токены префила до размера пакета (`ContextParams.BatchSize`).

Для работы с мультимодальным контентом исполнитель должен быть создан со спецификацией mtmd (`MtmdSpec`) или переведён в мультимодальный режим позже через `InitializeMtmdAsync` (см. раздел [Мультимодальность (mtmd)](#мультимодальность-mtmd)).

**Поля:**<br>
- `public LLamaContext Context { get; }` - контекст исполнителя. Через него доступны токенизация (`Context.Tokenize`), нативный хэндл (`Context.NativeHandle`), словарь (`Context.Vocab`) и параметры контекста (`Context.Params`).

**Доп. информация:**<br>
Исполнитель владеет контекстом: `Dispose()` исполнителя освобождает и контекст. Модель (`LLamaWeights`) при этом не освобождается и должна жить дольше исполнителя.

### ■ Метод создания последовательности

```csharp
public async Task<LLamaSeqId> CreateSequence()
```
**Что делает?**<br>
Создаёт в исполнителе новую последовательность и возвращает её идентификатор. Сначала используется пул ранее освобождённых идентификаторов (после `DeleteSequence`), затем выдаются новые — до тех пор, пока не будет достигнут `ContextParams.SeqMax`.

**Параметры:**<br>
Отсутствуют.

**Возврат:**<br>
`Task<LLamaSeqId>` - идентификатор созданной последовательности; `(LLamaSeqId)(-1)`, если лимит `SeqMax` исчерпан.

**Исключения:**<br>
- `OperationCanceledException` - токен жизни исполнителя отменён.
- `ObjectDisposedException` - исполнитель (или его семафор) уже освобождён.

**Использование:**<br>

```csharp
LLamaSeqId seq1 = await executor.CreateSequence();
```

**Что если:**<br>
- **создам вторую последовательность при SeqMax = 1 (значение по умолчанию)?**

метод вернёт `-1`, исключения не будет. Чтобы работать с несколькими последовательностями, задайте `SeqMax` в `ContextParams` при создании исполнителя.

- **вызову метод параллельно из нескольких потоков?**

это безопасно: все обращения синхронизированы общим семафором, каждая последовательность получит уникальный идентификатор.

### ■ Метод удаления последовательности

```csharp
public async Task DeleteSequence(LLamaSeqId id)
```
**Что делает?**<br>
Полностью удаляет последовательность: освобождает занятые ею ячейки KV-кэша, при необходимости прерывает её префил или генерацию (задача префила и канал генерации при этом завершаются) и возвращает идентификатор в пул свободных.

**Параметры:**<br>
- `LLamaSeqId id` - идентификатор удаляемой последовательности.

**Возврат:**<br>
`Task`

**Исключения:**<br>
- `IndexOutOfRangeException` - последовательности с таким id не существует:
```csharp
throw new IndexOutOfRangeException($"sequence {id} not exist");
```
- `OperationCanceledException` / `ObjectDisposedException` - исполнитель остановлен или освобождён.

**Использование:**<br>

```csharp
await executor.DeleteSequence(seq1);
```

### ■ Метод префила текста (ProcessPrompt)

```csharp
public async Task ProcessPrompt(LLamaSeqId seqId, string text, bool addBos = false, bool special = true)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, bool addBos = false, bool special = true)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessPrompt(List<LLamaSeqId> seqIds, List<string> texts, List<bool> addBos, List<bool> special)
```
**Что делает?**<br>
Токенизирует текст и ставит токены в очередь префила указанных последовательностей. Декодирование этих токенов выполняет фоновый цикл исполнителя: токены разных последовательностей объединяются в общие батчи (по кругу, до `ContextParams.BatchSize` токенов за проход).

**Параметры:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - последовательности, в которые добавляется текст. При пакетном вызове n-й текст добавляется n-й последовательности.
- `string text` / `List<string> texts` - добавляемый текст. Пустая строка допустима: для такой последовательности возвращается уже завершённая задача, ничего не добавляется.
- `bool addBos = false` (или список на каждую последовательность) - добавить ли BOS-токен перед текстом. В примерах обычно передаётся `model.Vocab.ShouldAddBOS`.
- `bool special = true` (или список на каждую последовательность) - токенизировать ли специальные токены, записанные прямо в тексте (например, `BOS`/`EOS`, служебные маркеры чат-шаблона и строки `BOM`/`EOM` mtmd). Если `false`, такие токены считаются обычным текстом.

**Возврат:**<br>
- одиночный вариант - `Task`, который завершается, когда префил последовательности полностью декодирован;
- пакетный вариант - `Dictionary<LLamaSeqId, Task>` с задачей на каждую последовательность. Метод возвращает управление сразу, задачи нужно дождаться (`await`/`Task.WhenAll`).

**Исключения:**<br>
- `ArgumentNullException` - передан `null` вместо списка.
- `ArgumentException` - передан пустой список:
```csharp
throw new ArgumentException("Count is 0");
```
- `Exception` - список текстов/флагов меньше, чем последовательностей:
```csharp
throw new Exception("There are few texts");
throw new Exception("There are few addBos");
throw new Exception("There are few special");
```
- `IndexOutOfRangeException` - последовательности с таким id не существует:
```csharp
throw new IndexOutOfRangeException($"There is not {seqId} sequence");
```
- `Exception` - последовательность уже занята другой операцией:
```csharp
throw new Exception($"{seqId} sequence using in another place: {seq.InferState.State}");
```
- `ContextFullException` - не хватает места в контексте. Исключение **не выбрасывается методом напрямую**, а помещается в возвращаемую задачу, поэтому проявится при её ожидании.

**Использование:**<br>

```csharp
LLamaSeqId seq1 = await executor.CreateSequence();

// Одиночный префил: дожидается полного декодирования текста
await executor.ProcessPrompt(seq1, "<system>\n You are a helpful assistant\n</system>\n<user>\n Hello! </user>\n<assistant>\n", model.Vocab.ShouldAddBOS);
```
```csharp
// Пакетный префил нескольких последовательностей
Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([seq1, seq2, seq3], [text1, text2, text3]);
await Task.WhenAll(prefills.Values);
```
```csharp
// Свой addBos/special для каждой последовательности
Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt(
    [seq1, seq2],
    [text1, text2],
    [model.Vocab.ShouldAddBOS, false],
    [true, true]);
await Task.WhenAll(prefills.Values);
```

**Что если:**<br>
- **вызову префил для последовательности, которая уже занята?** получите исключение `sequence using in another place` - дождитесь завершения текущей операции.
- **текст не помещается в контекст?** задача префила завершится исключением `ContextFullException`. Проверить занятое место заранее можно методом [`GetContextFilling`](#методы-чтения-состояния-последовательности).
- **нужно добавить текст к уже заполненной последовательности?** просто вызовите `ProcessPrompt` повторно: токены добавятся в конец, позиция последовательности продолжит расти.

### ■ Метод генерации (Generate)

```csharp
public async Task<Channel<string>> Generate(LLamaSeqId seqId, IInferenceParams inferenceParam)
public async Task<Dictionary<LLamaSeqId, Channel<string>>> Generate(List<LLamaSeqId> seqIds, List<IInferenceParams> inferenceParams)
```
**Что делает?**<br>
Запускает генерацию для указанных последовательностей и возвращает канал с потоком текста. Исполнитель сэмплирует следующий токен, декодирует его моделью и записывает полученный текстовый фрагмент в `Channel<string>` соответствующей последовательности.

**Параметры:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - последовательности, для которых запускается генерация. n-е параметры применяются к n-й последовательности.
- `IInferenceParams inferenceParam` / `List<IInferenceParams> inferenceParams` - параметры генерации (см. раздел [Параметры генерации](#параметры-генерации-inferenceparams)).

**Возврат:**<br>
- одиночный вариант - `Task<Channel<string>>`;
- пакетный вариант - `Task<Dictionary<LLamaSeqId, Channel<string>>>`.

Канал читается до его завершения:

```csharp
Channel<string> ch = await executor.Generate(seq1, inferenceParams);

await foreach (string text in ch.Reader.ReadAllAsync())
{
    Console.Write(text);
}
```

**Генерация завершается и канал закрывается, если:**<br>
1. сгенерировано `MaxTokens` токенов (`MaxTokens != -1`);
2. в вывод попала любая из строк `AntiPrompts` (поиск подстроки в накопленном выводе; фрагмент, содержащий антипромпт, ещё будет доставлен в канал);
3. модель выдала EOG-токен, а `AutoStopFromEOG = true`;
4. в контексте закончилось место;
5. вызваны `StopSeqGeneration`, `DeleteSequence`, `ClearSequence` или `Dispose` исполнителя.

**Исключения:**<br>
- `ArgumentNullException` / `ArgumentException("Count is 0")` - некорректные списки аргументов.
- `Exception("There are few inferenceParams")` - параметров меньше, чем последовательностей.
- `IndexOutOfRangeException` - последовательности с таким id не существует:
```csharp
throw new IndexOutOfRangeException($"There is not {seqId} sequence");
```
- `Exception` - последовательность занята другой операцией: `sequence using in another place`.
- `Exception("For prefill use ProcessPrompt")` - для последовательности заданы параметры с `MaxTokens = 0`.
- `InvalidOperationException` - попытка генерировать на пустой последовательности (сначала нужен префил):
```csharp
throw new InvalidOperationException($"Cannot generate on an empty sequence (id={seqId}). Please prefill some text first.");
```
- `LLamaDecodeError` - `llama_decode` вернул ошибку; в пакетном исполнителе ошибка возникает в фоновом цикле декодирования.

**Использование:**<br>

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
// Пакетная генерация: последовательности генерируются параллельно в общих батчах
Dictionary<LLamaSeqId, Channel<string>> channels = await executor.Generate([seq1, seq2], [inferenceParams, inferenceParams]);

await foreach (string text in channels[seq1].Reader.ReadAllAsync()) { /* ... */ }
await foreach (string text in channels[seq2].Reader.ReadAllAsync()) { /* ... */ }
```

**Доп. информация:**<br>
После завершения генерации состояние последовательности возвращается в `None`, а контекст сохраняется - можно снова вызвать `Generate` (без повторного префила) или добавить текст через `ProcessPrompt`.

**Что если:**<br>
- **`MaxTokens = -1`?** генерация не ограничена по количеству токенов и остановится только по антипромпту, EOG или вызову `StopSeqGeneration`.
- **вызову `Generate` сразу для нескольких последовательностей?** они будут генерироваться пакетно, совместно используя простаивающее время вычислений.
- **один и тот же экземпляр `InferenceParams` передан двум последовательностям?** это допустимо, но конвейер сэмплинга у них будет общим (см. раздел про сэмплинг).

### ■ Параметры генерации (InferenceParams)

```csharp
InferenceParams inferenceParams = new InferenceParams() {...};
```
**Что делает?**<br>
Задаёт параметры генерации одной последовательности. Реализует интерфейс `IInferenceParams`, поэтому при необходимости можно написать собственную реализацию.

**Параметры:**<br>
- `public int TokensKeep { get; init; } = 0;` - сколько токенов исходного префила сохранять. На данный момент в цикле генерации не применяется (зарезервировано).
- `public int MaxTokens { get; init; } = -1;` - сколько новых токенов сгенерировать; `-1` - без ограничения. Значение `0` означает «не генерировать»: `Generate` с такими параметрами вызовет исключение.
- `public bool DecodeSpecialTokens { get; init; } = true;` - преобразовывать ли специальные токены (BOS, EOS и т.п.) в их текстовое представление при выводе. Если `false`, они не попадают в текст.
- `public bool AutoStopFromEOG { get; init; } = true;` - автоматически останавливать генерацию при получении EOG-токена (проверка выполняется в постобработке).
- `public IReadOnlyList<string> AntiPrompts { get; init; } = [];` - строки, при появлении которых в выводе генерация останавливается.
- `public ISamplingPipeline SamplingPipeline { get; init; }` - конвейер сэмплинга. Значение по умолчанию:
```csharp
new TunableSamplerPipeline(new TunableSamplerPipelineSettings(
    [new TopKSampler()],
    new Mirostat2Sampler()));
```

**Возврат:**<br>
Экземпляр `InferenceParams` (это `record` с `init`-свойствами, поэтому задаётся через инициализатор объекта).

**Использование:**<br>

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

### ■ Конвейер сэмплинга (SamplingPipeline)

**Что делает?**<br>
Конвейер сэмплинга превращает логиты модели в один выбранный токен. Исполнитель сам вызывает его методы на каждом шаге генерации, поэтому в большинстве случаев достаточно настроить `TunableSamplerPipeline` через `InferenceParams.SamplingPipeline`.

**Контракт `ISamplingPipeline`:**<br>

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
- `Apply` - применяет цепочку сэмплеров к логитам и выбирает токен-кандидат;
- `Accept` - сообщает конвейеру, что токен принят (обновляет внутреннее состояние: наказания, Mirostat, грамматику);
- `Reset` - сбрасывает состояние конвейера;
- `Sample` - сэмплирует токен напрямую из контекста.

Собственная реализация `ISamplingPipeline` даёт полный контроль над сэмплингом; базовый класс `BaseSamplingPipeline` берёт на себя создание и освобождение нативной цепочки сэмплеров.

**Настройка `TunableSamplerPipeline`:**<br>

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
Сэмплеры из `Samplers` применяются к логитам по порядку, затем `FinalizeSampler` выбирает итоговый токен.

**Сэмплеры (`ISampler`):**<br>

| Сэмплер | Что делает | Значения по умолчанию |
|---|---|---|
| `TopKSampler` | оставляет K наиболее вероятных токенов | `K = 40`, `0` - отключено |
| `TopNSigmaSampler` | порог `mean + N * std` | `N = 1` |
| `TopPSampler` | нуклеус: кумулятивная вероятность не выше P | `P = 0.95`, `MinKeep = 1` |
| `MinPSampler` | оставляет токены с вероятностью не ниже `P * maxProb` | `P = 0.05`, `MinKeep = 1` |
| `TypicalSampler` | typical sampling | `P = 0.2`, `MinKeep = 1` |
| `TemperatureSampler` | делит логиты на температуру перед softmax | `T = 0.8` |
| `AdapTSampler` | адаптивная температура по энтропии распределения | `T = 0.8`, `Delta = 0.40`, `Exponent = 1` |
| `XTCSampler` | с вероятностью P отбрасывает самые вероятные токены | `P = 0.5`, `T = 0.1`, `MinKeep = 1`, `Seed = 42` |
| `GrammarSampler` | ограничивает вывод грамматикой GBNF | `Grammar`, обязательное `Vocab = model.Vocab` |
| `PenaltiesSampler` | наказания за повторы, частоту и присутствие | `PenaltyCount = 64` (`0` - отключено, `-1` - весь контекст), `RepeatPenalty = 1`, `FrequencyPenalty = 0`, `PresencePenalty = 0` |
| `LogitBiasSampler` | сдвигает логиты указанных токенов | словарь `LogitBias`, обязательное `Vocab = model.Vocab` |

Значения проверяются при создании сэмплера: например, `TopPSampler` с `P > 1` или `TemperatureSampler` с `T <= 0` вызовет `ArgumentOutOfRangeException`.

**Финальные сэмплеры (`IFinalizeSampler`):**<br>

| Сэмплер | Что делает | Значения по умолчанию |
|---|---|---|
| `GreedySampler` | всегда выбирает токен с максимальной вероятностью | - |
| `DistributionSampler` | случайный выбор по распределению | `Seed = 42` |
| `Mirostat2Sampler` | Mirostat v2 с целевой перплексией | `Seed = 42`, `Tau = 3`, `Eta = 0.1` |

**Грамматика:**<br>

```csharp
public record Grammar(string Gbnf, string Root);
```
`Gbnf` - текст грамматики в формате GBNF, `Root` - имя стартового правила.

**Использование:**<br>

```csharp
SamplingPipeline = new TunableSamplerPipeline(
    new TunableSamplerPipelineSettings(
        [
            new PenaltiesSampler() { RepeatPenalty = 1.1f },
            new TopKSampler() { K = 30 },
            new TemperatureSampler() { T = 0.7f },
            new TopPSampler() { P = 0.9f, MinKeep = 1 }
        ],
        new DistributionSampler() { Seed = 256 }
    )
)
```
```csharp
// Генерация строго по грамматике (например, JSON)
SamplingPipeline = new TunableSamplerPipeline(
    new TunableSamplerPipelineSettings(
        [new GrammarSampler { Vocab = model.Vocab, Grammar = new Grammar(jsonGbnf, "root") }],
        new GreedySampler()
    )
)
```

**Что если:**<br>
- **один и тот же `InferenceParams` используется двумя последовательностями?**

они будут делить один экземпляр конвейера вместе с его состоянием (наказания, Mirostat, позиция грамматики). Для независимой генерации создавайте отдельный `InferenceParams` (и свой конвейер) на последовательность.

- **конвейер больше не нужен?**

он владеет нативными ресурсами, поэтому вызовите `Dispose()` (например, `inferenceParams.SamplingPipeline.Dispose()`).

### ■ Методы остановки генерации и префила

```csharp
public async Task StopSeqGeneration(LLamaSeqId id)
public async Task StopSeqPrefill(LLamaSeqId id, LLamaPos startPos /* включительно */)
```
**Что делает?**<br>
- `StopSeqGeneration` - запрашивает остановку генерации последовательности; остановка применяется перед следующим шагом декодирования, канал генерации закрывается.
- `StopSeqPrefill` - прерывает выполняющийся префил и удаляет все токены, добавленные этим префилом, начиная с позиции `startPos` (включительно), возвращая состояние последовательности к этой позиции.

**Параметры:**<br>
- `LLamaSeqId id` - идентификатор последовательности.
- `LLamaPos startPos` - позиция (включительно), с которой удаляются токены; должна быть меньше текущего `NextDecodedTokenPos` последовательности.

**Возврат:**<br>
`Task`

**Исключения:**<br>
- `IndexOutOfRangeException` - последовательности нет (`sequence {id} not exist`), она не генерируется (`sequence {id} not in generate`) или не находится в префиле (`sequence {id} not in prefill`).
- `ArgumentException` - `startPos` больше или равен `NextDecodedTokenPos`:
```csharp
throw new ArgumentException("Start position must be lower then NextDecodedTokenPos");
```
- `Exception` - последовательность пуста (`sequence is empty`).
- `Exception` - для рекуррентных и гибридных моделей операция недоступна:
```csharp
throw new Exception("for recurrent and hybrid models, use state checkpoints(not yet implemented)");
```

**Использование:**<br>

```csharp
await executor.StopSeqGeneration(seq1);
```

### ■ Методы чтения состояния последовательности

```csharp
public async Task<int> GetSequenceNextDecodedTokenPos(LLamaSeqId seqId)
public async Task<IReadOnlyList<LLamaToken>> GetSequenceDecodedTokens(LLamaSeqId seqId)
public async Task<LLamaTokenDataArray?> GetSequenceLastLogits(LLamaSeqId seqId)
public async Task<int> GetContextFilling()
```
**Что делает?**<br>
- `GetSequenceNextDecodedTokenPos` - возвращает позицию следующего токена последовательности (фактически число уже обработанных токенов). Полезно, например, для клонирования префикса через `CopySeqPrefixTo`.
- `GetSequenceDecodedTokens` - возвращает список всех декодированных токенов последовательности. В мультимодальных сессиях на месте медиа-позиций стоят токены-заполнители (см. раздел [Токены-заполнители](#токены-заполнители-placeholders)).
- `GetSequenceLastLogits` - возвращает последние логиты последовательности или `null`, если они ещё не вычислялись (например, последовательность пуста).
- `GetContextFilling` - суммарное число реальных токенов, занимающих ячейки контекста во всех последовательностях исполнителя (сравнивайте с `Context.ContextSize`).

**Параметры:**<br>
- `LLamaSeqId seqId` - идентификатор последовательности (кроме `GetContextFilling`, у которого параметров нет).

**Возврат:**<br>
`Task<int>`, `Task<IReadOnlyList<LLamaToken>>` или `Task<LLamaTokenDataArray?>` в зависимости от метода.

**Исключения:**<br>
- `IndexOutOfRangeException` - последовательности с таким id не существует.
- `OperationCanceledException` / `ObjectDisposedException` - исполнитель остановлен или освобождён.

**Использование:**<br>

```csharp
int pos = await executor.GetSequenceNextDecodedTokenPos(seq1);
IReadOnlyList<LLamaToken> tokens = await executor.GetSequenceDecodedTokens(seq1);
LLamaTokenDataArray? logits = await executor.GetSequenceLastLogits(seq1);
int filling = await executor.GetContextFilling();
```

### ■ Метод пробного сэмплирования (SampleOneTokenWithoutDecode)

```csharp
public async Task<string> SampleOneTokenWithoutDecode(LLamaSeqId seqId, IInferenceParams inferenceParams)
```
**Что делает?**<br>
Сэмплирует один токен из последних логитов последовательности, но **не декодирует его и не фиксирует**: состояние модели и позиция последовательности не меняются. Удобно для быстрой проверки, например когда в контекст добавлен вопрос «да/нет» и нужно подсмотреть ответ, не влияя на него.

**Параметры:**<br>
- `LLamaSeqId seqId` - идентификатор последовательности.
- `IInferenceParams inferenceParams` - параметры сэмплинга (используется только конвейер сэмплинга).

**Возврат:**<br>
`Task<string>` - текстовый фрагмент, соответствующий сэмплированному токену.

**Исключения:**<br>
- `IndexOutOfRangeException` - последовательности не существует (`Sequence {seqId} does not exist`).
- `Exception` - последовательность занята другой операцией, у неё нет логитов или не переданы параметры:
```csharp
throw new Exception($"{seqId} sequence is being used in another place: {seq.InferState.State}");
throw new Exception($"Sequence {seqId} has no logits");
throw new Exception($"No inference parameters");
```
Метод вызывает `Apply`, но **не** вызывает `Accept`, поэтому принятое токеном состояние (наказания, грамматика) не обновляется.

**Использование:**<br>

```csharp
string probe = await executor.SampleOneTokenWithoutDecode(seq1, inferenceParams);
```

### ■ Метод копирования префикса последовательности (CopySeqPrefixTo)

```csharp
public async Task CopySeqPrefixTo(LLamaSeqId srcId, List<LLamaSeqId> targetIds, LLamaPos endPos)
```
**Что делает?**<br>
Копирует префикс последовательности-источника в каждую целевую последовательность. Целевые последовательности перед копированием полностью очищаются. Если `endPos` равно текущей позиции источника, копируется всё состояние целиком; иначе копируется только часть.

**Параметры:**<br>
- `LLamaSeqId srcId` - источник префикса.
- `List<LLamaSeqId> targetIds` - последовательности-получатели.
- `LLamaPos endPos` - позиция конца копируемого префикса (не включительно). Должна быть больше или равна 1 и не превышать `NextDecodedTokenPos` источника.

**Возврат:**<br>
`Task`

**Исключения:**<br>
- `ArgumentNullException` - `targetIds` равен `null`.
- `ArgumentException` - пустой список целевых последовательностей, `endPos < 1` или `endPos > NextDecodedTokenPos`:
```csharp
throw new ArgumentException("Count is 0");
throw new ArgumentException("End position must be 1 or greater");
throw new ArgumentException("End position must be lower or equal NextDecodedTokenPos");
```
- `IndexOutOfRangeException` - источника или одной из целей не существует.
- `Exception` - последовательность занята; для рекуррентных моделей частичное копирование недоступно:
```csharp
throw new Exception("for recurrent models, use state checkpoints(not yet implemented)");
```

**Доп. информация:**<br>
При неполном копировании первый сгенерированный целевой последовательностью токен повторяет токен, стоявший на позиции `endPos` источника (логиты этой позиции не копируются, так как они не сохранены).

**Использование:**<br>

```csharp
// Общий префил для двух веток диалога
await executor.ProcessPrompt(seq1, prompt, model.Vocab.ShouldAddBOS);

LLamaPos endPos = await executor.GetSequenceNextDecodedTokenPos(seq1);
await executor.CopySeqPrefixTo(seq1, [seq2], endPos);

Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
```

### ■ Методы редактирования и очистки последовательности

```csharp
public async Task DeleteSequenceEnd(LLamaSeqId seqId, LLamaPos startPos /* включительно */)
public async Task ClearSequence(LLamaSeqId id)
```
**Что делает?**<br>
- `DeleteSequenceEnd` - удаляет токены от `startPos` (включительно) до конца последовательности. При `startPos = 0` очищает последовательность целиком. Используется, чтобы откатить генерацию и переиспользовать префикс.
- `ClearSequence` - очищает KV-кэш и состояние последовательности, прерывая выполняющиеся для неё префил или генерацию. Работает для всех типов моделей (в отличие от `DeleteSequenceEnd`).

**Параметры:**<br>
- `LLamaSeqId seqId`/`id` - идентификатор последовательности.
- `LLamaPos startPos` - позиция (включительно), с которой удаляются токены; должна быть меньше `NextDecodedTokenPos`.

**Возврат:**<br>
`Task`

**Исключения:**<br>
- `IndexOutOfRangeException` - последовательности с таким id не существует.
- `ArgumentException` - `startPos` больше или равен `NextDecodedTokenPos`.
- `Exception` - последовательность занята (`sequence using in another place`) или пуста (`sequence is empty`).
- `Exception` - для рекуррентных и гибридных моделей операция `DeleteSequenceEnd` недоступна:
```csharp
throw new Exception("for recurrent and hybrid models, use state checkpoints(not yet implemented)");
```

**Использование:**<br>

```csharp
// Сгенерировали ответ, откатили его и сгенерировали заново
LLamaPos startPos = await executor.GetSequenceNextDecodedTokenPos(seq1);
Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
await foreach (string _ in ch1.Reader.ReadAllAsync()) { }

await executor.DeleteSequenceEnd(seq1, startPos);

Channel<string> ch2 = await executor.Generate(seq1, inferenceParams);
```
```csharp
await executor.ClearSequence(seq1);
```

### ■ Метод освобождения исполнителя

```csharp
public void Dispose()
```
**Что делает?**<br>
Останавливает фоновый цикл декодирования, освобождает синхронизацию исполнителя и его контекст (`Context`). Модель `LLamaWeights` при этом не освобождается.

**Использование:**<br>

```csharp
executor.Dispose();
model.Dispose();
```

**Доп. информация:**<br>
Освобождайте исполнители **до** модели. Если модель освобождена, а исполнитель остался, обращения к контексту приведут к исключению:
```csharp
throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");
```
Перед освобождением активные генерации стоит остановить (`StopSeqGeneration`) или дождаться их завершения.

### ■ Типичные исключения LlamaExecutor

| Исключение | Когда возникает |
|---|---|
| `IndexOutOfRangeException` | последовательности с указанным id нет, либо она не в нужном состоянии (`not in prefill` / `not in generate`) |
| `Exception` `sequence using in another place` | последовательность уже занята другой операцией |
| `ContextFullException` | не хватает места в контексте (передаётся через задачу префила или префила mtmd) |
| `InvalidOperationException` | попытка генерации на пустой последовательности |
| `ArgumentException` / `ArgumentNullException` | некорректные или неполные списки аргументов, неверные позиции |
| `Exception` `use state checkpoints(not yet implemented)` | операции редактирования состояния на рекуррентных и гибридных моделях |
| `LLamaDecodeError` | ошибка `llama_decode` в фоновом цикле декодирования |
| `OperationCanceledException` / `ObjectDisposedException` | исполнитель остановлен или освобождён |

## Мультимодальность (mtmd)

Поддержка мультимодальности (mtmd) позволяет подавать в модель не только текст, но и изображения и аудио. Используются три компонента:

1. библиотека `mtmd` (`mtmd.dll`/`mtmd.so`), подключаемая при инициализации;
2. файл `mmproj` - веса мультимодального проектора, соответствующие загружаемой модели;
3. объект `MtmdContext`, который кодирует медиа в эмбеддинги, и `LlamaExecutor`, который вставляет эти эмбеддинги в последовательность.

### ■ Подключение mtmd при инициализации

Библиотека mtmd подключается необязательным пятым параметром метода `LlamaCpp.Initialize` (подробнее в разделе [Инициализация библиотеки](#инициализация-библиотеки)):

```csharp
LlamaCpp.Initialize(
    "./llama/llama.dll",
    "./llama/ggml.dll",
    "./llama/ggml-base.dll",
    ["./llama/ggml-cpu-alderlake.dll"],
    "./llama/mtmd.dll" // подключает мультимодальность
);
```

**Что если:**<br>
- **mtmd не был подключён?** любой вызов мультимодального API выбросит исключение:
```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize() with MTMD dll/so");
```
Например, его выбросит `MtmdContextParams.Default()`, который вызывается при создании mtmd-контекста.

**Доп. информация:**<br>
Файлы `llama.dll`, `ggml*.dll` и `mtmd.dll` берутся из одного релиза [llama.cpp](https://github.com/ggml-org/llama.cpp/releases). Файл `mmproj` скачивается отдельно - он должен соответствовать модели (на странице модели на Hugging Face это файлы вида `mmproj-*.gguf`). Сама модель тоже должна быть мультимодальной (Qwen3-VL, Qwen3-ASR, Gemma и подобные).

### ■ Параметры создания mtmd-контекста

```csharp
MtmdParams @params = new MtmdParams() {...};
```
**Что делает?**<br>
Определяет настройки мультимодального проектора (`mmproj`). Все поля кроме `UseGpu` и `PrintTimings` - nullable: значение `null` означает «взять значение по умолчанию нативной стороны» (`MtmdContextParams.Default()`).

**Параметры:**<br>
- `public bool UseGpu { get; init; } = false;` - использовать ли загруженный GPU-бэкенд для энкодера медиа.
- `public bool PrintTimings { get; init; } = false;` - печатать ли тайминги кодирования.
- `public int? Threads { get; init; } = null;` - число потоков энкодера.
- `public LlamaFlashAttentionType? FlashAttention { get; init; } = null;` - тип flash attention для mtmd.
- `public bool? Warmup { get; init; } = null;` - выполнять ли warmup-проход после инициализации (нативное значение по умолчанию - `true`).
- `public int? ImageMinTokens { get; init; } = null;` - минимальное число токенов для изображения (по умолчанию берётся из метаданных `mmproj`).
- `public int? ImageMaxTokens { get; init; } = null;` - максимальное число токенов для изображения (по умолчанию из метаданных). Уменьшение ускоряет кодирование изображения и экономит контекст.
- `public int? BatchSize { get; init; } = null;` - максимальное число выходных токенов энкодера в одном батче (по умолчанию 1024). Это не жёсткий лимит: первое изображение добавляется всегда, даже если превышает его.

**Возврат:**<br>
Экземпляр `MtmdParams`.

**Использование:**<br>

```csharp
MtmdParams mtmdParams = new MtmdParams
{
    UseGpu = false,
    Threads = 8,
    ImageMaxTokens = 1000,
    BatchSize = 1024
};
```

### ■ Создание mtmd-контекста

```csharp
public static MtmdContext CreateFromFile(string mmprojFile, LLamaWeights llamaModel, IMtmdParams @params)
```
**Что делает?**<br>
Загружает веса мультимодального проектора из указанного файла, связывает их с моделью и создаёт `MtmdContext`, который запускает собственный фоновый цикл кодирования медиа.

**Параметры:**<br>
- `string mmprojFile` - путь к файлу `mmproj` (мультимодальный проектор, соответствующий модели).
- `LLamaWeights llamaModel` - уже загруженная модель.
- `IMtmdParams @params` - параметры, описанные выше.

**Возврат:**<br>
Экземпляр `MtmdContext`.

**Исключения:**<br>
- `InvalidOperationException` - mtmd не был подключён при инициализации:
```csharp
throw new InvalidOperationException("First of all call LlamaCpp.Initialize() with MTMD dll/so");
```
- `InvalidOperationException` - файл `mmproj` не читается:
```csharp
throw new InvalidOperationException($"Model file '{mmprojPath}' is not readable");
```
- `LoadWeightsFailedException` - движок не смог загрузить `mmproj`:
```csharp
throw new LoadWeightsFailedException(mmprojPath);
```

**Использование:**<br>

```csharp
LLamaWeights model = LLamaWeights.LoadFromFile(new ModelParams(modelPath) { GpuLayerCount = 99 });
MtmdContext ctx = MtmdContext.CreateFromFile(mmprojPath, model, mtmdParams);
```

**Доп. информация:**<br>
`mmproj` должен соответствовать модели: чужой проектор приведёт к ошибке загрузки или к некорректным эмбеддингам. `MtmdContext` не владеет моделью - `Dispose` mtmd-контекста модель не освобождает.

### ■ Свойства MtmdContext

```csharp
public bool NonCasualDecode { get; }  // полное (не причинное) внимание при декодировании изображений
public bool MropeDecode { get; }      // используется ли MROPE-позиционирование
public bool SupportVision { get; }    // поддерживает ли проектор изображения
public bool SupportAudio { get; }     // поддерживает ли проектор аудио
public int AudioSampleRate { get; }   // частота дискретизации, ожидаемая моделью
```
**Что делает?**<br>
Свойства доступны только для чтения и отражают возможности загруженного `mmproj`. `AudioSampleRate` используется методом `DecodeWavToMonoFloat` для ресемплинга аудио.

**Использование:**<br>

```csharp
Console.WriteLine($"Vision: {ctx.SupportVision}, Audio: {ctx.SupportAudio}, SampleRate: {ctx.AudioSampleRate}");
```

**Что если:**<br>
- **проектор не поддерживает изображения?** `EncodeImage*` выбросит `throw new Exception("Vision dont support");`
- **проектор не поддерживает аудио?** `EncodeAudio*` и `DecodeWavToMonoFloat` выбросят `throw new Exception("Audio dont support");`

### ■ Спецификация mtmd и подключение к исполнителю

```csharp
public MtmdSpec GetSpecification()
public async Task InitializeMtmdAsync(MtmdSpec mtmdSpec)
```
```csharp
public readonly record struct MtmdSpec (bool UseMrope);
```
**Что делает?**<br>
- `GetSpecification` возвращает компактную спецификацию mtmd-контекста (признак использования MROPE), которая нужна исполнителю, чтобы правильно рассчитывать позиции эмбеддингов.
- `InitializeMtmdAsync` переводит уже созданный исполнитель в мультимодальный режим - используется, если исполнитель был создан без спецификации.

**Использование:**<br>

```csharp
// Вариант 1: сразу при создании исполнителя
LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());
```
```csharp
// Вариант 2: подключение позже
LlamaExecutor executor = model.CreateExecutor(ctxParams);
await executor.InitializeMtmdAsync(ctx.GetSpecification());
```

**Что если:**<br>
- **исполнитель создан без mtmd и вызову `ProcessMtmdEmbeds`?**

получите исключение:
```csharp
throw new Exception("Mtmd was not initialized");
```

- **`UseMrope = true`?** позиции эмбеддингов кодируются четырьмя координатами (t, y, x, z) вместо одной - это учитывается исполнителем автоматически.

### ■ Кодирование изображений

```csharp
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromPath(string filePath)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromRGB(uint nx, uint ny, Memory<byte> image)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImageFromPaths(List<string> filePaths)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
```
**Что делает?**<br>
Токенизирует медиа через mtmd (получая маркеры начала и конца медиа и чанки изображения) и ставит эмбеддинги в очередь фонового кодирования. Одиночные методы дожидаются готовности эмбеддингов, пакетные - возвращают `Task` на каждое изображение (кодирование идёт в фоне и объединяется в общие батчи энкодера).

**Параметры:**<br>
- `string filePath` / `List<string> filePaths` - путь(и) к файлам изображений. Поддерживаются форматы, которые читает библиотека SixLabors.ImageSharp (png, jpeg, bmp, webp и др.).
- `uint nx, uint ny` - ширина и высота изображения в пикселях.
- `Memory<byte> image` - буфер RGB24 сверху вниз: `nx * ny * 3` байт, строка за строкой начиная с верхней.
- `List<uint> nxs, List<uint> nys, List<Memory<byte>> images` - то же для пакетной обработки: n-й элемент каждого списка описывает n-е изображение.

**Возврат:**<br>
Кортеж `(string BOM, string EOM, ...)`, где:
- `BOM` - строка специальных токенов начала медиа (Begin Of Media), её нужно вставить в промпт перед эмбеддингами;
- `EOM` - строка специальных токенов конца медиа (End Of Media), вставляется после эмбеддингов;
- `embeds` - у одиночных методов готовый массив `LlamaEmbedding[]`, у пакетных - `Task<LlamaEmbedding[]>`, который нужно дождаться.

**Исключения:**<br>
- `Exception` - проектор не поддерживает изображения: `throw new Exception("Vision dont support");` (см. раздел [Свойства MtmdContext](#свойства-mtmdcontext)).
- `ArgumentNullException` / `ArgumentException` - некорректные входные буферы (проверки `RegisterEncode`).
- `Exception` - нарушена структура чанков, полученных от mtmd (например, `start and end chunks must be TEXT`).

**Использование:**<br>

```csharp
// Одно изображение: эмбеддинги готовы сразу
var result = await ctx.EncodeImageFromPath("./photo.png");

// Несколько изображений: кодирование выполняется пакетно
var results = await ctx.EncodeImageFromPaths(["./1.png", "./2.png", "./3.png"]);
foreach (var item in results)
{
    LlamaEmbedding[] embeds = await item.embeds; // дожидаемся каждого результата
}
```

**Доп. информация:**<br>
Количество токенов, которое займёт изображение, регулируется параметрами `ImageMinTokens`/`ImageMaxTokens` из `MtmdParams`; при необходимости mtmd масштабирует изображение.

### ■ Кодирование аудио

```csharp
public float[] DecodeWavToMonoFloat(string path)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudioFromWav(string wavFilePath)
public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudio(Memory<float> audio)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudiosFromWav(List<string> wavFilePaths)
public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudios(List<Memory<float>> audio)
```
**Что делает?**<br>
- `DecodeWavToMonoFloat` - читает WAV-файл, преобразует его в моно (стерео усредняется), при необходимости ресемплит к `AudioSampleRate` и возвращает PCM float32 в диапазоне [-1, 1].
- Остальные методы работают так же, как методы кодирования изображений: одиночные дожидаются эмбеддингов, пакетные возвращают задачи.

**Параметры:**<br>
- `string path` / `string wavFilePath` / `List<string> wavFilePaths` - путь(и) к WAV-файлам.
- `Memory<float> audio` / `List<Memory<float>> audio` - уже подготовленный моно PCM float32.

**Возврат:**<br>
- `float[]` - моно PCM float32;
- кортежи `(BOM, EOM, embeds)` - аналогично изображениям.

**Исключения:**<br>
- `Exception` - проектор не поддерживает аудио: `Audio dont support`.
- `NotSupportedException` - больше двух каналов в файле:
```csharp
throw new NotSupportedException($"Unsupported channel count: {channels}");
```

**Использование:**<br>

```csharp
var result = await ctx.EncodeAudioFromWav("./audio.wav");
```

### ■ Вставка эмбеддингов в последовательность (ProcessMtmdEmbeds)

```csharp
public async Task ProcessMtmdEmbeds(LLamaSeqId seqId, LlamaEmbedding[] embeds)
public async Task<Dictionary<LLamaSeqId, Task>> ProcessMtmdEmbeds(List<LLamaSeqId> seqIds, List<LlamaEmbedding[]> embeds)
```
**Что делает?**<br>
Ставит эмбеддинги медиа в очередь мультимодального префила указанных последовательностей. Декодирование эмбеддингов выполняет фоновый цикл исполнителя, объединяя их в батчи (до `ContextParams.BatchSize` за проход); для не причинных (non-causal) чанков внимание на время батча переключается автоматически. После завершения позиция последовательности сдвигается на число эмбеддингов, в историю добавляются токены-заполнители, а логиты последней позиции становятся доступны - дальше можно продолжать обычным текстовым префилом (`EOM` + вопрос) или запускать генерацию.

**Параметры:**<br>
- `LLamaSeqId seqId` / `List<LLamaSeqId> seqIds` - последовательности.
- `LlamaEmbedding[] embeds` / `List<LlamaEmbedding[]> embeds` - эмбеддинги: n-му массиву соответствует n-я последовательность. В одном массиве должна быть **одна медиа-единица** (одно изображение либо одно аудио), так как между разными медиа должны стоять текстовые токены.

**Возврат:**<br>
- одиночный вариант - `Task`, завершающийся, когда эмбеддинги последовательности декодированы;
- пакетный вариант - `Dictionary<LLamaSeqId, Task>`, задачи нужно дождаться.

**Исключения:**<br>
- `Exception("Mtmd was not initialized")` - исполнитель не в мультимодальном режиме.
- `ArgumentNullException` / `ArgumentException("Count is 0")` / `Exception("There are few embeds")` - некорректные списки.
- `IndexOutOfRangeException` - последовательности с таким id не существует.
- `Exception` - последовательность занята другой операцией: `sequence using in another place`.
- `ContextFullException` - не хватает места в контексте; исключение помещается в возвращаемую задачу.

**Использование:**<br>

```csharp
// Порядок важен: текст с BOM -> эмбеддинги -> текст с EOM
await executor.ProcessPrompt(seq1, "<|im_start|>user\n " + result.BOM);
await executor.ProcessMtmdEmbeds(seq1, result.embeds);
await executor.ProcessPrompt(seq1, result.EOM + "What is shown in the image?\n<|im_end|>\n<|im_start|>assistant\n");
```
```csharp
// Пакетная вставка в несколько последовательностей
Dictionary<LLamaSeqId, Task> embedsTasks = await executor.ProcessMtmdEmbeds([seq1, seq2, seq3], [emb1, emb2, emb3]);
await Task.WhenAll(embedsTasks.Values);
```

**Что если:**<br>
- **передать пустой массив эмбеддингов?**

для такой последовательности вернётся уже завершённая задача - удобно, когда медиа в конкретном запросе нет.

- **не дождаться возвращённой задачи?**

эмбеддинги могут быть ещё не декодированы, и последующий префил/генерация увидит неполную последовательность(а точнее вообще пустую). Всегда дожидайтесь задач.

### ■ Эмбеддинги mtmd (LlamaEmbedding и LlamaEmbeddingType)

```csharp
public readonly record struct LlamaEmbedding
{
    public Memory<float> Data { get; }      // вектор эмбеддинга
    public LlamaEmbeddingType Type { get; } // тип модальности
}
```
```csharp
public enum LlamaEmbeddingType { Text, Image, Audio, HiddenState, Video }
```
**Что делает?**<br>
Описывает один вектор эмбеддинга медиа. `Data` содержит сам вектор (его длина совпадает с размерностью входных эмбеддингов модели), `Type` - модальность. Позиция MROPE и признак не причинного внимания хранятся внутри и заполняются mtmd автоматически.

**Доп. информация:**<br>
Создавать `LlamaEmbedding` вручную не нужно - они приходят из методов `Encode*` и передаются в `ProcessMtmdEmbeds`.

### ■ Токены-заполнители (placeholders)

```csharp
public static bool IsMtmdPlaceholder(this LLamaToken token)
public static LLamaToken CreateImagePlaceholder()  // -100
public static LLamaToken CreateVideoPlaceholder()  // -101
public static LLamaToken CreateAudioPlaceholder()  // -102
```
**Что делает?**<br>
После декодирования эмбеддингов исполнитель записывает в историю последовательности по одному токену-заполнителю на каждую медиа-позицию. Это внутренние токены: они не существуют в словаре модели, не сэмплируются и не декодируются, а нужны лишь для того, чтобы позиции последовательности и история оставались согласованными.

**Доп. информация:**<br>
Константы базовых значений лежат в `MtmdTokenConstants` (`MtmdImagePlaceholderBase = -100`, `MtmdVideoPlaceholderBase = -101`, `MtmdAudioPlaceholderBase = -102`); для заполнителей зарезервирован диапазон от -200 до -100, чтобы не пересечься с `InvalidToken` (-1) и специальными токенами модели (обычно >= 0).

**Что если:**<br>
- **токен-заполнитель встретился в `GetSequenceDecodedTokens`?**

это позиция медиа, а не настоящий токен словаря. Проверить можно расширением `token.IsMtmdPlaceholder()`; передавать такие токены в `ProcessPrompt` не нужно - для текста используются строки `BOM`/`EOM`.

### ■ Пример: вопрос по изображению

```csharp
// 1. Инициализация библиотек с mtmd
LlamaCpp.Initialize(
    Path.Combine(baseDllPath, "llama.dll"),
    Path.Combine(baseDllPath, "ggml.dll"),
    Path.Combine(baseDllPath, "ggml-base.dll"),
    [Path.Combine(baseDllPath, "ggml-cpu-alderlake.dll")],
    Path.Combine(baseDllPath, "mtmd.dll"));

// 2. Модель
LLamaWeights model = LLamaWeights.LoadFromFile(new ModelParams(modelPath));

// 3. Мультимодальный контекст
MtmdParams mtmdParams = new MtmdParams
{
    UseGpu = false,
    Threads = 8,
    ImageMaxTokens = 1000
};
MtmdContext ctx = MtmdContext.CreateFromFile(mmprojPath, model, mtmdParams);

// 4. Кодирование изображения
var result = await ctx.EncodeImageFromPath("./test.png");

// 5. Исполнитель с мультимодальной спецификацией
LlamaExecutor executor = model.CreateExecutor(
    new ContextParams() { ContextSize = 8000 },
    ctx.GetSpecification());

LLamaSeqId seq1 = await executor.CreateSequence();

// 6. Префил: текст с BOM, затем эмбеддинги, затем EOM и вопрос
await executor.ProcessPrompt(seq1, "<|im_start|>system\n you are a helpfull assistant\n<|im_end|>\n<|im_start|>user\n " + result.BOM);
await executor.ProcessMtmdEmbeds(seq1, result.embeds);
await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image?\n<|im_end|>\n<|im_start|>assistant\n");

// 7. Генерация
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

// 8. Освобождение
ctx.Dispose();
executor.Dispose();
model.Dispose();
```

Пакетный вариант (несколько изображений в нескольких последовательностях) использует ту же схему: общий текстовый префил в первой последовательности, `CopySeqPrefixTo` для остальных, затем `ProcessMtmdEmbeds` списком и параллельная генерация (пример есть в `Llama.csharp.IntegrationTest/TestMtmd.cs`).

### ■ Освобождение ресурсов mtmd

```csharp
public void Dispose()
```
**Что делает?**<br>
Останавливает фоновый цикл кодирования, освобождает нативный контекст mtmd и отменяет незавершённые задачи кодирования (ожидание таких задач завершится отменой). Модель не освобождается.

**Использование:**<br>

```csharp
ctx.Dispose();
executor.Dispose();
model.Dispose();
```

**Доп. информация:**<br>
Временные буферы (bitmap) освобождаются автоматически сразу после регистрации кодирования, поэтому дополнительно чистить их не нужно. Модель освобождайте последней: если освободить её раньше, обращения к `mtmd`- и языковому контексту приведут к `ObjectDisposedException`.