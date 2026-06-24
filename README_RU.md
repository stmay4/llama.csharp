<p align="center">
  <img src=".\llamacsharp_light.svg" alt="llamacsharp Logo" width="500"/>
</p>

# llama.csharp

**llama.csharp** - библиотека-обёртка над [llama.cpp](https://github.com/ggml-org/llama.cpp) для .NET, предоставляющая пакетную обработку (Continuous batching) и работу с кэшем последовательностей контекста.

Телеграм группа проекта - [Local AI Models](https://t.me/+5u17pJAAlSJlN2Zi) для получения уведомлений об обновлениях, общения, вопросов, примеров использования и т.п. Также это общая группа со всеми проектами, реализуемыми мною (и может в будущем не только мною) на основе данной библиотеки.

## Пример использования

Для простого чата *(немного сокращенный код)*

```csharp
var requiredFiles = new[] { *пути к файлам движка*};

// Инициализация библиотеки
LlamaCpp.Initialize(requiredFiles[0],
                    requiredFiles[1],
                    requiredFiles[2],
                    [requiredFiles[3]]);
// Загрузка модели
ModelParams parametres = new ModelParams(_modelPath) 
{
    //GpuLayerCount = 999, // Можно установить при использовании GPU
    //TensorBufferOverrides = [new TensorBufferOverride("blk\\.[0-35].*exps.*", "CPU")], // Если используется GPU, для MOE моделей можно выгрузить экспертов на CPU. В данном примере для слоев 0-35
    //TensorBufferOverrides = [new TensorBufferOverride(".*exps.*", "CPU")], // Здесь всех экспертов
    //... другие настройки (подробнее в документации)
};
LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

// Создаем испольнитель для работы с контекстом
ContextParams ctxParams = new ContextParams()
{
    ContextSize = 16000,
    //SeqMax = 1, // число последовательностей, доступных для создания, по умолчанию и так одна
    //NoKqvOffload = false // Если используется GPU и достаточно VRAM, можно выгрузить контекст на GPU
    //... другие настройки (подробнее в документации)
};
LlamaExecutor executor = model.CreateExecutor(ctxParams);

// Создаем последовательность (для простого чата достаточно одной)
LLamaSeqId mainSeq = await executor.CreateSequence();

//Сообщение, которое будет загружено в контекст последовательности. Содержит произвольно расставленные теги ролей
string startPrefill = "<system>\r\nYou are an expert translator. Before translating, you must analyze the input in a <think> block.\r\n</system>\r\n" +
    "<user>\r\n今天天气不错，我们去公园散步吧\r\n</user>\r\n" +
    "<assistant>\r\n<think>\r\nThe input is a casual Chinese sentence. " +
    "Breaking it down: 今天 (today), 天气 (weather), 不错 (not bad / pretty good), 我们 (we), 去 (go), 公园 (park), 散步 (take a walk / stroll), 吧 (suggestion particle). " +
    "The overall tone is friendly and suggestive. A natural English equivalent should maintain this casual, inviting tone.\r\n</think>\r\n" +
    "The weather is nice today, let's go for a walk in the park.\r\n</assistant>\r\n" +
    "<user>\r\n请把这份文件翻译成英文，注意保持正式语气\r\n</user>\r\n" +
    "<assistant>\r\n<think>\r\nThe user is asking to translate a document, but no document has been provided yet. " +
    "The instruction itself is in Chinese: 请 (please), 把这份文件 (this document), 翻译成英文 (translate into English), 注意保持正式语气 (pay attention to maintaining a formal tone). " +
    "Since the user only provided the instruction and not the actual document, I should acknowledge the request and ask for the document text.\r\n</think>\r\n" +
    "Please share the document text you would like me to translate, and I will ensure a formal tone in the English version.\r\n</assistant>";
//можно добавить после '</assistant>' EOS токен модели, полученный из model.Vocab.LLamaTokenToString(model.Vocab.EOS, true) если необходимо

// Заполняем контекст последовательности. Никакие токены не добавляются к вводу внутри, только BOS, если установлено в третьем аргументе (по умолчанию false)
await executor.ProcessPrompt(mainSeq, startPrefill, model.Vocab.ShouldAddBOS);

Console.WriteLine(startPrefill);

// Параметры генерации
InferenceParams inferenceParams = new InferenceParams()
{
    MaxTokens = -1,
    AutoStopFromEOG = true,
    DecodeSpecialTokens = true,
    AntiPrompts = ["</assistant>"],
    SamplingPipeline = new TunableSamplerPipeline(
        new TunableSamplerPipelineSettings(
            [
                // Лист сэмплеров в порядке применения к логитам
                new TopKSampler() { K=30 }
            ],
            // Финализирующий семплер: Greedy, Distribution или Mirostat2
            new Mirostat2Sampler() { Seed = 256}
        )
    )
};

string input = "";

// Цикл чата
while (true)
{
    Console.Write("Me: "); // не отправляется в модель, визуал
    *получение ввода, проверка на выход из цикла*

    // Текст, отправляемый в контекст последовательности
    string prefillInput = "\r\n<user> " + input + " </user>\r\n<assistant>\r\n<think>";

    Console.Write("\r\nNot me):\r\n<think>"); // не отправляется в модель, визуал

    // Заполняем ввод пользователя в контекст последовательности
    await executor.ProcessPrompt(mainSeq, prefillInput);

    // Получаем канал генерации для последовательности, передаем только ее id и параметры генерации
    Channel<string> ch = await executor.Generate(mainSeq, inferenceParams);

    await foreach (string token in ch.Reader.ReadAllAsync())
    {
        Console.Write(token); // Печать токенов, преобразованных в строки
    }
}

executor.Dispose(); // Освобождаем исполнитель
model.Dispose(); // и модель
```

Для пакетной генерации по общему префиксу (сильно сокращенный код)
```csharp
    LlamaCpp.Initialize(...);
    ModelParams parametres = new ModelParams(_modelPath) {...};
    LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

    ContextParams ctxParams = new ContextParams()
    {
        ContextSize = 16000,
        SeqMax = 3 // устанавливаем максимум три последовательности
    };
    LlamaExecutor executor = model.CreateExecutor(ctxParams);

    // Создаем три последовательности для пакетного перевода трех текстов
    LLamaSeqId seq1 = await executor.CreateSequence();
    LLamaSeqId seq2 = await executor.CreateSequence();
    LLamaSeqId seq3 = await executor.CreateSequence();

    string startPrefill = *такой же*;

    // Заполняем одну из последовательностей стартовым префилом (любую)
    await executor.ProcessPrompt(seq1, startPrefill, model.Vocab.ShouldAddBOS);

    // Получаем позицию следующего токена для заполненной последовательности
    LLamaPos endPos = await executor.GetSequenceNextDecodedTokenPos(seq1);

    // Разделяем кэш seq1 с seq2 и seq3 до указанной позиции последовательности
    await executor.CopySeqPrefixTo(seq1, [seq2, seq3], endPos);

    Console.WriteLine(startPrefill);

    // Данные трёх потоков генерации для отображения
    var contexts = new ConcurrentDictionary<LLamaSeqId, string>()
    {
        [seq1] = "",
        [seq2] = "",
        [seq3] = ""
    };

    //Три текста для перевода
    List<string> queries = [
        "🔬 模块一：量子计算：超越经典的新范式\n传统计算机以“比特”（0或1）为信息基本单位，而量子计算机利用“量子比特”的叠加态与量子纠缠特性，可在同一时刻探索海量计算路径。近年来，谷歌、IBM与中国科研团队相继实现“量子优越性”，在特定算法任务上显著超越经典超算。尽管量子纠错、相干时间与规模化集成仍是技术瓶颈，但量子计算有望在密码破译、新药分子模拟与高温超导材料设计中实现突破性应用。\n📌 核心提示：量子并行性是算力跃迁的关键，工程化落地仍需跨学科协同攻关。",
        "🧬 模块二：CRISPR-Cas9：精准改写生命密码\nCRISPR-Cas9是一种源自细菌适应性免疫系统的基因编辑工具，够像“分子剪刀”般在目标DNA位点进行精准切割与修复。自2012年技术成熟以来，它已广泛应用于作物抗病育种、遗病机制解析与肿瘤免疫治疗。2023年底，全球首款基于CRISPR的镰状细胞病基因疗法正式获批，标志着基因医学从验室走向临床。当前研究重点在于提升编辑特异性、降低脱靶效应，并探索体内递送系统的安全边界。\n📌 核心示：技术已进入临床转化期，伦理监管与长期安全性评估需同步完善。",
        "🤖 模块五：AI赋能科研：从AlphaFold到科学大模型\n人工智能正推动科学研究范式向“数据驱动+智能推演”转型DeepMind的AlphaFold成功预测超2亿种蛋白质三维结构，将结构生物学研究效率提升数个数量级。如今，面向材料选、气候模拟、催化反应与药物设计的科学大模型可自动解析文献、生成可验证假设并优化实验路径。AI并非替代科家，而是作为“高通量协作者”压缩试错周期，加速跨学科知识融合。\n📌 核心提示：人机协同科研已成常态，模型解释性与科学因果推断是下一阶段重点。"
    ];

    //задания на генерацию с имитацией прихода в разное время через Delay, также содержат еще по одному пользовательскому запросу после завершения перевода
    Task gen1 = GenerateAsync(executor, seq1, queries[0], contexts, 3000);
    Task gen2 = GenerateAsync(executor, seq2, queries[1], contexts, 1500);
    Task gen3 = GenerateAsync(executor, seq3, queries[2], contexts, 0);

    List<Task> tasks = [gen1, gen2, gen3];

    *Таблица из Spectre.Console для вывода генерации трех текстов одновременно*
}

//Метод для генерации двух раундов с задержкой
static async Task GenerateAsync(
    LlamaExecutor executor,
    LLamaSeqId seqId,
    string text, // текст на китайском для перевода
    ConcurrentDictionary<LLamaSeqId, string> contexts,
    int delay)
{
    // Параметры генерации
    InferenceParams inferenceParams = new InferenceParams() {*те же ...*};

    List<string> queries = new List<string>();
    queries.Add(text);
    queries.Add("thanks"); // добавляем второй запрос

    string input = queries[0];

    // чат из двух запросов: перевод и спасибо
    foreach (var query in queries)
    {
        // симуляция прихода запросов в разное время
        await Task.Delay(delay);

        string prefillInput = "\r\n<user> " + query + "</user> \r\n <assistant> <think>";
        contexts[seqId] += prefillInput; // для отображения в таблице

        await executor.ProcessPrompt(seqId, prefillInput, false, true);

        await foreach (string token in (await executor.Generate(seqId, inferenceParams)).Reader.ReadAllAsync())
        {
            contexts[seqId] += token;  // для отображения в таблице
        }
    }

}
```

Работа с последовательностями может происходить раздельно (хотя есть и функции для работы сразу с пачкой последовательностей за раз), в каждую можно отдельно вводить префилл и для каждой отдельно запускать генерацию, при этом **пакет** на обработку моделью собирается на каждом шагу обработки (метод DecodingLoop() LlamaExecutor) **из доступных на текущий момент заданий генерации и заполнения** одного исполнителя (кто-то называет это Continuous Batching).

**Последовательности могут делить между собой ячейки KV-кэша контекста** (обычно начало последовательности для GPT). При использовании CopySeqPrefixTo последовательность, в которую разделяем кэш, полностью очищается и она начинает ссылаться на кусок кэша другой последовательности (это также экономит кэш контекста, если три последовательности по 2000 токенов разделяют 1500 общих, то реально кэша занято только 1500 + 3*500 = 3000 вместо 6000). После разделения кэша последовательности также могут обрабатываться раздельно, удаление одной из последовательностей не очистит общий кэш (кэш очистится, когда на него никто не ссылается).

Подробнее в [документации](./doc/PublicDoc_RU.md).

Полный код примеров в подпроекте **test_program** репозитория в методах SimpleChat и BatchGenerator. Также там есть и будут добавляться другие примеры.

## Документация публичного интерфейса
[Документация](./doc/PublicDoc_RU.md) содержит описание функций основного класса для работы с контекстами моделей LlamaExecutor и другие данные, необходимые для их использования.

В интеграционных тестах подпроекта **IntegrationTest** есть доп. примеры использования и проверки ломки состояния последовательностей при заведомо странных вариантах использования.

## Добавление в проект
Исходные файлы библиотеки без подпроектов с тестами и примерами с поддержкой необходимого перечня версий llama.cpp могут быть скачаны в виде архива из [релизов](https://github.com/stmay4/llama.csharp/releases).

Файлы движка для инициализации библиотеки в коде могут быть скачаны из официальных [релизов](https://github.com/ggml-org/llama.cpp/releases) проекта llama.cpp и указаны в методе инита (подробнее в [документации](./doc/PublicDoc_RU.md)).

```csharp
// Инициализация библиотеки (загрузка движка и привязка функций)
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

Библиотека написана с использованием .NET 8

Зависимости:<br>
PackageReference Include="CommunityToolkit.HighPerformance" Version="8.4.0" с его SpanOwner (возможно заменю на ArrayPool<T>.Shared из System)

## Планы

- Поддержка мультимодальных llm (звук и изображения): функции токенизации мультимодального ввода и заполнения контекста такими токенами
- Добавление функции получения эмбеддингов (возможно, если установлен флаг в true, то получать вместе с логитами - подумать) 
- Поддержка сохранения состояний последовательностей (и контекста в целом?) для возможности выгрузки части последовательностей в память во время работы при нехватке кэша контекста
- Поддержка спекулятивного декодирования (MTP, eagle и тп)
- Поддержка адаптеров lora (пока можно мержить адаптеры в модель) 

## Проекты
Проекты, использующие библиотеку:

---

*(в данный момент репозиторий приватный, происходит подготовка к открытию)*<br>
[**LAIM**](https://github.com/stmay4/LAIM) - локальный сервер для предоставления программам на компьютере доступа к заданному в GUI списку LLM через низкоуровневый stateful api на именованных каналах с функциями для прямой работы с контекстом и последовательностями из данной библиотеки.<br>Проект предлагает готовую библиотеку интеграции Laim.Client для платформы .NET<br>Также в разработке десктоп GUI программа для прямой работы с моделями - LAIMCHAT

---

Для добавления своих проектов в эту графу пишите в Issues или на почту stasmayorov2004@mail.ru или в группе в телеграмме в теме Проекты-обсуждение.

## Спасибо
Структура проекта взята из [LlamaSharp](https://github.com/SciSharp/LLamaSharp).<br>
Для разработки используется документация и исходный код [llama.cpp](https://github.com/ggml-org/llama.cpp), для работы с llm при эксплуатации используются релизные сборки [llama.cpp](https://github.com/ggml-org/llama.cpp).