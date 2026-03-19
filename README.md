# llama.csharp

> ⚠️ **Статус проекта**: Библиотека находится в активной разработке.

C# библиотека-обертка над нативной библиотекой [`llama.cpp`](https://github.com/ggml-org/llama.cpp), основанная на проекте [LlamaSharp](https://github.com/SciSharp/LlamaSharp). Предназначена для интеграции больших языковых моделей (LLM) в приложения на .NET с минимальными накладными расходами и полным контролем над нативными зависимостями.

---

## 🔑 Ключевые особенности

- **Использование llama.cpp через NativeLibrary**:<br>
Прямой проброс нативных функций через `System.Runtime.InteropServices.NativeLibrary`. Пути к нативным библиотекам указываются явно при старте — полный контроль над версиями и бэкендами для разработчика.
- **Изменена логика Executor и SamplerPipeline** (относительно [LlamaSharp](https://github.com/SciSharp/LlamaSharp)): Описание TODO
- **Изменена логика AntipromptProcessor**: теперь он проверяет не конец строки, а производит поиск в конечной части строки.
- (планируется)**Автоматическое обновление**: автообновление обертки для использования актуальных версий [`llama.cpp`](https://github.com/ggml-org/llama.cpp).
---

## 🛠 Требования

| Компонент | Версия / Примечание |
|-----------|---------------------|
| **.NET SDK** | 8.0 или выше |
| **llama.cpp** | Протестировано с `b7552`, совместимость ожидается в диапазоне `b7552` – `b7628` |
| **ОС** | Windows (`.dll`), Linux (`.so`) |
| **Аппаратное обеспечение** | Обязательно: CPU-бэкенд. Опционально: Vulkan, CUDA и др. |

🔗 Ожидаемая совместимость с llama.cpp основана на таблице изменений API `llama.cpp`: [issue #9289](https://github.com/ggml-org/llama.cpp/issues/9289)

---

## 📦 Установка и настройка

### 1. Подключение исходного кода
Подключите библиотеку как подмодуль или скопируйте исходники в ваш проект.

### 2. Подготовка нативных библиотек
Скачайте необходимые бинарные файлы из [официальных релизов llama.cpp](https://github.com/ggml-org/llama.cpp/releases):

**Обязательные файлы:**
- `ggml.dll` / `libggml.so`
- `ggml-base.dll` / `libggml-base.so`

**Бэкенды (выберите под ваше железо):**
- `ggml-cpu-alderlake.dll` / `libggml-cpu-alderlake.so` *(обязательно хотя бы один CPU-бэкенд)*
- `ggml-vulkan.dll` / `libggml-vulkan.so` *(опционально, для GPU через Vulkan)*
- `ggml-cuda.dll` / `libggml-cuda.so` *(опционально, для NVIDIA)*
- и другие...

>  Можно подключать несколько бэкендов одновременно.

Класс `Native/LlamaCpp.cs` реализует загрузку через `NativeLibrary.Load`.

### 3. Инициализация в коде
Укажите пути к нативным библиотекам в коде перед использованием библиотеки:
```csharp
LlamaCpp.Initialize("./llama/llama.dll", "./llama/ggml.dll", "./llama/ggml-base.dll",
    ["./llama/ggml-cpu-alderlake.dll",
     "./llama/ggml-vulkan.dll"]);
```
### 4. Пример использования в новом проекте
(тестовый - переделать)
```csharp
using Llama.csharp.Native;
using Llama.csharp;
using System.Diagnostics;

class Program
{
    static async Task Main()
    {
        string modelPath = @"D:\LLMmodels\gemma-3-4b-it-Q4_K_M.gguf"; //пути будут отличаться у Вас

        LlamaCpp.Initialize("./llama/llama.dll", "./llama/ggml.dll", "./llama/ggml-base.dll",
            ["./llama/ggml-cpu-alderlake.dll",
             "./llama/ggml-vulkan.dll"]); //пути будут отличаться у Вас

        ModelParams parametres = new ModelParams(modelPath)
        {
            GpuLayerCount = 0,
        };

        LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

        ContextParams ctxParams = new ContextParams()
        {
            ContextSize = 16000,
            NoKqvOffload = true,
            BatchSize = 1024,
            UBatchSize = 1024
            //FlashAttention = LlamaFlashAttentionType.Disabled
        };

        LLamaContext context = new LLamaContext(model, ctxParams);

        LlamaExecutor executor = new LlamaExecutor(context);

        InferenceParams inferenceParams = new InferenceParams()
        {
            AutoStopFromEOG = true,
            DecodeSpecialTokens = true,
            AntiPrompts = ["<end_of_turn>", "Пользователь:", "User:"]
        };

        string chat = @"<|system|>
Ты — полезный и вежливый ассистент. Отвечай кратко, по делу, на русском языке. Если не знаешь ответа — так и скажи. Избегай лишних рассуждений.
<end_of_turn>

User: Привет! Как тебя зовут?
assistant: Привет! Я — локальная языковая модель, работающая через llama.csharp. Чем могу помочь?
<end_of_turn>

User: Сколько будет 2 + 2?
assistant: 2 + 2 = 4.
<end_of_turn>

User:"; // заполните контекст, например системное сообщение + пару вопросов ответов

        await executor.ProcessPrompt(chat, null, model.Vocab.ShouldAddBOS); // некоторые llm просят добавлять токен BOS в самом начале.
//нужен ли он моно узнать через model.Vocab.ShouldAddBOS

        string userInput = Console.ReadLine() ?? "";

        while (userInput != "exit")
        {
            await executor.ProcessPrompt("\nUser:" + userInput + "\n" + "\nassistant:");

            var watcher = Stopwatch.StartNew();
            await foreach (var text in executor.Generate(inferenceParams))
            {
                Console.Write(text);
            }

            watcher.Stop();
            Console.WriteLine(watcher.ElapsedMilliseconds + " ms");

            userInput = Console.ReadLine() ?? "";
        }

        context.Dispose();

        model.Dispose();
    }
}
```

---

## 🗺 Планы развития (Roadmap)

| Приоритет | Задача | Статус |
|-----------|--------|--------|
| 🔴 Высокий | Контроль над KV-кэшем контекста: удаление с конца и из произвольной позиции (для RoPE-моделей) | 🟡 В работе |
| 🔴 Высокий | Исследование и реализация батчевого декодирования | ⚪ Запланировано |
| 🟡 Средний | Поддержка мультимодальности: обработка аудио и изображений | ⚪ Запланировано |
| 🟡 Средний | Наполнение кода внутренней документацией: контракты, XML-комментарии, описания классов | ⚪ Запланировано |
| 🟢 Низкий | Автоматизация обновления обертки под новые версии `llama.cpp` через AI-агент | 💡 Идея |

---

## ⚠️ Ограничения и заметки

- **Совместимость версий**: Библиотека разрабатывалась под `llama.cpp@b7552`. Использование Других версий не тестировалось.
- **Платформенная зависимость**: Пути и имена нативных библиотек зависят от ОС. Убедитесь, что передаете корректные пути для вашей платформы.
---


*Разработано на C# / .NET 8.0. Основано на [llama.cpp](https://github.com/ggml-org/llama.cpp) и проекте [LlamaSharp](https://github.com/SciSharp/LlamaSharp).*
