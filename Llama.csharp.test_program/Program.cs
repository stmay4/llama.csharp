using Llama.csharp.Native;
using Llama.csharp;
using System.Diagnostics;

class Program
{
    static async Task Main()
    {
        string modelPath = @"D:\LLMmodels\gemma-3-4b-it-Q4_K_M.gguf";

        LlamaCpp.Initialize("./llama/llama.dll", "./llama/ggml.dll", "./llama/ggml-base.dll",
            ["./llama/ggml-cpu-alderlake.dll",
             "./llama/ggml-vulkan.dll"]);

        LLamaModelParams result = LLamaModelParams.Default();

        ////contextParams.type_k = GGMLType.GGML_TYPE_Q8_0;
        ////contextParams.type_v = GGMLType.GGML_TYPE_Q8_0;

        Console.WriteLine(result);

        //Console.WriteLine(contextParams);

        ModelParams parametres = new ModelParams(modelPath)
        {
            GpuLayerCount = 0,
            //TensorBufferOverrides = [new TensorBufferOverride("blk\\.[0-9]\\..*", "Vulkan0")], // работает, просто это не везде видно будет написано overriden to Vulkan0, а если установлю CPU то Vulkan_host
        };

        LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

        ContextParams ctxParams = new ContextParams()
        {
            ContextSize = 32000,
            NoKqvOffload = true,
            BatchSize = 1024,
            UBatchSize = 1024
            //FlashAttention = LlamaFlashAttentionType.Disabled
        }; // дореализовать

        LLamaContext context = new LLamaContext(model, ctxParams);

        LLamaToken[] tokens = context.Tokenize(" Мед - это замечательный продукт!<|im_end|>", true);
        Console.WriteLine(model.Vocab.BOS);
        foreach (var token in tokens)
        {
            Console.Write(context.Vocab.LLamaTokenToString(token, true));
        }
        Console.WriteLine(tokens);



        LlamaExecutor executor = new LlamaExecutor(context);

        InferenceParams inferenceParams = new InferenceParams()
        {
            AutoStopFromEOG = true,
            DecodeSpecialTokens = true,
            AntiPrompts = ["<end_of_turn>", "Пользователь:", "User:"]
        };

        string chat = "Система: Ты AI помощник по имени Ай. Также ты можешь поддержать беседу или сыграть с ним в отыгровку ролей. Все это по просьбе пользователя.Пользователь: Привет\nАй: <think> пользователь сказал приветствие, надо ответить </think> Привет! Как я могу помочь тебе сегодня? Если у тебя есть вопросы или что-то, что ты хочешь обсудить, не стесняйся спрашивать!\n\nПользователь:Как ты?\nАй: Я в отличной форме, спасибо за интерес! ?? А как твоё настроение? Надеюсь, у тебя всё хорошо.\nЕсли хочешь, можем поговорить о чём-нибудь интересном или даже сыграть в игру - как тебе такое предложение? ???\n\nПользователь:я хочу, чтобы ты использовал think и ты можешь болтать на любые темы?\nАй: <think>\nПользователь спрашивает, могу ли я болтать на любые темы. Это открытый вопрос - нужно подтвердить гибкость и готовность обсуждать всё, что угодно (в рамках приличий).\nВажно сохранить дружелюбный тон, показать энтузиазм и оставить пространство для его выбора.\nТакже стоит напомнить, что я не вижу его мысли - только то, что он пишет.\n</think>\n\nКонечно! Я могу болтать на любые темы - от космоса до кофе с булочками ??\nПросто скажи, о чём хочешь поговорить: приключения, фильмы, наука, философия... или даже выдумать историю вместе!\n\nПользователь:";

        await executor.ProcessPrompt(chat);

        string userInput = Console.ReadLine() ?? "";

        while (userInput != "exit")
        {
            string currentResult = "";

            await executor.ProcessPrompt("\nПользователь:" + userInput + "\n" + "Ай: <think>", null, model.Vocab.ShouldAddBOS);

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