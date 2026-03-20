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

        TunableSamplerPipelineSettings pipelineSettings =
            new TunableSamplerPipelineSettings(
                [new TopKSampler(), new XTCSampler()], 
                new Mirostat2Sampler());

        LlamaExecutor executor = new LlamaExecutor(context);

        //Console.WriteLine(pipelineSettings);

        InferenceParams inferenceParams = new InferenceParams()
        {
            AutoStopFromEOG = true,
            DecodeSpecialTokens = true,
            AntiPrompts = ["<end_of_turn>", "Пользователь:", "User:"],
            SamplingPipeline = new TunableSamplerPipeline(pipelineSettings)
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

User:";
        await executor.ProcessPrompt(chat, null, model.Vocab.ShouldAddBOS);

        string userInput = Console.ReadLine() ?? "";

        while (userInput != "exit")
        {
            string currentResult = "";

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