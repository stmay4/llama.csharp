using Llama.csharp;
using Llama.csharp.Abstractions;
using Llama.csharp.Native;
using Spectre.Console;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Text;
using System.Threading;
using System.Threading.Channels;

class Program
{
    private static readonly string _baseDllPath = @"D:\DownLoads\llama-b9756-bin-win-vulkan-x64"; // set the path to the file folder
    private static readonly string _modelPath = @"D:\LLMmodels\qwen35\Qwen3.5-9B-UD-Q5_K_XL.gguf";
    private static readonly string _сpuBackend = @"ggml-cpu-alderlake.dll"; // set a hardware-supported backend. more inf in docs
    static async Task Main()
    {
        // set for emoji
        Console.OutputEncoding = Encoding.UTF8;
        Console.InputEncoding = Encoding.UTF8;

        #region lib init
        var requiredFiles = new[]
        {
            Path.Combine(_baseDllPath, "llama.dll"), // on Linux, change to .so
            Path.Combine(_baseDllPath, "ggml.dll"),
            Path.Combine(_baseDllPath, "ggml-base.dll"),
            Path.Combine(_baseDllPath, _сpuBackend),
            Path.Combine(_baseDllPath, "ggml-vulkan.dll")
        };
        try
        {
            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3],
                                //requiredFiles[4]
                                ]);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(ex.ToString());
            Console.Error.WriteLine(ex.StackTrace);
            return;
        }
        #endregion

        Console.WriteLine("0 - simple chat \n1 - batch generation \n2 - perf test \n");
        int i = Convert.ToInt32(Console.ReadLine());
        switch (i)
        {
            case 0: await SimpleChat(); break;
            case 1: await BatchGenerator(); break;
            //case 2: await PerfTest(); break;
            default: break;
        }

    }

    private static async Task SimpleChat()
    {
        // Load model
        ModelParams parametres = new ModelParams(_modelPath)
        {
            //GpuLayerCount = 999, // Set this when using GPU; adjust according to available VRAM.
            //                     // Reduce if necessary. Layer count can be read from model.Metadata.
            //TensorBufferOverrides = [new TensorBufferOverride("blk\\.[0-35].*exps.*", "CPU")], // Use something like this for GPU offloading (with GpuLayerCount = 999)
            //                                                                                   // when loading an MoE model. This example offloads experts of layers 0–35 to CPU.
            //                                                                                   // More details in the docs under "Model Loading".
            //TensorBufferOverrides = [new TensorBufferOverride(".*exps.*", "CPU")], // Similar to the above, but offloads *all* expert layers to CPU.
        };
        LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

        // Create an executor for working with context
        ContextParams ctxParams = new ContextParams()
        {
            ContextSize = 16000,
            //SeqMax = 1, // number of sequences available to create, default is already one
            //NoKqvOffload = false // If using GPU and enough VRAM, you can offload the context to GPU
        };
        LlamaExecutor executor = model.CreateExecutor(ctxParams);

        // Create a sequence (one is enough for a simple chat)
        LLamaSeqId mainSeq = await executor.CreateSequence();

        // The message that will be loaded into the sequence context. Contains arbitrarily placed role tags
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
        // you can add the model's EOS token after '</assistant>', obtained from model.Vocab.LLamaTokenToString(model.Vocab.EOS, true) if needed

        // Fill the sequence context. No tokens are added to the input internally, only BOS if set in the third argument (default false)
        await executor.ProcessPrompt(mainSeq, startPrefill, model.Vocab.ShouldAddBOS);

        Console.WriteLine(startPrefill);

        // Inference parameters
        InferenceParams inferenceParams = new InferenceParams()
        {
            MaxTokens = -1,
            AutoStopFromEOG = true,
            DecodeSpecialTokens = true,
            AntiPrompts = ["</assistant>"],
            SamplingPipeline = new TunableSamplerPipeline(
                new TunableSamplerPipelineSettings(
                    [
                        // List of samplers in order of application to logits
                        new TopKSampler() { K=30 }
                    ],
                    // Finalizing sampler: Greedy, Distribution or Mirostat2
                    new Mirostat2Sampler() { Seed = 256 }
                )
            )
        };

        string input = "";

        Console.WriteLine("Press Enter, then Ctrl+Z to send a message. Press only Ctrl+Z to exit.");

        // Chat loop
        while (true)
        {
            Console.WriteLine();
            Console.Write("Me: ");// Not sent to the model, visual only

            // Press Ctrl+Z on a new line to send input
            input = Console.In.ReadToEnd();

            if (string.IsNullOrEmpty(input)) break; // End chat if input is empty

            // Text sent to the model
            string prefillInput = "\r\n<user> " + input + " </user>\r\n<assistant>\r\n<think>";

            // Visual indicator, not sent to the model
            Console.Write("\r\nNot me):\r\n<think>");

            // Prefill the current input, appending it to the specified context sequence
            await executor.ProcessPrompt(mainSeq, prefillInput);

            // Get the generation data stream
            Channel<string> ch = await executor.Generate(mainSeq, inferenceParams);

            await foreach (string token in ch.Reader.ReadAllAsync())
            {
                Console.Write(token); // Print tokens converted to strings
            }
        }

        executor.Dispose();
        model.Dispose();
    }

    private static async Task BatchGenerator()
    {
        // Load model
        ModelParams parametres = new ModelParams(_modelPath)
        {
            //GpuLayerCount = 999, // Set this when using GPU; adjust according to available VRAM.
            //                     // Reduce if necessary. Layer count can be read from model.Metadata.
            //TensorBufferOverrides = [new TensorBufferOverride("blk\\.[0-35].*exps.*", "CPU")], // Use something like this for GPU offloading (with GpuLayerCount = 999)
            //                                                                                   // when loading an MoE model. This example offloads experts of layers 0–35 to CPU.
            //                                                                                   // More details in the docs under "Model Loading".
            //TensorBufferOverrides = [new TensorBufferOverride(".*exps.*", "CPU")], // Similar to the above, but offloads *all* expert layers to CPU.
        };
        LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

        // Create LlamaExecutor to work with a context
        ContextParams ctxParams = new ContextParams()
        {
            ContextSize = 16000,
            SeqMax = 3, // set maximum three sequences
            //NoKqvOffload = false // If using GPU and have enough VRAM, load context onto GPU.
        };
        LlamaExecutor executor = model.CreateExecutor(ctxParams);

        // Create three sequences for batch translation of three texts
        LLamaSeqId seq1 = await executor.CreateSequence();
        LLamaSeqId seq2 = await executor.CreateSequence();
        LLamaSeqId seq3 = await executor.CreateSequence();

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

        // Fill one of the sequences with the starting prefix (any one)
        await executor.ProcessPrompt(seq1, startPrefill, model.Vocab.ShouldAddBOS);

        // Get the position of the next token for the filled sequence
        LLamaPos endPos = await executor.GetSequenceNextDecodedTokenPos(seq1);

        // Share the cache of seq1 with seq2 and seq3 up to the specified sequence position
        await executor.CopySeqPrefixTo(seq1, [seq2, seq3], endPos);

        Console.WriteLine(startPrefill);

        // Data of the three generation streams for display
        var contexts = new ConcurrentDictionary<LLamaSeqId, string>()
        {
            [seq1] = "",
            [seq2] = "",
            [seq3] = ""
        };

        // Three texts for translation
        List<string> queries = [
            "🔬 模块一：量子计算：超越经典的新范式\n传统计算机以“比特”（0或1）为信息基本单位，而量子计算机利用“量子比特”的叠加态与量子纠缠特性，可在同一时刻探索海量计算路径。近年来，谷歌、IBM与中国科研团队相继实现“量子优越性”，在特定算法任务上显著超越经典超算。尽管量子纠错、相干时间与规模化集成仍是技术瓶颈，但量子计算有望在密码破译、新药分子模拟与高温超导材料设计中实现突破性应用。\n📌 核心提示：量子并行性是算力跃迁的关键，工程化落地仍需跨学科协同攻关。",
                "🧬 模块二：CRISPR-Cas9：精准改写生命密码\nCRISPR-Cas9是一种源自细菌适应性免疫系统的基因编辑工具，能够像“分子剪刀”般在目标DNA位点进行精准切割与修复。自2012年技术成熟以来，它已广泛应用于作物抗病育种、遗传病机制解析与肿瘤免疫治疗。2023年底，全球首款基于CRISPR的镰状细胞病基因疗法正式获批，标志着基因医学从实验室走向临床。当前研究重点在于提升编辑特异性、降低脱靶效应，并探索体内递送系统的安全边界。\n📌 核心提示：技术已进入临床转化期，伦理监管与长期安全性评估需同步完善。",
                "🤖 模块五：AI赋能科研：从AlphaFold到科学大模型\n人工智能正推动科学研究范式向“数据驱动+智能推演”转型。DeepMind的AlphaFold成功预测超2亿种蛋白质三维结构，将结构生物学研究效率提升数个数量级。如今，面向材料筛选、气候模拟、催化反应与药物设计的科学大模型可自动解析文献、生成可验证假设并优化实验路径。AI并非替代科学家，而是作为“高通量协作者”压缩试错周期，加速跨学科知识融合。\n📌 核心提示：人机协同科研已成常态，模型可解释性与科学因果推断是下一阶段重点。"
        ];

        // generation tasks simulating arrival at different times via Delay, also containing one additional user query each after translation completes
        Task gen1 = GenerateAsync(executor, seq1, queries[0], contexts, 6000);
        Task gen2 = GenerateAsync(executor, seq2, queries[1], contexts, 3000);
        Task gen3 = GenerateAsync(executor, seq3, queries[2], contexts, 0);

        List<Task> tasks = [gen1, gen2, gen3];

        // Таблица вывода генерации
        await AnsiConsole.Live(new Table())
            .StartAsync(async ctx =>
            {
                while (tasks.Any(t => !t.IsCompleted)) //хоть одна не завершена
                {
                    // Создаём таблицу заново на каждом кадре – это быстро и не мерцает
                    var table = new Table()
                        .Border(TableBorder.Rounded)
                        .ShowRowSeparators()
                        .AddColumn("Последовательность")
                        .AddColumn("Сгенерированный текст");

                    // Заполняем строки (потокобезопасное чтение)
                    foreach (var (model, text) in contexts)
                    {
                        table.AddRow(
                            new Markup($"[bold]{model}[/]"),
                            new Markup(text)
                        );
                    }

                    // Обновляем отображение
                    ctx.UpdateTarget(table);

                    await Task.Delay(50); // частота обновления UI
                }
            });

        executor.Dispose();
        model.Dispose();
    }

    // Method for generating two rounds with a delay
    static async Task GenerateAsync(
        LlamaExecutor executor,
        LLamaSeqId seqId,
        string text,
        ConcurrentDictionary<LLamaSeqId, string> contexts,
        int delay)
    {
        // Inference parameters
        InferenceParams inferenceParams = new InferenceParams()
        {
            MaxTokens = -1,
            AutoStopFromEOG = true,
            DecodeSpecialTokens = true,
            AntiPrompts = ["</assistant>"],
            SamplingPipeline = new TunableSamplerPipeline(
                new TunableSamplerPipelineSettings(
                    [
                        // Samplers in the order they are applied
                        new TopKSampler() { K=30 }
                    ],
                    // Finalizing sampler: Greedy, Distribution, or Mirostat2
                    new Mirostat2Sampler() { Seed = 256 }
                )
            )
        };

        List<string> queries = new List<string>();
        queries.Add(text);
        queries.Add("thanks"); // add a second query

        string input = queries[0];

        // chat of two queries: translation and thanks
        foreach (var query in queries)
        {
            // simulate queries arriving at different times
            await Task.Delay(delay);

            string prefillInput = "\r\n<user> " + query + "</user> \r\n <assistant> <think>";
            contexts[seqId] += prefillInput; // for display in the table

            await executor.ProcessPrompt(seqId, prefillInput, false, true);

            await foreach (string token in (await executor.Generate(seqId, inferenceParams)).Reader.ReadAllAsync())
            {
                contexts[seqId] += token; // for display in the table
            }
        }

    }
}