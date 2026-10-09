using FluentAssertions;
using Llama.csharp.Abstractions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System.Threading.Channels;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    public class TestMtmd
    {
        private static readonly string _baseDllPath = @"D:\DownLoads\llama-b9851-bin-win-vulkan-x64"; // !set your path to the library!

        //QWEN 3
        private static readonly string _qwen3ModelPath = @"D:\LLMmodels\Qwen3-VL-4B-Instruct-UD-Q5_K_XL.gguf"; // !set your vision model path!
        private static readonly string _qwen3mmprojPath = @"D:\LLMmodels\Qwen3-VL-4B-Instruct-mmproj-F16.gguf"; // !set your mmproj path!

        //QWEN 3 ASR
        private static readonly string _qwen3ASRModelPath = @"D:\LLMmodels\Qwen3-ASR-1.7B-Q8_0.gguf"; // !set your vision model path!
        private static readonly string _qwen3ASRmmprojPath = @"D:\LLMmodels\mmproj-Qwen3-ASR-1.7B-bf16.gguf"; // !set your mmproj path!

        //QWEN 3.5
        private static readonly string _qwen35modelPath = @"D:\LLMmodels\Qwen3.5-4B-UD-Q5_K_XL.gguf"; // !set your vision model path!
        private static readonly string _qwen35mmprojPath = @"D:\LLMmodels\qwen3.5-4b-mmproj-F16.gguf"; // !set your mmproj path!

        //GEMMA 4
        private static readonly string _gemma4modelPath = @"D:\LLMmodels\gemma-4-E2B-it-UD-Q5_K_XL.gguf"; // !set your vision model path!
        private static readonly string _gemma4mmprojPath = @"D:\LLMmodels\gemma4-e2b-mmproj-F16.gguf"; // !set your mmproj path!

        //GEMMA 4 Uni
        private static readonly string _gemma4UnimodelPath = @"D:\LLMmodels\gemma-4-12b-it-UD-Q4_K_XL.gguf"; // !set your vision model path!
        private static readonly string _gemma4UnimmprojPath = @"D:\LLMmodels\gemma4-12b-mmproj-F16.gguf"; // !set your mmproj path!

        private static readonly string _сpuBackend = "ggml-cpu-alderlake.dll"; // !set the best CPU backend for your PC here!

        private static readonly string _testImagePath = "./assets/mtmdImageTest.png";
        private static readonly string _testAudioPath = "./assets/mtmdAudioTest(gen).wav";

        
        private static readonly string _testImagePath1 = _testImagePath;
        private static readonly string _testImagePath2 = _testImagePath;
        private static readonly string _testImagePath3 = _testImagePath;

        private readonly ITestOutputHelper _output;
        public TestMtmd(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void LlamaExecutor_ValidCreation()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            MtmdContextParams @params = MtmdContextParams.Default();

            PrintStructFields(@params);
        }


        [Fact]
        public void LlamaExecutor_InValidCreation_FromNotMtmdInit()
        {
            LibraryInitHelper.InitializeCpuOnly(_baseDllPath, _сpuBackend);

            var act = () => MtmdContextParams.Default();

            act.Should().Throw<InvalidOperationException>();
        }

        private void PrintStructFields(MtmdContextParams p)
        {
            _output.WriteLine("=== LlamaMtmdParams fields ===");
            var type = typeof(MtmdContextParams);

            // Все поля (включая приватные, например _use_gpu)
            foreach (var field in type.GetFields(System.Reflection.BindingFlags.Instance |
                                                 System.Reflection.BindingFlags.Public |
                                                 System.Reflection.BindingFlags.NonPublic))
            {
                var value = field.GetValue(p);
                _output.WriteLine($"[Field] {field.Name} = {value}");
            }

            // Все публичные свойства (use_gpu, print_timings, warmup)
            foreach (var prop in type.GetProperties(System.Reflection.BindingFlags.Instance |
                                                    System.Reflection.BindingFlags.Public))
            {
                if (prop.CanRead)
                {
                    var value = prop.GetValue(p);
                    _output.WriteLine($"[Property] {prop.Name} = {value}");
                }
            }
        }

        [Fact]
        public void ToLlamaContextParams_ValidParams_FillsStructCorrectly()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var @params = new MtmdParams
            {
                UseGpu = true,
                PrintTimings = true,
                Threads = 8,
                FlashAttention = LlamaFlashAttentionType.Auto,
                Warmup = true,
                ImageMinTokens = 256,
                ImageMaxTokens = 5120,
                BatchSize = 5120
            };

            // Act
            @params.ToMtmdContextParams(out var result);

            // Assert
            result.use_gpu.Should().BeTrue();
            result.print_timings.Should().BeTrue();
            result.n_threads.Should().Be(8);
            result.flash_attn_type.Should().Be(LlamaFlashAttentionType.Auto);
            result.warmup.Should().BeTrue();
            result.image_min_tokens.Should().Be(256);
            result.image_max_tokens.Should().Be(5120);
            result.batch_max_tokens.Should().Be(5120);

            PrintStructFields(result);
        }

        [Fact]
        public void ToLlamaContextParams_NullParams_UsesDefaults()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var @params = new MtmdParams(); // Все null-значения

            // Act
            @params.ToMtmdContextParams(out var result);

            // Assert
            result.use_gpu.Should().BeFalse(); // Default из LlamaMtmdParams.Default()
            result.print_timings.Should().BeFalse();
            result.warmup.Should().BeTrue();

            PrintStructFields(result);
            // Остальные поля должны быть равны значениям по умолчанию из Default()
        }

        [Fact]
        public void ToLlamaContextParams_PartialParams_UsesDefaultsForNull()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var @params = new MtmdParams
            {
                UseGpu = true,
                Threads = 8
            };

            // Act
            @params.ToMtmdContextParams(out var result);

            // Assert
            result.use_gpu.Should().BeTrue();
            result.n_threads.Should().Be(8);
            result.print_timings.Should().BeFalse(); // Default
            result.warmup.Should().BeTrue(); // Default

            PrintStructFields(result);
        }

        [Fact]
        public void CreateMtmdContext_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);
                ctx.Dispose();
            };

            act.Should().NotThrow();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdContext_EncodeImage_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                //ImageMinTokens = 8,
                //ImageMaxTokens = 8,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                foreach (var emded in result.embeds)
                {
                    _output.WriteLine(emded.Data.ToString());
                }

                _output.WriteLine(result.BOM);
                _output.WriteLine(result.EOM);

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdContext_EncodeAudio_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMinTokens = 8,
                ImageMaxTokens = 8,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeAudioFromWav(_testAudioPath);

                foreach (var emded in result.embeds)
                {
                    _output.WriteLine(emded.Data.ToString());
                }

                _output.WriteLine(result.BOM);
                _output.WriteLine(result.EOM);

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public void MtmdContext_CheckFields()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);
                _output.WriteLine("Support Vision: " + ctx.SupportVision.ToString());
                _output.WriteLine("Support Audio: " + ctx.SupportAudio.ToString());
                _output.WriteLine("AudioSampleRate: " + ctx.AudioSampleRate.ToString());
                _output.WriteLine("NonCasualDecode: " + ctx.NonCasualDecode.ToString());
                _output.WriteLine("MropeDecode: " + ctx.MropeDecode.ToString());

                ctx.Dispose();
            };

            act.Should().NotThrow();

            model.Dispose();
        }

        #region QWEN_3

        [Fact]
        public async Task MtmdImage_Qwen3_StandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|im_start|>system\n you are a helpfull assistant\n<|im_end|>" +
                    "\n<|im_start|>user\n " + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image? \n<|im_end|>\n<|im_start|>assistant\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen3_RandomTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n What displayed on image? " + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen3_MultipleImagesEncode_OnlyEncode_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
                BatchSize = 1024
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var results = await ctx.EncodeImageFromPaths(imagePaths);

                // Assert - verify we got 3 results
                results.Should().HaveCount(3);

                // Verify each result has BOM, EOM and embeds task
                foreach (var result in results)
                {
                    result.BOM.Should().NotBeNullOrEmpty();
                    result.EOM.Should().NotBeNullOrEmpty();
                    result.embeds.Should().NotBeNull();

                    // Wait for embeds to be calculated
                    LlamaEmbedding[] embeds = await result.embeds;
                    embeds.Should().NotBeNull();
                    embeds.Length.Should().BeGreaterThan(0);
                }

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen3_MultipleImagesEncodeAndBatchProcess_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen3ModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var encodeResults = await ctx.EncodeImageFromPaths(imagePaths);

                ContextParams ctxParams = new ContextParams() { ContextSize = 12000, SeqMax = 10 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                // Create 3 separate sequences for batch processing
                LLamaSeqId seq1 = await executor.CreateSequence();
                LLamaSeqId seq2 = await executor.CreateSequence();
                LLamaSeqId seq3 = await executor.CreateSequence();

                List<LLamaSeqId> seqIds = new List<LLamaSeqId> { seq1, seq2, seq3 };
                List<LlamaEmbedding[]> embedsList = new List<LlamaEmbedding[]>();

                // Process first image in seq1
                var result1 = encodeResults[0];
                var result2 = encodeResults[1];
                var result3 = encodeResults[2];

                await executor.ProcessPrompt(seq1, "<system>\n you are a helpfull assistant\n</system>\n<user>\n " + result1.BOM, model.Vocab.ShouldAddBOS);

                // Share the prefix
                await executor.CopySeqPrefixTo(seq1, [seq2, seq3], (LLamaPos)(await executor.GetSequenceNextDecodedTokenPos(seq1)));

                // Collect mtmd embeds
                embedsList.Add(await result1.embeds);
                embedsList.Add(await result2.embeds);
                embedsList.Add(await result3.embeds);

                // Batch process mtmd embeds to specified sequences
                var embedsTasksDict = await executor.ProcessMtmdEmbeds(seqIds, embedsList);

                // Wait for all embed processing tasks to complete (for test I use Task.WhenAll)
                await Task.WhenAll(embedsTasksDict.Values);

                // Register different questions in seqs
                Dictionary<LLamaSeqId, Task> prefilltasks = await executor.ProcessPrompt(
                    [seq1, seq2, seq3],
                    [
                    result1.EOM + "What colors are on the image? </user>\n<assistant>\n",
                    result2.EOM + "What test is on the image? </user>\n<assistant>\n",
                    result3.EOM + "What is on the image? </user>\n<assistant>\n"
                    ]
                );

                //(for test I use Task.WhenAll)
                await Task.WhenAll(prefilltasks.Values);

                // Generate text for each sequence
                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                // Generate for seq1
                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
                string genText1 = "";
                // Generate for seq2
                Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
                string genText2 = "";
                // Generate for seq3
                Channel<string> ch3 = await executor.Generate(seq3, inferenceParams);
                string genText3 = "";

                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText1 += text;
                }
                _output.WriteLine("Seq1 result: " + genText1);

                await foreach (var text in ch2.Reader.ReadAllAsync())
                {
                    genText2 += text;
                }
                _output.WriteLine("Seq2 result: " + genText2);

                await foreach (var text in ch3.Reader.ReadAllAsync())
                {
                    genText3 += text;
                }
                _output.WriteLine("Seq3 result: " + genText3);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        #endregion

        #region QWEN_3_ASR

        [Fact]
        public async Task MtmdAudio_Vulkan_Qwen3ASR_StandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = true,
                Threads = 8,
                ImageMaxTokens = 1000,
                BatchSize = 512
            };

            IModelParams modelParams = new ModelParams(_qwen3ASRModelPath)
            {
                GpuLayerCount = 99
            };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3ASRmmprojPath, model, mtmdParams);

                var result = await ctx.EncodeAudioFromWav(_testAudioPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 4000, NoKqvOffload = false };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|im_start|>system\n you are a helpfull ASR system, write text from audiofiles\n<|im_end|>" +
                    "\n<|im_start|>user\n " + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n<|im_end|>\n<|im_start|>assistant\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdAudio_Qwen3ASR_RandomTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen3ASRModelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen3ASRmmprojPath, model, mtmdParams);

                var result = await ctx.EncodeAudioFromWav(_testAudioPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n" + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\nlanguage English<asr_text>");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        #endregion

        #region QWEN_3.5

        [Fact]
        public async Task MtmdImage_Qwen35_StandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen35modelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen35mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|im_start|>system\n you are a helpfull assistant\n<|im_end|>" +
                    "\n<|im_start|>user\n " + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image? \n<|im_end|>\n<|im_start|>assistant\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen35_RandomTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen35modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen35mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n What displayed on image? " + result.BOM);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen35_MultipleImagesEncode_OnlyEncode_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen35modelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen35mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var results = await ctx.EncodeImageFromPaths(imagePaths);

                // Assert - verify we got 3 results
                results.Should().HaveCount(3);

                // Verify each result has BOM, EOM and embeds task
                foreach (var result in results)
                {
                    result.BOM.Should().NotBeNullOrEmpty();
                    result.EOM.Should().NotBeNullOrEmpty();
                    result.embeds.Should().NotBeNull();

                    // Wait for embeds to be calculated
                    LlamaEmbedding[] embeds = await result.embeds;
                    embeds.Should().NotBeNull();
                    embeds.Length.Should().BeGreaterThan(0);
                }

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Qwen35_MultipleImagesEncodeAndBatchProcess_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_qwen35modelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_qwen35mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var encodeResults = await ctx.EncodeImageFromPaths(imagePaths);

                ContextParams ctxParams = new ContextParams() { ContextSize = 12000, SeqMax = 10 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                // Create 3 separate sequences for batch processing
                LLamaSeqId seq1 = await executor.CreateSequence();
                LLamaSeqId seq2 = await executor.CreateSequence();
                LLamaSeqId seq3 = await executor.CreateSequence();

                List<LLamaSeqId> seqIds = new List<LLamaSeqId> { seq1, seq2, seq3 };
                List<LlamaEmbedding[]> embedsList = new List<LlamaEmbedding[]>();

                // Process first image in seq1
                var result1 = encodeResults[0];
                var result2 = encodeResults[1];
                var result3 = encodeResults[2];

                await executor.ProcessPrompt(seq1, "<system>\n you are a helpfull assistant\n</system>\n<user>\n " + result1.BOM, model.Vocab.ShouldAddBOS);

                // Share the prefix
                await executor.CopySeqPrefixTo(seq1, [seq2, seq3], (LLamaPos)(await executor.GetSequenceNextDecodedTokenPos(seq1)));

                // Collect mtmd embeds
                embedsList.Add(await result1.embeds);
                embedsList.Add(await result2.embeds);
                embedsList.Add(await result3.embeds);

                // Batch process mtmd embeds to specified sequences
                var embedsTasksDict = await executor.ProcessMtmdEmbeds(seqIds, embedsList);

                // Wait for all embed processing tasks to complete (for test I use Task.WhenAll)
                await Task.WhenAll(embedsTasksDict.Values);

                // Register different questions in seqs
                Dictionary<LLamaSeqId, Task> prefilltasks = await executor.ProcessPrompt(
                    [seq1, seq2, seq3], 
                    [
                    result1.EOM + "What colors are on the image? </user>\n<assistant>\n",
                    result2.EOM + "What test is on the image? </user>\n<assistant>\n",
                    result3.EOM + "What is on the image? </user>\n<assistant>\n"
                    ]
                );
                
                //(for test I use Task.WhenAll)
                await Task.WhenAll(prefilltasks.Values);

                // Generate text for each sequence
                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                // Generate for seq1
                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
                string genText1 = "";
                // Generate for seq2
                Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
                string genText2 = "";
                // Generate for seq3
                Channel<string> ch3 = await executor.Generate(seq3, inferenceParams);
                string genText3 = "";

                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText1 += text;
                }
                _output.WriteLine("Seq1 result: " + genText1);

                await foreach (var text in ch2.Reader.ReadAllAsync())
                {
                    genText2 += text;
                }
                _output.WriteLine("Seq2 result: " + genText2);

                await foreach (var text in ch3.Reader.ReadAllAsync())
                {
                    genText3 += text;
                }
                _output.WriteLine("Seq3 result: " + genText3);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        #endregion

        #region GEMMA4

        [Fact]
        public async Task MtmdImage_Gemma4_StandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|turn>system\n you are a helpfull assistant\n<turn|>\n" +
                    "<|turn>user\n " + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image? \n<turn|>\n<|turn>model\n<|channel>thought");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdAudio_Gemma4_SemistandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeAudioFromWav(_testAudioPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|turn>system\n you are a helpfull VLM assistant with vision and audio capabilities.\n<turn|>\n" +
                    "<|turn>user\n Listen to the audio, identify the speaker’s voice gender and text from audio, answer strictly in JSON.\n" + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n<turn|>\n<|turn>model\n[");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 1000,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    SamplingPipeline = new TunableSamplerPipeline(new TunableSamplerPipelineSettings([], new GreedySampler())),
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4_RandomTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n What displayed on image? " + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4_MultipleImagesEncode_OnlyEncode_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var results = await ctx.EncodeImageFromPaths(imagePaths);

                // Assert - verify we got 3 results
                results.Should().HaveCount(3);

                // Verify each result has BOM, EOM and embeds task
                foreach (var result in results)
                {
                    result.BOM.Should().NotBeNullOrEmpty();
                    result.EOM.Should().NotBeNullOrEmpty();
                    result.embeds.Should().NotBeNull();

                    // Wait for embeds to be calculated
                    LlamaEmbedding[] embeds = await result.embeds;
                    embeds.Should().NotBeNull();
                    embeds.Length.Should().BeGreaterThan(0);
                }

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4_MultipleImagesEncodeAndBatchProcess_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4mmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var encodeResults = await ctx.EncodeImageFromPaths(imagePaths);

                ContextParams ctxParams = new ContextParams() { ContextSize = 12000, SeqMax = 10 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                // Create 3 separate sequences for batch processing
                LLamaSeqId seq1 = await executor.CreateSequence();
                LLamaSeqId seq2 = await executor.CreateSequence();
                LLamaSeqId seq3 = await executor.CreateSequence();

                List<LLamaSeqId> seqIds = new List<LLamaSeqId> { seq1, seq2, seq3 };
                List<LlamaEmbedding[]> embedsList = new List<LlamaEmbedding[]>();

                // Process first image in seq1
                var result1 = encodeResults[0];
                var result2 = encodeResults[1];
                var result3 = encodeResults[2];

                await executor.ProcessPrompt(seq1, "<system>\n you are a helpfull assistant\n</system>\n<user>\n " + result1.BOM, model.Vocab.ShouldAddBOS);

                // Share the prefix
                await executor.CopySeqPrefixTo(seq1, [seq2, seq3], (LLamaPos)(await executor.GetSequenceNextDecodedTokenPos(seq1)));

                // Collect mtmd embeds
                embedsList.Add(await result1.embeds);
                embedsList.Add(await result2.embeds);
                embedsList.Add(await result3.embeds);

                // Batch process mtmd embeds to specified sequences
                var embedsTasksDict = await executor.ProcessMtmdEmbeds(seqIds, embedsList);

                // Wait for all embed processing tasks to complete (for test I use Task.WhenAll)
                await Task.WhenAll(embedsTasksDict.Values);

                // Register different questions in seqs
                Dictionary<LLamaSeqId, Task> prefilltasks = await executor.ProcessPrompt(
                    [seq1, seq2, seq3],
                    [
                    result1.EOM + "What colors are on the image? </user>\n<assistant>\n",
                    result2.EOM + "What test is on the image? </user>\n<assistant>\n",
                    result3.EOM + "What is on the image? </user>\n<assistant>\n"
                    ]
                );

                //(for test I use Task.WhenAll)
                await Task.WhenAll(prefilltasks.Values);

                // Generate text for each sequence
                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                // Generate for seq1
                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
                string genText1 = "";
                // Generate for seq2
                Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
                string genText2 = "";
                // Generate for seq3
                Channel<string> ch3 = await executor.Generate(seq3, inferenceParams);
                string genText3 = "";

                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText1 += text;
                }
                _output.WriteLine("Seq1 result: " + genText1);

                await foreach (var text in ch2.Reader.ReadAllAsync())
                {
                    genText2 += text;
                }
                _output.WriteLine("Seq2 result: " + genText2);

                await foreach (var text in ch3.Reader.ReadAllAsync())
                {
                    genText3 += text;
                }
                _output.WriteLine("Seq3 result: " + genText3);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        #endregion

        #region GEMMA4_Uni

        [Fact]
        public async Task MtmdImage_Gemma4Uni_StandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4UnimodelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4UnimmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|turn>system\n you are a helpfull assistant\n<turn|>\n" +
                    "<|turn>user\n " + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "What displayed on image? \n<turn|>\n<|turn>model\n<|channel>thought");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdAudio_Gemma4Uni_SemistandartTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
            };

            IModelParams modelParams = new ModelParams(_gemma4UnimodelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4UnimmprojPath, model, mtmdParams);

                var result = await ctx.EncodeAudioFromWav(_testAudioPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|turn>system\n you are a helpfull VLM assistant with vision and audio capabilities. Vision in <vision></vision> tags and audio in <audio></audio> tags, respectively. \n<turn|>\n" +
                    "<|turn>user\n Listen to the audio, identify the speaker’s voice, and describe it.\n<audio>" + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "</audio>\n<turn|>\n<|turn>model\n<|channel>thought\nThinking");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 1000,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = []
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4Uni_RandomTemplate_DecodeByLLM_Valid()
        {
            LibraryInitHelper.InitializeCpuAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4UnimodelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4UnimmprojPath, model, mtmdParams);

                var result = await ctx.EncodeImageFromPath(_testImagePath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n What displayed on image? " + result.BOM, model.Vocab.ShouldAddBOS);
                await executor.ProcessMtmdEmbeds(seq1, result.embeds);
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\n");

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);

                string genText = "";
                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText += text;
                }

                _output.WriteLine(genText);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4Uni_MultipleImagesEncode_OnlyEncode_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4UnimodelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4UnimmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var results = await ctx.EncodeImageFromPaths(imagePaths);

                // Assert - verify we got 3 results
                results.Should().HaveCount(3);

                // Verify each result has BOM, EOM and embeds task
                foreach (var result in results)
                {
                    result.BOM.Should().NotBeNullOrEmpty();
                    result.EOM.Should().NotBeNullOrEmpty();
                    result.embeds.Should().NotBeNull();

                    // Wait for embeds to be calculated
                    LlamaEmbedding[] embeds = await result.embeds;
                    embeds.Should().NotBeNull();
                    embeds.Length.Should().BeGreaterThan(0);
                }

                ctx.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }

        [Fact]
        public async Task MtmdImage_Gemma4Uni_MultipleImagesEncodeAndBatchProcess_Valid()
        {
            LibraryInitHelper.InitializeCpuVulkanAndMtmd(_baseDllPath, _сpuBackend);

            // Arrange
            var mtmdParams = new MtmdParams
            {
                UseGpu = false,
                Threads = 8,
                ImageMaxTokens = 1000,
            };

            IModelParams modelParams = new ModelParams(_gemma4UnimodelPath) { GpuLayerCount = 0 };
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);

            // Placeholder image paths - to be replaced with actual image paths
            List<string> imagePaths = new List<string>
            {
                _testImagePath1,
                _testImagePath2,
                _testImagePath3
            };

            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_gemma4UnimmprojPath, model, mtmdParams);

                // Encode 3 images in a single request
                var encodeResults = await ctx.EncodeImageFromPaths(imagePaths);

                ContextParams ctxParams = new ContextParams() { ContextSize = 12000, SeqMax = 10 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                // Create 3 separate sequences for batch processing
                LLamaSeqId seq1 = await executor.CreateSequence();
                LLamaSeqId seq2 = await executor.CreateSequence();
                LLamaSeqId seq3 = await executor.CreateSequence();

                List<LLamaSeqId> seqIds = new List<LLamaSeqId> { seq1, seq2, seq3 };
                List<LlamaEmbedding[]> embedsList = new List<LlamaEmbedding[]>();

                // Process first image in seq1
                var result1 = encodeResults[0];
                var result2 = encodeResults[1];
                var result3 = encodeResults[2];

                await executor.ProcessPrompt(seq1, "<system>\n you are a helpfull assistant\n</system>\n<user>\n " + result1.BOM, model.Vocab.ShouldAddBOS);

                // Share the prefix
                await executor.CopySeqPrefixTo(seq1, [seq2, seq3], (LLamaPos)(await executor.GetSequenceNextDecodedTokenPos(seq1)));

                // Collect mtmd embeds
                embedsList.Add(await result1.embeds);
                embedsList.Add(await result2.embeds);
                embedsList.Add(await result3.embeds);

                // Batch process mtmd embeds to specified sequences
                var embedsTasksDict = await executor.ProcessMtmdEmbeds(seqIds, embedsList);

                // Wait for all embed processing tasks to complete (for test I use Task.WhenAll)
                await Task.WhenAll(embedsTasksDict.Values);

                // Register different questions in seqs
                Dictionary<LLamaSeqId, Task> prefilltasks = await executor.ProcessPrompt(
                    [seq1, seq2, seq3],
                    [
                    result1.EOM + "What colors are on the image? </user>\n<assistant>\n",
                    result2.EOM + "What test is on the image? </user>\n<assistant>\n",
                    result3.EOM + "What is on the image? </user>\n<assistant>\n"
                    ]
                );

                //(for test I use Task.WhenAll)
                await Task.WhenAll(prefilltasks.Values);

                // Generate text for each sequence
                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 200,
                    AutoStopFromEOG = true,
                    DecodeSpecialTokens = true,
                    AntiPrompts = ["</assistant>"]
                };

                // Generate for seq1
                Channel<string> ch1 = await executor.Generate(seq1, inferenceParams);
                string genText1 = "";
                // Generate for seq2
                Channel<string> ch2 = await executor.Generate(seq2, inferenceParams);
                string genText2 = "";
                // Generate for seq3
                Channel<string> ch3 = await executor.Generate(seq3, inferenceParams);
                string genText3 = "";

                await foreach (var text in ch1.Reader.ReadAllAsync())
                {
                    genText1 += text;
                }
                _output.WriteLine("Seq1 result: " + genText1);

                await foreach (var text in ch2.Reader.ReadAllAsync())
                {
                    genText2 += text;
                }
                _output.WriteLine("Seq2 result: " + genText2);

                await foreach (var text in ch3.Reader.ReadAllAsync())
                {
                    genText3 += text;
                }
                _output.WriteLine("Seq3 result: " + genText3);

                ctx.Dispose();
                executor.Dispose();
            };

            await act.Should().NotThrowAsync();

            model.Dispose();
        }
        #endregion
    }
}
