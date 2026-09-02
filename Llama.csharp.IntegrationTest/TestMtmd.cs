using FluentAssertions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using Newtonsoft.Json.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Encodings;
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

        private readonly ITestOutputHelper _output;
        public TestMtmd(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public void LlamaExecutor_ValidCreation()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

            MtmdContextParams @params = MtmdContextParams.Default();

            PrintStructFields(@params);
        }


        [Fact]
        public void LlamaExecutor_InValidCreation_FromNotMtmdInit()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]]
                                );
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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

        #endregion

        #region QWEN_3_ASR

        [Fact]
        public async Task MtmdAudio_Vulkan_Qwen3ASR_StandartTemplate_DecodeByLLM_Valid()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "ggml-vulkan.dll"),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3], requiredFiles[4]],
                                requiredFiles[5]);
            #endregion

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

                ContextParams ctxParams = new ContextParams() { ContextSize = 4000, NoKqvOffload = false};

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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

        #endregion

        #region GEMMA4

        [Fact]
        public async Task MtmdImage_Gemma4_StandartTemplate_DecodeByLLM_Valid()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
        public async Task MtmdImage_Gemma4_RandomTemplate_DecodeByLLM_Valid()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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

        #endregion

        #region GEMMA4_Uni

        [Fact]
        public async Task MtmdImage_Gemma4Uni_StandartTemplate_DecodeByLLM_Valid()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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
                    "<|turn>user\n Listen to the audio, identify the speaker’s voice, and describe it.\n<audio>" + result.BOM, model.Vocab.ShouldAddBOS); //не могут описывать голос (угадывают не всегда), извлекают только текст, и то хуже qwen3asr
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
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "mtmd.dll"),
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]],
                                requiredFiles[4]);
            #endregion

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

        #endregion
    }
}
