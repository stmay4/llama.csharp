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
        private static readonly string _modelPath = @"D:\LLMmodels\Qwen3-VL-4B-Instruct-UD-Q5_K_XL.gguf"; // !set your vision model path!
        private static readonly string _mmprojPath = @"D:\LLMmodels\Qwen3-VL-4B-Instruct-mmproj-F16.gguf"; // !set your mmproj path!
        private static readonly string _сpuBackend = "ggml-cpu-alderlake.dll"; // !set the best CPU backend for your PC here!

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
            result.image_max_tokens.Should().Be(1024);
            result.batch_max_tokens.Should().Be(512);

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

            IModelParams modelParams = new ModelParams(_modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_mmprojPath, model, mtmdParams);
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
                ImageMinTokens = 8,
                ImageMaxTokens = 8,
            };

            IModelParams modelParams = new ModelParams(_modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_mmprojPath, model, mtmdParams);

                string imgPath = @"C:\Users\stasm\Pictures\Screenshots\Снимок экрана 2026-06-22 014446.png";
                var result = await ctx.EncodeImageFromPath(imgPath);

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

            IModelParams modelParams = new ModelParams(_modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_mmprojPath, model, mtmdParams);
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

        [Fact]
        public async Task MtmdImage_Qwen_StandartTemplate_DecodeByLLM_Valid()
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

            IModelParams modelParams = new ModelParams(_modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_mmprojPath, model, mtmdParams);

                string imgPath = @"C:\Users\stasm\Pictures\Screenshots\Снимок экрана 2026-06-22 014446.png";
                var result = await ctx.EncodeImageFromPath(imgPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, " <|im_start|>system\n you are a helpfull assistant\n<|im_end|>" +
                    "\n<|im_start|>user\n " + result.BOM);
                Dictionary<LLamaSeqId, Task> mtmdprefill = await executor.ProcessMtmdEmbeds([seq1], [result.embeds]);
                await mtmdprefill[seq1];
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
        public async Task MtmdImage_Qwen_RandomTemplate_DecodeByLLM_Valid()
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

            IModelParams modelParams = new ModelParams(_modelPath);
            LLamaWeights model = LLamaWeights.LoadFromFile(modelParams);
            // Act
            var act = async () =>
            {
                MtmdContext ctx = MtmdContext.CreateFromFile(_mmprojPath, model, mtmdParams);

                string imgPath = @"C:\Users\stasm\Pictures\Screenshots\Снимок экрана 2026-06-22 014446.png";
                var result = await ctx.EncodeImageFromPath(imgPath);

                ContextParams ctxParams = new ContextParams() { ContextSize = 8000 };

                LlamaExecutor executor = model.CreateExecutor(ctxParams, ctx.GetSpecification());

                LLamaSeqId seq1 = await executor.CreateSequence();

                await executor.ProcessPrompt(seq1, "<system> you are a helpfull assistant </system>" +
                    "\n<user>\n What displayed on image? " + result.BOM);
                Dictionary<LLamaSeqId, Task> mtmdprefill = await executor.ProcessMtmdEmbeds([seq1], [result.embeds]);
                await mtmdprefill[seq1];
                await executor.ProcessPrompt(seq1, result.EOM + "\n</user>\n<assistant>\n");

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
    }
}
