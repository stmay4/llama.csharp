using FluentAssertions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Text.Encodings;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using Xunit.Abstractions;
using System.Text;

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
                (Memory<byte> bmpData, int width, int height) img = ConvertToBmp(imgPath);
                var result = await ctx.EncodeImage((uint)img.width, (uint)img.height, img.bmpData);

                LlamaEmbedding[] imgEmdeds = await result.embeds;
                foreach (var emded in imgEmdeds)
                {
                    _output.WriteLine(emded.Data.ToString());
                }

                foreach (LLamaToken token in result.BOM)
                    _output.WriteLine(model.Vocab.LLamaTokenToString(token, true));
                foreach (LLamaToken token in result.EOM)
                    _output.WriteLine(model.Vocab.LLamaTokenToString(token, true));

                Encoding encoding = Encoding.UTF8;
                foreach (LLamaToken token in model.Vocab.Tokenize("<|vision_start|>", false, true, encoding))
                {
                    _output.WriteLine(token.ToString());
                }
                foreach (LLamaToken token in model.Vocab.Tokenize("<|vision_start|> ", false, true, encoding))
                {
                    _output.WriteLine(token.ToString());
                }
                foreach (LLamaToken token in model.Vocab.Tokenize(" <|vision_start|> ", false, true, encoding))
                {
                    _output.WriteLine(token.ToString());
                }

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

        private (Memory<byte> bmpData, int width, int height) ConvertToBmp(string imgPath)
        {
            // Загружаем исходное изображение
            using var original = new Bitmap(imgPath);

            // Создаём новое 24‑битное RGB‑изображение тех же размеров
            using var bmp = new Bitmap(original.Width, original.Height, PixelFormat.Format24bppRgb);
            using var g = Graphics.FromImage(bmp);
            // Рендерим исходник в новое изображение (сохраняя пропорции)
            g.DrawImage(original, 0, 0, original.Width, original.Height);

            // Сохраняем как BMP в поток памяти
            using var ms = new MemoryStream();
            bmp.Save(ms, ImageFormat.Bmp);

            // Возвращаем байты и размеры
            return (ms.ToArray().AsMemory(), bmp.Width, bmp.Height);
        }
    }
}
