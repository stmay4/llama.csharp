using FluentAssertions;
using Llama.csharp;
using Llama.csharp.Native;
using Xunit;

namespace Llama.csharp.IntegrationTest
{
    public class TestModelParamsConstructor
    {
        private static readonly string _baseDllPath = @"D:\DownLoads\llama-b7667-bin-win-vulkan-x64"; // !set your path to the library!

        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ModelParams_WithValidPath()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, "ggml-cpu-x64.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams("test.test") { };

            parametres.ModelPath.Should().Be("test.test");
        }
        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ModelParams_Default_GpuLayerCount_is0()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, "ggml-cpu-x64.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams("test.test") { };

            parametres.GpuLayerCount.Should().Be(0);
        }
        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ModelParams_Default_MainGpu_is0()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, "ggml-cpu-x64.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams("test.test") { };

            parametres.MainGpu.Should().Be(0);
        }
    }
}
