using Xunit;
using FluentAssertions;
using Llama.csharp;
using Llama.csharp.Native;

namespace Llama.csharp.IntegrationTest
{
    [Trait("Category", "Integration")]
    [Trait("Category", "NativeLibraryLoad")]
    public class TestLoadLibrary
    {
        private static readonly string _baseDllPath = "../../../llama_b7552/";

        /// <summary>
        /// Проверка загрузки нативных библиотек с неверными путями к файлам
        /// </summary>
        [Fact]
        public void LlamaCpp_Initialize_WithInvalidPath_Throws()
        {
            var act = () => LlamaCpp.Initialize(
                @"E:\NonExistent\llama.dll",
                @"E:\NonExistent\ggml.dll",
                @"E:\NonExistent\ggml-base.dll",
                [@"E:\NonExistent\ggml-cpu-x64.dll"]
            );

            act.Should().Throw<DllNotFoundException>("The test should be called alone. Or there is a bug");
        }

        /// <summary>
        /// Проверка загрузки нативных библиотек и получения всех функций из библиотек
        /// Загрузка с одним бэкендом - CPU
        /// </summary>
        [Fact]
        public void LlamaCpp_Initialize_onlyCPU()
        {
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

            var act = () => LlamaCpp.Initialize(requiredFiles[0],
                                                requiredFiles[1],
                                                requiredFiles[2],
                                               [requiredFiles[3]]);

            act.Should().NotThrow<Exception>("Native library initialization fail");
        }

        /// <summary>
        /// Проверка загрузки нативных библиотек и получения всех функций из библиотек
        /// Загрузка с бэкендами - CPU и Vulkan
        /// </summary>
        [Fact]
        public void LlamaCpp_Initialize_CPU_Vulkan()
        {
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, "ggml-cpu-x64.dll"),
                Path.Combine(_baseDllPath, "ggml-vulkan.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            var act = () => LlamaCpp.Initialize(requiredFiles[0],
                                                requiredFiles[1],
                                                requiredFiles[2],
                                               [requiredFiles[3], requiredFiles[4]]);

            act.Should().NotThrow<Exception>("Native library initialization fail");
        }
    }
}