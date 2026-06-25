using Xunit;
using FluentAssertions;
using Llama.csharp;
using Llama.csharp.Native;
using System.IO;
using System;

namespace Llama.csharp.IntegrationTest
{

    /// <summary>
    /// CAUTION Every test in this file should be called alone. CAUTION
    /// </summary>

    [Trait("Category", "Integration")]
    [Trait("Category", "NativeLibraryLoad")]
    public class TestLoadLibrary
    {
        private static readonly string _baseDllPath = @"D:\DownLoads\llama-b7667-bin-win-vulkan-x64"; // !set your path to the library!

        /// <summary>
        /// Load with invalid lib paths
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
        /// Load with valid lib paths
        /// only CPU
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
        /// Load with valid lib paths
        /// CPU + Vulkan
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