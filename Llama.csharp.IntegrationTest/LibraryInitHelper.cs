using FluentAssertions;
using Llama.csharp.Native;

namespace Llama.csharp.IntegrationTest
{
    /// <summary>
    /// Types of backends to initialize
    /// </summary>
    public enum BackendType
    {
        /// <summary>
        /// CPU only
        /// </summary>
        CpuOnly,

        /// <summary>
        /// CPU + Vulkan
        /// </summary>
        CpuAndVulkan,

        /// <summary>
        /// CPU + MTMD (no GPU backend)
        /// </summary>
        CpuAndMtmd,

        /// <summary>
        /// CPU + Vulkan + MTMD
        /// </summary>
        CpuAndVulkanAndMtmd
    }

    /// <summary>
    /// Helper class for library initialization to avoid repeating initialization code in tests
    /// </summary>
    public static class LibraryInitHelper
    {
        /// <summary>
        /// Initialize llama.cpp libraries with specified backend configuration
        /// </summary>
        /// <param name="baseDllPath">Path to the folder with llama.cpp DLLs</param>
        /// <param name="cpuBackend">Name of the CPU backend DLL (e.g., "ggml-cpu-alderlake.dll")</param>
        /// <param name="backendType">Type of backends to initialize</param>
        /// <param name="verifyFilesExist">Whether to verify that all required files exist before initialization</param>
        public static void Initialize(string baseDllPath, string cpuBackend, BackendType backendType, bool verifyFilesExist = true)
        {
            var requiredFiles = new System.Collections.Generic.List<string>
            {
                Path.Combine(baseDllPath, "llama.dll"),
                Path.Combine(baseDllPath, "ggml.dll"),
                Path.Combine(baseDllPath, "ggml-base.dll"),
                Path.Combine(baseDllPath, cpuBackend)
            };

            // Add Vulkan if needed
            bool useVulkan = backendType == BackendType.CpuAndVulkan || 
                             backendType == BackendType.CpuAndVulkanAndMtmd;
            
            if (useVulkan)
            {
                requiredFiles.Add(Path.Combine(baseDllPath, "ggml-vulkan.dll"));
            }

            // Add MTMD if needed
            bool useMtmd = backendType == BackendType.CpuAndMtmd || 
                           backendType == BackendType.CpuAndVulkanAndMtmd;
            
            if (useMtmd)
            {
                requiredFiles.Add(Path.Combine(baseDllPath, "mtmd.dll"));
            }

            // Verify files exist if requested
            if (verifyFilesExist)
            {
                foreach (var file in requiredFiles)
                {
                    File.Exists(file).Should().BeTrue($"Required native library {file} not found");
                }
            }

            // Prepare backend files array (cpu backend + optional vulkan)
            var backendFiles = new System.Collections.Generic.List<string>
            {
                Path.Combine(baseDllPath, cpuBackend)
            };

            if (useVulkan)
            {
                backendFiles.Add(Path.Combine(baseDllPath, "ggml-vulkan.dll"));
            }

            // Initialize based on whether MTMD is required
            if (useMtmd)
            {
                LlamaCpp.Initialize(
                    requiredFiles[0], // llama.dll
                    requiredFiles[1], // ggml.dll
                    requiredFiles[2], // ggml-base.dll
                    backendFiles,
                    requiredFiles[requiredFiles.Count - 1] // mtmd.dll
                );
            }
            else
            {
                LlamaCpp.Initialize(
                    requiredFiles[0], // llama.dll
                    requiredFiles[1], // ggml.dll
                    requiredFiles[2], // ggml-base.dll
                    backendFiles
                );
            }
        }

        /// <summary>
        /// Initialize llama.cpp libraries with CPU only backends
        /// </summary>
        /// <param name="baseDllPath">Path to the folder with llama.cpp DLLs</param>
        /// <param name="cpuBackend">Name of the CPU backend DLL</param>
        /// <param name="verifyFilesExist">Whether to verify that all required files exist</param>
        public static void InitializeCpuOnly(string baseDllPath, string cpuBackend, bool verifyFilesExist = true)
        {
            Initialize(baseDllPath, cpuBackend, BackendType.CpuOnly, verifyFilesExist);
        }

        /// <summary>
        /// Initialize llama.cpp libraries with CPU + Vulkan backends
        /// </summary>
        /// <param name="baseDllPath">Path to the folder with llama.cpp DLLs</param>
        /// <param name="cpuBackend">Name of the CPU backend DLL</param>
        /// <param name="verifyFilesExist">Whether to verify that all required files exist</param>
        public static void InitializeCpuAndVulkan(string baseDllPath, string cpuBackend, bool verifyFilesExist = true)
        {
            Initialize(baseDllPath, cpuBackend, BackendType.CpuAndVulkan, verifyFilesExist);
        }

        /// <summary>
        /// Initialize llama.cpp libraries with CPU + MTMD backends
        /// </summary>
        /// <param name="baseDllPath">Path to the folder with llama.cpp DLLs</param>
        /// <param name="cpuBackend">Name of the CPU backend DLL</param>
        /// <param name="verifyFilesExist">Whether to verify that all required files exist</param>
        public static void InitializeCpuAndMtmd(string baseDllPath, string cpuBackend, bool verifyFilesExist = true)
        {
            Initialize(baseDllPath, cpuBackend, BackendType.CpuAndMtmd, verifyFilesExist);
        }

        /// <summary>
        /// Initialize llama.cpp libraries with CPU + Vulkan + MTMD backends
        /// </summary>
        /// <param name="baseDllPath">Path to the folder with llama.cpp DLLs</param>
        /// <param name="cpuBackend">Name of the CPU backend DLL</param>
        /// <param name="verifyFilesExist">Whether to verify that all required files exist</param>
        public static void InitializeCpuVulkanAndMtmd(string baseDllPath, string cpuBackend, bool verifyFilesExist = true)
        {
            Initialize(baseDllPath, cpuBackend, BackendType.CpuAndVulkanAndMtmd, verifyFilesExist);
        }
    }
}
