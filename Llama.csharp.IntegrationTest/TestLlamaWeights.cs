using FluentAssertions;
using Llama.csharp;
using Llama.csharp.Exceptions;
using Llama.csharp.Native;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.IO;
using System.Reflection.Metadata;
using Xunit;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    [Trait("Category", "Integration")]
    [Trait("Category", "LlamaWeightsAPI")]
    public class TestLlamaWeights
    {
        private static readonly string _baseDllPath = "./llama_b7552";
        private static readonly string _baseModelDirPath = "./test_model";
        private static readonly string _modelPath = @"D:\LLMmodels\Baguettotron-Q8_0.gguf";
        private readonly ITestOutputHelper _output;
        public TestLlamaWeights(ITestOutputHelper output)
        {
            _output = output;
        }

        /// <summary>
        /// Проверка загрузки модели с неверно указанным путем к модели
        /// </summary>
        [Fact]
        public void LlamaWeights_LoadFromFile_WithInvalidPath_Throws()
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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelDirPath, "NonExistent.gguf")){};

            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose(); //на всякий случай
            };

            act.Should().Throw<FileNotFoundException>();
        }
        /// <summary>
        /// Проверка загрузки модели с неверным файлом модели
        /// </summary>
        [Fact]
        public void LlamaWeights_LoadFromFile_WithInvalidModelFile_Throws()
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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelDirPath, "strangeModel.txt")) { };

            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose(); //на всякий случай
            };

            act.Should().Throw<LoadWeightsFailedException>();
        }
        /// <summary>
        /// Проверка загрузки модели с верно указанным путем к модели
        /// </summary>
        [Fact]
        public void LlamaWeights_LoadFromFile_WithValidPath()
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

            ModelParams parametres = new ModelParams(_modelPath) { };

            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose();
            };

            act.Should().NotThrow<Exception>();
        }

        [Fact]
        public void LlamaWeights_LoadInfoNoAlloc_WithValidPath()
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

            NoAllocModelInfo info =
                LLamaWeights.LoadInfoNoAlloc(_modelPath);

            info.Metadata.Count.Should().BeGreaterThan(0);
            info.ContextSize.Should().BeGreaterThan(0);
        }

        [Fact]
        public void LlamaWeights_LoadInfoNoAlloc_InvalidPath()
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

            var act = () => LLamaWeights.LoadInfoNoAlloc(Path.Combine(_baseModelDirPath, "NonExistent.gguf"));

            act.Should().Throw<FileNotFoundException>();
        }

        [Fact]
        public void LlamaWeights_LoadInfoNoAlloc_WithInvalidModelFile_Throws()
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

            var act = () => LLamaWeights.LoadInfoNoAlloc(Path.Combine(_baseModelDirPath, "strangeModel.txt"));

            act.Should().Throw<LoadWeightsFailedException>();
        }

        /// <summary>
        /// Проверка верного создания исполнителя модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateExecutor_Valid()
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

            ModelParams parametres = new ModelParams(_modelPath) { };
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                LlamaExecutor executor = model.CreateExecutor(new ContextParams() { ContextSize = 1024 });
                executor.Dispose();
                model.Dispose();
            };

            act.Should().NotThrow();
        }

        /// <summary>
        /// Проверка неверного создания исполнителя выгруженной модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateExecutor_Invalid()
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

            ModelParams parametres = new ModelParams(_modelPath) { };
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose();
                LlamaExecutor executor = model.CreateExecutor(new ContextParams() { ContextSize = 1024 });
                executor.Dispose();
            };

            act.Should().Throw<ObjectDisposedException>();
        }

        /// <summary>
        /// Проверка верного создания исполнителя модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateOneSeqExecutor_Valid()
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

            ModelParams parametres = new ModelParams(_modelPath) { };
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                OneSeqLlamaExecutor executor = model.CreateOneSeqExecutor(new ContextParams() { ContextSize = 1024 });
                executor.Dispose();
                model.Dispose();
            };

            act.Should().NotThrow();
        }

        /// <summary>
        /// Проверка неверного создания исполнителя выгруженной модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateOneSeqExecutor_Invalid()
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

            ModelParams parametres = new ModelParams(_modelPath) { };
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose();
                OneSeqLlamaExecutor executor = model.CreateOneSeqExecutor(new ContextParams() { ContextSize = 1024 });
                executor.Dispose();
            };

            act.Should().Throw<ObjectDisposedException>();
        }
    }
}
