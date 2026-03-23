using FluentAssertions;
using Llama.csharp;
using Llama.csharp.Exceptions;
using Llama.csharp.Native;
using System;
using System.IO;
using System.Reflection.Metadata;
using Xunit;

namespace Llama.csharp.IntegrationTest
{
    [Trait("Category", "Integration")]
    [Trait("Category", "LlamaWeightsAPI")]
    public class TestLlamaWeights
    {
        private static readonly string _baseDllPath = "./llama_b7552";
        private static readonly string _baseModelPath = "./test_model";

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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelPath, "NonExistent.gguf")){};

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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelPath, "strangeModel.txt")) { };

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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelPath, "Baguettotron-Q8_0.gguf")) 
            {
            };

            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose();
            };

            act.Should().NotThrow<Exception>();
        }

        /// <summary>
        /// Проверка загрузки метаданных модели без загрузки весов
        /// </summary>
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
                LLamaWeights.LoadInfoNoAlloc(Path.Combine(_baseModelPath, "Baguettotron-Q8_0.gguf"));

            info.Metadata.Count.Should().BeGreaterThan(0);
        }

        /// <summary>
        /// Проверка верного создания контекста модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateContext_Valid()
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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelPath, "Baguettotron-Q8_0.gguf")){};
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                LLamaContext context = model.CreateContext(new ContextParams() { ContextSize = 1024 });
                context.Dispose();
                model.Dispose();
            };

            act.Should().NotThrow();
        }

        /// <summary>
        /// Проверка неверного создания контекста выгруженной модели
        /// </summary>
        [Fact]
        public void LlamaWeights_CreateContext_Invalid()
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

            ModelParams parametres = new ModelParams(Path.Combine(_baseModelPath, "Baguettotron-Q8_0.gguf")) { };
            var act = () =>
            {
                LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
                model.Dispose();
                LLamaContext context = model.CreateContext(new ContextParams() { ContextSize = 1024 });
                context?.Dispose();
            };

            act.Should().Throw<ObjectDisposedException>();
        }
    }
}
