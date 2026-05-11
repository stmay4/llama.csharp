using FluentAssertions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    public class TestOneSeqExecutor
    {
        private static readonly string _baseDllPath = "./llama_b7552";
        private static readonly string _baseModelDirPath = "./test_model";
        private static readonly string _modelPath = @"D:\LLMmodels\Baguettotron-Q8_0.gguf";
        private static readonly string _heavyModelPath = @"D:\LLMmodels\Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill.q8_0.gguf";
        private static readonly string _cpuBackend = "ggml-cpu-alderlake.dll";

        private readonly ITestOutputHelper _output;
        public TestOneSeqExecutor(ITestOutputHelper output)
        {
            _output = output;
        }

        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_OneSeq()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _cpuBackend)
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

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                SeqMax = 1,
                //KVunified = true,
                NoPerf = false
            };


            OneSeqLlamaExecutor executor = model.CreateOneSeqExecutor(ctxParams);

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 50; k++) //конечно это надо префилить за раз, здесь только для теста
            {
                await executor.ProcessPrompt(prompt, false, false);

                tokens.AddRange(executor.Context.Tokenize(prompt, false, false).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetNextDecodedTokenPos();
                IReadOnlyList<LLamaToken>? decoded = await executor.GetDecodedTokens();

                tokenCount.Should().Be(pos);
                for (int i = 0; i < tokenCount; i++)
                {
                    tokens[i].Should().Be(decoded[i]);
                }
            }

            LLamaPerfContextTimings timings = executor.Context.NativeHandle.GetTimings();
            _output.WriteLine((timings.Eval.TotalMilliseconds + timings.PromptEval.TotalMilliseconds).ToString() + " eval ms");

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");


            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_TwoTask_1from4Threads()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _cpuBackend)
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

            ModelParams parametres = new ModelParams(_heavyModelPath) { };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 2000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Math.Max(Environment.ProcessorCount / 4, 1) // четверть ядер
            };

            OneSeqLlamaExecutor executor1 = model.CreateOneSeqExecutor(ctxParams);
            OneSeqLlamaExecutor executor2 = model.CreateOneSeqExecutor(ctxParams);

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();
            // Создаем задачи для каждого потока генерации
            var task1 = RunGenerate_TwoTask(executor1, inferenceParams);
            var task2 = RunGenerate_TwoTask(executor2, inferenceParams);

            // Ждем завершения обеих задач одновременно
            await Task.WhenAll(task1, task2);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor1.Dispose();
            executor2.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_TwoTask_1from2Threads()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _cpuBackend)
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

            ModelParams parametres = new ModelParams(_heavyModelPath) { };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 2000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Math.Max(Environment.ProcessorCount / 2, 1) // половина ядер
            };

            OneSeqLlamaExecutor executor1 = model.CreateOneSeqExecutor(ctxParams);
            OneSeqLlamaExecutor executor2 = model.CreateOneSeqExecutor(ctxParams);

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();
            // Создаем задачи для каждого потока генерации
            var task1 = RunGenerate_TwoTask(executor1, inferenceParams);
            var task2 = RunGenerate_TwoTask(executor2, inferenceParams);

            // Ждем завершения обеих задач одновременно
            await Task.WhenAll(task1, task2);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor1.Dispose();
            executor2.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_TwoTask_AllThreadsForEach()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _cpuBackend)
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

            ModelParams parametres = new ModelParams(_heavyModelPath) { };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 2000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            OneSeqLlamaExecutor executor1 = model.CreateOneSeqExecutor(ctxParams);
            OneSeqLlamaExecutor executor2 = model.CreateOneSeqExecutor(ctxParams);

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();
            // Создаем задачи для каждого потока генерации
            var task1 = RunGenerate_TwoTask(executor1, inferenceParams);
            var task2 = RunGenerate_TwoTask(executor2, inferenceParams);

            // Ждем завершения обеих задач одновременно
            await Task.WhenAll(task1, task2);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor1.Dispose();
            executor2.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_TwoTask_OneThreadForEach()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _cpuBackend)
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

            ModelParams parametres = new ModelParams(_heavyModelPath) { };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 2000,
                SeqMax = 1,
                NoPerf = true,
                Threads = 1
            };

            OneSeqLlamaExecutor executor1 = model.CreateOneSeqExecutor(ctxParams);
            OneSeqLlamaExecutor executor2 = model.CreateOneSeqExecutor(ctxParams);

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();
            // Создаем задачи для каждого потока генерации
            var task1 = RunGenerate_TwoTask(executor1, inferenceParams);
            var task2 = RunGenerate_TwoTask(executor2, inferenceParams);

            // Ждем завершения обеих задач одновременно
            await Task.WhenAll(task1, task2);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor1.Dispose();
            executor2.Dispose();
            model.Dispose();
        }

        /// При использовании больших моделей (от 4 млрд) можно ставить использование максимального числа ядер, если поставим максимум то каждая по очереди сильно используют процессор, 
        /// если половину, то одновременно наполовину, так что без разницы. Упор в обмене данных с памятью
        /// 
        /// Для малых моделей до 1 млрд параметров уменьшение количества используемых ядер улучшает скорость обработки



        private async Task RunGenerate_TwoTask(OneSeqLlamaExecutor executor, InferenceParams parameters)
        {
            string prompt = "<system> You are a technical documentation assistant. " +
                "Your tone is clear, concise, and professional. Avoid markdown, lists, or bullet points unless explicitly requested. " +
                "Use plain English and short sentences. </system> \n\n\n" +
                "<user> Explain in one paragraph what an API rate limit is, why it exists, and what happens when a user exceeds it. </user> \n\n\n" +
                "<assistant> ";
            await executor.ProcessPrompt(prompt, executor.Context.Vocab.ShouldAddBOS);
            string genText = "";
            await foreach (var text in executor.Generate(parameters))
            {
                genText += text;
            }
            _output.WriteLine(genText);
        }
    }
}
