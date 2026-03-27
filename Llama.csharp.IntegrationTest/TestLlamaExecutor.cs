using FluentAssertions;
using Llama.csharp.Native;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    public class TestLlamaExecutor
    {
        private static readonly string _baseDllPath = "./llama_b7552";
        private static readonly string _baseModelDirPath = "./test_model";
        private static readonly string _modelPath = @"D:\LLMmodels\Baguettotron-Q8_0.gguf";
        private static readonly string _cpuBackend = "ggml-cpu-alderlake.dll";

        private readonly ITestOutputHelper _output;
        public TestLlamaExecutor(ITestOutputHelper output)
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

            ContextParams ctxParams = new ContextParams();

            var act = () =>
            {
                LlamaExecutor executor = model.CreateExecutor(ctxParams);
                executor.Dispose();
                model.Dispose();
            };

            act.Should().NotThrow();
        }

        [Fact]
        public void LlamaExecutor_ValidCreation_DefaultCheck()
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

            ContextParams ctxParams = new ContextParams();

            
            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            executor.Context.ContextSize.Should().Be((uint)model.ContextSize);
            executor.Context.Params.SeqMax.Should().Be(ctxParams.SeqMax);
            executor.Context.NativeHandle.BatchSize.Should().Be(ctxParams.BatchSize);
            
            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_CreateSequence_OneSeq_OneMax()
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

            ContextParams ctxParams = new ContextParams();


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId i = await executor.CreateSequence();

            i.Should().Be((LLamaSeqId)0);

            executor.Dispose();
            model.Dispose();
        }
        [Fact]
        public async Task LlamaExecutor_CreateSequence_TwoSeq_OneMax()
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

            ContextParams ctxParams = new ContextParams();


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId i = await executor.CreateSequence();

            i.Should().Be((LLamaSeqId)0);

            LLamaSeqId j = await executor.CreateSequence();

            j.Should().Be((LLamaSeqId) (-1)); // нельзя создать больше одной

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_CreateSequence_TwoSeq_TwoMax()
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
                SeqMax = 2
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId i = await executor.CreateSequence();

            i.Should().Be((LLamaSeqId)0);

            LLamaSeqId j = await executor.CreateSequence();

            j.Should().Be((LLamaSeqId)(1));

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_CreateSequence_TwoSeqInit_RaceCheck()
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
                SeqMax = 2
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            var task1 = executor.CreateSequence();
            var task2 = executor.CreateSequence();

            // Ждем завершения обеих задач одновременно
            var results = await Task.WhenAll(task1, task2);

            var id1 = results[0];
            var id2 = results[1];

            id1.Should().NotBe(id2);

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_InvalidArguments()
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
                SeqMax = 2
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            try
            {
                LLamaSeqId main = await executor.CreateSequence();
                LLamaSeqId second = await executor.CreateSequence();
                var act = async () =>
                {
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([main], null, null, null);

                };

                await act.Should().ThrowAsync<ArgumentNullException>();

                var act2 = async () =>
                {
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([main], ["test"], null, null);

                };

                await act2.Should().ThrowAsync<ArgumentNullException>();

                var act3 = async () =>
                {
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([main], ["test"], [], []);

                };

                await act3.Should().ThrowAsync<ArgumentException>();

                var act4 = async () =>
                {
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([main,second], ["test"], [true], [true]);

                };

                await act4.Should().ThrowAsync<Exception>().WithMessage("*texts*");
            }
            finally
            {
                executor.Dispose();
                model.Dispose();
            }

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
                NoPerf = false
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId main = await executor.CreateSequence();

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 10; k++) //конечно это надо префилить за раз, здесь только для теста
            {
                Task prefill = await executor.ProcessPrompt(main, prompt, false, false);
                await prefill;

                tokens.AddRange(executor.Context.Tokenize(prompt, false, false).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetSequenceNextDecodedTokenPos(main);
                IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(main);

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
        public async Task LlamaExecutor_ProcessPrompt_FiveSeq()
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
                SeqMax = 5,
                NoPerf = false
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId s1 = await executor.CreateSequence();
            LLamaSeqId s2 = await executor.CreateSequence();
            LLamaSeqId s3 = await executor.CreateSequence();
            LLamaSeqId s4 = await executor.CreateSequence();
            LLamaSeqId s5 = await executor.CreateSequence();

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 10; k++) //конечно это надо префилить за раз, здесь только для теста
            {
                Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1,s2,s3,s4,s5], [prompt, prompt, prompt, prompt, prompt], [false, false, false, false, false], [false, false, false, false, false]);
                await Task.WhenAll(prefills.Values.ToArray());

                tokens.AddRange(executor.Context.Tokenize(prompt).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetSequenceNextDecodedTokenPos(s3);
                IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(s3);

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
        public async Task LlamaExecutor_ProcessPrompt_FiveSeq_TwoEnterPoint()
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
                SeqMax = 5,
                NoPerf = false
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId s1 = await executor.CreateSequence();
            LLamaSeqId s2 = await executor.CreateSequence();
            LLamaSeqId s3 = await executor.CreateSequence();
            LLamaSeqId s4 = await executor.CreateSequence();
            LLamaSeqId s5 = await executor.CreateSequence();

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 10; k++) //конечно это надо префилить за раз, здесь только для теста
            {
                //может случиться рассинхрон в несколько токенов, но с определенного момента буду обрабатываться вместе
                Dictionary<LLamaSeqId, Task> prefills1 = await executor.ProcessPrompt([s1, s2], [prompt, prompt], [false, false], [false, false]);
                Dictionary<LLamaSeqId, Task> prefills2 = await executor.ProcessPrompt([s3, s4, s5], [prompt, prompt, prompt], [false, false, false], [false, false, false]);

                await Task.WhenAll(prefills1.Values.ToArray());
                await Task.WhenAll(prefills2.Values.ToArray());//синхронизация

                tokens.AddRange(executor.Context.Tokenize(prompt).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetSequenceNextDecodedTokenPos(s3);
                IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(s3);

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

        //        using System;
        //        using System.Threading.Channels;
        //        using System.Threading.Tasks;

        //        public static async Task Run()
        //        {
        //            var channel = Channel.CreateUnbounded<string>();
        //            // Производитель
        //            var producer = Task.Run(async () => {
        //                for (int i = 0; i < 5; i++)
        //                {
        //                    var message = $"Message {i}";
        //                    await channel.Writer.WriteAsync(message);
        //                    Console.WriteLine($"Produced: {message}");
        //                    await Task.Delay(100); // Имитировать работу
        //                }
        //                channel.Writer.Complete();
        //            });
        //            // Потребитель
        //            var consumer = Task.Run(async () => {
        //                await foreach (var message in channel.Reader.ReadAllAsync())
        //                {
        //                    Console.WriteLine($"Consumed: {message}");
        //                    await Task.Delay(150); // Имитировать обработку
        //                }
        //            });
        //            await Task.WhenAll(producer, consumer);
        //            Console.WriteLine("Обработка завершена.");
        //        }
    }
}
