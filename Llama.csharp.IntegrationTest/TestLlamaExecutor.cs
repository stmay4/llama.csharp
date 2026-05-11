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
        private static readonly string _modelPath = @"D:\LLMmodels\Baguettotron-Q8_0.gguf";
        private static readonly string _heavyModelPath = @"D:\LLMmodels\Qwen3-4B-Thinking-2507-Claude-4.5-Opus-High-Reasoning-Distill.q8_0.gguf";
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

                var act5 = async () =>
                {
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([(LLamaSeqId)5], ["test"], [true], [true]);

                };

                await act5.Should().ThrowAsync<IndexOutOfRangeException>().WithMessage("*5*");
            }
            finally
            {
                executor.Dispose();
                model.Dispose();
            }

        }

        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_SequenceAlreadyInPrefill()
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

                var act = async () =>
                {
                    Task prefill1 = await executor.ProcessPrompt(main, "test prompt");
                    Task prefill2 = await executor.ProcessPrompt(main, "test prompt"); //должен вызвать ошибку, так как последовательность уже в префилле

                    await prefill1;
                    await prefill2;
                };

                await act.Should().ThrowAsync<Exception>().WithMessage("*sequence using in another place*");

                //var act7 = async () =>
                //{
                //    Task t1 = executor.ProcessPrompt(main, "test prompt");
                //    Task t2 = executor.ProcessPrompt(main, "test prompt"); //должен вызвать ошибку, так как последовательность уже в префилле, и не вызвать Race Condition
                //    await Task.WhenAll(t1, t2);
                //};

                //await act7.Should().ThrowAsync<Exception>().WithMessage("*sequence using in another place*");
            }
            finally
            {
                executor.Dispose();
                model.Dispose();
            }

        }

        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_SequenceAlreadyInPrefill_WithoutRaceCondition()
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

                var act = async () =>
                {
                    Task t1 = executor.ProcessPrompt(main, "test prompt");
                    Task t2 = executor.ProcessPrompt(main, "test prompt"); //должен вызвать ошибку, так как последовательность уже в префилле, и не вызвать Race Condition
                    await Task.WhenAll(t1, t2);
                };

                await act.Should().ThrowAsync<Exception>().WithMessage("*sequence using in another place*");
            }
            finally
            {
                executor.Dispose();
                model.Dispose();
            }
        }



        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_OneSeq_EmptyPrompt()
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

            LlamaExecutor executor = model.CreateExecutor(new ContextParams());

            LLamaSeqId main = await executor.CreateSequence();

            string prompt = "";

            var watcher = Stopwatch.StartNew();

            Task prefill = await executor.ProcessPrompt(main, prompt, false, false);
            await prefill;

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");


            int? pos = await executor.GetSequenceNextDecodedTokenPos(main);
            IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(main);

            pos.Should().Be(0);
            decoded.Count.Should().Be(0);

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_ProcessPrompt_OneSeq_KVUnified_False()
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
                KVunified = false,
                NoPerf = false
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId main = await executor.CreateSequence();

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 50; k++) //конечно это надо префилить за раз, здесь только для теста
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
        public async Task LlamaExecutor_ProcessPrompt_OneSeq_KVUnified_True()
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
                KVunified = true,
                NoPerf = false
            };


            LlamaExecutor executor = model.CreateExecutor(ctxParams);

            LLamaSeqId main = await executor.CreateSequence();

            string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
                "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

            List<LLamaToken> tokens = new List<LLamaToken>();

            var watcher = Stopwatch.StartNew();

            for (int k = 0; k < 50; k++) //конечно это надо префилить за раз, здесь только для теста
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
                KVunified = true,
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

        [Fact]
        public async Task LlamaExecutor_Generate_TwoSeq_AllThreads()
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
                ContextSize = 4000,
                SeqMax = 2,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();
            LLamaSeqId s2 = await executor.CreateSequence();

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();

            string prompt = "<system> You are a technical documentation assistant. " +
                "Your tone is clear, concise, and professional. Avoid markdown, lists, or bullet points unless explicitly requested. " +
                "Use plain English and short sentences. </system> \n\n\n" +
                "<user> Explain in one paragraph what an API rate limit is, why it exists, and what happens when a user exceeds it. </user> \n\n\n" +
                "<assistant> ";
            Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1, s2],[prompt, prompt], executor.Context.Vocab.ShouldAddBOS);
            await Task.WhenAll(prefills.Values.ToArray());

            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);
            Channel<string> ch2 = await executor.Generate(s2, inferenceParams);

            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
            {
                genText += text;
            }
            await foreach (var text in ch2.Reader.ReadAllAsync())
            {
                genText += text;
            }
            _output.WriteLine(genText);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_TwoSeq_SharePrefill_AllThreads()
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
                ContextSize = 4000,
                SeqMax = 2,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();
            LLamaSeqId s2 = await executor.CreateSequence();

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 100,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = []
            };

            var watcher = Stopwatch.StartNew();

            string prompt = "<system> You are a technical documentation assistant. " +
                "Your tone is clear, concise, and professional. Avoid markdown, lists, or bullet points unless explicitly requested. " +
                "Use plain English and short sentences. </system> \n\n\n" +
                "<user> Explain in one paragraph what an API rate limit is, why it exists, and what happens when a user exceeds it. </user> \n\n\n" +
                "<assistant> ";
            //Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1, s2],[prompt, prompt], executor.Context.Vocab.ShouldAddBOS);
            //await Task.WhenAll(prefills.Values.ToArray());

            await await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            LLamaPos endpos = await executor.GetSequenceNextDecodedTokenPos(s1) ?? 0;

            await executor.CopySeqPrefixTo(s1, [s2], endpos);

            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);
            Channel<string> ch2 = await executor.Generate(s2, inferenceParams);

            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
            {
                genText += text;
            }
            await foreach (var text in ch2.Reader.ReadAllAsync())
            {
                genText += text;
            }
            _output.WriteLine(genText);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");

            executor.Dispose();
            model.Dispose();
        }


        //[Fact]
        //public async Task LlamaExecutor_CloneSeqBeginning_FiveSeq()
        //{
        //    #region init
        //    var requiredFiles = new[]
        //    {
        //        Path.Combine(_baseDllPath, "llama.dll"),
        //        Path.Combine(_baseDllPath, "ggml.dll"),
        //        Path.Combine(_baseDllPath, "ggml-base.dll"),
        //        Path.Combine(_baseDllPath, _cpuBackend)
        //    };

        //    foreach (var file in requiredFiles)
        //    {
        //        File.Exists(file).Should().BeTrue($"Required native library {file} not found");
        //    }

        //    LlamaCpp.Initialize(requiredFiles[0],
        //                        requiredFiles[1],
        //                        requiredFiles[2],
        //                       [requiredFiles[3]]);
        //    #endregion

        //    ModelParams parametres = new ModelParams(_modelPath) { };

        //    LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

        //    ContextParams ctxParams = new ContextParams()
        //    {
        //        SeqMax = 5,
        //        KVunified = true,
        //        NoPerf = false
        //    };


        //    LlamaExecutor executor = model.CreateExecutor(ctxParams);

        //    LLamaSeqId s1 = await executor.CreateSequence();
        //    LLamaSeqId s2 = await executor.CreateSequence();
        //    LLamaSeqId s3 = await executor.CreateSequence();
        //    LLamaSeqId s4 = await executor.CreateSequence();
        //    LLamaSeqId s5 = await executor.CreateSequence();

        //    string prompt = "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here " +
        //        "prompt is here prompt is here prompt is here prompt is here prompt is here prompt is here";

        //    List<LLamaToken> tokens = new List<LLamaToken>();

        //    var watcher = Stopwatch.StartNew();

        //    for (int k = 0; k < 10; k++) //конечно это надо префилить за раз, здесь только для теста
        //    {
        //        //Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1, s2, s3, s4, s5], [prompt, prompt, prompt, prompt, prompt], [false, false, false, false, false], [false, false, false, false, false]);
        //        //await Task.WhenAll(prefills.Values.ToArray());

        //        //tokens.AddRange(executor.Context.Tokenize(prompt).ToList());
        //        //int tokenCount = tokens.Count;
        //        //int? pos = await executor.GetSequenceNextDecodedTokenPos(s3);
        //        //IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(s3);

        //        //tokenCount.Should().Be(pos);
        //        //for (int i = 0; i < tokenCount; i++)
        //        //{
        //        //    tokens[i].Should().Be(decoded[i]);
        //        //}
        //    }

        //    LLamaPerfContextTimings timings = executor.Context.NativeHandle.GetTimings();
        //    _output.WriteLine((timings.Eval.TotalMilliseconds + timings.PromptEval.TotalMilliseconds).ToString() + " eval ms");

        //    watcher.Stop();
        //    _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");


        //    executor.Dispose();
        //    model.Dispose();
        //}
    }
}
