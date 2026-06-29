using FluentAssertions;
using Llama.csharp.Abstractions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using Microsoft.VisualStudio.TestPlatform.CommunicationUtilities;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    public class TestLlamaExecutor
    {
        private static readonly string _baseDllPath = @"D:\DownLoads\llama-b9756-bin-win-vulkan-x64"; // !set your path to the library!
        private static readonly string _modelPath = @"D:\LLMmodels\Baguettotron-Q8_0.gguf"; // !set your model path!
        private static readonly string _heavyModelPath = @"D:\LLMmodels\qwen35\Qwen3.5-9B-UD-Q5_K_XL.gguf"; // !set your model path!
        private static readonly string _moeModelPath = @"D:\LLMmodels\Qwen_Qwen3-30B-A3B-Q4_K_M.gguf"; // !set your MOE model path!
        private static readonly string _сpuBackend = "ggml-cpu-alderlake.dll"; // !set the best CPU backend for your PC here!
        private static readonly string _badCpuBackend = "ggml-cpu-x64.dll";
        private static readonly string _sseCpuBackend = "ggml-cpu-sse42.dll";

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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            j.Should().Be((LLamaSeqId)(-1)); // cannot create more than one if SeqMax = 1 (default).

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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                    Dictionary<LLamaSeqId, Task> prefill = await executor.ProcessPrompt([main, second], ["test"], [true], [true]);

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
                Path.Combine(_baseDllPath, _сpuBackend)
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
                    Task prefill1 = executor.ProcessPrompt(main, "test prompt");
                    Task prefill2 = executor.ProcessPrompt(main, "test prompt"); //must throw, because sequence already in prefill State

                    await prefill1;
                    await prefill2;
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            await executor.ProcessPrompt(main, prompt, false, false);

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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            for (int k = 0; k < 50; k++) // Loop used only for perf test — I don't want to create a large prompt.
            {
                await executor.ProcessPrompt(main, prompt, false, false);

                tokens.AddRange(executor.Context.Tokenize(prompt, false, false).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetSequenceNextDecodedTokenPos(main);
                IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(main);

                tokenCount.Should().Be(pos);
                for (int i = 0; i < tokenCount; i++)
                {
                    tokens[i].Should().Be(decoded[i]); // Compare tokens in the sequence with the tokens obtained from the prompt via executor.Context.Tokenize.
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            for (int k = 0; k < 50; k++) // Loop used only for perf test — I don't want to create a large prompt.
            {
                await executor.ProcessPrompt(main, prompt, false, false);

                tokens.AddRange(executor.Context.Tokenize(prompt, false, false).ToList());
                int tokenCount = tokens.Count;
                int? pos = await executor.GetSequenceNextDecodedTokenPos(main);
                IReadOnlyList<LLamaToken>? decoded = await executor.GetSequenceDecodedTokens(main);

                tokenCount.Should().Be(pos);
                for (int i = 0; i < tokenCount; i++)
                {
                    tokens[i].Should().Be(decoded[i]); // Compare tokens in the sequence with the tokens obtained from the prompt via executor.Context.Tokenize.
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            for (int k = 0; k < 10; k++) // Loop used only for perf test — I don't want to create a large prompt.
            {
                Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1, s2, s3, s4, s5], [prompt, prompt, prompt, prompt, prompt], [false, false, false, false, false], [false, false, false, false, false]);
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            for (int k = 0; k < 10; k++) // Loop used only for perf test — I don't want to create a large prompt.
            {
                //TwoEnterPoint. Share batch may be not from the first token
                Dictionary<LLamaSeqId, Task> prefills1 = await executor.ProcessPrompt([s1, s2], [prompt, prompt], [false, false], [false, false]);
                Dictionary<LLamaSeqId, Task> prefills2 = await executor.ProcessPrompt([s3, s4, s5], [prompt, prompt, prompt], [false, false, false], [false, false, false]);

                await Task.WhenAll(prefills1.Values.ToArray());
                await Task.WhenAll(prefills2.Values.ToArray());

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
                Path.Combine(_baseDllPath, _сpuBackend)
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
            Dictionary<LLamaSeqId, Task> prefills = await executor.ProcessPrompt([s1, s2], [prompt, prompt], executor.Context.Vocab.ShouldAddBOS);
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
                Path.Combine(_baseDllPath, _сpuBackend)
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

            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            LLamaPos endpos = await executor.GetSequenceNextDecodedTokenPos(s1);

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

        [Fact]
        public async Task LlamaExecutor_SpecCPU_Generate_TwoSeq_SharePrefill_AllThreads()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, _badCpuBackend),
                Path.Combine(_baseDllPath, _sseCpuBackend)
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3], // When loading two CPU backends, the first one is used.
                                requiredFiles[4],
                                requiredFiles[5]]);
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

            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            LLamaPos endpos = await executor.GetSequenceNextDecodedTokenPos(s1);

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

        [Fact]
        public async Task LlamaExecutor_Generate_DeleteEnd_Generate()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_heavyModelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams() { ContextSize = 4000 };
            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();

            InferenceParams inferenceParams = new InferenceParams()
            {
                MaxTokens = 15,
                AutoStopFromEOG = false,
                DecodeSpecialTokens = true,
                AntiPrompts = [],
                SamplingPipeline = new TunableSamplerPipeline(
                    new TunableSamplerPipelineSettings(
                        [new TopKSampler()],
                        new DistributionSampler() { Seed = 256 }
                    )
                )
            };

            string prompt = "<system> You are a helpful assistant. </system> \n\n\n" +
                            "<user> count from 1 to 50 </user> \n\n\n" +
                            "<assistant> ";
            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            LLamaPos startPos = await executor.GetSequenceNextDecodedTokenPos(s1);

            // first generation
            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);
            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
                genText += text;
            _output.WriteLine(genText);

            // Check model type
            bool isRecurrentOrHybrid = model.NativeHandle.IsHybrid || model.NativeHandle.IsRecurrent;

            if (isRecurrentOrHybrid)
            {
                // DeleteSequenceEnd must throw exception
                Func<Task> act = async () => await executor.DeleteSequenceEnd(s1, startPos);
                await act.Should().ThrowAsync<Exception>()
                         .WithMessage("*for recurrent and hybrid models, use state checkpoints(not yet implemented)*");

                // Check seq life
                int posAfter = await executor.GetSequenceNextDecodedTokenPos(s1);
                posAfter.Should().BeGreaterThan(0);
            }
            else
            {
                // seq ending delete
                await executor.DeleteSequenceEnd(s1, startPos);

                var watcher = Stopwatch.StartNew();
                Channel<string> ch2 = await executor.Generate(s1, inferenceParams); // generate numbers from 1 again
                genText = "";
                await foreach (var text in ch2.Reader.ReadAllAsync())
                    genText += text;
                _output.WriteLine(genText);
                watcher.Stop();
                _output.WriteLine(watcher.ElapsedMilliseconds + " watcher ms");
            }

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_GenerateMOE_WithoutTensorOverride()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "ggml-vulkan.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3], requiredFiles[4]]);
            #endregion

            ModelParams parametres = new ModelParams(_moeModelPath)
            {
                GpuLayerCount = 999,
            };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 4000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();

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
            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);

            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
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
        public async Task LlamaExecutor_GenerateMOE_WithTensorOverride()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "ggml-vulkan.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3], requiredFiles[4]]);
            #endregion

            ModelParams parametres = new ModelParams(_moeModelPath)
            {
                GpuLayerCount = 999,
                TensorBufferOverrides = [new TensorBufferOverride("blk\\.[0-35].*exps.*", "CPU")]
            };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 4000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();

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
            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);

            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
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
        public async Task LlamaExecutor_GenerateMOE_OnlyCPU()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend),
                Path.Combine(_baseDllPath, "ggml-vulkan.dll")
            };

            foreach (var file in requiredFiles)
            {
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");
            }

            LlamaCpp.Initialize(requiredFiles[0],
                                requiredFiles[1],
                                requiredFiles[2],
                               [requiredFiles[3], requiredFiles[4]]);
            #endregion

            ModelParams parametres = new ModelParams(_moeModelPath)
            {
                GpuLayerCount = 0,
                //TensorBufferOverrides = [new TensorBufferOverride(".*exps.*", "CPU")]
            };

            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);

            ContextParams ctxParams = new ContextParams()
            {
                ContextSize = 4000,
                SeqMax = 1,
                NoPerf = true,
                Threads = Environment.ProcessorCount
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();

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
            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            Channel<string> ch1 = await executor.Generate(s1, inferenceParams);

            string genText = "";
            await foreach (var text in ch1.Reader.ReadAllAsync())
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
        public async Task Perf_vs_LMstudio_test()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
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
                ContextSize = 8000,
                SeqMax = 3
            };

            LlamaExecutor executor = model.CreateExecutor(ctxParams);
            LLamaSeqId s1 = await executor.CreateSequence();
            LLamaSeqId s2 = await executor.CreateSequence();
            LLamaSeqId s3 = await executor.CreateSequence();

            string prompt = "<system> Ты — экспертный переводчик с китайского на русский язык. Переводи точно, сохраняя стиль, контекст и культурные особенности. " +
            "Избегай калькирования, адаптируй идиомы и термины под естественный русский язык. Если встречаются неоднозначности, выбирай наиболее вероятный по контексту " +
            "вариант. Всегда сначала анализируй исходный текст в блоке <think>, затем выдавай только готовый перевод.  </system>\r\n" +
                "<user> 今天天气不错，我们去公园散步吧。顺便买杯咖啡，听说街角那家新开的店味道很好 </user>\r\n" +
                "<assistant> <think>\r\nАнализирую структуру: предложение состоит из двух частей. Первая — предложение пойти гулять из-за хорошей погоды. " +
                "Вторая — предложение купить кофе и упоминание новой кофейни. Тон: дружеский, повседневный. " +
                "Ключевые моменты: «天气不错» (погода хорошая), «散步» (прогулка), «街角那家新开的店» (новый магазин на углу улицы). " +
                "Нужно передать естественный разговорный стиль на русском.\r\n</think>\r\n" +
                "Сегодня отличная погода, давай сходим погулять в парк. Заодно купим кофе, говорят, в новой кофейне на углу варят очень вкусно. </assistant>" +
                "<user> 这个项目的截止日期是下周五，请大家务必在此之前提交最终版本。如果遇到技术问题，随时联系技术支持团队 </user>\r\n" +
                "<assistant> <think>\r\nТекст носит официально-деловой характер. Ключевые элементы: дедлайн (下周五), требование предоставить финальную версию (提交最终版本), инструкция по техподдержке. " +
                "Важно сохранить строгий, но вежливый тон. «务必» усиливает требование, переведу как «обязательно» или «просьба строго». «技术支持团队» — команда технической поддержки.\r\n</think>\r\nСрок сдачи проекта — следующая пятница. " +
                "Просьба обязательно отправить финальную версию до этого времени. Если возникнут технические вопросы, обращайтесь в службу поддержки. </assistant>";

            var watcher = Stopwatch.StartNew(); //начало замера всего

            await executor.ProcessPrompt(s1, prompt, executor.Context.Vocab.ShouldAddBOS);

            LLamaPos endPos = await executor.GetSequenceNextDecodedTokenPos(s1);

            await executor.CopySeqPrefixTo(s1, [s2, s3], endPos);

            string genText = "";

            List<List<string>> messages = [
                ["🔬 模块一：量子计算：超越经典的新范式\n传统计算机以" + "比特" + "（0或1）为信息基本单位，而量子计算机利用" + "量子比特" + "的叠加态与量子纠缠特性，可在同一时刻探索海量计算路径。近年来，谷歌、IBM与中国科研团队相继实现" + "量子优越性" + "，在特定算法任务上显著超越经典超算。尽管量子纠错、相干时间与规模化集成仍是技术瓶颈，但量子计算有望在密码破译、新药分子模拟与高温超导材料设计中实现突破性应用。\n📌 核心提示：量子并行性是算力跃迁的关键，工程化落地仍需跨学科协同攻关。",
                    "🤖 模块五：AI赋能科研：从AlphaFold到科学大模型\n人工智能正推动科学研究范式向" + "数据驱动+智能推演" + "转型。DeepMind的AlphaFold成功预测超2亿种蛋白质三维结构，将结构生物学研究效率提升数个数量级。如今，面向材料筛选、气候模拟、催化反应与药物设计的科学大模型可自动解析文献、生成可验证假设并优化实验路径。AI并非替代科学家，而是作为" + "高通量协作者" + "压缩试错周期，加速跨学科知识融合。\n📌 核心提示：人机协同科研已成常态，模型可解释性与科学因果推断是下一阶段重点。"],
                ["🧬 模块二：CRISPR-Cas9：精准改写生命密码\nCRISPR-Cas9是一种源自细菌适应性免疫系统的基因编辑工具，能够像" + "分子剪刀" + "般在目标DNA位点进行精准切割与修复。自2012年技术成熟以来，它已广泛应用于作物抗病育种、遗传病机制解析与肿瘤免疫治疗。2023年底，全球首款基于CRISPR的镰状细胞病基因疗法正式获批，标志着基因医学从实验室走向临床。当前研究重点在于提升编辑特异性、降低脱靶效应，并探索体内递送系统的安全边界。\n📌 核心提示：技术已进入临床转化期，伦理监管与长期安全性评估需同步完善。",
                    "🔬 模块一：量子计算：超越经典的新范式\n传统计算机以" + "比特" + "（0或1）为信息基本单位，而量子计算机利用" + "量子比特" + "的叠加态与量子纠缠特性，可在同一时刻探索海量计算路径。近年来，谷歌、IBM与中国科研团队相继实现" + "量子优越性" + "，在特定算法任务上显著超越经典超算。尽管量子纠错、相干时间与规模化集成仍是技术瓶颈，但量子计算有望在密码破译、新药分子模拟与高温超导材料设计中实现突破性应用。\n📌 核心提示：量子并行性是算力跃迁的关键，工程化落地仍需跨学科协同攻关。"],
                ["🤖 模块五：AI赋能科研：从AlphaFold到科学大模型\n人工智能正推动科学研究范式向" + "数据驱动+智能推演" + "转型。DeepMind的AlphaFold成功预测超2亿种蛋白质三维结构，将结构生物学研究效率提升数个数量级。如今，面向材料筛选、气候模拟、催化反应与药物设计的科学大模型可自动解析文献、生成可验证假设并优化实验路径。AI并非替代科学家，而是作为" + "高通量协作者" + "压缩试错周期，加速跨学科知识融合。\n📌 核心提示：人机协同科研已成常态，模型可解释性与科学因果推断是下一阶段重点。",
                    "🔬 模块一：量子计算：超越经典的新范式\n传统计算机以" + "比特" + "（0或1）为信息基本单位，而量子计算机利用" + "量子比特" + "的叠加态与量子纠缠特性，可在同一时刻探索海量计算路径。近年来，谷歌、IBM与中国科研团队相继实现" + "量子优越性" + "，在特定算法任务上显著超越经典超算。尽管量子纠错、相干时间与规模化集成仍是技术瓶颈，但量子计算有望在密码破译、新药分子模拟与高温超导材料设计中实现突破性应用。\n📌 核心提示：量子并行性是算力跃迁的关键，工程化落地仍需跨学科协同攻关。"]
                ];

            Task<string> task1 = GenerateOneContextAsync(executor, s1, messages[0], "seq1");
            Task<string> task2 = GenerateOneContextAsync(executor, s2, messages[1], "seq2");
            Task<string> task3 = GenerateOneContextAsync(executor, s3, messages[2], "seq3");

            string[] results = await Task.WhenAll(task1, task2, task3);
            genText += string.Concat(results);

            _output.WriteLine(genText);

            watcher.Stop();
            _output.WriteLine(watcher.ElapsedMilliseconds + " общее время ожидания watcher ms");

            executor.Dispose();
            model.Dispose();
        }

        private async Task<string> GenerateOneContextAsync(LlamaExecutor executor, LLamaSeqId seqId, List<string> Messages, string name)
        {
            string genText = "";
            int k = 0;
            foreach (var message in Messages)
            {
                string input = "<user> " + message + " </user> <assistent> <think>";
                k++;
                var stopwatch = Stopwatch.StartNew();
                long? firstTokenTime = null;
                double? ttft = null;

                await executor.ProcessPrompt(seqId, message, executor.Context.Vocab.ShouldAddBOS);

                InferenceParams inferenceParams = new InferenceParams()
                {
                    MaxTokens = 100,
                    AutoStopFromEOG = false,
                    DecodeSpecialTokens = true,
                    AntiPrompts = [],
                    SamplingPipeline = new TunableSamplerPipeline(
                        new TunableSamplerPipelineSettings(
                            [new TopKSampler()], new DistributionSampler()
                            {
                                Seed = 256
                            }
                            )
                        )
                };

                Channel<string> ch = await executor.Generate(seqId, inferenceParams);

                genText += $" \n\n {name} R{k} \n\n";

                await foreach (var text in ch.Reader.ReadAllAsync())
                {
                    if (firstTokenTime == null)
                    {
                        firstTokenTime = stopwatch.ElapsedMilliseconds;
                        ttft = firstTokenTime.Value;
                        genText += $"TTFT = {ttft:F0}мс \n\n";
                    }

                    genText += text;
                }

                stopwatch.Stop();
                var totalTime = stopwatch.ElapsedMilliseconds;
                var genTime = firstTokenTime.HasValue ? totalTime - firstTokenTime.Value : 0;

                genText += $" \n\nРаунд {k}: всего {totalTime:F0}мс, генерация {genTime:F0}мс";
            }

            return genText;
        }

        [Fact]
        public async Task LlamaExecutor_DeleteSequence_DuringPrefill()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0], requiredFiles[1], requiredFiles[2], [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_modelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
            LlamaExecutor executor = model.CreateExecutor(new ContextParams { SeqMax = 2 });

            LLamaSeqId seq = await executor.CreateSequence();

            // A long enough text so that the prefill does not finish instantly.
            string bigText = string.Join(" ", Enumerable.Repeat("prompt is here", 200));

            // Start the prefill and immediately delete the sequence.
            Task prefillTask = executor.ProcessPrompt(seq, bigText);
            await executor.DeleteSequence(seq);

            // The prefill task should complete without exception.
            await prefillTask;

            // Verify that the sequence has indeed been deleted.
            Func<Task> act = async () => await executor.DeleteSequence(seq);
            await act.Should().ThrowAsync<IndexOutOfRangeException>().WithMessage("*not exist*");

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_DeleteSequence_DuringGeneration()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0], requiredFiles[1], requiredFiles[2], [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_modelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
            LlamaExecutor executor = model.CreateExecutor(new ContextParams { SeqMax = 2 });

            LLamaSeqId seq = await executor.CreateSequence();

            // Provide any non‑empty context so that generation makes sense.
            await executor.ProcessPrompt(seq, "test");

            InferenceParams infParams = new InferenceParams { MaxTokens = 50 };

            // Start generation and immediately delete the sequence.
            Channel<string> channel = await executor.Generate(seq, infParams);
            await executor.DeleteSequence(seq);

            // The channel should be closed; reading it should not hang the test.
            await foreach (string token in channel.Reader.ReadAllAsync())
                _output.WriteLine(token);

            // Generation was stopped, so there are few or no tokens – that is expected.

            // The sequence must be absent.
            Func<Task> act = async () => await executor.DeleteSequence(seq);
            await act.Should().ThrowAsync<IndexOutOfRangeException>().WithMessage("*not exist*");

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_StopPrefill_RevertsSequencePosition()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0], requiredFiles[1], requiredFiles[2], [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_modelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
            LlamaExecutor executor = model.CreateExecutor(new ContextParams { SeqMax = 2 });

            LLamaSeqId seq = await executor.CreateSequence();

            // First, fill the sequence with a short initial prompt
            await executor.ProcessPrompt(seq, "initial prompt");
            int posBefore = await executor.GetSequenceNextDecodedTokenPos(seq);

            // Prepare a long text so the second prefill doesn't finish instantly
            string bigText = string.Join(" ", Enumerable.Repeat("prompt is here", 200));

            bool isRecurrentOrHybrid = model.NativeHandle.IsHybrid || model.NativeHandle.IsRecurrent;

            // Start the second prefill
            Task prefillTask = executor.ProcessPrompt(seq, bigText);

            if (isRecurrentOrHybrid)
            {
                // For recurrent/hybrid models StopSeqPrefill must throw
                Func<Task> act = async () => await executor.StopSeqPrefill(seq, posBefore);
                await act.Should().ThrowAsync<Exception>()
                         .WithMessage("*for recurrent and hybrid models, use state checkpoints(not yet implemented)*");

                // Wait for the prefill task to complete (it should finish normally, but not be stopped)
                await prefillTask;

                // The sequence position should not have been reverted
                int posAfter = await executor.GetSequenceNextDecodedTokenPos(seq);
                posAfter.Should().BeGreaterThan(posBefore); // prefill actually happened

                // The sequence is still usable
                await executor.ProcessPrompt(seq, "another prompt");
                int posFinal = await executor.GetSequenceNextDecodedTokenPos(seq);
                posFinal.Should().BeGreaterThan(posAfter);
            }
            else
            {
                // For transformer models – stop the prefill and revert
                await executor.StopSeqPrefill(seq, posBefore);

                // Wait for the prefill task – it should complete without exception
                await prefillTask;

                // The sequence position should be reverted to the saved position
                int posAfter = await executor.GetSequenceNextDecodedTokenPos(seq);
                posAfter.Should().Be(posBefore);

                // Verify that the sequence is still usable: another prefill should advance the position
                await executor.ProcessPrompt(seq, "another prompt");
                int posFinal = await executor.GetSequenceNextDecodedTokenPos(seq);
                posFinal.Should().BeGreaterThan(posBefore);
            }

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_StopPrefill_RevertsToEmptyAndContinues()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0], requiredFiles[1], requiredFiles[2], [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_modelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
            LlamaExecutor executor = model.CreateExecutor(new ContextParams { SeqMax = 2 });

            LLamaSeqId seq = await executor.CreateSequence();

            // The sequence is empty at the start
            int posEmpty = await executor.GetSequenceNextDecodedTokenPos(seq);
            posEmpty.Should().Be(0);

            // Start a long prefill that won't finish instantly
            string bigText = string.Join(" ", Enumerable.Repeat("prompt is here", 200));
            Task prefillTask = executor.ProcessPrompt(seq, bigText);

            // Immediately stop the prefill, reverting to the beginning
            await executor.StopSeqPrefill(seq, posEmpty);

            // The prefill task should complete without exception
            await prefillTask;

            // Position should be back to 0
            int posAfter = await executor.GetSequenceNextDecodedTokenPos(seq);
            posAfter.Should().Be(0);

            // The sequence should still be functional: run a new prefill and then generate
            await executor.ProcessPrompt(seq, "Hello, how are you?");

            int posAfterNewPrefill = await executor.GetSequenceNextDecodedTokenPos(seq);
            posAfterNewPrefill.Should().BeGreaterThan(0);

            // Attempt generation to confirm the sequence is healthy
            InferenceParams infParams = new InferenceParams { MaxTokens = 10, AutoStopFromEOG = false, AntiPrompts = [] };
            Channel<string> ch = await executor.Generate(seq, infParams);
            var tokens = new List<string>();
            await foreach (string token in ch.Reader.ReadAllAsync())
                tokens.Add(token);

            // Even a few tokens mean generation works
            tokens.Should().NotBeNull();

            executor.Dispose();
            model.Dispose();
        }

        [Fact]
        public async Task LlamaExecutor_Generate_EmptySequence_ThrowsInvalidOperation()
        {
            #region init
            var requiredFiles = new[]
            {
                Path.Combine(_baseDllPath, "llama.dll"),
                Path.Combine(_baseDllPath, "ggml.dll"),
                Path.Combine(_baseDllPath, "ggml-base.dll"),
                Path.Combine(_baseDllPath, _сpuBackend)
            };

            foreach (var file in requiredFiles)
                File.Exists(file).Should().BeTrue($"Required native library {file} not found");

            LlamaCpp.Initialize(requiredFiles[0], requiredFiles[1], requiredFiles[2], [requiredFiles[3]]);
            #endregion

            ModelParams parametres = new ModelParams(_modelPath) { };
            LLamaWeights model = LLamaWeights.LoadFromFile(parametres);
            LlamaExecutor executor = model.CreateExecutor(new ContextParams { ContextSize = 1024 });
            LLamaSeqId seq = await executor.CreateSequence();

            // Make sure the sequence is empty
            int pos = await executor.GetSequenceNextDecodedTokenPos(seq);
            pos.Should().Be(0);

            InferenceParams infParams = new InferenceParams { MaxTokens = 10 };

            // Generating on an empty sequence should throw an exception
            Func<Task> act = async () => await executor.Generate(seq, infParams);
            await act.Should().ThrowAsync<InvalidOperationException>()
                             .WithMessage("*empty sequence*");

            executor.Dispose();
            model.Dispose();
        }
    }
}
