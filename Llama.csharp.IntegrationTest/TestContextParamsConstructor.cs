using FluentAssertions;
using Llama.csharp.Native;
using Microsoft.VisualStudio.TestPlatform.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit.Abstractions;

namespace Llama.csharp.IntegrationTest
{
    public class TestContextParamsConstructor
    {
        private static readonly string _baseDllPath = "./llama_b7552";
        
        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ContextParams_CheckDefault()
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

            ContextParams parametres = new ContextParams() {};

            parametres.SeqMax.Should().Be(1);
            parametres.ContextSize.Should().Be(0);
            parametres.BatchSize.Should().Be(512);
            parametres.UBatchSize.Should().Be(512);
        }
        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ContextParams_CheckInit()
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

            ContextParams parametres = new ContextParams()
            {
                SeqMax = 8,
                KVunified = true,
                OPoffload = false,
                ContextSize = 32000
            };

            parametres.SeqMax.Should().Be(8);
            parametres.KVunified.Should().Be(true);
            parametres.OPoffload.Should().Be(false);
            parametres.ContextSize.Should().Be(32000);
        }
    }
}
