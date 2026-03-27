using FluentAssertions;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.IntegrationTest
{
    public class TestContextParamsConstructor
    {
        private static readonly string _baseDllPath = "./llama_b7552";

        /// <summary>
        /// 
        /// </summary>
        [Fact]
        public void ModelParams_WithValidPath()
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

            parametres.KVunified.Should().Be(true);
        }
    }
}
