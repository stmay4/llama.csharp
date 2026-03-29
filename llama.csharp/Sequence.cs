using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    internal class Sequence(LLamaContext context)
    {
        public required LLamaSeqId Id { get; init; }
        public int NextDecodedTokenPos { get; set; } = 0;
        public LLamaTokenDataArray? LastLogits { get; set; } = null;
        public List<LLamaToken> DecodedTokens { get; set; } = new List<LLamaToken>();
        /// <summary>
        /// токены для обработки декодом
        /// </summary>
        public List<LLamaToken> Embeds = new();

        /// <summary>
        /// Tracks anti-prompts across streamed output.
        /// </summary>
        public required AntipromptProcessor AntipromptProc { get; init; }

        public InferStateArgs InferState { get; set; } = new InferStateArgs();

        public IInferenceParams InferParams { get; set; } = new InferenceParams();

        public readonly StreamingTokenDecoder Decoder = new StreamingTokenDecoder(context);
    }

    /// <summary>
    /// State arguments that are used in single inference
    /// </summary>
    internal class InferStateArgs
    {
        /// <summary>
        /// Lock help
        /// </summary>
        public SeqState State { get; set; } = SeqState.None;
        /// <summary>
        /// Sequence must be postprocessed
        /// </summary>
        public bool TokenSampledAndDecoded { get; set; } = false;
        /// <summary>
        /// Tokens count remained to be used. (n_remain)
        /// </summary>
        public int RemainedTokens { get; set; }
        /// <summary>
        /// 
        /// </summary>
        public bool AutoStopFromEOG { get; set; }
    }
    internal enum SeqState
    {
        None,
        Generation,
        Prefill
    }
}
