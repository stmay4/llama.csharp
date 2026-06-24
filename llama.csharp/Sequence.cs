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
        /// Tokens that should be decoded by the model
        /// </summary>
        public List<LLamaToken> TokensToPrefill = new();

        /// <summary>
        /// Number of physical tokens in the context that belong to this sequence
        /// (i.e., not shared with other sequences).
        /// This count also includes tokens that have just been placed into <see cref="TokensToPrefill"/>
        /// during a prefill request. This allows pre‑checking whether a generation or prefill
        /// job will fit into the context before actually submitting it to the decoder loop.
        /// Additionally, when the decoder loop adds freshly generated tokens to the context,
        /// it must perform a final check to prevent adding tokens if the context space
        /// is already fully occupied by prefill tokens.
        /// </summary>
        public int RealTokensCount = 0;

        /// <summary>
        /// Tracks anti-prompts across streamed output.
        /// </summary>
        public required AntipromptProcessor AntipromptProc { get; init; }

        public InferStateArgs InferState { get; set; } = new InferStateArgs();

        public IInferenceParams InferParams { get; set; } = new InferenceParams();

        public readonly StreamingTokenDecoder Decoder = new StreamingTokenDecoder(context);

        public bool EndDeleted { get; set; } = false;
        public LLamaToken DeletedNextToken { get; set; }

        internal void ClearSequenceTokens()
        {
            NextDecodedTokenPos = 0;
            LastLogits = null;
            DecodedTokens.Clear();
            RealTokensCount = 0;
        }
        internal void CopyStateFrom(Sequence seq)
        {
            NextDecodedTokenPos = seq.NextDecodedTokenPos;
            LastLogits = seq.LastLogits;
            DecodedTokens.AddRange(seq.DecodedTokens);
        }

        internal void SetNewEnd(int nextDecodedTokenPos)
        {
            if (NextDecodedTokenPos <= nextDecodedTokenPos) return; // new end must be smaller than current

            // Remove the truncated tokens from the count of tokens stored in the cache
            // (only tokens from the end can be "real", not shared)
            RealTokensCount -= (NextDecodedTokenPos - nextDecodedTokenPos);
            // Prevent negative count if some of the removed tokens were shared
            RealTokensCount = Math.Max(0, RealTokensCount);

            NextDecodedTokenPos = nextDecodedTokenPos;
            LastLogits = null;

            EndDeleted = true;
            DeletedNextToken = DecodedTokens[nextDecodedTokenPos];

            DecodedTokens = DecodedTokens.GetRange(0, nextDecodedTokenPos);
        }
    }

    /// <summary>
    /// State arguments used during one sequence generation task
    /// </summary>
    internal class InferStateArgs
    {
        /// <summary>
        /// Indicates whether the sequence is currently in use (serves as a lightweight lock).
        /// </summary>
        public SeqState State { get; set; } = SeqState.None;
        /// <summary>
        /// Set to true when a token has been sampled and decoded, meaning the sequence needs postprocessing.
        /// </summary>
        public bool TokenSampledAndDecoded { get; set; } = false;
        /// <summary>
        /// Number of tokens remaining to be generated (n_remain for max tokens limit).
        /// </summary>
        public int RemainedTokens { get; set; }
        /// <summary>
        /// If true, generation stops automatically when an End‑of‑Generation (EOG) token is produced.
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
