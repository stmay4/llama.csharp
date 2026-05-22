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
        /// Количество физических токенов в контексте (не разделенных между другими и принадлежащих этой последовательности)
        /// В это число входит также количество токенов, которые только попали в Embeds при задании на заполнение
        /// Это сделано для того, чтобы проверять задания на генерацию и заполнение на помещаемость в контекст до реальной отправки обрбаотчику
        /// Также проверка должна быть в обработчике при добавлении в контекст только сгенерированных токенов, чтобы
        /// они не добавлялись если все место уже занято префилом.
        /// В данном случае префилл первостепенен
        /// В будущем можно добавить настройка о первостепенности генерации, 
        /// хотя легче уж просто удалить ненужную последовательность или кусок в данной для освобождения места
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
            if (NextDecodedTokenPos <= nextDecodedTokenPos) return; // новый конец должен быть меньше

            RealTokensCount -= (NextDecodedTokenPos - nextDecodedTokenPos); // удаляем токены убранные из числа тех, что зранятся в кэше
            RealTokensCount = Math.Max(0, RealTokensCount); //если были среди них разделенные, чтобы не ушло в минус

            NextDecodedTokenPos = nextDecodedTokenPos;
            LastLogits = null;

            EndDeleted = true;
            DeletedNextToken = DecodedTokens[nextDecodedTokenPos];

            DecodedTokens = DecodedTokens.GetRange(0, nextDecodedTokenPos-1);
        }
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
