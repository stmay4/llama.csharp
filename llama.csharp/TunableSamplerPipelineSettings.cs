using CommunityToolkit.HighPerformance.Buffers;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public class TunableSamplerPipelineSettings
    {
        /// <summary>
        /// List of samplers in the order they are applied.
        /// </summary>
        private List<ISampler> _samplers = new List<ISampler>();
        public List<ISampler> Samplers
        {
            get { return _samplers; }
            set { _samplers = value; }
        }

        /// <summary>
        /// Final sampler that selects a single token from the distribution.
        /// </summary>
        private IFinalizeSampler _finalizeSampler = new GreedySampler();
        public IFinalizeSampler FinalizeSampler
        {
            get { return _finalizeSampler; }
            set { _finalizeSampler = value; }
        }

        /// <summary>
        /// Initializes a new instance of <see cref="TunableSamplerPipelineSettings"/> with the specified samplers and finalizer.
        /// </summary>
        /// <param name="samplers">The list of samplers to apply in sequence.</param>
        /// <param name="finalizeSampler">The sampler that makes the final token choice.</param>
        public TunableSamplerPipelineSettings(List<ISampler> samplers, IFinalizeSampler finalizeSampler) 
        {
            Samplers = samplers;
            FinalizeSampler = finalizeSampler;
        }
    }

    /// <summary>
    /// https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
    /// </summary>
    public interface ISampler 
    {
        /// <summary>
        /// Контракт: метод добавления семплера в конвеер
        /// </summary>
        /// <param name="chain"> конвеер в который добавляем семплер, передается для использования методов SafeLLamaSamplerChainHandle для добавления семплеров </param>
        public void AddToChain(SafeLLamaSamplerChainHandle chain);
    }

    /// <summary>
    /// Sorts the token distribution in descending order and selects the first K tokens.
    /// </summary>
    public class TopKSampler : ISampler
    { 
        public int K
        {
            get => _k;
            init
            {
                if (value < 0) // 0 means disabled
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(K)} cant be less then 0");
                _k = value;
            }
        }
        private readonly int _k = 40;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddTopK(K);
        }
    }

    /// <summary>
    /// Calculates the mean and standard deviation of the token probabilities,
    /// sets a threshold at mean + N * standard deviation, and keeps tokens
    /// whose probability is above that threshold.
    /// </summary>
    public class TopNSigmaSampler : ISampler
    {
        public float N
        {
            get => _n;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(N)} cant be less then 0");
                _n = value;
            }
        }
        private readonly float _n = 1; // -1 = disabled

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddTopNSigma(N);
        }
    }

    /// <summary>
    /// Filters tokens so that only those with a cumulative probability not exceeding
    /// the given threshold P are kept.
    /// </summary>
    public class TopPSampler : ISampler
    {
        
        public float P
        {
            get => _p;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be greate then 1");
                _p = value;
            }
        }
        private readonly float _p = 0.95f; // 1 = disabled
        public int MinKeep
        {
            get => _minKeep;
            init
            {
                if (value < 0) 
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(MinKeep)} cant be less then 0");
                _minKeep = value;
            }
        }
        private readonly int _minKeep = 1;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddTopP(P, (nint)MinKeep);
        }
    }

    /// <summary>
    /// Keeps only tokens whose probability is at least P * (probability of the most likely token).
    /// </summary>
    public class MinPSampler : ISampler
    {

        public float P
        {
            get => _p;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be greate then 1");
                _p = value;
            }
        }
        private readonly float _p = 0.05f; // 0 = disabled
        public int MinKeep
        {
            get => _minKeep;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(MinKeep)} cant be less then 0");
                _minKeep = value;
            }
        }
        private readonly int _minKeep = 1;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddMinP(P, (nint)MinKeep);
        }
    }

    public class TypicalSampler : ISampler
    {

        public float P
        {
            get => _p;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be greate then 1");
                _p = value;
            }
        }
        private readonly float _p = 0.2f; // 0 = disabled
        public nint MinKeep
        {
            get => _minKeep;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(MinKeep)} cant be less then 0");
                _minKeep = value;
            }
        }
        private readonly nint _minKeep = 1;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddTypical(P, MinKeep);
        }
    }

    /// <summary>
    /// Divides the logits by the temperature before applying softmax.
    /// Higher temperature increases diversity, lower makes the output more deterministic.
    /// </summary>
    public class TemperatureSampler : ISampler
    {

        public float T
        {
            get => _t;
            init
            {
                if (value <= 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(T)} cant be less or equel 0");
                _t = value;
            }
        }
        private readonly float _t = 0.8f;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddTemperature(T);
        }
    }

    /// <summary>
    /// Adaptive temperature: adjusts the temperature dynamically based on the entropy
    /// of the token distribution. Automatically increases diversity in uncertain contexts
    /// and decreases it in confident ones.
    /// </summary>
    public class AdapTSampler : ISampler
    {
        public float T
        {
            get => _t;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(T)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(T)} cant be greate then 1");

                _t = value;
            }
        }
        private readonly float _t = 0.8f;

        public float Delta
        {
            get => _delta;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(Delta)} cant be less then 0");
                if (value >= 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(Delta)} cant be greate or equal 1");

                _delta = value;
            }
        }
        private readonly float _delta = 0.40f;

        public float Exponent {get;init;} = 1f;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddAdapT(T, Delta, Exponent);
        }
    }

    /// <summary>
    /// XTC (eXtreme Truncation) sampler: with a given probability, discards the most likely tokens
    /// and keeps only tokens from the "tail" of the distribution. Used to increase creativity
    /// and unpredictability of the generation.
    /// </summary>
    public class XTCSampler : ISampler
    {
        public float P
        {
            get => _p;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(P)} cant be greate then 1");

                _p = value;
            }
        }
        private readonly float _p = 0.5f;
        public float T
        {
            get => _treshold;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(T)} cant be less then 0");
                if (value > 1)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(T)} cant be greate then 1");

                _treshold = value;
            }
        }
        private readonly float _treshold = 0.1f;

        public int MinKeep
        {
            get => _minKeep;
            init
            {
                if (value < 0)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(MinKeep)} cant be less then 0");
                _minKeep = value;
            }
        }
        private readonly int _minKeep = 1;

        public uint Seed { get; init; } = 42;


        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddXTC(P,T, MinKeep, Seed);
        }
    }

    /// <summary>
    /// Grammar-guided sampler. Constrains the generated tokens to follow a formal grammar
    /// (e.g., GBNF format).
    /// </summary>
    public class GrammarSampler : ISampler
    {
        public Grammar Grammar { get; init; } = new Grammar("", "");
        public required SafeLlamaModelHandle.Vocabulary Vocab {  get; init; }

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddGrammar(Vocab, Grammar.Gbnf, Grammar.Root);
        }
    }

    /// <summary>
    /// Applies repetition, frequency, and presence penalties to the token logits.
    /// </summary>
    public class PenaltiesSampler : ISampler
    {
        /// <summary>
        /// на какое количество токенов обращать внимание при наказании, при -1 - весь контекст
        /// </summary>
        public int PenaltyCount { get; init; } = 64; //0 = disabled, -1 = ctx_size
        public float RepeatPenalty { get; init; } = 1;

        /// <summary>
        /// Frequency penalty as described by OpenAI: https://platform.openai.com/docs/api-reference/chat/create<br />
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text
        /// so far, decreasing the model's likelihood to repeat the same line verbatim.
        /// </summary>
        public float FrequencyPenalty
        {
            get => _frequencyPenalty;
            init
            {
                if (value < -2)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(FrequencyPenalty)} must be greater than -2");
                if (value > 2)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(FrequencyPenalty)} must be less than 2");
                _frequencyPenalty = value;
            }
        }
        private readonly float _frequencyPenalty = 0;

        /// <summary>
        /// Presence penalty as described by OpenAI: https://platform.openai.com/docs/api-reference/chat/create<br />
        /// Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the
        /// text so far, increasing the model's likelihood to talk about new topics.
        /// </summary>
        public float PresencePenalty
        {
            get => _presencePenalty;
            init
            {
                if (value < -2)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(PresencePenalty)} must be greater than -2");
                if (value > 2)
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(PresencePenalty)} must be less than 2");
                _presencePenalty = value;
            }
        }
        private readonly float _presencePenalty = 0;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddPenalties(PenaltyCount, RepeatPenalty, FrequencyPenalty, PresencePenalty);
        }
    }

    /// <summary>
    /// Applies bias values to the logits of specific tokens, manually increasing or decreasing
    /// their probability of being selected.
    /// </summary>
    public class LogitBiasSampler : ISampler
    {
        public IReadOnlyDictionary<LLamaToken, float> LogitBias { get; init; } = new Dictionary<LLamaToken, float>();
        public required SafeLlamaModelHandle.Vocabulary Vocab { get; init; }

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            if (LogitBias.Count > 0)
            {
                using var biases = SpanOwner<LLamaLogitBias>.Allocate(LogitBias.Count);

                // copy the biases into it
                var index = 0;
                foreach (var bias in LogitBias)
                {
                    biases.Span[index++] = new LLamaLogitBias
                    {
                        Token = bias.Key,
                        Bias = bias.Value
                    };
                }

                // Add the biases to the sampler
                chain.AddLogitBias(Vocab.Count, biases.Span);

            }
        }
    }

    public interface IFinalizeSampler{
        public void AddToChain(SafeLLamaSamplerChainHandle chain);
    }

    /// <summary>
    /// Greedy sampler: always picks the token with the highest probability.
    /// </summary>
    public class GreedySampler: IFinalizeSampler
    {
        public GreedySampler() { }

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddGreedySampler();
        }
    }

    /// <summary>
    /// Random sampling: selects a token according to the probability distribution.
    /// </summary>
    public class DistributionSampler : IFinalizeSampler
    {
        public uint Seed { get; init; } = 42;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddDistributionSampler(Seed);
        }
    }

    /// <summary>
    /// Mirostat v2: a perplexity‑controlled sampling algorithm that adjusts the token distribution
    /// to maintain a target perplexity level.
    /// </summary>
    public class Mirostat2Sampler : IFinalizeSampler
    {
        public uint Seed { get; init; } = 42;
        /// <summary>
        /// Target perplexity level.
        /// </summary>
        public float Tau
        {
            get => _tau;
            init
            {
                if (value < 0) //0 - disabled
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(Tau)} cant be less then 0");
                _tau = value;
            }
        }
        private readonly float _tau = 3;
        /// <summary>
        /// Learning rate for perplexity adjustment.
        /// </summary>
        public float Eta
        {
            get => _eta;
            init
            {
                if (value < 0) //0 - disabled
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(Eta)} cant be less then 0");
                _eta = value;
            }
        }
        private readonly float _eta = 0.1f;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddMirostat2Sampler(Seed, Tau, Eta);
        }
    }

}
