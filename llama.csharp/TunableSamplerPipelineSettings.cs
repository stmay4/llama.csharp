using CommunityToolkit.HighPerformance.Buffers;
using Llama.csharp.Native;

namespace Llama.csharp
{
    public class TunableSamplerPipelineSettings
    {
        //список семплеров и их характеристик

        private List<ISampler> _samplers = new List<ISampler>();
        public List<ISampler> Samplers
        {
            get { return _samplers; }
            set { _samplers = value; }
        } 

        private IFinalizeSampler _finalizeSampler = new GreedySampler();

        public IFinalizeSampler FinalizeSampler
        {
            get { return _finalizeSampler; }
            set { _finalizeSampler = value; }
        }

        public TunableSamplerPipelineSettings(List<ISampler> samplers, IFinalizeSampler finalizeSampler) 
        {
            Samplers = samplers;
            FinalizeSampler = finalizeSampler;
        }
    }

    public interface ISampler 
    {
        /// <summary>
        /// 
        /// </summary>
        /// <param name="chain"></param>
        public void AddToChain(SafeLLamaSamplerChainHandle chain);
    }

    public class TopKSampler : ISampler
    {
        public int K
        {
            get => _k;
            init
            {
                if (value < 0) //0 - не применяется
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

    public class TopNSigmaSampler : ISampler
    {
        public float N
        {
            get => _n;
            init
            {
                if (value < 0) //Число указывает количество стандартных отклонений от среднего значения вероятности, которое допускается для выбора токена.
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
            chain.AddTopP(P, MinKeep);
        }
    }

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
            chain.AddMinP(P, MinKeep);
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
        private readonly float _treshold = 1f;

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

    public class GrammarSampler : ISampler
    {
        public Grammar Grammar { get; init; } = new Grammar("", "");
        public required SafeLlamaModelHandle.Vocabulary Vocab {  get; init; }

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddGrammar(Vocab, Grammar.Gbnf, Grammar.Root);
        }
    }

    public class PenaltiesSampler : ISampler
    {
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

    public class GreedySampler: IFinalizeSampler
    {
        public GreedySampler() { }

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            throw new NotImplementedException();
        }
    }

    public class DistributionSampler : IFinalizeSampler
    {
        public uint Seed { get; init; } = 42;

        public void AddToChain(SafeLLamaSamplerChainHandle chain)
        {
            chain.AddDistributionSampler(Seed);
        }
    }

    public class Mirostat2Sampler : IFinalizeSampler
    {
        public uint Seed { get; init; } = 42;
        public float Tau
        {
            get => _tau;
            init
            {
                if (value < 0) //0 - не применяется
                    throw new ArgumentOutOfRangeException(nameof(value), $"{nameof(Tau)} cant be less then 0");
                _tau = value;
            }
        }
        private readonly float _tau = 3;
        public float Eta
        {
            get => _eta;
            init
            {
                if (value < 0) //0 - не применяется
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
