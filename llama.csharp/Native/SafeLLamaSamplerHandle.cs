using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;

namespace Llama.csharp.Native
{
    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //
    // TODO: In the future, llama_sampler will be utilized to offload the sampling to the backends (e.g. GPU).
    //

    //цепочка Pipeline для обработки декода и выбора токена
    public sealed class SafeLLamaSamplerChainHandle
    : SafeLLamaHandleBase
    {
        /// <summary>
        /// Get the number of samplers in this chain
        /// </summary>
        public int Count
        {
            get
            {
                return LlamaCpp.Llama_SamplerChainN(this);
            }
        }

        /// <inheritdoc />
        protected override bool ReleaseHandle()
        {
            if (handle != nint.Zero)
                LlamaCpp.Llama_SamplerFree(handle);
            return true;
        }

        /// <summary>
        /// Apply this sampler to a set of candidates
        /// </summary>
        /// <param name="candidates"></param>
        public void Apply(ref LLamaTokenDataArrayNative candidates)
        {
            LlamaCpp.Llama_SamplerApply(this, ref candidates);
        }

        /// <summary>
        /// Sample and accept a token from the idx-th output of the last evaluation. Shorthand for:
        ///
        /// <code>
        ///    var logits = ctx.GetLogitsIth(idx);
        ///    var token_data_array = LLamaTokenDataArray.Create(logits);
        ///    using LLamaTokenDataArrayNative.Create(token_data_array, out var native_token_data);
        ///    sampler_chain.Apply(native_token_data);
        ///    var token = native_token_data.Data.Span[native_token_data.Selected];
        ///    sampler_chain.Accept(token);
        ///    return token;
        /// </code>
        /// </summary>
        /// <param name="context"></param>
        /// <param name="index"></param>
        public LLamaToken Sample(SafeLLamaContextHandle context, int index)
        {
            return LlamaCpp.Llama_SamplerSample(this, context, index);
        }

        /// <summary>
        /// Reset the state of this sampler
        /// </summary>
        public void Reset()
        {
            LlamaCpp.Llama_SamplerReset(this);
        }

        /// <summary>
        /// Accept a token and update the internal state of this sampler
        /// </summary>
        /// <param name="token"></param>
        public void Accept(LLamaToken token)
        {
            LlamaCpp.Llama_SamplerAccept(this, token);
        }

        /// <summary>
        /// Get the name of the sampler at the given index
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public string GetName(int index)
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            return Marshal.PtrToStringAnsi(LlamaCpp.Llama_SamplerName(LlamaCpp.Llama_SamplerChainGet(this, index))) ?? "Unknown Name";
        }

        /// <summary>
        /// Get the seed of the sampler at the given index if applicable. returns LLAMA_DEFAULT_SEED otherwise
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public uint GetSeed(int index)
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            return LlamaCpp.Llama_SamplerGetSeed(LlamaCpp.Llama_SamplerChainGet(this, index));
        }

        #region add/remove samplers
        /// <summary>
        /// Create a new sampler chain
        /// </summary>
        /// <param name="params"></param>
        /// <returns></returns>
        public static SafeLLamaSamplerChainHandle Create(LLamaSamplerChainParams @params)
        {
            return LlamaCpp.Llama_SamplerChainInit(@params);
        }


        /// <summary>
        /// Clone a sampler stage from another chain and add it to this chain
        /// </summary>
        /// <param name="src">The chain to clone a stage from</param>
        /// <param name="index">The index of the stage to clone</param>
        public void AddClone(SafeLLamaSamplerChainHandle src, int index)
        {
            if (index < 0 || index >= src.Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            LlamaCpp.Llama_SamplerChainAdd(
                this,
                LlamaCpp.Llama_SamplerClone(
                    LlamaCpp.Llama_SamplerChainGet(src, index)
                )
            );
        }

        /// <summary>
        /// Remove a sampler stage from this chain
        /// </summary>
        /// <param name="index"></param>
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public void Remove(int index)
        {
            if (index < 0 || index >= Count)
                throw new ArgumentOutOfRangeException(nameof(index));

            LlamaCpp.Llama_SamplerFree(LlamaCpp.Llama_SamplerChainRemove(this, index));
        }

        //добавление своих семплеров
        ///// <summary>
        ///// Add a custom sampler stage
        ///// </summary>
        ///// <typeparam name="TSampler"></typeparam>
        ///// <param name="sampler"></param>
        //public void AddCustom<TSampler>(TSampler sampler)
        //    where TSampler : class, ICustomSampler
        //{
        //    unsafe
        //    {
        //        var samplerHandle = CustomSamplerHandle.Create(sampler);
        //        LlamaCpp.Llama_SamplerChainAdd(this, (IntPtr)samplerHandle.GetLLamaSamplerPointer());
        //    }
        //}


        /// <summary>
        /// Add a sampler which picks the most likely token.
        /// </summary>
        /// <returns></returns>
        public void AddGreedySampler()
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitGreedy());
        }

        /// <summary>
        /// Add a sampler which picks from the probability distribution of all tokens
        /// </summary>
        /// <returns></returns>
        public void AddDistributionSampler(uint seed)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitDist(seed));
        }

        /// <summary>
        /// Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
        /// </summary>
        /// <param name="seed"></param>
        /// <param name="tau">The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.</param>
        /// <param name="eta">The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.</param>
        /// <param name="m">The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.</param>
        /// <param name="vocabCount"></param>
        /// <returns></returns>
        public void AddMirostat1Sampler(int vocabCount, uint seed, float tau, float eta, int m)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitMirostat(vocabCount, seed, tau, eta, m));
        }

        /// <summary>
        /// Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
        /// </summary>
        /// <param name="seed"></param>
        /// <param name="tau">The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.</param>
        /// <param name="eta">The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.</param>
        /// <returns></returns>
        public void AddMirostat2Sampler(uint seed, float tau, float eta)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitMirostatV2(seed, tau, eta));
        }

        /// <summary>
        /// Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
        /// </summary>
        /// <remarks>Setting k &lt;= 0 makes this a noop</remarks>
        /// <returns></returns>
        public void AddTopK(int k)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTopK(k));
        }

        /// <summary>
        /// Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
        /// </summary>
        /// <returns></returns>
        public void AddTopNSigma(float n)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTopNSigma(n));
        }

        /// <summary>
        /// Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
        /// </summary>
        /// <returns></returns>
        public void AddTopP(float p, nint minKeep)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTopP(p, minKeep));
        }

        /// <summary>
        /// Minimum P sampling as described in https://github.com/ggerganov/llama.cpp/pull/3841
        /// </summary>
        public void AddMinP(float p, nint minKeep)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitMinP(p, minKeep));
        }

        /// <summary>
        /// Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
        /// </summary>
        public void AddTypical(float p, nint minKeep)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTypical(p, minKeep));
        }

        /// <summary>
        /// Apply temperature to the logits.
        /// If temperature is less than zero the maximum logit is left unchanged and the rest are set to -infinity
        /// </summary>
        /// <param name="t"></param>
        public void AddTemperature(float t)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTemp(t));
        }

        /// <summary>
        /// Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
        /// </summary>
        /// <param name="t"></param>
        /// <param name="delta"></param>
        /// <param name="exponent"></param>
        public void AddAdapT(float t, float delta, float exponent)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitTempExt(t, delta, exponent));
        }

        /// <summary>
        /// XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
        /// </summary>
        /// <param name="p"></param>
        /// <param name="t"></param>
        /// <param name="minKeep"></param>
        /// <param name="seed"></param>
        public void AddXTC(float p, float t, int minKeep, uint seed)
        {
            LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitXTC(p, t, minKeep, seed));
        }

        /// <summary>
        /// This sampler is meant to be used for fill-in-the-middle infilling, after top_k + top_p sampling
        ///<br />
        /// 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG<br />
        /// 2. combine probs of tokens that have the same prefix<br />
        /// <br />
        /// example:<br />
        /// <br />
        /// - before:<br />
        ///   "abc":   0.5<br />
        ///   "abcd":  0.2<br />
        ///   "abcde": 0.1<br />
        ///   "dummy": 0.1<br />
        ///<br />
        /// - after:<br />
        ///   "abc":   0.8<br />
        ///   "dummy": 0.1<br />
        ///<br />
        /// 3. discard non-EOG tokens with low prob<br />
        /// 4. if no tokens are left -> pick EOT
        /// </summary>
        /// <param name="model"></param>
        public void AddFillInMiddleInfill(SafeLlamaModelHandle model)
        {
            unsafe
            {
                LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitInfill(model.Vocab.VocabNative));
            }
        }

        /// <summary>
        /// Create a sampler which makes tokens impossible unless they match the grammar.
        /// </summary>
        /// <param name="model">The model that this grammar will be used with</param>
        /// <param name="grammar"></param>
        /// <param name="root">Root rule of the grammar</param>
        /// <returns></returns>
        public void AddGrammar(SafeLlamaModelHandle model, string grammar, string root)
        {
            AddGrammar(model.Vocab, grammar, root);
        }

        /// <summary>
        /// Create a sampler which makes tokens impossible unless they match the grammar.
        /// </summary>
        /// <param name="vocab">The vocabulary that this grammar will be used with</param>
        /// <param name="grammar"></param>
        /// <param name="root">Root rule of the grammar</param>
        /// <returns></returns>
        public void AddGrammar(SafeLlamaModelHandle.Vocabulary vocab, string grammar, string root)
        {
            unsafe
            {
                LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitGrammar(vocab.VocabNative, grammar, root));
            }
        }

        /// <summary>
        /// Create a sampler using lazy grammar sampling: https://github.com/ggerganov/llama.cpp/pull/9639
        /// </summary>
        /// <param name="model"></param>
        /// <param name="grammar">Grammar in GBNF form</param>
        /// <param name="root">Root rule of the grammar</param>
        /// <param name="patterns">A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.</param>
        /// <param name="triggerTokens">A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included..</param>
        /// <returns></returns>
        public void AddLazyGrammar(
            SafeLlamaModelHandle model,
            string grammar, string root,
            ReadOnlySpan<string> patterns,
            ReadOnlySpan<LLamaToken> triggerTokens)
        {
            unsafe
            {
                LlamaCpp.Llama_SamplerChainAdd(
                    this,
                    LlamaCpp.Llama_SamplerInitGrammarLazyPatterns(
                        model.Vocab.VocabNative,
                        grammar, root,
                        patterns.ToArray(),
                        triggerTokens.ToArray(), (nuint)triggerTokens.Length
                    )
                );
            }
        }

        /// <summary>
        /// Create a sampler that applies various repetition penalties.
        ///
        /// Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
        /// </summary>
        /// <param name="penaltyCount">How many tokens of history to consider when calculating penalties</param>
        /// <param name="repeat">Repetition penalty</param>
        /// <param name="freq">Frequency penalty</param>
        /// <param name="presence">Presence penalty</param>
        /// <returns></returns>
        public void AddPenalties(int penaltyCount, float repeat, float freq, float presence)
        {
            LlamaCpp.Llama_SamplerChainAdd(
                this,
                LlamaCpp.Llama_SamplerInitPenalties(
                    penaltyCount,
                    repeat,
                    freq,
                    presence
                )
            );
        }

        /// <summary>
        /// DRY sampler, designed by p-e-w, as described in: <a href="https://github.com/oobabooga/text-generation-webui/pull/5677">https://github.com/oobabooga/text-generation-webui/pull/5677</a>.
        /// Porting Koboldcpp implementation authored by pi6am: <a href="https://github.com/LostRuins/koboldcpp/pull/982">https://github.com/LostRuins/koboldcpp/pull/982</a>
        /// </summary>
        /// <param name="model">The model this sampler will be used with</param>
        /// <param name="sequenceBreakers"></param>
        /// <param name="multiplier">penalty multiplier, 0.0 = disabled</param>
        /// <param name="base">exponential base</param>
        /// <param name="allowedLength">repeated sequences longer than this are penalized</param>
        /// <param name="penaltyLastN">how many tokens to scan for repetitions (0 = entire context)</param>
        public void AddDry(SafeLlamaModelHandle model, ReadOnlySpan<string> sequenceBreakers, float multiplier = 0.8f, float @base = 1.75f, int allowedLength = 2, int penaltyLastN = 0)
        {
            unsafe
            {
                // Convert strings, fix memory in place, build array of pointers
                var handles = new List<MemoryHandle>();
                var breakers = stackalloc byte*[sequenceBreakers.Length];
                for (var i = 0; i < sequenceBreakers.Length; i++)
                {
                    var chars = Encoding.Default.GetBytes(sequenceBreakers[i]);
                    handles.Add(chars.AsMemory().Pin());

                    breakers[i] = (byte*)handles[i].Pointer;
                }

                LlamaCpp.Llama_SamplerChainAdd(
                    this,
                    LlamaCpp.Llama_SamplerInitDry(
                        model.Vocab.VocabNative,
                        model.ContextSize,
                        multiplier,
                        @base,
                        allowedLength,
                        penaltyLastN,
                        breakers,
                        (nuint)sequenceBreakers.Length
                    )
                );

                // Clear up all the handles fixing the memory in place
                for (var i = 0; i < handles.Count; i++)
                    handles[i].Dispose();
            }
        }

        /// <summary>
        /// Create a sampler that applies a bias directly to the logits
        /// </summary>
        /// <param name="vocabSize"></param>
        /// <param name="biases"></param>
        /// <returns></returns>
        public void AddLogitBias(int vocabSize, Span<LLamaLogitBias> biases)
        {
            unsafe
            {
                fixed (LLamaLogitBias* biasPtr = biases)
                {
                    LlamaCpp.Llama_SamplerChainAdd(this, LlamaCpp.Llama_SamplerInitLogitBias(vocabSize, biases.Length, biasPtr));
                }
            }
        }
        #endregion
    }
}
