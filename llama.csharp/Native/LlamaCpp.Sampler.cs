using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    public static partial class LlamaCpp
    {
        #region LLAMA API functions

        #region delegates

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int llama_sampler_chain_n(SafeLLamaSamplerChainHandle chain);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_sampler_apply(SafeLLamaSamplerChainHandle smpl, ref LLamaTokenDataArrayNative cur_p);

        // <summary>
        // Sample and accept a token from the idx-th output of the last evaluation
        //
        // Shorthand for:
        // <code>
        //    const auto * logits = llama_get_logits_ith(ctx, idx);
        //    llama_token_data_array cur_p = { ... init from logits ... };
        //    llama_sampler_apply(smpl, cur_p);
        //    auto token = cur_p.data[cur_p.selected].id;
        //    llama_sampler_accept(smpl, token);
        //    return token;
        // </code>
        // </summary>
        // <param name="smpl"></param>
        // <param name="ctx"></param>
        // <param name="idx"></param>
        // <returns>The sampled token</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaToken llama_sampler_sample(SafeLLamaSamplerChainHandle smpl, SafeLLamaContextHandle ctx, int idx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_sampler_reset(SafeLLamaSamplerChainHandle smpl);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_sampler_accept(SafeLLamaSamplerChainHandle smpl, LLamaToken token);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_name(IntPtr smpl);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint llama_sampler_get_seed(IntPtr smpl);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeLLamaSamplerChainHandle llama_sampler_chain_init(LLamaSamplerChainParams @params);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LLamaSamplerChainParams llama_sampler_chain_default_params();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_clone(IntPtr chain);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_chain_remove(SafeLLamaSamplerChainHandle chain, int i);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_greedy();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_dist(uint seed);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_mirostat(int nVocab, uint seed, float tau, float eta, int m);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_mirostat_v2(uint seed, float tau, float eta);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_top_k(int k);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_top_n_sigma(float n);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_top_p(float p, nint min_keep);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_min_p(float p, nint min_keep);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_typical(float p, nint min_keep);

        // #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_temp(float t);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_temp_ext(float t, float delta, float exponent);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_xtc(float p, float t, nint minKeep, uint seed);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate IntPtr llama_sampler_init_infill(LLamaVocabNative* vocab);

        // @details Initializes a GBNF grammar, see grammars/README.md for details.
        // @param vocab The vocabulary that this grammar will be used with.
        // @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
        // @param grammar_root The name of the start symbol for the grammar.
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate IntPtr llama_sampler_init_grammar(LLamaVocabNative* model, string grammar_str, string grammar_root);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate IntPtr llama_sampler_init_grammar_lazy_patterns(
            LLamaVocabNative* model,
            string grammar_str,
            string grammar_root,
            string[] triggerPatterns,
            LLamaToken[] triggerTokens,
            nuint num_trigger_tokens
        );

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_init_penalties(
            int penalty_last_n,     // last n tokens to penalize (0 = disable penalty, -1 = context size)
            float penalty_repeat,   // 1.0 = disabled
            float penalty_freq,     // 0.0 = disabled
            float penalty_present   // 0.0 = disabled
        );

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate IntPtr llama_sampler_init_dry(
            LLamaVocabNative* vocab,
            int n_ctx_train,
            float dry_multiplier,
            float dry_base,
            int dry_allowed_length,
            int dry_penalty_last_n,
            byte** seq_breakers,
            nuint num_breakers
        );

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate IntPtr llama_sampler_init_logit_bias(
            int n_vocab,
            int n_logit_bias,
            LLamaLogitBias* logit_bias
        );

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        internal unsafe delegate LLamaSampler* llama_sampler_init(LLamaSamplerI* iface, IntPtr ctx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_sampler_free(IntPtr sampler);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void llama_sampler_chain_add(SafeLLamaSamplerChainHandle chain, IntPtr smpl);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate IntPtr llama_sampler_chain_get(SafeLLamaSamplerChainHandle chain, int i);

        #endregion

        #region functions

        private static llama_sampler_chain_n _llama_sampler_chain_n;
        private static llama_sampler_apply _llama_sampler_apply;
        private static llama_sampler_sample _llama_sampler_sample;
        private static llama_sampler_reset _llama_sampler_reset;
        private static llama_sampler_accept _llama_sampler_accept;
        private static llama_sampler_name _llama_sampler_name;
        private static llama_sampler_get_seed _llama_sampler_get_seed;
        private static llama_sampler_chain_init _llama_sampler_chain_init;
        private static llama_sampler_chain_default_params _llama_sampler_chain_default_params;
        private static llama_sampler_clone _llama_sampler_clone;
        private static llama_sampler_chain_remove _llama_sampler_chain_remove;
        private static llama_sampler_init_greedy _llama_sampler_init_greedy;
        private static llama_sampler_init_dist _llama_sampler_init_dist;
        private static llama_sampler_init_mirostat _llama_sampler_init_mirostat;
        private static llama_sampler_init_mirostat_v2 _llama_sampler_init_mirostat_v2;
        private static llama_sampler_init_top_k _llama_sampler_init_top_k;
        private static llama_sampler_init_top_n_sigma _llama_sampler_init_top_n_sigma;
        private static llama_sampler_init_top_p _llama_sampler_init_top_p;
        private static llama_sampler_init_min_p _llama_sampler_init_min_p;
        private static llama_sampler_init_typical _llama_sampler_init_typical;
        private static llama_sampler_init_temp _llama_sampler_init_temp;
        private static llama_sampler_init_temp_ext _llama_sampler_init_temp_ext;
        private static llama_sampler_init_xtc _llama_sampler_init_xtc;
        private static llama_sampler_init_infill _llama_sampler_init_infill;
        private static llama_sampler_init_grammar _llama_sampler_init_grammar;
        private static llama_sampler_init_grammar_lazy_patterns _llama_sampler_init_grammar_lazy_patterns;
        private static llama_sampler_init_penalties _llama_sampler_init_penalties;
        private static llama_sampler_init_dry _llama_sampler_init_dry;
        private static llama_sampler_init_logit_bias _llama_sampler_init_logit_bias;
        private static llama_sampler_init _llama_sampler_init;
        private static llama_sampler_free _llama_sampler_free;
        private static llama_sampler_chain_add _llama_sampler_chain_add;
        private static llama_sampler_chain_get _llama_sampler_chain_get;

        #endregion

        #endregion

        private static void LoadSamplerFunctions()
        {
            _llama_sampler_chain_n = GetLibFunction<llama_sampler_chain_n>(_llamaHandle, "llama_sampler_chain_n");
            _llama_sampler_apply = GetLibFunction<llama_sampler_apply>(_llamaHandle, "llama_sampler_apply");
            _llama_sampler_sample = GetLibFunction<llama_sampler_sample>(_llamaHandle, "llama_sampler_sample");
            _llama_sampler_reset = GetLibFunction<llama_sampler_reset>(_llamaHandle, "llama_sampler_reset");
            _llama_sampler_accept = GetLibFunction<llama_sampler_accept>(_llamaHandle, "llama_sampler_accept");
            _llama_sampler_name = GetLibFunction<llama_sampler_name>(_llamaHandle, "llama_sampler_name");
            _llama_sampler_get_seed = GetLibFunction<llama_sampler_get_seed>(_llamaHandle, "llama_sampler_get_seed");
            _llama_sampler_chain_init = GetLibFunction<llama_sampler_chain_init>(_llamaHandle, "llama_sampler_chain_init");
            _llama_sampler_chain_default_params = GetLibFunction<llama_sampler_chain_default_params>(_llamaHandle, "llama_sampler_chain_default_params");
            _llama_sampler_clone = GetLibFunction<llama_sampler_clone>(_llamaHandle, "llama_sampler_clone");
            _llama_sampler_chain_remove = GetLibFunction<llama_sampler_chain_remove>(_llamaHandle, "llama_sampler_chain_remove");
            _llama_sampler_init_greedy = GetLibFunction<llama_sampler_init_greedy>(_llamaHandle, "llama_sampler_init_greedy");
            _llama_sampler_init_dist = GetLibFunction<llama_sampler_init_dist>(_llamaHandle, "llama_sampler_init_dist");
            _llama_sampler_init_mirostat = GetLibFunction<llama_sampler_init_mirostat>(_llamaHandle, "llama_sampler_init_mirostat");
            _llama_sampler_init_mirostat_v2 = GetLibFunction<llama_sampler_init_mirostat_v2>(_llamaHandle, "llama_sampler_init_mirostat_v2");
            _llama_sampler_init_top_k = GetLibFunction<llama_sampler_init_top_k>(_llamaHandle, "llama_sampler_init_top_k");
            _llama_sampler_init_top_n_sigma = GetLibFunction<llama_sampler_init_top_n_sigma>(_llamaHandle, "llama_sampler_init_top_n_sigma");
            _llama_sampler_init_top_p = GetLibFunction<llama_sampler_init_top_p>(_llamaHandle, "llama_sampler_init_top_p");
            _llama_sampler_init_min_p = GetLibFunction<llama_sampler_init_min_p>(_llamaHandle, "llama_sampler_init_min_p");
            _llama_sampler_init_typical = GetLibFunction<llama_sampler_init_typical>(_llamaHandle, "llama_sampler_init_typical");
            _llama_sampler_init_temp = GetLibFunction<llama_sampler_init_temp>(_llamaHandle, "llama_sampler_init_temp");
            _llama_sampler_init_temp_ext = GetLibFunction<llama_sampler_init_temp_ext>(_llamaHandle, "llama_sampler_init_temp_ext");
            _llama_sampler_init_xtc = GetLibFunction<llama_sampler_init_xtc>(_llamaHandle, "llama_sampler_init_xtc");
            _llama_sampler_init_infill = GetLibFunction<llama_sampler_init_infill>(_llamaHandle, "llama_sampler_init_infill");
            _llama_sampler_init_grammar = GetLibFunction<llama_sampler_init_grammar>(_llamaHandle, "llama_sampler_init_grammar");
            _llama_sampler_init_grammar_lazy_patterns = GetLibFunction<llama_sampler_init_grammar_lazy_patterns>(_llamaHandle, "llama_sampler_init_grammar_lazy_patterns");
            _llama_sampler_init_penalties = GetLibFunction<llama_sampler_init_penalties>(_llamaHandle, "llama_sampler_init_penalties");
            _llama_sampler_init_dry = GetLibFunction<llama_sampler_init_dry>(_llamaHandle, "llama_sampler_init_dry");
            _llama_sampler_init_logit_bias = GetLibFunction<llama_sampler_init_logit_bias>(_llamaHandle, "llama_sampler_init_logit_bias");
            _llama_sampler_init = GetLibFunction<llama_sampler_init>(_llamaHandle, "llama_sampler_init");
            _llama_sampler_free = GetLibFunction<llama_sampler_free>(_llamaHandle, "llama_sampler_free");
            _llama_sampler_chain_add = GetLibFunction<llama_sampler_chain_add>(_llamaHandle, "llama_sampler_chain_add");
            _llama_sampler_chain_get = GetLibFunction<llama_sampler_chain_get>(_llamaHandle, "llama_sampler_chain_get");
        }

        #region Public Methods

        public static int Llama_SamplerChainN(SafeLLamaSamplerChainHandle chain)
        {
            EnsureInitialized();
            return _llama_sampler_chain_n(chain);
        }

        public static void Llama_SamplerApply(SafeLLamaSamplerChainHandle smpl, ref LLamaTokenDataArrayNative cur_p)
        {
            EnsureInitialized();
            _llama_sampler_apply(smpl, ref cur_p);
        }

        public static LLamaToken Llama_SamplerSample(SafeLLamaSamplerChainHandle smpl, SafeLLamaContextHandle ctx, int idx)
        {
            EnsureInitialized();
            return _llama_sampler_sample(smpl, ctx, idx);
        }

        public static void Llama_SamplerReset(SafeLLamaSamplerChainHandle smpl)
        {
            EnsureInitialized();
            _llama_sampler_reset(smpl);
        }

        public static void Llama_SamplerAccept(SafeLLamaSamplerChainHandle smpl, LLamaToken token)
        {
            EnsureInitialized();
            _llama_sampler_accept(smpl, token);
        }

        public static IntPtr Llama_SamplerName(IntPtr smpl)
        {
            EnsureInitialized();
            return _llama_sampler_name(smpl);
        }

        public static uint Llama_SamplerGetSeed(IntPtr smpl)
        {
            EnsureInitialized();
            return _llama_sampler_get_seed(smpl);
        }

        public static SafeLLamaSamplerChainHandle Llama_SamplerChainInit(LLamaSamplerChainParams @params)
        {
            EnsureInitialized();
            return _llama_sampler_chain_init(@params);
        }

        public static LLamaSamplerChainParams Llama_SamplerChainDefaultParams()
        {
            EnsureInitialized();
            return _llama_sampler_chain_default_params();
        }

        public static IntPtr Llama_SamplerClone(IntPtr chain)
        {
            EnsureInitialized();
            return _llama_sampler_clone(chain);
        }

        public static IntPtr Llama_SamplerChainRemove(SafeLLamaSamplerChainHandle chain, int i)
        {
            EnsureInitialized();
            return _llama_sampler_chain_remove(chain, i);
        }

        public static IntPtr Llama_SamplerInitGreedy()
        {
            EnsureInitialized();
            return _llama_sampler_init_greedy();
        }

        public static IntPtr Llama_SamplerInitDist(uint seed)
        {
            EnsureInitialized();
            return _llama_sampler_init_dist(seed);
        }

        public static IntPtr Llama_SamplerInitMirostat(int nVocab, uint seed, float tau, float eta, int m)
        {
            EnsureInitialized();
            return _llama_sampler_init_mirostat(nVocab, seed, tau, eta, m);
        }

        public static IntPtr Llama_SamplerInitMirostatV2(uint seed, float tau, float eta)
        {
            EnsureInitialized();
            return _llama_sampler_init_mirostat_v2(seed, tau, eta);
        }

        public static IntPtr Llama_SamplerInitTopK(int k)
        {
            EnsureInitialized();
            return _llama_sampler_init_top_k(k);
        }

        public static IntPtr Llama_SamplerInitTopNSigma(float n)
        {
            EnsureInitialized();
            return _llama_sampler_init_top_n_sigma(n);
        }

        public static IntPtr Llama_SamplerInitTopP(float p, nint min_keep)
        {
            EnsureInitialized();
            return _llama_sampler_init_top_p(p, min_keep);
        }

        public static IntPtr Llama_SamplerInitMinP(float p, nint min_keep)
        {
            EnsureInitialized();
            return _llama_sampler_init_min_p(p, min_keep);
        }

        public static IntPtr Llama_SamplerInitTypical(float p, nint min_keep)
        {
            EnsureInitialized();
            return _llama_sampler_init_typical(p, min_keep);
        }

        public static IntPtr Llama_SamplerInitTemp(float t)
        {
            EnsureInitialized();
            return _llama_sampler_init_temp(t);
        }

        public static IntPtr Llama_SamplerInitTempExt(float t, float delta, float exponent)
        {
            EnsureInitialized();
            return _llama_sampler_init_temp_ext(t, delta, exponent);
        }

        public static IntPtr Llama_SamplerInitXTC(float p, float t, nint minKeep, uint seed)
        {
            EnsureInitialized();
            return _llama_sampler_init_xtc(p, t, minKeep, seed);
        }

        public static unsafe IntPtr Llama_SamplerInitInfill(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_sampler_init_infill(vocab);
        }

        public static unsafe IntPtr Llama_SamplerInitGrammar(LLamaVocabNative* model, string grammar_str, string grammar_root)
        {
            EnsureInitialized();
            return _llama_sampler_init_grammar(model, grammar_str, grammar_root);
        }

        public static unsafe IntPtr Llama_SamplerInitGrammarLazyPatterns(
            LLamaVocabNative* model,
            string grammar_str,
            string grammar_root,
            string[] triggerPatterns,
            LLamaToken[] triggerTokens,
            nuint num_trigger_tokens)
        {
            EnsureInitialized();
            return _llama_sampler_init_grammar_lazy_patterns(model, grammar_str, grammar_root, triggerPatterns, triggerTokens, num_trigger_tokens);
        }

        public static IntPtr Llama_SamplerInitPenalties(
            int penalty_last_n,
            float penalty_repeat,
            float penalty_freq,
            float penalty_present)
        {
            EnsureInitialized();
            return _llama_sampler_init_penalties(penalty_last_n, penalty_repeat, penalty_freq, penalty_present);
        }

        public static unsafe IntPtr Llama_SamplerInitDry(
            LLamaVocabNative* vocab,
            int n_ctx_train,
            float dry_multiplier,
            float dry_base,
            int dry_allowed_length,
            int dry_penalty_last_n,
            byte** seq_breakers,
            nuint num_breakers)
        {
            EnsureInitialized();
            return _llama_sampler_init_dry(vocab, n_ctx_train, dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n, seq_breakers, num_breakers);
        }

        public static unsafe IntPtr Llama_SamplerInitLogitBias(
            int n_vocab,
            int n_logit_bias,
            LLamaLogitBias* logit_bias)
        {
            EnsureInitialized();
            return _llama_sampler_init_logit_bias(n_vocab, n_logit_bias, logit_bias);
        }

        public static unsafe LLamaSampler* Llama_SamplerInit(LLamaSamplerI* iface, IntPtr ctx)
        {
            EnsureInitialized();
            return _llama_sampler_init(iface, ctx);
        }

        public static void Llama_SamplerFree(IntPtr sampler)
        {
            EnsureInitialized();
            _llama_sampler_free(sampler);
        }

        public static void Llama_SamplerChainAdd(SafeLLamaSamplerChainHandle chain, IntPtr smpl)
        {
            EnsureInitialized();
            _llama_sampler_chain_add(chain, smpl);
        }

        public static IntPtr Llama_SamplerChainGet(SafeLLamaSamplerChainHandle chain, int i)
        {
            EnsureInitialized();
            return _llama_sampler_chain_get(chain, i);
        }

        #endregion

    }
}
