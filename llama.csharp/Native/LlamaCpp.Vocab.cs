using CommunityToolkit.HighPerformance.Buffers;
using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    public static partial class LlamaCpp
    {

        #region LLAMA API functions

        #region delegates

        /// <summary>
        /// Total number of tokens in this vocabulary
        /// </summary>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_vocab_n_tokens(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaVocabType llama_vocab_type(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate byte* llama_vocab_get_text(LLamaVocabNative* vocab, LLamaToken token);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float llama_vocab_get_score(LLamaVocabNative* vocab, LLamaToken token);

        /// <summary>
        /// Get attributes for a specific token
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="token"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaTokenAttr llama_vocab_get_attr(LLamaVocabNative* vocab, LLamaToken token);

        /// <summary>
        /// Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="token"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private unsafe delegate bool llama_vocab_is_eog(LLamaVocabNative* vocab, LLamaToken token);


        /// <summary>
        /// Identify if Token Id is a control token or a render-able token
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="token"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private unsafe delegate bool llama_vocab_is_control(LLamaVocabNative* vocab, LLamaToken token);

        /// <summary>
        /// beginning-of-sentence
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_bos(LLamaVocabNative* vocab);

        /// <summary>
        /// end-of-sentence
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_eos(LLamaVocabNative* vocab);

        /// <summary>
        /// end-of-turn
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_eot(LLamaVocabNative* vocab);

        /// <summary>
        /// sentence separator
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_sep(LLamaVocabNative* vocab);

        /// <summary>
        /// next-line
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_nl(LLamaVocabNative* vocab);

        /// <summary>
        /// padding
        /// </summary>
        /// <param name="vocab"></param>
        /// <returns></returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_pad(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_pre(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_suf(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_mid(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_pad(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_rep(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken llama_vocab_fim_sep(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private unsafe delegate bool llama_vocab_get_add_bos(LLamaVocabNative* vocab);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private unsafe delegate bool llama_vocab_get_add_eos(LLamaVocabNative* vocab);

        /// <summary>
        /// Convert text into tokens
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="text"></param>
        /// <param name="text_len"></param>
        /// <param name="tokens">The tokens pointer must be large enough to hold the resulting tokens.</param>
        /// <param name="n_max_tokens"></param>
        /// <param name="add_special">add_special Allow to add BOS and EOS tokens if model is configured to do so.</param>
        /// <param name="parse_special">Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext. Does not insert a leading space.</param>
        /// <returns>Returns the number of tokens on success, no more than n_max_tokens.
        /// Returns a negative number on failure - the number of tokens that would have been returned
        /// </returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_tokenize(LLamaVocabNative* vocab, byte* text, int text_len, LLamaToken* tokens, int n_max_tokens, [MarshalAs(UnmanagedType.U1)] bool add_special, [MarshalAs(UnmanagedType.U1)] bool parse_special);

        /// <summary>
        /// Convert the provided tokens into text (inverse of llama_tokenize()).
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="tokens"></param>
        /// <param name="nTokens"></param>
        /// <param name="textOut">The char pointer must be large enough to hold the resulting text.</param>
        /// <param name="textLengthMax"></param>
        /// <param name="removeSpecial">remove_special Allow to remove BOS and EOS tokens if model is configured to do so.</param>
        /// <param name="unparseSpecial">unparse_special If true, special tokens are rendered in the output.</param>
        /// <returns>Returns the number of chars/bytes on success, no more than textLengthMax. Returns a negative number on failure - the number of chars/bytes that would have been returned.</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_detokenize(LLamaVocabNative* vocab, LLamaToken* tokens, int nTokens, byte* textOut, int textLengthMax, [MarshalAs(UnmanagedType.U1)] bool removeSpecial, [MarshalAs(UnmanagedType.U1)] bool unparseSpecial);

        /// <summary>
        /// Convert a single token into text
        /// </summary>
        /// <param name="vocab"></param>
        /// <param name="llamaToken"></param>
        /// <param name="buffer">buffer to write string into</param>
        /// <param name="lstrip">User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')</param>
        /// <param name="special">If true, special tokens are rendered in the output</param>
        /// <returns>The length written, or if the buffer is too small a negative that indicates the length required</returns>
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int llama_token_to_piece(LLamaVocabNative* vocab, LLamaToken llamaToken, byte* buffer, int length, int lstrip, [MarshalAs(UnmanagedType.U1)] bool special);


        #endregion

        #region functions

        private static llama_vocab_n_tokens _llama_vocab_n_tokens;
        private static llama_vocab_type _llama_vocab_type;
        private static llama_vocab_get_text _llama_vocab_get_text;
        private static llama_vocab_get_score _llama_vocab_get_score;
        private static llama_vocab_get_attr _llama_vocab_get_attr;
        private static llama_vocab_is_eog _llama_vocab_is_eog;
        private static llama_vocab_is_control _llama_vocab_is_control;
        private static llama_vocab_bos _llama_vocab_bos;
        private static llama_vocab_eos _llama_vocab_eos;
        private static llama_vocab_eot _llama_vocab_eot;
        private static llama_vocab_sep _llama_vocab_sep;
        private static llama_vocab_nl _llama_vocab_nl;
        private static llama_vocab_pad _llama_vocab_pad;
        private static llama_vocab_fim_pre _llama_vocab_fim_pre;
        private static llama_vocab_fim_suf _llama_vocab_fim_suf;
        private static llama_vocab_fim_mid _llama_vocab_fim_mid;
        private static llama_vocab_fim_pad _llama_vocab_fim_pad;
        private static llama_vocab_fim_rep _llama_vocab_fim_rep;
        private static llama_vocab_fim_sep _llama_vocab_fim_sep;
        private static llama_vocab_get_add_bos _llama_vocab_get_add_bos;
        private static llama_vocab_get_add_eos _llama_vocab_get_add_eos;
        private static llama_tokenize _llama_tokenize;
        private static llama_detokenize _llama_detokenize;
        private static llama_token_to_piece _llama_token_to_piece;

        #endregion
        #endregion

        #region GGML API functions

        #region delegates
        #endregion
        #region functions
        #endregion
        #endregion

        private static void LoadVocabFunctions()
        {
            _llama_vocab_n_tokens = GetLibFunction<llama_vocab_n_tokens>(_llamaHandle, "llama_vocab_n_tokens");
            _llama_vocab_type = GetLibFunction<llama_vocab_type>(_llamaHandle, "llama_vocab_type");
            _llama_vocab_get_text = GetLibFunction<llama_vocab_get_text>(_llamaHandle, "llama_vocab_get_text");
            _llama_vocab_get_score = GetLibFunction<llama_vocab_get_score>(_llamaHandle, "llama_vocab_get_score");
            _llama_vocab_get_attr = GetLibFunction<llama_vocab_get_attr>(_llamaHandle, "llama_vocab_get_attr");
            _llama_vocab_is_eog = GetLibFunction<llama_vocab_is_eog>(_llamaHandle, "llama_vocab_is_eog");
            _llama_vocab_is_control = GetLibFunction<llama_vocab_is_control>(_llamaHandle, "llama_vocab_is_control");
            _llama_vocab_bos = GetLibFunction<llama_vocab_bos>(_llamaHandle, "llama_vocab_bos");
            _llama_vocab_eos = GetLibFunction<llama_vocab_eos>(_llamaHandle, "llama_vocab_eos");
            _llama_vocab_eot = GetLibFunction<llama_vocab_eot>(_llamaHandle, "llama_vocab_eot");
            _llama_vocab_sep = GetLibFunction<llama_vocab_sep>(_llamaHandle, "llama_vocab_sep");
            _llama_vocab_nl = GetLibFunction<llama_vocab_nl>(_llamaHandle, "llama_vocab_nl");
            _llama_vocab_pad = GetLibFunction<llama_vocab_pad>(_llamaHandle, "llama_vocab_pad");
            _llama_vocab_fim_pre = GetLibFunction<llama_vocab_fim_pre>(_llamaHandle, "llama_vocab_fim_pre");
            _llama_vocab_fim_suf = GetLibFunction<llama_vocab_fim_suf>(_llamaHandle, "llama_vocab_fim_suf");
            _llama_vocab_fim_mid = GetLibFunction<llama_vocab_fim_mid>(_llamaHandle, "llama_vocab_fim_mid");
            _llama_vocab_fim_pad = GetLibFunction<llama_vocab_fim_pad>(_llamaHandle, "llama_vocab_fim_pad");
            _llama_vocab_fim_rep = GetLibFunction<llama_vocab_fim_rep>(_llamaHandle, "llama_vocab_fim_rep");
            _llama_vocab_fim_sep = GetLibFunction<llama_vocab_fim_sep>(_llamaHandle, "llama_vocab_fim_sep");
            _llama_vocab_get_add_bos = GetLibFunction<llama_vocab_get_add_bos>(_llamaHandle, "llama_vocab_get_add_bos");
            _llama_vocab_get_add_eos = GetLibFunction<llama_vocab_get_add_eos>(_llamaHandle, "llama_vocab_get_add_eos");
            _llama_tokenize = GetLibFunction<llama_tokenize>(_llamaHandle, "llama_tokenize");
            _llama_detokenize = GetLibFunction<llama_detokenize>(_llamaHandle, "llama_detokenize");
            _llama_token_to_piece = GetLibFunction<llama_token_to_piece>(_llamaHandle, "llama_token_to_piece");
        }

        public static unsafe int Llama_VocabNTokens(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_n_tokens(vocab);
        }

        public static unsafe LLamaVocabType Llama_VocabType(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_type(vocab);
        }

        public static unsafe byte* Llama_VocabGetText(LLamaVocabNative* vocab, LLamaToken token)
        {
            EnsureInitialized();
            return _llama_vocab_get_text(vocab, token);
        }

        public static unsafe float Llama_VocabGetScore(LLamaVocabNative* vocab, LLamaToken token)
        {
            EnsureInitialized();
            return _llama_vocab_get_score(vocab, token);
        }

        public static unsafe LLamaTokenAttr Llama_VocabGetAttr(LLamaVocabNative* vocab, LLamaToken token)
        {
            EnsureInitialized();
            return _llama_vocab_get_attr(vocab, token);
        }

        public static unsafe bool Llama_VocabIsEog(LLamaVocabNative* vocab, LLamaToken token)
        {
            EnsureInitialized();
            return _llama_vocab_is_eog(vocab, token);
        }

        public static unsafe bool Llama_VocabIsControl(LLamaVocabNative* vocab, LLamaToken token)
        {
            EnsureInitialized();
            return _llama_vocab_is_control(vocab, token);
        }

        public static unsafe LLamaToken Llama_VocabBos(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_bos(vocab);
        }

        public static unsafe LLamaToken Llama_VocabEos(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_eos(vocab);
        }

        public static unsafe LLamaToken Llama_VocabEot(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_eot(vocab);
        }

        public static unsafe LLamaToken Llama_VocabSep(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_sep(vocab);
        }

        public static unsafe LLamaToken Llama_VocabNl(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_nl(vocab);
        }

        public static unsafe LLamaToken Llama_VocabPad(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_pad(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimPre(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_pre(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimSuf(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_suf(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimMid(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_mid(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimPad(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_pad(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimRep(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_rep(vocab);
        }

        public static unsafe LLamaToken Llama_VocabFimSep(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_fim_sep(vocab);
        }

        public static unsafe bool Llama_VocabGetAddBos(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_get_add_bos(vocab);
        }

        public static unsafe bool Llama_VocabGetAddEos(LLamaVocabNative* vocab)
        {
            EnsureInitialized();
            return _llama_vocab_get_add_eos(vocab);
        }

        public static unsafe int Llama_VocabTokenize(LLamaVocabNative* vocab, SpanOwner<byte> text, int text_len, LLamaToken[]? tokens, int n_max_tokens, bool add_special, bool parse_special)
        {
            EnsureInitialized();
            unsafe
            {
                fixed(byte* textPtr = text.Span)
                fixed (LLamaToken* tokensPtr = tokens)
                {
                    return _llama_tokenize(vocab, textPtr, text_len, tokensPtr, n_max_tokens, add_special, parse_special);
                }
            }
            
        }

        public static unsafe int Llama_VocabDetokenize(LLamaVocabNative* vocab, LLamaToken* tokens, int nTokens, byte* textOut, int textLengthMax, bool removeSpecial, bool unparseSpecial)
        {
            EnsureInitialized();
            return _llama_detokenize(vocab, tokens, nTokens, textOut, textLengthMax, removeSpecial, unparseSpecial);
        }

        public static unsafe int Llama_VocabTokenToPiece(LLamaVocabNative* vocab, LLamaToken llamaToken, Span<byte> buffer, int lstrip, bool special)
        {
            // Handle invalid tokens
            if ((int)llamaToken < 0)
                return 0;

            unsafe
            {
                fixed (byte* bufferPtr = buffer)
                {
                    return _llama_token_to_piece(vocab, llamaToken, bufferPtr, buffer.Length, lstrip, special);
                }
            }
        }
    }
}
