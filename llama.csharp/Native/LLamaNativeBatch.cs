using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// A llama_batch object can contain input about one or many sequences
    /// The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    ///
    /// - token  : the token ids of the input (used when embd is NULL)
    /// - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    /// - pos    : the positions of the respective token in the sequence
    ///            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    /// - seq_id : the sequence to which the respective token belongs
    ///            (if set to NULL, the sequence ID will be assumed to be 0)
    /// - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    ///            (if set to NULL:
    ///               - if embeddings: all tokens are output
    ///               - if not:        only the last token is output
    ///            )
    ///
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct LLamaNativeBatch
    {
        /// <summary>
        /// The number of items pointed at by pos, seq_id and logits.
        /// </summary>
        public int n_tokens;

        /// <summary>
        /// Either `n_tokens` of `llama_token`, or `NULL`, depending on how this batch was created
        /// the token ids of the input (used when embd is NULL)
        /// </summary>
        public LLamaToken* token;

        /// <summary>
        /// Either `n_tokens * embd * sizeof(float)` or `NULL`, depending on how this batch was created
        /// token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
        /// </summary>
        public float* embd;

        /// <summary>
        /// the positions of the respective token in the sequence
        /// (if set to NULL, the token position will be tracked automatically by llama_decode)
        /// </summary>
        public LLamaPos* pos;

        /// <summary>
        /// the sequence to which the respective token belongs
        /// (if set to NULL, the sequence ID will be assumed to be 0)
        /// </summary>
        public int* n_seq_id;

        /// <summary>
        /// the sequence to which the respective token belongs
        /// (if set to NULL, the sequence ID will be assumed to be 0)
        /// </summary>
        public LLamaSeqId** seq_id;

        /// <summary>
        /// if zero, the logits (and/or the embeddings) for the respective token will not be output
        /// (if set to NULL, only the logits for last token will be returned)
        /// </summary>
        public byte* logits;
    }
}