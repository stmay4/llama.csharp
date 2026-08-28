namespace Llama.csharp.Native
{
    /// <summary>
    /// An embeddings batch allows submitting embeddings to multiple sequences simultaneously.
    /// This class is strictly immutable after initialization.
    /// </summary>
    public class LLamaBatchEmbeddings
    {
        private readonly byte[] _logits;
        private readonly float[] _embeddings;
        private readonly LLamaPos[] _positions;
        private readonly int[] _sequenceIdCount;
        private readonly LLamaSeqId[][] _sequenceIds;
        private readonly nint[] _sequenceIdsPtrs;

        public int EmbeddingsCount;

        /// <summary>
        /// Create a new batch fully initialized with all data.
        /// </summary>
        /// <param name="embeddingDimensions">Size of a single embedding vector.</param>
        /// <param name="embeddings">Flat array of all embeddings. Length must be embeddingsCount * embeddingDimensions.</param>
        /// <param name="positions">Array of positions for each embedding. Length must be embeddingsCount.</param>
        /// <param name="nPosPerEmbedding">Count of pos per one embedding (for mrope)</param>
        /// <param name="sequenceIds">Jagged array of sequence IDs for each embedding. Length must be embeddingsCount.</param>
        /// <param name="generateLogits">Array indicating whether to generate logits for each embedding. Length must be embeddingsCount.</param>
        /// <exception cref="ArgumentNullException"></exception>
        /// <exception cref="ArgumentException"></exception>
        /// <exception cref="Exception"></exception>
        public LLamaBatchEmbeddings(
            int embeddingDimensions,
            float[] embeddings,
            LLamaPos[] positions,
            int nPosPerEmbedding,
            LLamaSeqId[][] sequenceIds,
            bool[] generateLogits)
        {
            if (embeddings == null) throw new ArgumentNullException(nameof(embeddings));
            if (positions == null) throw new ArgumentNullException(nameof(positions));
            if (sequenceIds == null) throw new ArgumentNullException(nameof(sequenceIds));
            if (generateLogits == null) throw new ArgumentNullException(nameof(generateLogits));

            if (positions.Length % nPosPerEmbedding != 0)
                throw new ArgumentException($"positions.Length ({positions.Length}) must be a multiple of nPosPerEmbedding ({nPosPerEmbedding})", nameof(positions));

            int count = positions.Length / nPosPerEmbedding;
            if (count == 0)
            {
                throw new Exception("empty batch exception");
            }

            if (embeddings.Length != count * embeddingDimensions)
                throw new ArgumentException($"Embeddings array length must be exactly embeddingsCount * embeddingDimensions ({count} * {embeddingDimensions} = {count * embeddingDimensions}), but was {embeddings.Length}.", nameof(embeddings));
            if (sequenceIds.Length != count)
                throw new ArgumentException($"SequenceIds array length must match positions length ({count}).", nameof(sequenceIds));
            if (generateLogits.Length != count)
                throw new ArgumentException($"GenerateLogits array length must match positions length ({count}).", nameof(generateLogits));

            // Direct assignment or safe initialization of fields
            _embeddings = embeddings;
            _positions = positions;
            _sequenceIds = sequenceIds;

            EmbeddingsCount = count;

            _sequenceIdCount = new int[count];
            _logits = new byte[count];
            _sequenceIdsPtrs = new nint[count];

            for (int i = 0; i < count; i++)
            {
                _sequenceIdCount[i] = sequenceIds[i].Length;
                _logits[i] = generateLogits[i] ? (byte)1 : (byte)0;
            }
        }

        /// <summary>
        /// ToNativeBatch convert
        /// </summary>
        /// <param name="batch"></param>
        /// <returns></returns>
        internal GroupDisposable ToNativeBatch(out LLamaNativeBatch batch)
        {
            // This group holds all of the memory pins
            var group = new GroupDisposable();

            unsafe
            {
                batch = new LLamaNativeBatch
                {
                    n_tokens = EmbeddingsCount,
                    logits = (byte*)group.Add(_logits.AsMemory().Pin()).Pointer,

                    n_seq_id = (int*)group.Add(_sequenceIdCount.AsMemory().Pin()).Pointer,
                    pos = (LLamaPos*)group.Add(_positions.AsMemory().Pin()).Pointer,
                    seq_id = (LLamaSeqId**)group.Add(_sequenceIdsPtrs.AsMemory().Pin()).Pointer,

                    embd = (float*)group.Add(_embeddings.AsMemory().Pin()).Pointer,
                    token = null,
                };

                // Create pointers to each of the arrays in turns
                for (var i = 0; i < _sequenceIdsPtrs.Length; i++)
                    _sequenceIdsPtrs[i] = (nint)group.Add(_sequenceIds[i].AsMemory().Pin()).Pointer;
            }

            return group;
        }
    }
}