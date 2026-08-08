namespace Llama.csharp.Native
{
    internal class SafeMtmdBatchHandle : SafeLLamaHandleBase
    {
        private int _embeddingSize;
        private SafeMtmdBatchHandle() { }
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Mtmd_BatchFree(handle);
            return true;
        }

        internal static SafeMtmdBatchHandle Init(SafeMtmdContextHandle ctx)
        {
            SafeMtmdBatchHandle batch = LlamaCpp.Mtmd_BatchInit(ctx);
            batch._embeddingSize = ctx.EmbeddingSize;
            return batch;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="chunk"></param>
        /// <returns></returns>
        internal unsafe int AddChunk(MtmdInputChunkPtr* chunk)
        {
            return LlamaCpp.Mtmd_BatchAddChunk(this, chunk);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        internal int Encode()
        {
            return LlamaCpp.Mtmd_BatchEncode(this);
        }

        internal unsafe LlamaEmbedding[] GetChunkEmbeddings(MtmdInputChunkPtr* chunk, LlamaEmbeddingType chunkType)
        {
            // Получаем данные чанка: количестов токенов
            int n_tokens = (int)LlamaCpp.Mtmd_InputChunkGetNTokens(chunk);

            int totalFloats = n_tokens * _embeddingSize;

            // Выделяем управляемый непрерывный массив
            float[] embeddings = new float[totalFloats];

            // Получаем указатель на нативные данные
            float* nativeData = LlamaCpp.Mtmd_BatchGetOutputEmbed(this, chunk);

            // Копируем из нативной памяти в управляемую
            fixed (float* managedPtr = embeddings)
            {
                Buffer.MemoryCopy(nativeData, managedPtr,
                                  totalFloats * sizeof(float),
                                  totalFloats * sizeof(float));
            }

            // Создаем запись Memory о данных
            Memory<float> embedsMemory = embeddings.AsMemory();

            // Создаем результирующий массив длиной с количество эмбеддингов, равное колву входных токенов. Структура LlamaEmbedding содержит запись Memory и Тип эмбеддинга
            LlamaEmbedding[] result = new LlamaEmbedding[n_tokens];

            for (int i = 0; i < n_tokens; i++) 
            {
                result[i] = new LlamaEmbedding(embedsMemory.Slice(i * _embeddingSize, _embeddingSize), chunkType);
            }

            return result;
        }
    }
}
