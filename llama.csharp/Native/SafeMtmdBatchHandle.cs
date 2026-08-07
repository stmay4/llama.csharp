using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata;
using System.Text;
using System.Threading.Tasks;

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

        internal unsafe Memory<float> GetChunkEmbeddings(MtmdInputChunkPtr* chunk)
        {
            nuint n_tokens = LlamaCpp.mtmd_input_chunk_get_n_tokens(chunk);
            int totalFloats = (int)n_tokens * _embeddingSize;

            // Выделяем управляемый массив
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

            return new Memory<float>(embeddings);
        }
    }
}
