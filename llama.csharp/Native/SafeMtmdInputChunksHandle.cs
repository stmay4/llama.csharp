using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    // инициализируется пустым, потом передается в токенизацию, там заполняется и обратно уже вовзращается с чанками текстовыми и медиа, текстовые окружают медиа и представляют собой открывающую и закрывающие токены текстовые
    // проверить при free одного из чанков, размер данного листа чанков уменьшится или нет?
    /// <summary>
    /// mtmd_input_chunks
    ///
    /// this is simply a list of mtmd_input_chunk
    /// the elements can only be populated via mtmd_tokenize()
    /// </summary>
    internal class SafeMtmdInputChunksHandle : SafeLLamaHandleBase
    {
        private SafeMtmdInputChunksHandle() { }
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Mtmd_InputChunksFree(handle); 
            return true;
        }

        internal static SafeMtmdInputChunksHandle Init()
        {
            return LlamaCpp.Mtmd_InputChunksInit();
        }

        internal int GetChunkCount()
        {
            return (int)LlamaCpp.Mtmd_InputChunksSize(this);
        }

        internal unsafe MtmdInputChunkPtr* GetChunk(int id)
        {
            return LlamaCpp.Mtmd_InputChunksGet(this, (nuint)id);
        }
    }
}
