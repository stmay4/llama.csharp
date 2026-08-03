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
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Mtmd_InputChunksFree(handle); 
            return true;
        }
    }
}
