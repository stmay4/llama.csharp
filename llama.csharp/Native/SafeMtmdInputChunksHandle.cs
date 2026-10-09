using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    // Initialized as empty, then passed to tokenization where it gets populated
    // and returned with text and media chunks. Text chunks surround media chunks
    // and represent opening and closing text tokens.
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

        /// <summary>
        /// Initializes an empty container that gets populated with chunks on the C++ side
        /// </summary>
        /// <returns></returns>
        internal static SafeMtmdInputChunksHandle Init()
        {
            return LlamaCpp.Mtmd_InputChunksInit();
        }

        /// <summary>
        /// Returns the number of chunks in the container populated on the C++ side
        /// </summary>
        /// <returns></returns>
        internal int GetChunkCount()
        {
            return (int)LlamaCpp.Mtmd_InputChunksSize(this);
        }

        /// <summary>
        /// Returns a reference to a chunk
        /// </summary>
        /// <param name="id"></param>
        /// <returns></returns>
        internal unsafe MtmdInputChunkPtr* GetChunk(int id)
        {
            return LlamaCpp.Mtmd_InputChunksGet(this, (nuint)id);
        }
    }
}
