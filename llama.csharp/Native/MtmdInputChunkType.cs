using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    /// <summary>
    /// C++ : mtmd_input_chunk_type
    /// </summary>
    internal enum MtmdInputChunkType
    {
        MTMD_INPUT_CHUNK_TYPE_TEXT = 0,
        MTMD_INPUT_CHUNK_TYPE_IMAGE = 1,
        MTMD_INPUT_CHUNK_TYPE_AUDIO = 2,
    }
}
