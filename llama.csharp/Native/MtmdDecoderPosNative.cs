using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// C++ mtmd_decoder_pos
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal struct MtmdDecoderPosNative
    {
        uint t;
        uint x;
        uint y;
        uint z; // unused for now, reserved for future use
    }
}
