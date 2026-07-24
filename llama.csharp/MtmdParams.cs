using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class MtmdParams : IMtmdParams
    {
        public bool UseGpu { get; init; } = false;

        public bool PrintTimings { get; init; } = false;

        public int? Threads { get; init; } = null;

        public nint? MediaMarker { get; init; } = null;

        public LlamaFlashAttentionType? FlashAttention { get; init; } = null;

        public bool? Warmup { get; init; } = null;

        public int? ImageMinTokens { get; init; } = null;

        public int? ImageMaxTokens { get; init; } = null;

        public int? BatchSize { get; init; } = null;
    }
}
