using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class NoAllocModelInfo
    {
        public required IReadOnlyDictionary<string, string> Metadata { get; init; }
        public required int ContextSize { get; init; }
    }
}
