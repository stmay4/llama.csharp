using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public readonly record struct MtmdSpec (bool UseNonCausal, bool UseMrope);
}
