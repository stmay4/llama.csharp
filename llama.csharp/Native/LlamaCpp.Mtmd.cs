using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    public static partial class LlamaCpp
    {
        #region MTMD API functions

        #region delegates

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate LlamaMtmdParams mtmd_context_params_default();

        #endregion

        #region functions

        private static mtmd_context_params_default _mtmd_context_params_default;

        #endregion

        #endregion

        private static void LoadMtmdFunctions()
        {
            _mtmd_context_params_default = GetLibFunction<mtmd_context_params_default>(_mtmdHandle, "mtmd_context_params_default");
        }

        public static LlamaMtmdParams Llama_MtmdDefaultParams()
        {
            EnsureMtmdInitialized();
            return _mtmd_context_params_default();
        }
    }
}
