using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class MtmdContext : IDisposable
    {
        public SafeMtmdContextHandle NativeHandle;

        private MtmdContext(SafeMtmdContextHandle nativeHandle)
        {
            NativeHandle = nativeHandle;
        }

        public static MtmdContext CreateFromFile(string mmprojFile, LLamaWeights llamaModel, IMtmdParams @params)
        {
            @params.ToMtmdContextParams(out var nativeParams);
            var weights = SafeMtmdContextHandle.LoadFromFile(mmprojFile, llamaModel.NativeHandle, nativeParams);
            return new MtmdContext(weights);
        }

        // метод регистрации картинок, видео и звука на энкод, на вход спаны не битмапы



        public void Dispose()
        {
            NativeHandle.Dispose();
        }
    }
}
