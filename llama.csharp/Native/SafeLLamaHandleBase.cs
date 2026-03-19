using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    public abstract class SafeLLamaHandleBase
        : SafeHandle
    {
        private protected SafeLLamaHandleBase()
            : base(nint.Zero, ownsHandle: true)
        {
        }

        private protected SafeLLamaHandleBase(nint handle, bool ownsHandle)
            : base(nint.Zero, ownsHandle)
        {
            SetHandle(handle);
        }

        /// <inheritdoc />
        public override bool IsInvalid => handle == nint.Zero;

        /// <inheritdoc />
        public override string ToString() => handle.ToString();
    }
}
