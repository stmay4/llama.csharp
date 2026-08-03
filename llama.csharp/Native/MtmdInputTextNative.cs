using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// struct mtmd_input_text {
    /// const char* text;
    /// bool add_special;
    /// bool parse_special;
    /// };
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    internal unsafe struct MtmdInputTextNative
    {
        
        byte* text;

        sbyte add_special = 1;

        sbyte parse_special = 1;

        public MtmdInputTextNative(byte* text)
        {
            this.text = text;
        }
    }
}
