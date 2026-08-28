using System.Runtime.InteropServices;

namespace Llama.csharp.Native
{
    /// <summary>
    /// struct mtmd_input_text {
    /// const char* text;
    /// bool add_special;
    /// bool parse_special;
    /// };
    /// 
    /// Used for retrieving chunks; image markers are inserted here by the code. Not used by library users.
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
