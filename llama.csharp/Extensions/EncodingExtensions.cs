using System.Text;

namespace Llama.csharp.Extensions
{
    internal static class EncodingExtensions
    {
        internal static int GetBytesImpl(Encoding encoding, ReadOnlySpan<char> chars, Span<byte> output)
        {
            if (chars.Length == 0)
                return 0;

            unsafe
            {
                fixed (char* charPtr = chars)
                fixed (byte* bytePtr = output)
                {
                    return encoding.GetBytes(charPtr, chars.Length, bytePtr, output.Length);
                }
            }
        }

        internal static int GetCharsImpl(Encoding encoding, ReadOnlySpan<byte> bytes, Span<char> output)
        {
            if (bytes.Length == 0)
                return 0;

            unsafe
            {
                fixed (byte* bytePtr = bytes)
                fixed (char* charPtr = output)
                {
                    return encoding.GetChars(bytePtr, bytes.Length, charPtr, output.Length);
                }
            }
        }

        internal static int GetCharCountImpl(Encoding encoding, ReadOnlySpan<byte> bytes)
        {
            if (bytes.Length == 0)
                return 0;

            unsafe
            {
                fixed (byte* bytePtr = bytes)
                {
                    return encoding.GetCharCount(bytePtr, bytes.Length);
                }
            }
        }

        internal static string GetStringFromSpan(this Encoding encoding, ReadOnlySpan<byte> bytes)
        {
            unsafe
            {
                fixed (byte* bytesPtr = bytes)
                {
                    return encoding.GetString(bytesPtr, bytes.Length);
                }
            }
        }
    }
}
