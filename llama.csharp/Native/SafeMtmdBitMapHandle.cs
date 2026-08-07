using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    /// <summary>
    /// mtmd_bitmap
    ///
    /// if bitmap is image:
    ///     length of data must be nx * ny * 3
    ///     the data is in RGBRGBRGB... format
    ///     note: some video-capable models (i.e. qwen-vl) can merge consecutive bitmaps
    ///           into one chunk, mtmd_tokenize() will automatically handle this
    /// if bitmap is audio:
    ///     length of data must be n_samples * sizeof(float)
    ///     the data is in float format (PCM F32)
    ///
    /// if data == nullptr:
    ///     the bitmap is considered "empty", and will be treated as a placeholder for counting tokens
    ///     you can pass the bitmap via mtmd_tokenize(), then call mtmd_*_get_n_tokens() to count the tokens
    ///     note: passing a placeholder bitmap to mtmd_encode() will return an error
    /// </summary>
    internal class SafeMtmdBitMapHandle : SafeLLamaHandleBase
    {
        private SafeMtmdBitMapHandle() { }
        public static SafeMtmdBitMapHandle InitFromImage(uint nx, uint ny, Span<byte> image)
        {
            return LlamaCpp.Mtmd_BitMapInitFromImage(nx, ny, image);
        }

        public static SafeMtmdBitMapHandle InitFromAudio(nuint n_samples, Span<float> audio)
        {
            return LlamaCpp.Mtmd_BitMapInitFromAudio(n_samples, audio);
        }
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Mtmd_BitMapFree(handle);
            return true;
        }
    }
}
