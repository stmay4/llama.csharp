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
        private delegate MtmdContextParams mtmd_context_params_default();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeMtmdContextHandle mtmd_init_from_file(string mmproj_fname,
                                            SafeLlamaModelHandle text_model,
                                            MtmdContextParams ctx_params);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void mtmd_free(IntPtr ctx);

        // whether we need to set non-causal mask before llama_decode
        // if chunk is nullptr, we assume the default case where chunk is an image chunk
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool mtmd_decode_use_non_causal(SafeMtmdContextHandle ctx, IntPtr chunk);

        // whether the current model use M-RoPE for llama_decode
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool mtmd_decode_use_mrope(SafeMtmdContextHandle ctx);

        // whether the current model supports vision input
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool mtmd_support_vision(SafeMtmdContextHandle ctx);

        // whether the current model supports audio input
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool mtmd_support_audio(SafeMtmdContextHandle ctx);

        // get audio sample rate in Hz, for example 16000 for Whisper
        // return -1 if audio is not supported
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int mtmd_get_audio_sample_rate(SafeMtmdContextHandle ctx);

        // get the current marker string
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate string mtmd_get_marker(SafeMtmdContextHandle ctx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void mtmd_input_chunk_free(IntPtr chunk);

        #region bitmap

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate SafeMtmdBitMapHandle mtmd_bitmap_init(uint nx, uint ny, byte* data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate SafeMtmdBitMapHandle mtmd_bitmap_init_from_audio(nuint n_samples, float* data);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint mtmd_bitmap_get_nx(SafeMtmdBitMapHandle bitmap);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate uint mtmd_bitmap_get_ny(SafeMtmdBitMapHandle bitmap);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint mtmd_bitmap_get_n_bytes(SafeMtmdBitMapHandle bitmap);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.U1)]
        private delegate bool mtmd_bitmap_is_audio(SafeMtmdBitMapHandle bitmap);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void mtmd_bitmap_free(IntPtr bitmap);

        #endregion

        #region chunks

        // mtmd_input_chunks
        //
        // this is simply a list of mtmd_input_chunk
        // the elements can only be populated via mtmd_tokenize()

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeMtmdInputChunksHandle mtmd_input_chunks_init();

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate nuint mtmd_input_chunks_size(SafeMtmdInputChunksHandle chunks);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate MtmdInputChunkPtr* mtmd_input_chunks_get(SafeMtmdInputChunksHandle chunks, nuint idx);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void mtmd_input_chunks_free(IntPtr chunks);

        #endregion

        #endregion

        #region functions

        private static mtmd_context_params_default _mtmd_context_params_default;
        private static mtmd_init_from_file _mtmd_init_from_file;

        private static mtmd_decode_use_non_causal _mtmd_decode_use_non_causal;
        private static mtmd_decode_use_mrope _mtmd_decode_use_mrope;
        private static mtmd_support_vision _mtmd_support_vision;
        private static mtmd_support_audio _mtmd_support_audio;
        private static mtmd_get_audio_sample_rate _mtmd_get_audio_sample_rate;
        private static mtmd_get_marker _mtmd_get_marker;

        #region bitmap

        private static mtmd_bitmap_init _mtmd_bitmap_init;
        private static mtmd_bitmap_init_from_audio _mtmd_bitmap_init_from_audio;
        private static mtmd_bitmap_get_nx _mtmd_bitmap_get_nx;
        private static mtmd_bitmap_get_ny _mtmd_bitmap_get_ny;
        private static mtmd_bitmap_get_n_bytes _mtmd_bitmap_get_n_bytes;
        private static mtmd_bitmap_is_audio _mtmd_bitmap_is_audio;
        private static mtmd_bitmap_free _mtmd_bitmap_free;

        #endregion

        #region chunks

        private static mtmd_input_chunks_init _mtmd_input_chunks_init;
        private static mtmd_input_chunks_size _mtmd_input_chunks_size;
        private static mtmd_input_chunks_get _mtmd_input_chunks_get;
        private static mtmd_input_chunks_free _mtmd_input_chunks_free;

        #endregion

        private static mtmd_free _mtmd_free;
        private static mtmd_input_chunk_free _mtmd_input_chunk_free;

        #endregion

        #endregion

        private static void LoadMtmdFunctions()
        {
            _mtmd_context_params_default = GetLibFunction<mtmd_context_params_default>(_mtmdHandle, "mtmd_context_params_default");
            _mtmd_init_from_file = GetLibFunction<mtmd_init_from_file>(_mtmdHandle, "mtmd_init_from_file");

            _mtmd_decode_use_non_causal = GetLibFunction<mtmd_decode_use_non_causal>(_mtmdHandle, "mtmd_decode_use_non_causal");
            _mtmd_decode_use_mrope = GetLibFunction<mtmd_decode_use_mrope>(_mtmdHandle, "mtmd_decode_use_mrope");
            _mtmd_support_vision = GetLibFunction<mtmd_support_vision>(_mtmdHandle, "mtmd_support_vision");
            _mtmd_support_audio = GetLibFunction<mtmd_support_audio>(_mtmdHandle, "mtmd_support_audio");
            _mtmd_get_audio_sample_rate = GetLibFunction<mtmd_get_audio_sample_rate>(_mtmdHandle, "mtmd_get_audio_sample_rate");
            _mtmd_get_marker = GetLibFunction<mtmd_get_marker>(_mtmdHandle, "mtmd_get_marker");

            #region bitmap

            _mtmd_bitmap_init = GetLibFunction<mtmd_bitmap_init>(_mtmdHandle, "mtmd_bitmap_init");
            _mtmd_bitmap_init_from_audio = GetLibFunction<mtmd_bitmap_init_from_audio>(_mtmdHandle, "mtmd_bitmap_init_from_audio");
            _mtmd_bitmap_get_nx = GetLibFunction<mtmd_bitmap_get_nx>(_mtmdHandle, "mtmd_bitmap_get_nx");
            _mtmd_bitmap_get_ny = GetLibFunction<mtmd_bitmap_get_ny>(_mtmdHandle, "mtmd_bitmap_get_ny");
            _mtmd_bitmap_get_n_bytes = GetLibFunction<mtmd_bitmap_get_n_bytes>(_mtmdHandle, "mtmd_bitmap_get_n_bytes");
            _mtmd_bitmap_is_audio = GetLibFunction<mtmd_bitmap_is_audio>(_mtmdHandle, "mtmd_bitmap_is_audio");
            _mtmd_bitmap_free = GetLibFunction<mtmd_bitmap_free>(_mtmdHandle, "mtmd_bitmap_free");

            #endregion

            #region chunks

            _mtmd_input_chunks_init = GetLibFunction<mtmd_input_chunks_init>(_mtmdHandle, "mtmd_input_chunks_init");
            _mtmd_input_chunks_size = GetLibFunction<mtmd_input_chunks_size>(_mtmdHandle, "mtmd_input_chunks_size");
            _mtmd_input_chunks_get = GetLibFunction<mtmd_input_chunks_get>(_mtmdHandle, "mtmd_input_chunks_get");
            _mtmd_input_chunks_free = GetLibFunction<mtmd_input_chunks_free>(_mtmdHandle, "mtmd_input_chunks_free");

            #endregion

            _mtmd_free = GetLibFunction<mtmd_free>(_mtmdHandle, "mtmd_free");
            _mtmd_input_chunk_free = GetLibFunction<mtmd_input_chunk_free>(_mtmdHandle, "mtmd_input_chunk_free");
        }

        internal static MtmdContextParams Mtmd_DefaultContextParams()
        {
            EnsureMtmdInitialized();
            return _mtmd_context_params_default();
        }

        internal static SafeMtmdContextHandle Mtmd_InitFromFile(string mmproj_fname, SafeLlamaModelHandle text_model, MtmdContextParams ctx_params)
        {
            EnsureMtmdInitialized();
            return _mtmd_init_from_file(mmproj_fname, text_model, ctx_params);
        }

        #region mtmd_context_fields
        internal static bool Mtmd_DecodeUseNonCausal(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_decode_use_non_causal(ctx, IntPtr.Zero);
        }

        internal unsafe static bool Mtmd_DecodeUseNonCausal(SafeMtmdContextHandle ctx, MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_decode_use_non_causal(ctx, (IntPtr)chunk);
        }

        internal static bool Mtmd_DecodeUseMrope(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_decode_use_mrope(ctx);
        }

        internal static bool Mtmd_SupportVision(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_support_vision(ctx);
        }

        internal static bool Mtmd_SupportAudio(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_support_audio(ctx);
        }

        internal static int Mtmd_GetAudioSampleRate(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_get_audio_sample_rate(ctx);
        }

        internal static string Mtmd_GetMarker(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_get_marker(ctx);
        }

        #endregion

        internal static void Mtmd_Free(IntPtr ctx)
        {
            EnsureMtmdInitialized();
            _mtmd_free(ctx);
        }

        #region bitmap

        internal static SafeMtmdBitMapHandle Mtmd_BitMapInitFromImage(uint nx, uint ny, Span<byte> image)
        {
            EnsureMtmdInitialized();

            // Проверка размера буфера
            nuint expectedSize = (nuint)nx * (nuint)ny * 3;
            if ((nuint)image.Length < expectedSize)
                throw new ArgumentException(
                    $"Image buffer too small: expected {expectedSize}, got {image.Length}");

            unsafe
            {
                fixed (byte* img = image)
                {
                    return _mtmd_bitmap_init(nx, ny, img);
                }
            }
        }

        internal static SafeMtmdBitMapHandle Mtmd_BitMapInitFromAudio(nuint n_samples, Span<float> audio)
        {
            EnsureMtmdInitialized();

            // Проверка размера буфера
            if ((nuint)audio.Length < n_samples)
                throw new ArgumentException(
                    $"Audio buffer too small: expected {n_samples}, got {audio.Length}");

            unsafe
            {
                fixed (float* aud = audio)
                {
                    return _mtmd_bitmap_init_from_audio(n_samples, aud);
                }
            }
        }

        internal static void Mtmd_BitMapFree(IntPtr bitmap)
        {
            EnsureMtmdInitialized();
            _mtmd_bitmap_free(bitmap);
        }
        #endregion

        #region chunks

        internal static void Mtmd_InputChunksFree(IntPtr chunks)
        {
            EnsureMtmdInitialized();
            _mtmd_input_chunks_free(chunks);
        }

        internal static SafeMtmdInputChunksHandle Mtmd_InputChunksInit()
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunks_init();
        }

        internal static nuint Mtmd_InputChunksSize(SafeMtmdInputChunksHandle chunks)
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunks_size(chunks);
        }

        internal unsafe static MtmdInputChunkPtr* Mtmd_InputChunksGet(SafeMtmdInputChunksHandle chunks, nuint idx)
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunks_get(chunks, idx);
        }
        #endregion

        internal static void Mtmd_FreeInputChunk(IntPtr chunk)
        {
            EnsureMtmdInitialized();
            _mtmd_input_chunk_free(chunk);
        }
    }
}
