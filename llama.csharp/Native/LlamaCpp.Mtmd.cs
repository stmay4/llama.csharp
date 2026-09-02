using CommunityToolkit.HighPerformance.Buffers;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

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
        private delegate IntPtr mtmd_get_marker(SafeMtmdContextHandle ctx);

        // tokenize an input text prompt and a list of bitmaps (images/audio)
        // the prompt must have the input image marker (default: "<__media__>") in it
        // the default marker is defined by mtmd_default_marker()
        // the marker will be replaced with the image/audio chunk
        // for example:
        //   "here is an image: <__media__>\ndescribe it in detail."
        //   this will gives 3 chunks:
        //   1. "here is an image: <start_of_image>"
        //   2. (image/audio tokens)
        //   3. "<end_of_image>\ndescribe it in detail."
        // number of bitmaps must be equal to the number of markers in the prompt
        // this function is thread-safe (shared ctx)
        // return values:
        //   0 on success
        //   1 on number of bitmaps not matching the number of markers
        //   2 on image preprocessing error
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int mtmd_tokenize(SafeMtmdContextHandle ctx,
                                                    SafeMtmdInputChunksHandle output,
                                                    MtmdInputTextNative* text,
                                                    IntPtr* bitmaps,
                                                    nuint n_bitmaps);

        #region batch

        // batch encoding API
        // chunks are not owned by the batch, they will not be freed by mtmd_batch_free()
        // batch is valid for a given context, cannot be shared across contexts
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate SafeMtmdBatchHandle mtmd_batch_init(SafeMtmdContextHandle ctx);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate void mtmd_batch_free(IntPtr batch);

        // only media chunks are allowed, text chunks will be rejected
        // returns 0 on success
        // returns 1 on generic error
        // returns 2 if the batch is too large (chunk won't be added)
        // returns 3 if it cannot be batched with the existing chunks in the batch
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate int mtmd_batch_add_chunk(SafeMtmdBatchHandle batch, MtmdInputChunkPtr* chunk);

        // returns 0 on success
        // returns 1 on generic error
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private delegate int mtmd_batch_encode(SafeMtmdBatchHandle batch);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate float* mtmd_batch_get_output_embd(SafeMtmdBatchHandle batch, MtmdInputChunkPtr* chunk);

        #endregion

        //[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        //private delegate void mtmd_input_chunk_free(IntPtr chunk);

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

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate nuint mtmd_input_chunk_get_n_tokens(MtmdInputChunkPtr* chunk);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate MtmdInputChunkType mtmd_input_chunk_get_type(MtmdInputChunkPtr* chunk);

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate LLamaToken* mtmd_input_chunk_get_tokens_text (MtmdInputChunkPtr * chunk, nuint* n_tokens_output); // возвращает указатель на массив токенов (структур)

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate MtmdImageTokensPtr* mtmd_input_chunk_get_tokens_image(MtmdInputChunkPtr* chunk);

        // get position for decoder attention, to be used by M-RoPE models
        // i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
        // pos_0 is the absolute position of the first token
        // return relative position (for example, embedding 0 will have position (0, 0, 0); remember to adjust it to the current absolute position)
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        private unsafe delegate MtmdDecoderPosNative mtmd_image_tokens_get_decoder_pos(MtmdImageTokensPtr* image_tokens, LLamaPos pos_0, nuint i);

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

        private static mtmd_tokenize _mtmd_tokenize;

        #region batch

        private static mtmd_batch_init _mtmd_batch_init;
        private static mtmd_batch_free _mtmd_batch_free;
        private static mtmd_batch_add_chunk _mtmd_batch_add_chunk;
        private static mtmd_batch_encode _mtmd_batch_encode;
        private static mtmd_batch_get_output_embd _mtmd_batch_get_output_embd;

        #endregion

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

        private static mtmd_input_chunk_get_n_tokens _mtmd_input_chunk_get_n_tokens;
        private static mtmd_input_chunk_get_type _mtmd_input_chunk_get_type;
        private static mtmd_input_chunk_get_tokens_text _mtmd_input_chunk_get_tokens_text;

        private static mtmd_input_chunk_get_tokens_image _mtmd_input_chunk_get_tokens_image;
        private static mtmd_image_tokens_get_decoder_pos _mtmd_image_tokens_get_decoder_pos;

        #endregion

        private static mtmd_free _mtmd_free;
        //private static mtmd_input_chunk_free _mtmd_input_chunk_free;

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

            _mtmd_tokenize = GetLibFunction<mtmd_tokenize>(_mtmdHandle, "mtmd_tokenize");

            #region batch

            _mtmd_batch_init = GetLibFunction<mtmd_batch_init>(_mtmdHandle, "mtmd_batch_init");
            _mtmd_batch_free = GetLibFunction<mtmd_batch_free>(_mtmdHandle, "mtmd_batch_free");
            _mtmd_batch_add_chunk = GetLibFunction<mtmd_batch_add_chunk>(_mtmdHandle, "mtmd_batch_add_chunk");
            _mtmd_batch_encode = GetLibFunction<mtmd_batch_encode>(_mtmdHandle, "mtmd_batch_encode");
            _mtmd_batch_get_output_embd = GetLibFunction<mtmd_batch_get_output_embd>(_mtmdHandle, "mtmd_batch_get_output_embd");

            #endregion

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

            _mtmd_input_chunk_get_n_tokens = GetLibFunction<mtmd_input_chunk_get_n_tokens>(_mtmdHandle, "mtmd_input_chunk_get_n_tokens");
            _mtmd_input_chunk_get_type = GetLibFunction<mtmd_input_chunk_get_type>(_mtmdHandle, "mtmd_input_chunk_get_type");
            _mtmd_input_chunk_get_tokens_text = GetLibFunction<mtmd_input_chunk_get_tokens_text>(_mtmdHandle, "mtmd_input_chunk_get_tokens_text");
            _mtmd_input_chunk_get_tokens_image = GetLibFunction<mtmd_input_chunk_get_tokens_image>(_mtmdHandle, "mtmd_input_chunk_get_tokens_image");
            _mtmd_image_tokens_get_decoder_pos = GetLibFunction<mtmd_image_tokens_get_decoder_pos>(_mtmdHandle, "mtmd_image_tokens_get_decoder_pos");

            #endregion

            _mtmd_free = GetLibFunction<mtmd_free>(_mtmdHandle, "mtmd_free");
            //_mtmd_input_chunk_free = GetLibFunction<mtmd_input_chunk_free>(_mtmdHandle, "mtmd_input_chunk_free");
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

        internal static string? Mtmd_GetMarker(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            IntPtr ptr = _mtmd_get_marker(ctx);
            return ptr == IntPtr.Zero ? null : Marshal.PtrToStringUTF8(ptr);
        }

        #endregion

        internal static int Mtmd_Tokenize(SafeMtmdContextHandle ctx,
                                          SafeMtmdInputChunksHandle output,
                                          IReadOnlyList<SafeMtmdBitMapHandle> bitmaps)
        {
            if (bitmaps == null || bitmaps.Count == 0)
                throw new ArgumentException("Mtmd_Tokenize: bitmaps");

            //маркер для строки ввода
            string? marker = Mtmd_GetMarker(ctx);

            if (string.IsNullOrEmpty(marker))
                throw new Exception("Mtmd_Tokenize: Null marker");

            EnsureMtmdInitialized();

            var acquired = new bool[bitmaps.Count];

            try
            {
                // Увеличиваем счётчики ссылок у всех SafeHandle на время вызова
                for (int i = 0; i < bitmaps.Count; i++)
                {
                    bool success = false;
                    bitmaps[i].DangerousAddRef(ref success);
                    if (!success)
                        throw new ObjectDisposedException($"Mtmd_Tokenize: Bitmap at index {i} is disposed");
                    acquired[i] = true;
                }

                // Массив IntPtr для хранения нативных указателей
                nuint size = (nuint)bitmaps.Count;

                var ptrs = new IntPtr[bitmaps.Count];
                for (int i = 0; i < bitmaps.Count; i++)
                    ptrs[i] = bitmaps[i].DangerousGetHandle(); // Get raw handle

                // Собираем строку с маркерами для умного чанкового токенизатора mtmd
                string text = "";
                for (int i = 0; i < bitmaps.Count; i++)
                    text += marker;

                Encoding encoding = Encoding.UTF8;
                var bytesCount = encoding.GetByteCount(text);
                using var bytes = SpanOwner<byte>.Allocate(bytesCount + 1, AllocationMode.Clear);

                encoding.GetBytes(text, bytes.Span);

                unsafe
                {
                    fixed (IntPtr* p = ptrs)
                    fixed (byte* textPtr = bytes.Span)
                    {
                        MtmdInputTextNative inputTextNative = new MtmdInputTextNative(textPtr);
                        MtmdInputTextNative* inputTextPtr = &inputTextNative;
                        return _mtmd_tokenize(ctx, output, inputTextPtr, p, size);
                    }
                }
            }
            finally
            {
                // Снимаем удержание для корректных, теперь SafeHandle могут быть освобождены
                for (int i = 0; i < bitmaps.Count; i++)
                {
                    if (acquired[i])
                        bitmaps[i].DangerousRelease();
                }
            }
        }

        #region batch

        internal static SafeMtmdBatchHandle Mtmd_BatchInit(SafeMtmdContextHandle ctx)
        {
            EnsureMtmdInitialized();
            return _mtmd_batch_init(ctx);
        }
        internal static void Mtmd_BatchFree(IntPtr batch)
        {
            EnsureMtmdInitialized();
            _mtmd_batch_free(batch);
        }

        /// <summary>
        /// only media chunks are allowed, text chunks will be rejected
        /// returns 0 on success
        /// returns 1 on generic error
        /// returns 2 if the batch is too large (chunk won't be added)
        /// returns 3 if it cannot be batched with the existing chunks in the batch
        /// </summary>
        /// <param name="batch"></param>
        /// <param name="chunk"></param>
        /// <returns></returns>
        internal static unsafe int Mtmd_BatchAddChunk(SafeMtmdBatchHandle batch, MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_batch_add_chunk(batch, chunk);
        }

        /// <summary>
        /// returns 0 on success
        /// returns 1 on generic error
        /// </summary>
        /// <param name="batch"></param>
        /// <returns></returns>
        internal static int Mtmd_BatchEncode(SafeMtmdBatchHandle batch)
        {
            EnsureMtmdInitialized();
            return _mtmd_batch_encode(batch);
        }

        internal static unsafe float* Mtmd_BatchGetOutputEmbed(SafeMtmdBatchHandle batch, MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_batch_get_output_embd(batch, chunk);
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

        internal static SafeMtmdBitMapHandle Mtmd_BitMapInitFromAudio(Span<float> audio)
        {
            EnsureMtmdInitialized();
            nuint n_samples = (nuint)audio.Length;
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

        internal unsafe static nuint Mtmd_InputChunkGetNTokens(MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunk_get_n_tokens(chunk);
        }

        internal unsafe static MtmdInputChunkType Mtmd_InputChunkGetType(MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunk_get_type(chunk);
        }

        internal unsafe static LLamaToken* Mtmd_InputChunkGetTokensText(MtmdInputChunkPtr* chunk, out nuint n_tokens_output)
        {
            EnsureMtmdInitialized();
            nuint nTokens;
            var tokensPtr = _mtmd_input_chunk_get_tokens_text(chunk, &nTokens);
            n_tokens_output = nTokens;
            return tokensPtr;
        }

        internal unsafe static MtmdImageTokensPtr* Mtmd_InputChunkGetTokensImage(MtmdInputChunkPtr* chunk)
        {
            EnsureMtmdInitialized();
            return _mtmd_input_chunk_get_tokens_image(chunk);
        }

        // get position for decoder attention, to be used by M-RoPE models
        // i is the index of the embedding token, ranging from 0 to mtmd_image_tokens_get_n_tokens() - 1
        // pos_0 is the absolute position of the first token
        // return relative position (for example, embedding 0 will have position (0, 0, 0); remember to adjust it to the current absolute position)
        internal unsafe static MtmdDecoderPosNative Mtmd_ImageTokensGetDecoderPos(MtmdImageTokensPtr* image_tokens, LLamaPos pos_0, nuint i)
        {
            EnsureMtmdInitialized();
            return _mtmd_image_tokens_get_decoder_pos(image_tokens, pos_0, i);
        }


        #endregion

        // не нужен изза общего освобождения всех чанков в chunks
        //internal static void Mtmd_FreeInputChunk(IntPtr chunk)
        //{
        //    EnsureMtmdInitialized();
        //    _mtmd_input_chunk_free(chunk);
        //}
    }
}
