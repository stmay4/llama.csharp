using CommunityToolkit.HighPerformance;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class MtmdContext : IDisposable
    {
        private Dictionary<SafeMtmdInputChunksHandle /*work (one media of three chunks), must be Disposed in the end of work*/,
            TaskCompletionSource<LlamaEmbedding[]> /*return result task*/> _works = new();

        private readonly CancellationTokenSource _mtmdContextLifeToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _encodeSemaphore = new SemaphoreSlim(1);

        /// <summary>
        /// Signal for the endless encode loop
        /// Set and Reset only under _encodeSemaphore
        /// </summary>
        private readonly ManualResetEventSlim _workSignal = new ManualResetEventSlim(false);

        public SafeMtmdContextHandle NativeHandle;

        private MtmdContext(SafeMtmdContextHandle nativeHandle)
        {
            NativeHandle = nativeHandle;

            _ = EncodingLoop();
        }

        public static MtmdContext CreateFromFile(string mmprojFile, LLamaWeights llamaModel, IMtmdParams @params)
        {
            @params.ToMtmdContextParams(out var nativeParams);
            var weights = SafeMtmdContextHandle.LoadFromFile(mmprojFile, llamaModel.NativeHandle, nativeParams);
            return new MtmdContext(weights);
        }

        // метод регистрации картинок, видео и звука на энкод, на вход спаны не битмапы

        public async Task<(LLamaToken[] BOM, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)> EncodeImage(uint nx, uint ny, Memory<byte> image)
        {
            return (await EncodeImages([nx], [ny], [image]))[0];
        }

        //для автоудаления битмапов
        public async Task<(LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
        {
            //Support Vision check here

            List<SafeMtmdBitMapHandle> bitmaps = new List<SafeMtmdBitMapHandle>();
            for (int i = 0; i < images.Count; i++)
            {
                bitmaps.Add(SafeMtmdBitMapHandle.InitFromImage(nxs[i], nys[i], images[i].Span));
            }

            (LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[] result = await EncodeMedias(bitmaps);

            foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
            {
                bitmap.Dispose();
            }

            return result;
        }

        // общий для всех медиа
        private async Task<(LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeMedias(List<SafeMtmdBitMapHandle> bitmaps)
        {
            //проверка параметров
            //создать SafeMtmdInputChunksHandle для каждой картинки
            SafeMtmdInputChunksHandle[] outputs = new SafeMtmdInputChunksHandle[bitmaps.Count];

            var result = new (LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[bitmaps.Count];

            for (int i = 0; i < bitmaps.Count; i++)
            {
                outputs[i] = SafeMtmdInputChunksHandle.Init();

                //вызвать токенизацию для каждой картинки, чтобы не объединилось в видео и вернуло в outputs по 3 чанка: токен начала + медиа эмбеддинги + токен конца
                NativeHandle.Tokenize(outputs[i], [bitmaps[i]]);

                int chankCount = outputs[i].GetChunkCount();

                for (int j = 0; j < chankCount; j++)
                {
                    unsafe
                    {
                        MtmdInputChunkPtr* chunk = outputs[i].GetChunk(j);
                        MtmdInputChunkType type = LlamaCpp.Mtmd_InputChunkGetType(chunk);
                        if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_TEXT)
                        {
                            nuint tokensCount;

                            // Получаем указатель на нативные данные
                            LLamaToken* nativeData = LlamaCpp.Mtmd_InputChunkGetTokensText(chunk, out tokensCount);

                            // Выделяем управляемый непрерывный массив
                            LLamaToken[] tokens = new LLamaToken[(int)tokensCount];

                            // Копируем из нативной памяти в управляемую
                            fixed (LLamaToken* managedPtr = tokens)
                            {
                                Buffer.MemoryCopy(nativeData, managedPtr,
                                                  (int)tokensCount * sizeof(LLamaToken),
                                                  (int)tokensCount * sizeof(LLamaToken));
                            }

                            if (result[i].BOM == null)
                                result[i].BOM = tokens;
                            else
                                result[i].EOM = tokens;

                        }
                        else if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_AUDIO)
                        {
                            // передача в encode chunks
                        }
                    }
                }
            }
            return result;
        }

        private async Task EncodingLoop()
        {
            while (!_mtmdContextLifeToken.Token.IsCancellationRequested) // One pass generates one token where needed and prefills as much as fits
            {
                await _workSignal.WaitAsync(_mtmdContextLifeToken.Token); // Checks if set with fast return inside WaitAsync

                await _encodeSemaphore.WaitAsync(_mtmdContextLifeToken.Token); // sync Lock 
                try
                {
                    if (_works.Count == 0) // Nothing to do
                    {
                        _workSignal.Reset(); // Reset the work signal
                    }
                    else
                    {
                        //await encodeInternal();
                    }
                }
                finally
                {
                    _encodeSemaphore.Release();
                }
            }
        }

        public void Dispose()
        {
            _mtmdContextLifeToken.Cancel();
            _mtmdContextLifeToken.Dispose();
            _encodeSemaphore.Dispose();

            NativeHandle.Dispose();
        }
    }
}
