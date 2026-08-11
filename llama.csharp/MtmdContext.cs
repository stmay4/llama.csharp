using CommunityToolkit.HighPerformance;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp
{
    public class MtmdContext : IDisposable
    {
        public bool NonCasualDecode => NativeHandle.NonCasualDecode;
        public bool MropeDecode => NativeHandle.MropeDecode;
        public bool SupportVision => NativeHandle.SupportVision;
        public bool SupportAudio => NativeHandle.SupportAudio;
        public int AudioSampleRate => NativeHandle.AudioSampleRate;
        public string CurrentMarker => NativeHandle.CurrentMarker;

        private Dictionary<SafeMtmdInputChunksHandle /*work (chunks[1] is MEDIA), must be Disposed in the end of work*/,
            (TaskCompletionSource<LlamaEmbedding[]>/*return result task*/, LlamaEmbedding[] /* заготовка, частично заполненная*/ )> _visionWorks = new();

        private readonly CancellationTokenSource _mtmdContextLifeToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _visionEncodeSemaphore = new SemaphoreSlim(1);

        /// <summary>
        /// Signal for the endless encode loop
        /// Set and Reset only under _encodeSemaphore
        /// </summary>
        private readonly ManualResetEventSlim _visionWorkSignal = new ManualResetEventSlim(false);

        private Task? _visionLoopTask;

        internal SafeMtmdContextHandle NativeHandle;

        private MtmdContext(SafeMtmdContextHandle nativeHandle)
        {
            NativeHandle = nativeHandle;

            if (SupportVision)
                _visionLoopTask = VisionEncodingLoop();
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

            (LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[] result = await RegisterVisionEncode(bitmaps);

            foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
            {
                bitmap.Dispose();
            }

            return result;
        }

        // для видео и картинок, именно для них может быть испольован MROPE, позиции для MROPE получают здесь после токенизации, получая токены изображения и передавая их в функцию полученмия позиций
        private async Task<(LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[]> RegisterVisionEncode(List<SafeMtmdBitMapHandle> bitmaps)
        {
            //проверка параметров
            //создать SafeMtmdInputChunksHandle для каждой картинки
            SafeMtmdInputChunksHandle[] outputs = new SafeMtmdInputChunksHandle[bitmaps.Count];

            try
            {

                var result = new (LLamaToken[] BOM /* Begin of media */, LLamaToken[] EOM, Task<LlamaEmbedding[]> embeds)[bitmaps.Count];

                //отправляем все задания под одной блокировкой для одновременного отправления в энкод
                await _visionEncodeSemaphore.WaitAsync(_mtmdContextLifeToken.Token);
                try
                {
                    for (int i = 0; i < bitmaps.Count; i++)
                    {
                        outputs[i] = SafeMtmdInputChunksHandle.Init();

                        //вызвать токенизацию для каждой картинки, чтобы не объединилось в видео и вернуло в outputs по 3 чанка: токен начала + медиа эмбеддинги + токен конца
                        NativeHandle.Tokenize(outputs[i], [bitmaps[i]]);
                        // Для видео по другому !!! все вместе

                        int chunkCount = outputs[i].GetChunkCount();

                        if (chunkCount != 3)
                            throw new Exception("EncodeMedias: Strange behaviour. Chunk count is not 3");

                        for (int j = 0; j < chunkCount; j++)
                        {
                            unsafe
                            {
                                MtmdInputChunkPtr* chunk = outputs[i].GetChunk(j);
                                MtmdInputChunkType type = LlamaCpp.Mtmd_InputChunkGetType(chunk);

                                if (j == 0 || j == 2)
                                    if (type != MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_TEXT)
                                        throw new Exception("Chunks 0 and 2 must be TEXT");
                                if (j == 1)
                                    if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_TEXT)
                                        throw new Exception("Chunk 1 must be MEDIA");

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

                                    if (j == 0)
                                        result[i].BOM = tokens;
                                    else
                                        result[i].EOM = tokens;

                                }
                                else if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_IMAGE || type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_AUDIO)
                                {
                                    // передача в encode chunks
                                    TaskCompletionSource<LlamaEmbedding[]> tcs = new TaskCompletionSource<LlamaEmbedding[]>();

                                    int imageTokensCount = (int)LlamaCpp.Mtmd_InputChunkGetNTokens(chunk);

                                    LlamaEmbedding[] futureResult = new LlamaEmbedding[imageTokensCount];

                                    // для получения позиций при MROPE
                                    MtmdImageTokensPtr* imageTokens = null;
                                    if (MropeDecode)
                                    {
                                        imageTokens = LlamaCpp.Mtmd_InputChunkGetTokensImage(chunk);
                                    }

                                    for (int frCounter = 0; frCounter < futureResult.Length; frCounter++)
                                    {
                                        // для MROPE заполняем позиции
                                        if (MropeDecode)
                                        {
                                            futureResult[frCounter] = new LlamaEmbedding(Memory<float>.Empty, LlamaEmbeddingType.Image, LlamaCpp.Mtmd_ImageTokensGetDecoderPos(imageTokens, 0, (nuint)frCounter));
                                        }
                                        else
                                        {
                                        futureResult[frCounter] = new LlamaEmbedding(Memory<float>.Empty, LlamaEmbeddingType.Image);
                                        }
                                    }

                                    _visionWorks.Add(outputs[i], (tcs, futureResult));

                                    _visionWorkSignal.Set();

                                    result[i].embeds = tcs.Task;
                                }
                            }
                        }

                    }
                }
                finally
                {
                    _visionEncodeSemaphore.Release();
                }
                return result;
            }
            catch
            {
                foreach (var o in outputs)
                {
                    if (o != null && !o.IsClosed && !_visionWorks.ContainsKey(o))
                        o.Dispose();
                }
                throw;
            }
        }

        private async Task VisionEncodingLoop()
        {
            while (!_mtmdContextLifeToken.Token.IsCancellationRequested) // One pass generates one token where needed and prefills as much as fits
            {
                await _visionWorkSignal.WaitAsync(_mtmdContextLifeToken.Token); // Checks if set with fast return inside WaitAsync

                await _visionEncodeSemaphore.WaitAsync(_mtmdContextLifeToken.Token); // sync Lock 
                try
                {
                    if (_visionWorks.Count == 0) // Nothing to do
                    {
                        _visionWorkSignal.Reset(); // Reset the work signal
                    }
                    else
                    {
                        await VisionEncodeInternal();
                    }
                }
                finally
                {
                    _visionEncodeSemaphore.Release();
                }
            }
        }

        private async Task VisionEncodeInternal()
        {
            // собираем батч из того что доступно в works 
            using (SafeMtmdBatchHandle mtmdBatch = SafeMtmdBatchHandle.Init(this.NativeHandle))
            {
                List<SafeMtmdInputChunksHandle> acceptedChunks = new();
                foreach (SafeMtmdInputChunksHandle chunks in _visionWorks.Keys)
                {
                    unsafe
                    {
                        MtmdInputChunkPtr* chunk = chunks.GetChunk(1);

                        int addResult = mtmdBatch.AddChunk(chunk);
                        if (addResult == 0)
                        {
                            acceptedChunks.Add(chunks);
                            continue;
                        }
                        else break;
                    }

                }

                // производим энкод
                int encodeResult = await Task.Run(() => mtmdBatch.Encode());

                // ошибка пока так обрабатывается
                if (encodeResult != 0)
                {
                    foreach (SafeMtmdInputChunksHandle chunk in acceptedChunks)
                    {
                        _visionWorks[chunk].Item1.TrySetCanceled();
                        _visionWorks.Remove(chunk);
                        chunk.Dispose();
                    }
                    acceptedChunks.Clear();
                }

                // постобработка с возвратом результата и удалением работы
                foreach (SafeMtmdInputChunksHandle chunks in acceptedChunks)
                {
                    unsafe
                    {
                        MtmdInputChunkPtr* chunk = chunks.GetChunk(1);

                        Memory<float>[] encodedEmdeddings = mtmdBatch.GetChunkEmbeddings(chunk);
                        TaskCompletionSource<LlamaEmbedding[]> tcs = _visionWorks[chunks].Item1;

                        LlamaEmbedding[] encodedResult = _visionWorks[chunks].Item2;
                        for (int i = 0; i < encodedResult.Length; i++)
                        {
                            encodedResult[i] = new LlamaEmbedding(encodedEmdeddings[i], encodedResult[i].Type, encodedResult[i].Pos);
                        }

                        tcs.TrySetResult(encodedResult);
                    }

                    // очистка
                    _visionWorks.Remove(chunks);
                    chunks.Dispose();
                }
            }
        }

        public void Dispose()
        {
            _mtmdContextLifeToken.Cancel();

            //// Дожидаемся завершения фонового цикла
            //try { _visionLoopTask?.Wait(TimeSpan.FromSeconds(2)); }
            //catch (Exception) { }

            _mtmdContextLifeToken.Dispose();
            _visionEncodeSemaphore.Dispose();

            foreach (var item in _visionWorks)
            {
                item.Key.Dispose(); //освобождение чанков
                item.Value.Item1.TrySetCanceled(); // возврат отмены отмененным заданиям
            }
            NativeHandle.Dispose();
        }
    }
}
