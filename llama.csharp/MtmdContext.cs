using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using SixLabors.ImageSharp.PixelFormats;
using System.Reflection;

namespace Llama.csharp
{
    public class MtmdContext : IDisposable
    {
        /// <summary>
        /// Indicates whether causal or full attention is used for decoding image embeddings by the language model
        /// </summary>
        public bool NonCasualDecode => NativeHandle.NonCasualDecode;

        /// <summary>
        /// Mrope positioning?
        /// </summary>
        public bool MropeDecode => NativeHandle.MropeDecode;

        /// <summary>
        /// Indicates whether images are supported
        /// </summary>
        public bool SupportVision => NativeHandle.SupportVision;

        /// <summary>
        /// Indicates whether audio is supported
        /// </summary>
        public bool SupportAudio => NativeHandle.SupportAudio;

        /// <summary>
        /// Sample rate for audio
        /// </summary>
        public int AudioSampleRate => NativeHandle.AudioSampleRate;

        //public string CurrentMarker => NativeHandle.CurrentMarker;

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

        /// <summary>
        /// Initializes the MTMD model and its runtime environment (context)
        /// </summary>
        /// <param name="mmprojFile"></param>
        /// <param name="llamaModel"></param>
        /// <param name="params"></param>
        /// <returns></returns>
        public static MtmdContext CreateFromFile(string mmprojFile, LLamaWeights llamaModel, IMtmdParams @params)
        {
            @params.ToMtmdContextParams(out var nativeParams);
            var weights = SafeMtmdContextHandle.LoadFromFile(mmprojFile, llamaModel.NativeHandle, nativeParams);
            return new MtmdContext(weights);
        }

        /// <summary>
        /// Gets the specification from the initialized MTMD for LlamaExecutor
        /// </summary>
        /// <returns></returns>
        public MtmdSpec GetSpecification()
        {
            return new MtmdSpec(NonCasualDecode, MropeDecode);
        }

        private (Memory<byte> rgb, int w, int h) ConvertToRgbTopDown(string path)
        {
            // Загружаем изображение в формате RGB24 (8 бит на канал, без альфы)
            using SixLabors.ImageSharp.Image<Rgb24> image = SixLabors.ImageSharp.Image.Load<Rgb24>(path);

            int width = image.Width;
            int height = image.Height;

            // Выделяем буфер нужного размера: ширина * высота * 3 байта
            byte[] rgbData = new byte[width * height * 3];

            // Копируем пиксельные данные в буфер.
            // ImageSharp хранит пиксели построчно, начиная с верхней строки (top‑down),
            // поэтому порядок соответствует требуемому.
            image.CopyPixelDataTo(rgbData);

            return (new Memory<byte>(rgbData), width, height);
        }

        #region ImageRegistration

        public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromPath(string filePath)
        {
            (Memory<byte> image, int nx, int ny) = ConvertToRgbTopDown(filePath);
            (string BOM, string EOM, Task<LlamaEmbedding[]> embeds) = (await EncodeImages([(uint)nx], [(uint)ny], [image]))[0];
            LlamaEmbedding[] calculatedEmbeds = await embeds;
            return (BOM, EOM, calculatedEmbeds);
        }
        // метод регистрации картинок, видео и звука на энкод, на вход спаны не битмапы
        /// <summary>
        /// Принимает top-down RGB буфер
        /// </summary>
        /// <param name="nx"></param>
        /// <param name="ny"></param>
        /// <param name="image"></param>
        /// <returns></returns>
        public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromRGB(uint nx, uint ny, Memory<byte> image)
        {
            (string BOM, string EOM, Task<LlamaEmbedding[]> embeds) = (await EncodeImages([nx], [ny], [image]))[0];
            LlamaEmbedding[] calculatedEmbeds = await embeds;
            return (BOM, EOM, calculatedEmbeds);
        }

        //для автоудаления битмапов
        public async Task<(string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
        {
            List<SafeMtmdBitMapHandle> bitmaps = new List<SafeMtmdBitMapHandle>();

            for (int i = 0; i < images.Count; i++)
            {
                bitmaps.Add(SafeMtmdBitMapHandle.InitFromImage(nxs[i], nys[i], images[i].Span));
            }

            (string BOM /* Begin Of Media */, string EOM, Task<LlamaEmbedding[]> embeds)[] result = await RegisterVisionEncode(bitmaps);

            // dispose all bitmaps
            foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
            {
                bitmap.Dispose();
            }

            return result;
        }

        // для видео и картинок, именно для них может быть испольован MROPE, позиции для MROPE получают здесь после токенизации, получая токены изображения и передавая их в функцию полученмия позиций
        private async Task<(string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[]> RegisterVisionEncode(List<SafeMtmdBitMapHandle> bitmaps)
        {
            //Support Vision check
            if (!SupportVision) throw new Exception("Model dont support Vision");

            //проверка параметров
            if (bitmaps == null) throw new ArgumentNullException("RegisterVisionEncode: bitmaps is null");
            if (bitmaps.Count == 0) throw new ArgumentException("RegisterVisionEncode: bitmaps count is 0");
            foreach (var bitmap in bitmaps)
            {
                if (bitmap == null || bitmap.IsClosed || bitmap.IsInvalid) throw new ArgumentException("RegisterVisionEncode: bitmap error");
            }

            //создать SafeMtmdInputChunksHandle для каждой картинки
            SafeMtmdInputChunksHandle[] outputs = new SafeMtmdInputChunksHandle[bitmaps.Count];

            try
            {
                var result = new (string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[bitmaps.Count];

                //отправляем все задания под одной блокировкой для ОДНОВРЕМЕННОГО отправления в энкод
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

                                    string tokensToString = "";

                                    foreach (LLamaToken token in tokens)
                                        tokensToString += NativeHandle.ModelHandle.Vocab.LLamaTokenToString(token, true);

                                    if (j == 0)
                                        result[i].BOM = tokensToString;
                                    else
                                        result[i].EOM = tokensToString;

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
                // очистка при ошибке
                foreach (var o in outputs)
                {
                    if (o != null && !o.IsClosed && !_visionWorks.ContainsKey(o))
                        o.Dispose();
                }
                throw;
            }
        }

        #endregion

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
            // собираем батч из того что доступно в works и энкодим
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
