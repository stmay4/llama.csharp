using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using NAudio.Wave;
using NAudio.Wave.SampleProviders;
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

        private Dictionary<(SafeMtmdInputChunksHandle chunksHandle, int pos),
            (TaskCompletionSource<LlamaEmbedding[]>/*return result task*/, LlamaEmbedding[] /* заготовка, частично заполненная*/ )> _visionWorks = new();

        private Dictionary<(SafeMtmdInputChunksHandle chunksHandle, int pos),
    (TaskCompletionSource<LlamaEmbedding[]>/*return result task*/, LlamaEmbedding[] /* заготовка, частично заполненная*/ )> _audioWorks = new();

        private readonly CancellationTokenSource _mtmdContextLifeToken = new CancellationTokenSource();

        private readonly SemaphoreSlim _encodeSemaphore = new SemaphoreSlim(1);

        /// <summary>
        /// Signal for the endless encode loop
        /// Set and Reset only under _encodeSemaphore
        /// </summary>
        private readonly ManualResetEventSlim _workSignal = new ManualResetEventSlim(false);

        private Task? _loopTask;

        internal SafeMtmdContextHandle NativeHandle;

        private MtmdContext(SafeMtmdContextHandle nativeHandle)
        {
            NativeHandle = nativeHandle;

            _loopTask = EncodingLoop();
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
            return new MtmdSpec(MropeDecode);
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
        /// <summary>
        /// using SixLabors.ImageSharp v2.1.13 (Apache 2.0) for read all popular formats
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeImageFromPath(string filePath)
        {
            (Memory<byte> image, int nx, int ny) = await Task.Run(() => ConvertToRgbTopDown(filePath));
            return await EncodeImageFromRGB((uint)nx, (uint)ny, image);
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

        public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImageFromPaths(List<string> filePaths)
        {
            List<uint> nxs = new();
            List<uint> nys = new();
            List<Memory<byte>> images = new();
            foreach (string filePath in filePaths)
            {
                (Memory<byte> image, int nx, int ny) = await Task.Run(() => ConvertToRgbTopDown(filePath));
                nxs.Add((uint)nx);
                nys.Add((uint)ny);
                images.Add(image);
            }
            return await EncodeImages(nxs, nys, images);
        }

        public async Task<(string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
        {
            if (!SupportVision) throw new Exception("Vision dont support");
            List<SafeMtmdBitMapHandle> bitmaps = new List<SafeMtmdBitMapHandle>();

            for (int i = 0; i < images.Count; i++)
            {
                bitmaps.Add(SafeMtmdBitMapHandle.InitFromImage(nxs[i], nys[i], images[i].Span));
            }

            try
            {
                (string BOM /* Begin Of Media */, string EOM, Task<LlamaEmbedding[]> embeds)[] result = await RegisterEncode(bitmaps);
                return result;
            }
            finally
            {
                // dispose all bitmaps
                foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
                {
                    bitmap.Dispose();
                }
            }
        }

        #endregion

        #region AudioRegistration
        /// <summary>
        /// Читает WAV, конвертирует в моно встроенным StereoToMonoSampleProvider,
        /// ресемплит WdlResamplingSampleProvider при несовпадении rate.
        /// Возвращает mono PCM float32 в [-1, 1].
        /// </summary>
        public float[] DecodeWavToMonoFloat(string path)
        {
            if (!SupportAudio) throw new Exception("Audio dont support");
            using var reader = new WaveFileReader(path);
            int channels = reader.WaveFormat.Channels;
            int sourceRate = reader.WaveFormat.SampleRate;

            // любой PCM (16/24/32 int, IEEE float) → float32
            ISampleProvider samples = reader.ToSampleProvider();

            // встроенное преобразование в моно
            ISampleProvider mono = channels switch
            {
                1 => samples,
                2 => new StereoToMonoSampleProvider(samples), // по умолчанию усредняет каналы
                _ => throw new NotSupportedException($"Unsupported channel count: {channels}")
            };

            // ресемплинг только при несовпадении rate
            if (sourceRate != AudioSampleRate)
                mono = new WdlResamplingSampleProvider(mono, AudioSampleRate);

            return ReadAll(mono);
        }
        private float[] ReadAll(ISampleProvider provider)
        {
            var buffer = new float[8192];
            var result = new List<float>();
            int read;
            while ((read = provider.Read(buffer)) > 0)
                result.AddRange(buffer.AsSpan(0, read));
            return result.ToArray();
        }

        public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudioFromWav(string wavFilePath)
        {
            float[] audio = await Task.Run(() => DecodeWavToMonoFloat(wavFilePath));
            return await EncodeAudio(audio.AsMemory());
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="audio">mono PCM f32 buffer</param>
        /// <returns></returns>
        public async Task<(string BOM, string EOM, LlamaEmbedding[] embeds)> EncodeAudio(Memory<float> audio)
        {
            (string BOM, string EOM, Task<LlamaEmbedding[]> embeds) = (await EncodeAudios([audio]))[0];
            LlamaEmbedding[] calculatedEmbeds = await embeds;
            return (BOM, EOM, calculatedEmbeds);
        }

        public async Task<(string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudiosFromWav(List<string> wavFilePaths)
        {
            List<Memory<float>> audios = new List<Memory<float>>();
            foreach (var filePath in wavFilePaths)
            {
                float[] audio = await Task.Run(() => DecodeWavToMonoFloat(filePath));
                audios.Add(audio.AsMemory());
            }
            return await EncodeAudios(audios);

        }

        //для автоудаления битмапов
        public async Task<(string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[]> EncodeAudios(List<Memory<float>> audio)
        {
            if (!SupportAudio) throw new Exception("Audio dont support");
            List<SafeMtmdBitMapHandle> bitmaps = new List<SafeMtmdBitMapHandle>();

            for (int i = 0; i < audio.Count; i++)
            {
                bitmaps.Add(SafeMtmdBitMapHandle.InitFromAudio(audio[i].Span));
            }

            try
            {
                (string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[] result = await RegisterEncode(bitmaps);
                return result;
            }
            finally
            {
                // dispose all bitmaps
                foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
                {
                    bitmap.Dispose();
                }
            }
        }

        #endregion

        private async Task<(string BOM /* Begin of media */, string EOM, Task<LlamaEmbedding[]> embeds)[]> RegisterEncode(List<SafeMtmdBitMapHandle> bitmaps)
        {
            //проверка параметров
            if (bitmaps == null) throw new ArgumentNullException("RegisterEncode: bitmaps is null");
            if (bitmaps.Count == 0) throw new ArgumentException("RegisterEncode: bitmaps count is 0");
            foreach (var bitmap in bitmaps)
            {
                if (bitmap == null || bitmap.IsClosed || bitmap.IsInvalid) throw new ArgumentException("RegisterEncode: bitmap error");
            }

            //создать SafeMtmdInputChunksHandle для каждого bitmap
            SafeMtmdInputChunksHandle[] outputs = new SafeMtmdInputChunksHandle[bitmaps.Count];

            //отправляем все задания под одной блокировкой для ОДНОВРЕМЕННОГО отправления в энкод
            await _encodeSemaphore.WaitAsync(_mtmdContextLifeToken.Token);
            try
            {
                var result = new (string BOM, string EOM, Task<LlamaEmbedding[]> embeds)[bitmaps.Count];

                for (int i = 0; i < bitmaps.Count; i++)
                {
                    outputs[i] = SafeMtmdInputChunksHandle.Init();
                    List<Task<LlamaEmbedding[]>> works = new();

                    //вызвать токенизацию для каждого bitmap(audio or image), чтобы не объединилось в видео и вернуло в outputs по 3 чанка: токен начала + медиа эмбеддинги + токен конца
                    NativeHandle.Tokenize(outputs[i], [bitmaps[i]]);

                    int chunkCount = outputs[i].GetChunkCount();

                    for (int j = 0; j < chunkCount; j++)
                    {
                        unsafe
                        {
                            MtmdInputChunkPtr* chunk = outputs[i].GetChunk(j);
                            MtmdInputChunkType type = LlamaCpp.Mtmd_InputChunkGetType(chunk);

                            //проверка, что между текстовыми только медиа
                            if (j == 0 || j == chunkCount - 1)
                            {
                                if (type != MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_TEXT)
                                    throw new Exception("start and end chunks must be TEXT");
                            }
                            else
                            {
                                if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_TEXT)
                                    throw new Exception("Chunks between text chunks must be MEDIA");
                            }

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
                            else if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_IMAGE)
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

                                bool useNonCausal = NativeHandle.UseNonCasualDecodeForChunk(chunk);

                                for (int frCounter = 0; frCounter < futureResult.Length; frCounter++)
                                {
                                    // для MROPE заполняем позиции
                                    if (MropeDecode)
                                    {
                                        futureResult[frCounter] = new LlamaEmbedding(Memory<float>.Empty, LlamaEmbeddingType.Image, useNonCausal, LlamaCpp.Mtmd_ImageTokensGetDecoderPos(imageTokens, 0, (nuint)frCounter));
                                    }
                                    else
                                    {
                                        futureResult[frCounter] = new LlamaEmbedding(Memory<float>.Empty, LlamaEmbeddingType.Image, useNonCausal);
                                    }
                                }

                                _visionWorks.Add((outputs[i], j), (tcs, futureResult));

                                works.Add(tcs.Task);
                            }
                            else if (type == MtmdInputChunkType.MTMD_INPUT_CHUNK_TYPE_AUDIO)
                            {
                                // передача в encode chunks
                                TaskCompletionSource<LlamaEmbedding[]> tcs = new TaskCompletionSource<LlamaEmbedding[]>();

                                int tokensCount = (int)LlamaCpp.Mtmd_InputChunkGetNTokens(chunk);

                                LlamaEmbedding[] futureResult = new LlamaEmbedding[tokensCount];

                                bool useNonCausal = NativeHandle.UseNonCasualDecodeForChunk(chunk);

                                for (int frCounter = 0; frCounter < futureResult.Length; frCounter++)
                                {
                                    futureResult[frCounter] = new LlamaEmbedding(Memory<float>.Empty, LlamaEmbeddingType.Audio, useNonCausal);
                                }

                                _audioWorks.Add((outputs[i], j), (tcs, futureResult));

                                works.Add(tcs.Task);
                            }
                            else
                            {
                                throw new Exception("Unsupported chunk type");
                            }
                        }
                    }

                    

                    MtmdContextWork mtmdWork = new MtmdContextWork(works);
                    result[i].embeds = mtmdWork.GetWork();
                }
                _workSignal.Set();

                return result;
            }
            catch
            {
                // очистка при ошибке
                foreach (var o in outputs)
                {
                    if (o != null && !o.IsClosed && !IsHandleStillInUse(_visionWorks, o) && !IsHandleStillInUse(_audioWorks, o))
                        o.Dispose();
                }
                throw;
            }
            finally
            {
                _encodeSemaphore.Release();
            }
        }

        private async Task EncodingLoop()
        {
            while (!_mtmdContextLifeToken.Token.IsCancellationRequested) // One pass generates one token where needed and prefills as much as fits
            {
                await _workSignal.WaitAsync(_mtmdContextLifeToken.Token); // Checks if set with fast return inside WaitAsync

                await _encodeSemaphore.WaitAsync(_mtmdContextLifeToken.Token); // sync Lock 
                try
                {
                    if (_visionWorks.Count == 0 && _audioWorks.Count == 0) // Nothing to do
                    {
                        _workSignal.Reset(); // Reset the work signal
                    }
                    else
                    {
                        await EncodeInternal();
                    }
                }
                finally
                {
                    _encodeSemaphore.Release();
                }
            }
        }

        private async Task EncodeInternal()
        {
            for (int w = 0; w < 2; w++)
            {
                var currentWorks = _visionWorks;
                if (w == 0)
                {
                    if (_visionWorks.Count == 0) continue;
                    currentWorks = _visionWorks;
                }
                else
                {
                    if (_audioWorks.Count == 0) continue;
                    currentWorks = _audioWorks;
                }
                // собираем батч из того что доступно в currentWorks и энкодим
                using (SafeMtmdBatchHandle mtmdBatch = SafeMtmdBatchHandle.Init(this.NativeHandle))
                {
                    List<(SafeMtmdInputChunksHandle, int)> acceptedChunks = new();
                    foreach ((SafeMtmdInputChunksHandle chunksHandle, int pos) in currentWorks.Keys)
                    {
                        unsafe
                        {
                            MtmdInputChunkPtr* chunk = chunksHandle.GetChunk(pos);

                            int addResult = mtmdBatch.AddChunk(chunk);
                            if (addResult == 0)
                            {
                                acceptedChunks.Add((chunksHandle, pos));
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
                        foreach ((SafeMtmdInputChunksHandle chunksHandle, int pos) item in acceptedChunks)
                        {
                            currentWorks[item].Item1.TrySetCanceled();
                            
                            // очистка
                            currentWorks.Remove(item);

                            if (!IsHandleStillInUse(currentWorks, item.chunksHandle))
                            {
                                item.chunksHandle.Dispose();
                            }
                        }
                        acceptedChunks.Clear();
                    }

                    // постобработка с возвратом результата и удалением работы
                    foreach ((SafeMtmdInputChunksHandle chunksHandle, int pos) item in acceptedChunks)
                    {
                        unsafe
                        {
                            MtmdInputChunkPtr* chunk = item.chunksHandle.GetChunk(item.pos);

                            Memory<float>[] encodedEmdeddings = mtmdBatch.GetChunkEmbeddings(chunk);
                            TaskCompletionSource<LlamaEmbedding[]> tcs = currentWorks[item].Item1;

                            LlamaEmbedding[] encodedResult = currentWorks[item].Item2;
                            for (int i = 0; i < encodedResult.Length; i++)
                            {
                                encodedResult[i] = new LlamaEmbedding(encodedEmdeddings[i], encodedResult[i].Type, encodedResult[i].UseNonCausal, encodedResult[i].Pos);
                            }

                            tcs.TrySetResult(encodedResult);
                        }

                        // очистка
                        currentWorks.Remove(item);

                        if (!IsHandleStillInUse(currentWorks, item.chunksHandle))
                        {
                            item.chunksHandle.Dispose();
                        }
                    }
                }
            }
        }

        // Возвращает true, если хендл все еще используется (есть в словаре)
        private bool IsHandleStillInUse(
            Dictionary<(SafeMtmdInputChunksHandle chunksHandle, int pos), (TaskCompletionSource<LlamaEmbedding[]>, LlamaEmbedding[])> dict,
            SafeMtmdInputChunksHandle handle)
        {
            foreach (var key in dict.Keys)
            {
                if (key.chunksHandle == handle) return true;
            }
            return false;
        }

        public void Dispose()
        {
            _mtmdContextLifeToken.Cancel();

            //// Дожидаемся завершения фонового цикла
            //try { _visionLoopTask?.Wait(TimeSpan.FromSeconds(2)); }
            //catch (Exception) { }

            _mtmdContextLifeToken.Dispose();
            _encodeSemaphore.Dispose();

            var disposedHandles = new HashSet<SafeMtmdInputChunksHandle>();

            foreach (var item in _visionWorks)
            {
                if (disposedHandles.Add(item.Key.chunksHandle))
                    item.Key.chunksHandle.Dispose();
                item.Value.Item1.TrySetCanceled();
            }
            foreach (var item in _audioWorks)
            {
                if (disposedHandles.Add(item.Key.chunksHandle))
                    item.Key.chunksHandle.Dispose();
                item.Value.Item1.TrySetCanceled();
            }

            NativeHandle.Dispose();
        }
    }
}
