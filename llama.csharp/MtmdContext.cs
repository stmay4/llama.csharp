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
            TaskCompletionSource<Memory<float>> /*return result task*/> _works = new();

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

        public async Task<(string BOM, string EOM, Task<Memory<float>> embeds)> EncodeImage(uint nx, uint ny, Memory<byte> image)
        {
            return (await EncodeImages([nx], [ny], [image]))[0];
        }

        //для автоудаления битмапов
        public async Task<Dictionary<int, (string BOM, string EOM, Task<Memory<float>> embeds)>> EncodeImages(List<uint> nxs, List<uint> nys, List<Memory<byte>> images)
        {
            List<SafeMtmdBitMapHandle> bitmaps = new List<SafeMtmdBitMapHandle>();
            for (int i = 0; i < images.Count; i++)
            {
                bitmaps.Add(SafeMtmdBitMapHandle.InitFromImage(nxs[i], nys[i], images[i].Span));
            }

            Dictionary<int, (string BOM, string EOM, Task<Memory<float>> embeds)> result = await EncodeMedias(bitmaps);

            foreach (SafeMtmdBitMapHandle bitmap in bitmaps)
            {
                bitmap.Dispose();
            }

            return result;
        }

        // общий для всех медиа
        private async Task<Dictionary<int, (string BOM, string EOM, Task<Memory<float>> embeds)>> EncodeMedias(List<SafeMtmdBitMapHandle> bitmaps)
        {
            //проверка параметров
            //создать SafeMtmdInputChunksHandle для каждой картинки
            List<SafeMtmdInputChunksHandle> outputs = new List<SafeMtmdInputChunksHandle>();
            for (int i = 0; i < bitmaps.Count; i++)
            {
                outputs[i] = SafeMtmdInputChunksHandle.Init();

                //вызвать токенизацию для каждой картинки, чтобы не объединилось в видео и вернуло в outputs по 3 чанка: токен начала + медиа эмбеддинги + токен конца
                NativeHandle.Tokenize(outputs[i], [bitmaps[i]]);

                //перебрать полученные чанки, текстовые вернуть, а медиа отправить на энкод
                // Запись string BOI, string EOI, остается проверить, что для всех моделей колво чанков равно колву медиа

                // добавляем энкод в задачи Dictionary<SafeMtmdInputChunksHandle, TaskCompletionSource<float[]>>
                
                // за норму считаем, что в чанках только один является медиа и всего их 3 (2 текстовых + 1 медиа)

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
