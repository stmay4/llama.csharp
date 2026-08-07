using Llama.csharp.Exceptions;
using Llama.csharp.Extensions;
using Llama.csharp.Interfaces;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using static System.Net.Mime.MediaTypeNames;

namespace Llama.csharp.Native
{
    /// <summary>
    /// A reference to a mmproj context
    /// </summary>
    public sealed class SafeMtmdContextHandle : SafeLLamaHandleBase
    {
        public bool NonCasualDecode => LlamaCpp.Mtmd_DecodeUseNonCausal(this);

        internal unsafe bool UseNonCasualDecodeForChunk(MtmdInputChunkPtr* chunk) => LlamaCpp.Mtmd_DecodeUseNonCausal(this, chunk);

        public bool MropeDecode => LlamaCpp.Mtmd_DecodeUseMrope(this);

        public bool SupportVision => LlamaCpp.Mtmd_SupportVision(this);

        public bool SupportAudio => LlamaCpp.Mtmd_SupportAudio(this);

        public int AudioSampleRate => LlamaCpp.Mtmd_GetAudioSampleRate(this);

        public string CurrentMarker => LlamaCpp.Mtmd_GetMarker(this);


        /// <summary>
        /// Dimension of embedding vectors
        /// </summary>
        public int EmbeddingSize => ThrowIfDisposed().EmbeddingSize;

        /// <summary>
        /// Get the model which this context is using
        /// </summary>
        public SafeLlamaModelHandle ModelHandle => ThrowIfDisposed();

        private SafeLlamaModelHandle? _model;

        private SafeLlamaModelHandle ThrowIfDisposed()
        {
            if (IsClosed)
                throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - it has been disposed");
            if (_model == null || _model.IsClosed)
                throw new ObjectDisposedException("Cannot use this `SafeLLamaContextHandle` - `SafeLlamaModelHandle` has been disposed");

            return _model!;
        }

        /// переопределенный метод класса SafeHandle из System.Runtime.InteropServices, который выполняется при Dispose
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Mtmd_Free(handle);
            SetHandle(nint.Zero);

            // Decrement refcount on model
            _model?.DangerousRelease();
            _model = null!;

            return true;
        }

        public static SafeMtmdContextHandle LoadFromFile(string mmprojPath, SafeLlamaModelHandle llamaModel, MtmdContextParams nativeParams) //mtmdcontextparams
        {
            // Try to open the mmproj file, this will check:
            // - File exists (automatically throws FileNotFoundException)
            // - File is readable (explicit check)
            using (var fs = new FileStream(mmprojPath, FileMode.Open, FileAccess.Read))
                if (!fs.CanRead)
                    throw new InvalidOperationException($"Model file '{mmprojPath}' is not readable");

            var mtmdCtx = LlamaCpp.Mtmd_InitFromFile(mmprojPath, llamaModel, nativeParams);//LlamaCpp.Llama_ModelLoadFromFile(modelPath, lparams); //mtmd_init_from_file
            if (mtmdCtx.IsInvalid)
                throw new LoadWeightsFailedException(mmprojPath);

            // Increment the model reference count while this context exists.
            // DangerousAddRef throws if it fails, so there is no need to check "success"
            mtmdCtx._model = llamaModel;
            var success = false;
            mtmdCtx._model.DangerousAddRef(ref success);

            return mtmdCtx;
        }


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
        internal void Tokenize(SafeMtmdInputChunksHandle outputChunks, List<SafeMtmdBitMapHandle> bitmaps)
        {
            LlamaCpp.Mtmd_Tokenize(this, outputChunks, bitmaps);
        } 

        
            
    }
}
