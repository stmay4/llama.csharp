using System.Diagnostics;
using System.Text;
using CommunityToolkit.HighPerformance.Buffers;
using Llama.csharp.Extensions;
using Llama.csharp.Exceptions;

namespace Llama.csharp.Native
{
    /// <summary>
    /// A reference to a set of llama model weights
    /// </summary>
    public sealed class SafeLlamaModelHandle
        : SafeLLamaHandleBase
    {
        /// <summary>
        /// Get the rope (positional embedding) type for this model
        /// </summary>
        public LLamaRopeType RopeType => LlamaCpp.Llama_ModelRopeType(this);

        /// <summary>
        /// The number of tokens in the context that this model was trained for
        /// </summary>
        public int ContextSize => LlamaCpp.Llama_ModelNContextTrain(this);

        /// <summary>
        /// Get the rope frequency this model was trained with
        /// </summary>
        public float RopeFrequency => LlamaCpp.Llama_ModelRopeFreqScaleTrain(this);

        /// <summary>
        /// Dimension of embedding vectors
        /// </summary>
        public int EmbeddingSize => LlamaCpp.Llama_ModelNEmbd(this);

        /// <summary>
        /// Get the size of this model in bytes
        /// </summary>
        public ulong SizeInBytes => LlamaCpp.Llama_ModelSize(this);

        /// <summary>
        /// Get the number of parameters in this model
        /// </summary>
        public ulong ParameterCount => LlamaCpp.Llama_ModelNParams(this);

        /// <summary>
        /// Get the number of layers in this model
        /// </summary>
        public int LayerCount => LlamaCpp.Llama_ModelNLayer(this);

        /// <summary>
        /// Get the number of heads in this model
        /// </summary>
        public int HeadCount => LlamaCpp.Llama_ModelNHead(this);

        /// <summary>
        /// Get the number of KV heads in this model
        /// </summary>
        public int KVHeadCount => LlamaCpp.Llama_ModelNHeadKV(this);

        /// <summary>
        /// Returns true if the model contains an encoder that requires llama_encode() call
        /// </summary>
        public bool HasEncoder => LlamaCpp.Llama_ModelHasEncoder(this);

        /// <summary>
        /// Returns true if the model contains a decoder that requires llama_decode() call
        /// </summary>
        public bool HasDecoder => LlamaCpp.Llama_ModelHasDecoder(this);

        /// <summary>
        /// Returns true if the model is recurrent (like Mamba, RWKV, etc.)
        /// </summary>
        public bool IsRecurrent => LlamaCpp.Llama_ModelIsRecurrent(this);

        /// <summary>
        /// Returns true if the model is hybrid (like Jamba, Granite, etc.)
        /// </summary>
        public bool IsHybrid => LlamaCpp.Llama_ModelIsHybrid(this);

        /// <summary>
        /// Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
        /// </summary>
        public bool IsDiffusion => LlamaCpp.Llama_ModelIsDiffusion(this);

        /// <summary>
        /// Get a description of this model
        /// </summary>
        public string Description
        {
            get
            {
                unsafe
                {
                    // Get description length
                    var size = LlamaCpp.Llama_ModelDesc(this, null, 0);
                    var buf = new byte[size + 1];
                    fixed (byte* bufPtr = buf)
                    {
                        size = LlamaCpp.Llama_ModelDesc(this, bufPtr, buf.Length);
                        return Encoding.UTF8.GetString(buf, 0, size);
                    }
                }
            }
        }

        /// <summary>
        /// Get the number of metadata key/value pairs
        /// </summary>
        /// <returns></returns>
        public int MetadataCount => LlamaCpp.Llama_ModelMetaCount(this);

        private Vocabulary? _vocab;

        /// <summary>
        /// Get the vocabulary of this model
        /// </summary>
        public Vocabulary Vocab => _vocab ??= new Vocabulary(this);

        /// переопределенный метод класса SafeHandle из System.Runtime.InteropServices, который выполняется при Dispose
        protected override bool ReleaseHandle()
        {
            LlamaCpp.Llama_ModelFree(handle);
            return true;
        }

        private SafeLlamaModelHandle() { }

        /// <summary>
        /// Load a model from the given file path into memory
        /// </summary>
        /// <param name="modelPath"></param>
        /// <param name="lparams"></param>
        /// <returns></returns>
        /// <exception cref="RuntimeError"></exception>
        public static SafeLlamaModelHandle LoadFromFile(string modelPath, LLamaModelParams lparams)
        {
            // Try to open the model file, this will check:
            // - File exists (automatically throws FileNotFoundException)
            // - File is readable (explicit check)
            // This provides better error messages that llama.cpp, which would throw an access violation exception in both cases.
            using (var fs = new FileStream(modelPath, FileMode.Open, FileAccess.Read))
                if (!fs.CanRead)
                    throw new InvalidOperationException($"Model file '{modelPath}' is not readable");

            var handle = LlamaCpp.Llama_ModelLoadFromFile(modelPath, lparams);
            if (handle.IsInvalid)
                throw new LoadWeightsFailedException(modelPath);

            return handle;
        }

        #region LoRA
        /// <summary>
        /// Load a LoRA adapter from file. The adapter will be associated with this model but will not be applied
        /// </summary>
        /// <param name="path"></param>
        /// <returns></returns>
        /// <exception cref="InvalidOperationException"></exception>
        //public LoraAdapter LoadLoraFromFile(string path)
        //{
        //    path = Path.GetFullPath(path);

        //    // Try to open the model file, this will check:
        //    // - File exists (automatically throws FileNotFoundException)
        //    // - File is readable (explicit check)
        //    // This provides better error messages that llama.cpp, which would throw an access violation exception in both cases.
        //    using (var fs = new FileStream(path, FileMode.Open))
        //        if (!fs.CanRead)
        //            throw new InvalidOperationException($"LoRA file '{path}' is not readable");

        //    var ptr = llama_adapter_lora_init(this, path);
        //    return new LoraAdapter(this, path, ptr);
        //}
        #endregion
    

        #region metadata
        /// <summary>
        /// Get the metadata value for the given key
        /// </summary>
        /// <param name="key">The key to fetch</param>
        /// <returns>The value, null if there is no such key</returns>
        public Memory<byte>? MetadataValueByKey(string key)
        {
            // Check if the key exists, without getting any bytes of data
            var keyLength = LlamaCpp.Llama_ModelMetaValStr(this, key, []);
            if (keyLength < 0)
                return null;

            // get a buffer large enough to hold it
            var buffer = new byte[keyLength + 1];
            keyLength = LlamaCpp.Llama_ModelMetaValStr(this, key, buffer);
            Debug.Assert(keyLength >= 0);

            return buffer.AsMemory().Slice(0, keyLength);
        }

        /// <summary>
        /// Get the metadata key for the given index
        /// </summary>
        /// <param name="index">The index to get</param>
        /// <returns>The key, null if there is no such key or if the buffer was too small</returns>
        public Memory<byte>? MetadataKeyByIndex(int index)
        {
            // Check if the key exists, without getting any bytes of data
            var keyLength = LlamaCpp.Llama_ModelMetaKeyByIndex(this, index, Array.Empty<byte>());
            if (keyLength < 0)
                return null;

            // get a buffer large enough to hold it
            var buffer = new byte[keyLength + 1];
            keyLength = LlamaCpp.Llama_ModelMetaKeyByIndex(this, index, buffer);
            Debug.Assert(keyLength >= 0);

            return buffer.AsMemory().Slice(0, keyLength);
        }

        /// <summary>
        /// Get the metadata value for the given index
        /// </summary>
        /// <param name="index">The index to get</param>
        /// <returns>The value, null if there is no such value or if the buffer was too small</returns>
        public Memory<byte>? MetadataValueByIndex(int index)
        {
            // Check if the key exists, without getting any bytes of data
            var valueLength = LlamaCpp.Llama_ModelMetaValStrByIndex(this, index, Array.Empty<byte>());
            if (valueLength < 0)
                return null;

            // get a buffer large enough to hold it
            var buffer = new byte[valueLength + 1];
            valueLength = LlamaCpp.Llama_ModelMetaValStrByIndex(this, index, buffer);
            Debug.Assert(valueLength >= 0);

            return buffer.AsMemory().Slice(0, valueLength);
        }

        internal IReadOnlyDictionary<string, string> ReadMetadata()
        {
            var result = new Dictionary<string, string>();

            for (var i = 0; i < MetadataCount; i++)
            {
                var keyBytes = MetadataKeyByIndex(i);
                if (keyBytes == null)
                    continue;
                var key = Encoding.UTF8.GetString(keyBytes.Value.Span);

                var valBytes = MetadataValueByIndex(i);
                if (valBytes == null)
                    continue;
                var val = Encoding.UTF8.GetString(valBytes.Value.Span);

                result[key] = val;
            }

            return result;
        }
        #endregion

        #region template
        /// <summary>
        /// Get the default chat template. Returns nullptr if not available
        /// If name is NULL, returns the default chat template
        /// </summary>
        /// <param name="name">The name of the template, in case there are many or differently named. Set to 'null' for the default behaviour of finding an appropriate match.</param>
        /// <param name="strict">Setting this to true will cause the call to throw if no valid templates are found.</param>
        /// <returns></returns>
        public string? GetTemplate(string? name = null, bool strict = true)
        {
            unsafe
            {
                var bytesPtr = LlamaCpp.Llama_ModelChatTemplate(this, name);
                if (bytesPtr == null)
                {
                    if (strict)
                        throw new TemplateNotFoundException(name ?? "default template");
                    else
                        return null;

                }

                // Find null terminator
                var spanBytes = new Span<byte>(bytesPtr, int.MaxValue);
                spanBytes = spanBytes.Slice(0, spanBytes.IndexOf((byte)0));

                return Encoding.UTF8.GetString(spanBytes);
            }
        }
        #endregion

        /// <summary>
        /// Get tokens for a model
        /// </summary>
        public sealed class Vocabulary
        {
            private readonly unsafe LLamaVocabNative* _vocabNative;
            internal unsafe LLamaVocabNative* VocabNative => _vocabNative;

            internal Vocabulary(SafeLlamaModelHandle model)
            {
                unsafe
                {
                    _vocabNative = LlamaCpp.Llama_ModelGetVocab(model);
                }
                DecoderStartToken = Normalize(LlamaCpp.Llama_ModelDecoderStartToken(model));
            }

            private static LLamaToken? Normalize(LLamaToken token)
            {
                return token == -1 ? null : token;
            }

            /// <summary>
            /// Convert a single llama token into bytes
            /// </summary>
            /// <param name="token">Token to decode</param>
            /// <param name="dest">A span to attempt to write into. If this is too small nothing will be written</param>
            /// <param name="lstrip">User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')</param>
            /// <param name="special">If true, special characters will be converted to text. If false they will be invisible.</param>
            /// <returns>The size of this token. **nothing will be written** if this is larger than `dest`</returns>
            public uint TokenToSpan(LLamaToken token, Span<byte> dest, int lstrip = 0, bool special = false)
            {
                unsafe
                {
                    var length = LlamaCpp.Llama_VocabTokenToPiece(_vocabNative, token, dest, lstrip, special);
                    return (uint)Math.Abs(length);
                }
            }

            /// <summary>
            /// Convert a string of text into tokens
            /// </summary>
            /// <param name="text"></param>
            /// <param name="addBos"></param>
            /// <param name="encoding"></param>
            /// <param name="special">Allow tokenizing special and/or control tokens which otherwise are not exposed and treated as plaintext.</param>
            /// <returns></returns>
            public LLamaToken[] Tokenize(string text, bool addBos, bool special, Encoding encoding)
            {
                // Early exit if there's no work to do
                if (text == string.Empty && !addBos)
                    return [];

                // Convert string to bytes, adding one extra byte to the end (null terminator)
                var bytesCount = encoding.GetByteCount(text);
                using var bytes = SpanOwner<byte>.Allocate(bytesCount + 1, AllocationMode.Clear);

                encoding.GetBytes(text, bytes.Span);

                unsafe
                {
                    // Tokenize once with no output, to get the token count. Output will be negative (indicating that there was insufficient space)
                    var count = -LlamaCpp.Llama_VocabTokenize(_vocabNative, bytes, bytesCount, null, 0, addBos, special);

                    // Tokenize again, this time outputting into an array of exactly the right size
                    var tokens = new LLamaToken[count];

                    _ = LlamaCpp.Llama_VocabTokenize(_vocabNative, bytes, bytesCount, tokens, count, addBos, special);
                    return tokens;
                }
            }

            /// <summary>
            /// Translate LLamaToken to String
            /// </summary>
            /// <param name="token"></param>
            /// <param name="isSpecialToken"></param>
            /// <returns></returns>
            public string? LLamaTokenToString(LLamaToken? token, bool isSpecialToken)
            {
                if (!token.HasValue)
                    return null;

                // Try to convert using a fixed size buffer
                const int buffSize = 32;
                Span<byte> buff = stackalloc byte[buffSize];
                var tokenLength = TokenToSpan((LLamaToken)token, buff, special: isSpecialToken);

                // Negative indicates that there was no result
                if (tokenLength <= 0)
                    return null;

                // if the original buffer wasn't large enough, try again with one that's the right size
                if (tokenLength > buffSize)
                {
                    buff = stackalloc byte[(int)tokenLength];
                    _ = TokenToSpan((LLamaToken)token, buff, special: isSpecialToken);
                }

                var slice = buff.Slice(0, (int)tokenLength);
                return Encoding.UTF8.GetString(slice);
            }

            /// <summary>
            /// Total number of tokens in this vocabulary
            /// </summary>
            public int Count
            {
                get
                {
                    unsafe
                    {
                        return LlamaCpp.Llama_VocabNTokens(_vocabNative);
                    }
                }
            }

            /// <summary>
            /// Get the the type of this vocabulary
            /// </summary>
            public LLamaVocabType Type
            {
                get
                {
                    unsafe
                    {
                        return LlamaCpp.Llama_VocabType(_vocabNative);
                    }
                }
            }

            /// <summary>
            /// Get the Beginning of Sentence token for this model
            /// </summary>
            public LLamaToken? BOS
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabBos(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Get the End of Sentence token for this model
            /// </summary>
            public LLamaToken? EOS
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabEos(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Get the newline token for this model
            /// </summary>
            public LLamaToken? Newline
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabNl(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Get the padding token for this model
            /// </summary>
            public LLamaToken? Pad
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabPad(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Get the sentence separator token for this model
            /// </summary>
            public LLamaToken? SEP
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabSep(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama beginning of infill prefix
            /// </summary>
            public LLamaToken? InfillPrefix
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimPre(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama beginning of infill middle
            /// </summary>
            public LLamaToken? InfillMiddle
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimMid(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama beginning of infill suffix
            /// </summary>
            public LLamaToken? InfillSuffix
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimSuf(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama pad
            /// </summary>
            public LLamaToken? InfillPad
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimPad(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama rep
            /// </summary>
            public LLamaToken? InfillRep
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimRep(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// Codellama rep
            /// </summary>
            public LLamaToken? InfillSep
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabFimSep(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// end-of-turn token
            /// </summary>
            public LLamaToken? EOT
            {
                get
                {
                    unsafe
                    {
                        return Normalize(LlamaCpp.Llama_VocabEot(_vocabNative));
                    }
                }
            }

            /// <summary>
            /// For encoder-decoder models, this function returns id of the token that must be provided
            /// to the decoder to start generating output sequence.
            /// </summary>
            public LLamaToken? DecoderStartToken;

            /// <summary>
            /// Check if the current model requires a BOS token added
            /// </summary>
            public bool ShouldAddBOS
            {
                get
                {
                    unsafe
                    {
                        return LlamaCpp.Llama_VocabGetAddBos(_vocabNative);
                    }
                }
            }

            /// <summary>
            /// Check if the current model requires a EOS token added
            /// </summary>
            public bool ShouldAddEOS
            {
                get
                {
                    unsafe
                    {
                        return LlamaCpp.Llama_VocabGetAddEos(_vocabNative);
                    }
                }
            }
        }
    }
}
