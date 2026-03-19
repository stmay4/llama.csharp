
namespace Llama.csharp.Interfaces
{
    public interface IInferenceParams
    {
        /// <summary>
        /// number of tokens to keep from initial prompt
        /// </summary>
        public int TokensKeep { get; set; } // елси реализую словарь сообщений и он будет работать то не нужно

        /// <summary>
        /// how many new tokens to predict (n_predict), set to -1 to inifinitely generate response
        /// until it complete.
        /// </summary>
        public int MaxTokens { get; set; }

        /// <summary>
		/// if true print bos, eos and etc.
		/// </summary>
        public bool DecodeSpecialTokens { get; set; }

        /// <summary>
		/// automatically stop ingerence when eog received. It happens in PostProcess
		/// </summary>
        public bool AutoStopFromEOG { get; set; }

        /// <summary>
        /// Sequences where the model will stop generating further tokens.
        /// </summary>
        public IReadOnlyList<string> AntiPrompts { get; set; }

        /// <summary>
        /// Set a custom sampling pipeline to use.
        /// </summary>
        ISamplingPipeline SamplingPipeline { get; set; }
    }
}
