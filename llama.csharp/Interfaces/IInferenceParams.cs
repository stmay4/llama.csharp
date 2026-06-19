
namespace Llama.csharp.Interfaces
{
    public interface IInferenceParams
    {
        /// <summary>
        /// number of tokens to keep from initial prompt
        /// </summary>
        public int TokensKeep { get; }

        /// <summary>
        /// how many new tokens to predict (n_predict), set to -1 to inifinitely generate response
        /// until it complete.
        /// </summary>
        public int MaxTokens { get; }

        /// <summary>
		/// if true print bos, eos and etc.
		/// </summary>
        public bool DecodeSpecialTokens { get; }

        /// <summary>
		/// automatically stop ingerence when eog received. It happens in PostProcess
		/// </summary>
        public bool AutoStopFromEOG { get; }

        /// <summary>
        /// Sequences where the model will stop generating further tokens.
        /// </summary>
        public IReadOnlyList<string> AntiPrompts { get; }

        /// <summary>
        /// Set a custom sampling pipeline to use.
        /// </summary>
        ISamplingPipeline SamplingPipeline { get; }
    }
}
