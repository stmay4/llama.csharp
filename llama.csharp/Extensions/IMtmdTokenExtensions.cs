using Llama.csharp.Native;

namespace Llama.csharp.Extensions
{
    public static class MtmdTokenConstants
    {
        // Reserve a range of negative numbers for internal placeholder tokens
        // -1 is already taken by InvalidToken, so start from -100 to avoid conflicts
        // with special model tokens (BOS, EOS, PAD), which are typically >= 0.
        public const int MtmdImagePlaceholderBase = -100;
        public const int MtmdVideoPlaceholderBase = -101;
        public const int MtmdAudioPlaceholderBase = -102;
    }

    public static class MtmdTokenExtensions
    {
        /// <summary>
        /// Checks whether the token is an internal MTMD placeholder
        /// </summary>
        public static bool IsMtmdPlaceholder(this LLamaToken token)
        {
            int val = (int)token;
            return val <= -100 && val >= -200; // Проверяем зарезервированный диапазон
        }

        /// <summary>
        /// Creates an image placeholder token.
        /// </summary>
        public static LLamaToken CreateImagePlaceholder()
        {
            return (LLamaToken)(MtmdTokenConstants.MtmdImagePlaceholderBase);
        }

        /// <summary>
        /// Creates a video placeholder token.
        /// </summary>
        public static LLamaToken CreateVideoPlaceholder()
        {
            return (LLamaToken)(MtmdTokenConstants.MtmdVideoPlaceholderBase);
        }

        /// <summary>
        /// Creates an audio placeholder token.
        /// </summary>
        public static LLamaToken CreateAudioPlaceholder()
        {
            return (LLamaToken)(MtmdTokenConstants.MtmdAudioPlaceholderBase);
        }
    }
}
