namespace Llama.csharp
{
    /// <summary>
    /// AntipromptProcessor keeps track of past tokens looking for any set Anti-Prompts
    /// </summary>
    public sealed class AntipromptProcessor
    {
        private int _longestAntiprompt;
        private readonly List<string> _antiprompts = new();

        private string? _string;


        /// <summary>
        /// Initializes a new instance of the <see cref="AntipromptProcessor"/> class.
        /// </summary>
        /// <param name="antiprompts">The antiprompts.</param>
        public AntipromptProcessor(IEnumerable<string>? antiprompts = null)
        {
            if (antiprompts != null)
                SetAntiprompts(antiprompts);
        }

        /// <summary>
        /// Add an antiprompt to the collection
        /// </summary>
        /// <param name="antiprompt"></param>
        public void AddAntiprompt(string antiprompt)
        {
            _antiprompts.Add(antiprompt);
            _longestAntiprompt = Math.Max(_longestAntiprompt, antiprompt.Length);
        }

        /// <summary>
        /// Overwrite all current antiprompts with a new set
        /// </summary>
        /// <param name="antiprompts"></param>
        public void SetAntiprompts(IEnumerable<string> antiprompts)
        {
            _antiprompts.Clear();
            _antiprompts.AddRange(antiprompts);

            _longestAntiprompt = 0;
            foreach (var antiprompt in _antiprompts)
                _longestAntiprompt = Math.Max(_longestAntiprompt, antiprompt.Length);
        }

        /// <summary>
        /// Clear current string for find antiprompt
        /// </summary>
        public void ClearString()
        {
            _string = "";
        }

        /// <summary>
        /// Add some text and check if the buffer now CONTAINS any antiprompt
        /// </summary>
        /// <param name="text"></param>
        /// <returns>true if the text buffer ends with any antiprompt</returns>
        public bool Add(string text)
        {
            _string += text;

            var maxLength = Math.Max(32, _longestAntiprompt * 4);
            var trimLength = Math.Max(16, _longestAntiprompt * 2);
            if (_string.Length > maxLength)
                _string = _string.Substring(_string.Length - trimLength);

            foreach (var antiprompt in _antiprompts)
                if (_string.Contains(antiprompt)) //Contains Instead of EndsWith
                    return true;

            return false;
        }
    }
}