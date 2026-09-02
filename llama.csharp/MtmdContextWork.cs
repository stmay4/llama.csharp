namespace Llama.csharp
{
    internal class MtmdContextWork
    {
        private readonly Task<LlamaEmbedding[]> workTask;

        internal MtmdContextWork(List<Task<LlamaEmbedding[]>> tasks)
        {
            this.workTask = CombineTasksAsync(tasks);
        }

        private async Task<LlamaEmbedding[]> CombineTasksAsync(List<Task<LlamaEmbedding[]>> tasks)
        {
            // Ждем выполнения всех кусков
            LlamaEmbedding[][] results = await Task.WhenAll(tasks);

            if (results.Length == 0) return Array.Empty<LlamaEmbedding>();
            if (results.Length == 1) return results[0];

            // Считаем общий размер, чтобы выделить финальный массив
            int totalLength = 0;
            foreach (var res in results) totalLength += res.Length;

            LlamaEmbedding[] finalResult = new LlamaEmbedding[totalLength];

            // Быстрое копирование без лишних List и AddRange
            int offset = 0;
            foreach (var res in results)
            {
                Array.Copy(res, 0, finalResult, offset, res.Length);
                offset += res.Length;
            }

            return finalResult;
        }

        internal Task<LlamaEmbedding[]> GetWork() => workTask;
    }
}
