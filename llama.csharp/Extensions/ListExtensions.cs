namespace Llama.csharp.Extensions
{
    internal static class ListExtensions
    {
        public static void AddSpan<T>(this List<T> list, ReadOnlySpan<T> items)
        {
            list.EnsureCapacity(list.Count + items.Length);

            for (var i = 0; i < items.Length; i++)
                list.Add(items[i]);
        }
    }
}
