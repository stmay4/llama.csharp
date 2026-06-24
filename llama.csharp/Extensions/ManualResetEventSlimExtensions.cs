namespace Llama.csharp.Extensions
{
    /// <summary>
    /// Extension for the ability to use Await with a work availability signal in the executor.
    /// </summary>
    internal static class ManualResetEventSlimExtensions
    {
        /// <summary>
        /// Custom WaitAsync for ManualResetEventSlim, which does not natively support async waiting.
        /// </summary>
        /// <param name="mres"></param>
        /// <param name="cancellationToken"></param>
        /// <returns></returns>
        public static Task WaitAsync(this ManualResetEventSlim mres, CancellationToken cancellationToken = default)
        {
            // Fast path: if already set, return a completed task immediately,
            // so as not to waste thread pool resources unnecessarily.
            if (mres.IsSet)
                return Task.CompletedTask;

            return WaitAsync(mres.WaitHandle, cancellationToken);
        }

        public static Task WaitAsync(this WaitHandle waitHandle, CancellationToken cancellationToken = default)
        {
            var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            RegisteredWaitHandle? registeredHandle = null;
            CancellationTokenRegistration? registration = null;

            // Flag to determine which completed the task first (event or cancellation)
            bool isCompleted = false;

            registeredHandle = ThreadPool.RegisterWaitForSingleObject(
                waitObject: waitHandle,
                callBack: (state, timedOut) =>
                {
                    // Event signaled
                    if (!isCompleted)
                    {
                        isCompleted = true;
                        registration?.Unregister();
                        tcs.TrySetResult();
                    }

                    // IMPORTANT: Always release the resource, even if the task has already been completed by cancellation
                    registeredHandle?.Unregister(waitHandle);
                },
                state: null,
                timeout: Timeout.InfiniteTimeSpan,
                executeOnlyOnce: true);

            if (cancellationToken.CanBeCanceled)
            {
                registration = cancellationToken.Register(() =>
                {
                    if (!isCompleted)
                    {
                        isCompleted = true;
                        tcs.TrySetCanceled(cancellationToken);
                        registeredHandle?.Unregister(waitHandle);
                    }
                });
            }

            // Ensure cleanup of the token registration when the task completes (successfully or not)
            _ = tcs.Task.ContinueWith(_ =>
            {
                registration?.Dispose(); // Extra safety measure
            }, TaskScheduler.Default);

            return tcs.Task;
        }
    }
}
