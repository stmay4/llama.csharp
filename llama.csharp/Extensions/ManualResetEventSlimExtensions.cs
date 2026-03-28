using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Extensions
{
    internal static class ManualResetEventSlimExtensions
    {
        public static Task WaitAsync(this ManualResetEventSlim mres, CancellationToken cancellationToken = default)
        {
            // Быстрый путь: если уже установлен, возвращаем завершенную задачу сразу, 
            // чтобы не тратить ресурсы пула потоков напрасно.
            if (mres.IsSet)
                return Task.CompletedTask;

            return WaitAsync(mres.WaitHandle, cancellationToken);
        }

        public static Task WaitAsync(this WaitHandle waitHandle, CancellationToken cancellationToken = default)
        {
            var tcs = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            RegisteredWaitHandle? registeredHandle = null;
            CancellationTokenRegistration? registration = null;

            // Флаг, чтобы понять, кто первым завершил задачу (событие или отмена)
            bool isCompleted = false;

            registeredHandle = ThreadPool.RegisterWaitForSingleObject(
                waitObject: waitHandle,
                callBack: (state, timedOut) =>
                {
                    // Событие сработало
                    if (!isCompleted)
                    {
                        isCompleted = true;
                        registration?.Unregister();
                        tcs.TrySetResult();
                    }

                    // ВАЖНО: Всегда освобождаем ресурс, даже если задача уже завершена отменой
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

            // Гарантируем очистку регистрации токена, когда задача завершится (успешно или нет)
            // Это нужно, чтобы delegate токена не держал ссылку на tcs бесконечно, если событие сработало первым
            _ = tcs.Task.ContinueWith(_ =>
            {
                registration?.Dispose();// Двойная страховка
            }, TaskScheduler.Default);

            return tcs.Task;
        }
    }
}
