using Llama.csharp.Interfaces;
using Llama.csharp.Native;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Extensions
{
    public static class IMtmdParamsExtensions
    {
        /// <summary>
        /// Convert the given `IMtmdParams` into a `LlamaMtmdParams`
        /// </summary>
        /// <param name="params"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public static void ToLlamaContextParams(this IMtmdParams @params, out LlamaMtmdParams result)
        {
            result = LlamaMtmdParams.Default();
            result.use_gpu = @params.UseGpu;
            result.print_timings = @params.PrintTimings;
            result.n_threads = @params.Threads ?? result.n_threads;
            result.media_marker = @params.MediaMarker ?? result.media_marker;
            result.flash_attn_type = @params.FlashAttention ?? result.flash_attn_type;
            result.warmup = @params.Warmup ?? result.warmup;
            result.image_min_tokens = @params.ImageMinTokens ?? result.image_min_tokens;
            result.image_max_tokens = @params.ImageMaxTokens ?? result.image_max_tokens;
            result.batch_max_tokens = @params.BatchSize ?? result.batch_max_tokens;
        }
    }
}
