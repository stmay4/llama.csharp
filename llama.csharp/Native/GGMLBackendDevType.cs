using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Llama.csharp.Native
{
    /// <summary>
    /// C++ enum ggml_backend_dev_type
    /// </summary>
    internal enum GGMLBackendDevType
    {

        // CPU device using system memory
        GGML_BACKEND_DEVICE_TYPE_CPU,
        // GPU device using dedicated memory
        GGML_BACKEND_DEVICE_TYPE_GPU,
        // integrated GPU device using host memory
        GGML_BACKEND_DEVICE_TYPE_IGPU,
        // accelerator devices intended to be used together with the CPU backend (e.g. BLAS or AMX)
        GGML_BACKEND_DEVICE_TYPE_ACCEL
    }
}
