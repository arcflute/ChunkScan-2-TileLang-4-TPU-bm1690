#include "kernel.h"

#include <ppl_mem.h>
#include <tpuv7_rt.h>

#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern tpuRtStream_t stream;
extern tpuRtKernelModule_t tpu_module;

#define MIN(x, y) (((x)) < ((y)) ? (x) : (y))
#define MAX(x, y) (((x)) > ((y)) ? (x) : (y))

namespace {{

int requested_core_num() {{
  const char* text = std::getenv("CHUNKSCAN_CORE_NUM");
  if (text == nullptr || *text == '\0') {{
    return 1;
  }}

  errno = 0;
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value > INT_MAX) {{
    return -1;
  }}
  if (value != 1 && value != 2 && value != 4 && value != 8) {{
    return -1;
  }}
  return static_cast<int>(value);
}}

}}  // namespace

int {function_name}_check_mem({func_params}) {{
  return 0;
}}

int {function_name}_check_mem_s(tpu_kernel_api_{function_name}_t* api) {{
  return 0;
}}

tpu_kernel_api_{function_name}_t fill_{function_name}_struct({func_params}) {{
  tpu_kernel_api_{function_name}_t api;
  {struct_assignments}
  return api;
}}

int {function_name}({func_params}) {{
  const int core_num = requested_core_num();
  if (core_num < 0) {{
    std::fprintf(
        stderr,
        "CHUNKSCAN_CORE_NUM must be one of 1, 2, 4, or 8\n");
    return -100;
  }}

  tpu_kernel_api_{function_name}_t api;
  {struct_assignments}
  std::vector<tpu_kernel_api_{function_name}_t> apis(
      static_cast<size_t>(core_num), api);

  const uint64_t group_num = 1;
  const uint64_t block_num = static_cast<uint64_t>(core_num);
  const uint32_t argument_bytes = static_cast<uint32_t>(
      apis.size() * sizeof(tpu_kernel_api_{function_name}_t));

  int ret = tpuRtKernelLaunch(
      tpu_module,
      "{function_name}",
      apis.data(),
      argument_bytes,
      group_num,
      block_num,
      stream);
  if (ret != 0) {{
    std::fprintf(stderr, "TPU kernel launch failed: %d\n", ret);
    return ret;
  }}

  ret = tpuRtStreamSynchronize(stream);
  if (ret != 0) {{
    std::fprintf(stderr, "TPU stream synchronize failed: %d\n", ret);
    return ret;
  }}
  return 0;
}}
