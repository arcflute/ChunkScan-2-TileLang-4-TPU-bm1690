#include <tpuv7_rt.h>
#include "host_test_utils.h"
#include "kernel.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

tpuRtStream_t stream;
tpuRtKernelModule_t tpu_module;

namespace {{

int env_int(const char* name, int fallback) {{
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') {{
    return fallback;
  }}
  char* end = nullptr;
  long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max()) {{
    std::cerr << "Invalid positive integer in " << name << ": " << text
              << '\n';
    return -1;
  }}
  return static_cast<int>(value);
}}

double env_double(const char* name, double fallback) {{
  const char* text = std::getenv(name);
  if (text == nullptr || *text == '\0') {{
    return fallback;
  }}
  char* end = nullptr;
  double value = std::strtod(text, &end);
  if (end == text || *end != '\0' || !std::isfinite(value) || value <= 0.0) {{
    std::cerr << "Invalid positive number in " << name << ": " << text
              << '\n';
    return -1.0;
  }}
  return value;
}}

double percentile(const std::vector<double>& sorted, double q) {{
  if (sorted.empty()) {{
    return std::numeric_limits<double>::quiet_NaN();
  }}
  const double position = q * static_cast<double>(sorted.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double fraction = position - static_cast<double>(lower);
  return sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction;
}}

int init_runtime() {{
  tpuRtStatus_t ret = tpuRtInit();
  if (ret != tpuRtSuccess) {{
    std::cerr << "tpuRtInit failed: " << ret << '\n';
    return -1;
  }}

  const char* device_id_text = std::getenv("CHUNKSCAN_DEVICE_ID");
  int device_id = device_id_text ? std::atoi(device_id_text) : 0;
  ret = tpuRtSetDevice(device_id);
  if (ret != tpuRtSuccess) {{
    std::cerr << "tpuRtSetDevice(" << device_id << ") failed: " << ret
              << '\n';
    return -2;
  }}

  ret = tpuRtStreamCreate(&stream);
  if (ret != tpuRtSuccess) {{
    std::cerr << "tpuRtStreamCreate failed: " << ret << '\n';
    return -3;
  }}

  const char* kernel_path = std::getenv("PPL_KERNEL_PATH");
  if (kernel_path == nullptr || *kernel_path == '\0') {{
    std::cerr << "PPL_KERNEL_PATH is not set\n";
    tpuRtStreamDestroy(stream);
    return -4;
  }}
  tpu_module = tpuRtKernelLoadModuleFile(kernel_path, stream);
  if (tpu_module == nullptr) {{
    std::cerr << "tpuRtKernelLoadModuleFile failed: " << kernel_path << '\n';
    tpuRtStreamDestroy(stream);
    return -5;
  }}
  return 0;
}}

void close_runtime() {{
  tpuRtKernelUnloadModule(tpu_module, stream);
  tpuRtStreamDestroy(stream);
}}

}}  // namespace

extern "C" int tilelang_tpu_run(void** args) {{
{arg_declarations}

  int status = init_runtime();
  if (status != 0) {{
    return status;
  }}

{device_declarations}
{malloc_statements}
{memcpy_s2d_statements}

  auto cleanup = [&]() {{
{free_statements}
    close_runtime();
  }};

  int rst = 0;
  const char* mode_text = std::getenv("CHUNKSCAN_BENCH_MODE");
  const std::string mode = mode_text ? mode_text : "benchmark";

  if (mode == "correctness") {{
{pure_kernel_call}
    if (rst != 0) {{
      std::cerr << "correctness kernel launch failed: " << rst << '\n';
      cleanup();
      return 10;
    }}
{memcpy_d2s_statements}
    rst = tpuRtStreamSynchronize(stream);
    if (rst != 0) {{
      std::cerr << "correctness stream synchronize failed: " << rst << '\n';
      cleanup();
      return 11;
    }}
    cleanup();
    return 0;
  }}

  if (mode != "benchmark") {{
    std::cerr << "Unsupported CHUNKSCAN_BENCH_MODE: " << mode << '\n';
    cleanup();
    return 12;
  }}

  const int warmup_runs = env_int("CHUNKSCAN_BENCH_WARMUP", 100);
  const int min_runs = env_int("CHUNKSCAN_BENCH_MIN_RUNS", 100);
  const int max_runs = env_int("CHUNKSCAN_BENCH_MAX_RUNS", 2000000);
  const double min_seconds = env_double("CHUNKSCAN_BENCH_MIN_SECONDS", 5.0);
  if (warmup_runs < 0 || min_runs < 0 || max_runs < min_runs ||
      min_seconds <= 0.0) {{
    cleanup();
    return 13;
  }}

  for (int i = 0; i < warmup_runs; ++i) {{
{pure_kernel_call}
    if (rst != 0) {{
      std::cerr << "warm-up kernel launch failed at iteration " << i
                << ": " << rst << '\n';
      cleanup();
      return 14;
    }}
  }}

  using clock_type = std::chrono::steady_clock;
  std::vector<double> samples_us;
  samples_us.reserve(static_cast<size_t>(std::max(min_runs, 4096)));
  const auto measurement_start = clock_type::now();

  while (true) {{
    const auto run_start = clock_type::now();
{pure_kernel_call}
    const auto run_end = clock_type::now();
    if (rst != 0) {{
      std::cerr << "measured kernel launch failed at sample "
                << samples_us.size() << ": " << rst << '\n';
      cleanup();
      return 15;
    }}

    const double elapsed_us =
        std::chrono::duration<double, std::micro>(run_end - run_start).count();
    samples_us.push_back(elapsed_us);
    const double wall_seconds =
        std::chrono::duration<double>(run_end - measurement_start).count();

    if (samples_us.size() >= static_cast<size_t>(min_runs) &&
        wall_seconds >= min_seconds) {{
      break;
    }}
    if (samples_us.size() >= static_cast<size_t>(max_runs)) {{
      std::cerr << "CHUNKSCAN_BENCH_MAX_RUNS reached before minimum duration\n";
      cleanup();
      return 16;
    }}
  }}

  const auto measurement_end = clock_type::now();
  const double measurement_wall_seconds =
      std::chrono::duration<double>(measurement_end - measurement_start).count();

{memcpy_d2s_statements}
  rst = tpuRtStreamSynchronize(stream);
  if (rst != 0) {{
    std::cerr << "post-benchmark stream synchronize failed: " << rst << '\n';
    cleanup();
    return 17;
  }}

  std::vector<double> sorted = samples_us;
  std::sort(sorted.begin(), sorted.end());
  const double sum = std::accumulate(samples_us.begin(), samples_us.end(), 0.0);
  const double mean = sum / static_cast<double>(samples_us.size());
  double squared_error_sum = 0.0;
  for (double value : samples_us) {{
    const double delta = value - mean;
    squared_error_sum += delta * delta;
  }}
  const double stddev = std::sqrt(
      squared_error_sum / static_cast<double>(samples_us.size()));
  const double cv = mean == 0.0 ? 0.0 : stddev / mean;

  const char* result_path = std::getenv("CHUNKSCAN_BENCH_RESULT_PATH");
  if (result_path == nullptr || *result_path == '\0') {{
    std::cerr << "CHUNKSCAN_BENCH_RESULT_PATH is not set\n";
    cleanup();
    return 18;
  }}
  const char* stage_text = std::getenv("CHUNKSCAN_BENCH_STAGE");
  const std::string stage = stage_text ? stage_text : "unknown";

  std::ofstream output(result_path, std::ios::out | std::ios::trunc);
  if (!output) {{
    std::cerr << "Cannot open benchmark result: " << result_path << '\n';
    cleanup();
    return 19;
  }}

  output << std::setprecision(17);
  output << "{{\n";
  output << "  \"schema_version\": 1,\n";
  output << "  \"status\": \"PASS\",\n";
  output << "  \"stage\": \"" << stage << "\",\n";
  output << "  \"timer\": \"steady_clock_sync_kernel_call\",\n";
  output << "  \"includes_launch_and_sync\": true,\n";
  output << "  \"includes_h2d_d2h\": false,\n";
  output << "  \"warmup_runs\": " << warmup_runs << ",\n";
  output << "  \"min_requested_runs\": " << min_runs << ",\n";
  output << "  \"min_requested_seconds\": " << min_seconds << ",\n";
  output << "  \"sample_count\": " << samples_us.size() << ",\n";
  output << "  \"measurement_wall_seconds\": "
         << measurement_wall_seconds << ",\n";
  output << "  \"min_us\": " << sorted.front() << ",\n";
  output << "  \"p50_us\": " << percentile(sorted, 0.50) << ",\n";
  output << "  \"p90_us\": " << percentile(sorted, 0.90) << ",\n";
  output << "  \"p95_us\": " << percentile(sorted, 0.95) << ",\n";
  output << "  \"max_us\": " << sorted.back() << ",\n";
  output << "  \"mean_us\": " << mean << ",\n";
  output << "  \"stddev_us\": " << stddev << ",\n";
  output << "  \"cv\": " << cv << ",\n";
  output << "  \"samples_us\": [";
  for (size_t i = 0; i < samples_us.size(); ++i) {{
    if (i != 0) {{
      output << ',';
    }}
    output << samples_us[i];
  }}
  output << "]\n";
  output << "}}\n";
  output.close();
  if (!output) {{
    std::cerr << "Failed while writing benchmark result: " << result_path
              << '\n';
    cleanup();
    return 20;
  }}

  std::cout << "CHUNKSCAN_BENCH_PASS stage=" << stage
            << " samples=" << samples_us.size()
            << " p50_us=" << percentile(sorted, 0.50)
            << " p95_us=" << percentile(sorted, 0.95)
            << " cv=" << cv << '\n';

  cleanup();
  return 0;
}}
