# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
from typing import Optional, Literal
from .utils import is_cuda_target, is_hip_target, is_cpu_target, is_tpu_target
from tilelang import tvm as tvm
from tilelang.contrib.nvcc import get_target_compute_version
from tvm.target import Target
import ctypes
import os
import tempfile
import subprocess
import logging
from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR
from tilelang.jit.adapter.utils import get_tpu_template_dir

logger = logging.getLogger(__name__)


class LibraryGenerator(object):
    srcpath: Optional[str] = None
    libpath: Optional[str] = None
    lib_code: Optional[str] = None
    mode: Literal["pcie", "cmodel"] = "pcie"

    def __init__(self, target: Target, mode: Literal["pcie", "cmodel"] = "pcie"):
        self.target = target
        self.mode = mode

    def update_lib_code(self, lib_code: str):
        self.lib_code = lib_code

    # Assume currently we only support CUDA compilation
    def load_lib(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = self.libpath
        return ctypes.CDLL(lib_path)

    def compile_lib(self, timeout: float = None, with_tl: bool = True):
        target = self.target
        mode = self.mode
        if is_cuda_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete=False)
            compute_version = "".join(get_target_compute_version(target).split("."))
            if compute_version == "90":
                compute_version = "90a"
            libpath = src.name.replace(".cu", ".so")

            command = [
                "nvcc",
                "-std=c++17",
                "-w",  # Disable all warning messages
                "-Xcudafe",
                "--diag_suppress=177",
                "--compiler-options",
                "'-fPIC'",
                "-lineinfo",
                "--shared",
                src.name,
                "-lcuda",
                "-gencode",
                f"arch=compute_{compute_version},code=sm_{compute_version}",
            ]

        elif is_hip_target(target):
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [
                "hipcc",
                "-std=c++17",
                "-fPIC",
                "--shared",
                src.name,
            ]
        elif is_cpu_target(target):
            from tilelang.contrib.cc import get_cplus_compiler
            src = tempfile.NamedTemporaryFile(mode="w", suffix=".cpp", delete=False)
            libpath = src.name.replace(".cpp", ".so")

            command = [get_cplus_compiler(), "-std=c++17", "-fPIC", "-shared", src.name]
            with_tl = False
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
            ]
        elif is_tpu_target(target):

            src = tempfile.NamedTemporaryFile(mode="w", suffix=".c", delete=False)
            libpath = src.name.replace(".c", ".so")


            import os
            # 设置环境变量
            PPL_TOP = os.environ.get("PPL_PROJECT_ROOT", None)
            if not PPL_TOP:
                raise EnvironmentError("PPL_PROJECT_ROOT environment variable is not set.")
            CHIP = "bm1690"

            if mode=="pcie":
                self.tpu_compile_pcie(timeout=timeout, PPL_TOP=PPL_TOP, CHIP=CHIP)
            elif mode=="cmodel":
                self.tpu_compile_cmodel(timeout=timeout, PPL_TOP=PPL_TOP, CHIP=CHIP)
            else:
                raise ValueError(f"Unsupported compile mode: {mode}")
            self.srcpath = src.name
            self.libpath = f"{get_tpu_template_dir()}/main.so"
            return

        else:
            raise ValueError(f"Unsupported target: {target}")


        if with_tl:
            command += [
                "-I" + TILELANG_TEMPLATE_PATH,
                "-I" + CUTLASS_INCLUDE_DIR,
            ]
            command += ["-diag-suppress=20013"]
        command += ["-o", libpath]

        src.write(self.lib_code)
        src.flush()
        try:
            ret = subprocess.run(command, timeout=timeout)
        except Exception as e:
            raise RuntimeError(f"Compile kernel failed because of {e}") from e

        if ret.returncode != 0:
            raise RuntimeError(f"Compilation Failed! {command}")

        self.srcpath = src.name
        self.libpath = libpath

    def remove_lib(self):
        if self.libpath:
            os.remove(self.libpath)
        self.libpath = None

    def get_source_path(self):
        return self.srcpath

    def get_lib_path(self):
        return self.libpath

    def set_lib_path(self, libpath):
        self.libpath = libpath

    def set_src_path(self, srcpath):
        self.srcpath = srcpath

    def _prepare_cmodel_kernel_source(self, kernel_path: str):
        with open(kernel_path, "r") as f:
            kernel_code = f.read()

        sanitized = kernel_code.replace("      tpu_parallel_start(); \n", "")
        sanitized = sanitized.replace("      tpu_parallel_end(); \n", "")
        sanitized = sanitized.replace("tpu_parallel_start(); \n", "")
        sanitized = sanitized.replace("tpu_parallel_end(); \n", "")

        if sanitized != kernel_code:
            logger.info("Stripping TPU pipeline parallel markers for cmodel execution")
            with open(kernel_path, "w") as f:
                f.write(sanitized)

    def tpu_compile_pcie(self, timeout, PPL_TOP, CHIP):
        if CHIP != "bm1690":
            raise ValueError(f"Unsupported PCIe chip: {CHIP}")

        runtime_root = os.environ.get("CHUNKSCAN_DEVICE_RUNTIME_ROOT")
        toolchain_root = os.environ.get("CHUNKSCAN_RISCV_TOOLCHAIN_ROOT")
        if not runtime_root or not toolchain_root:
            raise EnvironmentError(
                "Set CHUNKSCAN_DEVICE_RUNTIME_ROOT and "
                "CHUNKSCAN_RISCV_TOOLCHAIN_ROOT for BM1690 PCIe mode."
            )

        cross_gcc = os.path.join(
            toolchain_root, "bin", "riscv64-unknown-linux-gnu-gcc"
        )
        runtime_header = os.path.join(runtime_root, "include", "tpuv7_rt.h")
        runtime_lib = os.path.join(runtime_root, "lib", "libtpuv7_rt.so")
        device_archive = os.path.join(
            PPL_TOP, "runtime", CHIP, "lib", f"lib{CHIP}.a"
        )
        helper_source = os.path.join(
            PPL_TOP, "runtime", "customize", "src", "ppl_helper.c"
        )

        required = {
            "RISC-V compiler": cross_gcc,
            "device runtime header": runtime_header,
            "device runtime library": runtime_lib,
            "PPL device archive": device_archive,
            "PPL helper source": helper_source,
        }
        for name, path in required.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{name} not found: {path}")
        if not os.access(cross_gcc, os.X_OK):
            raise PermissionError(f"RISC-V compiler is not executable: {cross_gcc}")

        src_dir = get_tpu_template_dir()
        includes = [
            f"-I{src_dir}",
            f"-I{PPL_TOP}/include",
            f"-I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include",
            f"-I{PPL_TOP}/runtime/kernel",
            f"-I{PPL_TOP}/runtime/customize/include",
            f"-I{runtime_root}/include",
        ]

        def run(command):
            logger.info("BM1690 PCIe compile: %s", " ".join(command))
            subprocess.run(command, check=True, timeout=timeout)

        for source, output in (
            (os.path.join(src_dir, "kernel.c"),
             os.path.join(src_dir, "kernel.o")),
            (helper_source, os.path.join(src_dir, "ppl_helper.o")),
        ):
            run([
                cross_gcc,
                f"-D__{CHIP}__",
                "-Dlibkernel_EXPORTS",
                *includes,
                "-O2",
                "-fPIC",
                "-c",
                source,
                "-o",
                output,
            ])

        run([
            cross_gcc,
            "-shared",
            "-fPIC",
            "-Wl,--no-undefined",
            "-Wl,-soname,libkernel.so",
            "-o",
            os.path.join(src_dir, "libkernel.so"),
            os.path.join(src_dir, "kernel.o"),
            os.path.join(src_dir, "ppl_helper.o"),
            "-Wl,--whole-archive",
            device_archive,
            "-Wl,--no-whole-archive",
            "-lm",
        ])

        for source, output in (
            (os.path.join(src_dir, "kernel.cpp"),
             os.path.join(src_dir, "kernel_host.o")),
            (os.path.join(src_dir, "main.cpp"),
             os.path.join(src_dir, "main.o")),
        ):
            run([
                "g++",
                f"-D__{CHIP}__",
                *includes,
                "-std=c++11",
                "-O2",
                "-fPIC",
                "-c",
                source,
                "-o",
                output,
            ])

        run([
            "g++",
            "-shared",
            "-fPIC",
            "-Wl,--no-undefined",
            "-o",
            os.path.join(src_dir, "main.so"),
            os.path.join(src_dir, "kernel_host.o"),
            os.path.join(src_dir, "main.o"),
            f"-L{runtime_root}/lib",
            f"-Wl,-rpath,{runtime_root}/lib",
            "-ltpuv7_rt",
            "-lpthread",
        ])

    def tpu_compile_cmodel(self, PPL_TOP, CHIP, timeout):
        src_dir = get_tpu_template_dir()
        KERNEL_C = f"{src_dir}/kernel.c"
        KERNEL_CPP = f"{src_dir}/kernel.cpp"
        MAIN_CPP = f"{src_dir}/main.cpp"
        CHIP = "bm1690"
        OUTPUT_PATH = f"{src_dir}"

        def execute_command(cmd, task_name, timeout):
            """Execute a shell command and handle errors"""
            # for debug
            # print(f"\n[{task_name}]")
            # print(f"Command: {cmd}")
            
            try:
                _ = subprocess.run(cmd, timeout= timeout, shell=True, check=True, text=True)
                print(f"{task_name} completed")
                return True
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Compile kernel failed because of {e}") from e

        print("=" * 60)
        print("PPL COMPILATION STARTING")
        print(f"Chip: {CHIP}")
        print(f"Output: {OUTPUT_PATH}")
        print("Mode: cmodel")
        print("=" * 60)

        self._prepare_cmodel_kernel_source(KERNEL_C)
        
        # 1. Compile kernel-cpp (kernel_cpp)
        cmd1 = f"""/usr/bin/c++ -D__{CHIP}__ \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -Wl,--no-undefined -O3 -DNDEBUG -O3 -fPIC -std=c++11 \
        -c {KERNEL_CPP} \
        -o {OUTPUT_PATH}/kernel_cpp.o"""
        
        execute_command(cmd1, "Compile kernel cpp", timeout)
        
        # 2. Compile main-cpp (main_cpp)
        cmd2 = f"""/usr/bin/c++ -D__{CHIP}__ \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -Wl,--no-undefined -O3 -DNDEBUG -O3 -fPIC -std=c++11 \
        -c {MAIN_CPP} \
        -o {OUTPUT_PATH}/main_cpp.o"""
        
        execute_command(cmd2, "Compile main cpp", timeout)
        
        # 3. Compile kernel-c (kernel_c)
        cmd3 = f"""/usr/bin/cc -D__{CHIP}__ -Dkernel_EXPORTS \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -I{OUTPUT_PATH}/include \
        -I{PPL_TOP}/include \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/common/include \
        -Wl,--no-undefined -O3 -DNDEBUG -fPIC -O3 \
        -c {KERNEL_C} \
        -o {OUTPUT_PATH}/kernel_c.o"""
        
        execute_command(cmd3, "Compile C kernel", timeout)
        
        # 4. Compile ppl_helper.c
        PPL_HELPER = f"{PPL_TOP}/runtime/customize/src/ppl_helper.c"
        cmd4 = f"""/usr/bin/cc -D__{CHIP}__ -Dkernel_EXPORTS \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/kernel/include \
        -I{PPL_TOP}/runtime/customize/include \
        -I{PPL_TOP}/runtime/kernel \
        -I{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/include \
        -I{OUTPUT_PATH}/include \
        -I{OUTPUT_PATH}/include \
        -I{PPL_TOP}/include \
        -I{PPL_TOP}/runtime/{CHIP}/TPU1686/common/include \
        -Wl,--no-undefined -O3 -DNDEBUG -fPIC -O3 \
        -c {PPL_HELPER} \
        -o {OUTPUT_PATH}/ppl_helper_c.o"""
        
        execute_command(cmd4, "Compile PPL helper", timeout)
        
        # 5. Link libkernel.so
        cmd5 = f"""/usr/bin/cc -fPIC -Wl,--no-undefined -O3 -DNDEBUG -shared -Wl,-soname,libkernel.so \
        -o {OUTPUT_PATH}/libkernel.so \
        {OUTPUT_PATH}/kernel_c.o \
        {OUTPUT_PATH}/ppl_helper_c.o \
        -L{PPL_TOP}/runtime/{CHIP}/lib \
        -L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        -Wl,-rpath,{PPL_TOP}/runtime/{CHIP}/lib:{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        {PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib/libtpuv7_emulator.so -lm"""
        
        execute_command(cmd5, "Link libkernel.so", timeout)
        
        # 6. Link executable
        cmd6 = f"""/usr/bin/c++ -O3 -DNDEBUG -fPIC -shared \
        {OUTPUT_PATH}/kernel_cpp.o \
        {OUTPUT_PATH}/main_cpp.o \
        -o {OUTPUT_PATH}/main.so \
        -L{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib \
        -L{PPL_TOP}/runtime/{CHIP}/lib \
        -Wl,--disable-new-dtags,-rpath,{PPL_TOP}/runtime/{CHIP}/tpuv7-runtime-emulator/lib:{PPL_TOP}/runtime/{CHIP}/lib \
        -ltpuv7_rt -lcdm_daemon_emulator -lpthread"""
                
        execute_command(cmd6, "Link main.so lib", timeout)
