
## Introduction

sgl-deep-gemm is a pypi package built from SGLang's customized branch of DeepGemm. Comparing with origina DeepGemm, it supports the following features to better support SGLang:
1. ABI support: with the help of tvm-ffi wrappers, a single wheel can run on different python versions.
2. pypi support: easy installation with `pip install sgl-deep-gemm`. No need to manually search for wheel links.
3. Fast iteration: add custom kernels and bump versions at no time.

## Usage
Build requirements include CUDA 12.9 or newer, a C++20 compiler with `std::format` support (for example GCC 13), and the elfutils development headers (`apt-get install libdw-dev` on Debian/Ubuntu). Initialize the CUTLASS and DeepJIT submodules with `git submodule update --init --recursive`.

To build it locally, run `bash build_sgl_deep_gemm.sh`, then pip install the wheel generated under `dist`.

### Installing a fork revision with pip

The `sgl_deep_gemm/source` directory provides a pip VCS entry point that calls
the existing wheel builder. It installs the distribution `sgl-deep-gemm` and
keeps the Python import name `deep_gemm`. No manual patch application is needed.
After pushing the desired revision to your public fork, install it with:

```bash
TVM_FFI_CUDA_ARCH_LIST=9.0a python -m pip install \
  "sgl-deep-gemm @ git+https://github.com/xuantengh/DeepGEMM.git@feat/sm90-w4a8-pip#subdirectory=sgl_deep_gemm/source"
```

Replace the branch with its full commit SHA when declaring a reproducible
dependency in `pyproject.toml`. The architecture above is for Hopper; select
the architecture of your GPU. pip fetches the Git submodules automatically.
The CUDA toolkit, C++ compiler, and elfutils headers listed above are still
required. This is a source build, so the first installation compiles `_C.so`.

The isolated build uses Torch 2.14.1, matching the SGLang wheel build. In an
existing SGLang environment, `--no-build-isolation` reuses the installed Torch
and build tools instead; ensure the packages listed in
`source/pyproject.toml` are installed. A wheel must be used with a compatible
Torch/CUDA runtime. For a local checkout, use
`python -m pip install --no-build-isolation ./sgl_deep_gemm/source`.

The VCS entry point builds wheels directly from the full repository. Standalone
sdist archives and editable installs are not supported by this adapter. The
existing `bash build_sgl_deep_gemm.sh` release workflow is unchanged.

K-grouped FP8 GEMM and its scale-packing helper preserve compact per-group scaling factors by default. Pass `use_padded_sf_layout=True` to both packing and GEMM when packed scales cover each group's padded K extent. The flag is appended to existing arguments, so existing positional calls retain their behavior.

Run wheel validation with `bash sgl_deep_gemm/run_tests.sh`. The Mega MoE reference comparisons require DeepEP with `ElasticBuffer`; Mega Gate and Mega mHC also require TileKernels and TileLang. Install the validation extras with `pip install 'sgl-deep-gemm[dev]'` (TileLang 0.1.9 and TileKernels 1.0.0). The runner enables NVML-based CUDA discovery to preserve fork compatibility while retaining TVM FFI's DLPack fast path. It runs legacy, lazy-init, and compute-sanitizer checks; `--skip-sanitizer` is available for ordinary development runs. Distributed MegaMoE runs include separate Hopper numerical accuracy coverage; the sanitizer's automatic discovery covers single-process tests and reports distributed entrypoints separately.

Run `bash sgl_deep_gemm/run_tests.sh --release` for reduced attention coverage and focused `memcheck`/`synccheck`, targeting a release gate under one hour. Selected cases retain their original assertions; other ordinary and distributed tests remain unchanged. The default command keeps the full suite; the table shows release / full attention case counts.

| GPU architecture | Dense MQA | Paged MQA | Sparse MQA |
| --- | ---: | ---: | ---: |
| Hopper (SM90) | 8 / 192 | 8 / 24 | unsupported |
| Blackwell (SM100/SM103) | 36 / 2304 | 72 / 4320 | 12 / 161 |
| Blackwell (SM120) | 16 / 256 | 16 / 48 | unsupported |

Use Compute Sanitizer 2025.4.1 or newer for Python tests; older releases can retain Python tensors while collecting host backtraces and exhaust GPU memory. Set `COMPUTE_SANITIZER` to select a separately installed executable. See [NVIDIA's release notes](https://docs.nvidia.com/compute-sanitizer/ReleaseNotes/index.html#updates-in-2025-4-1).

To release a new set of wheels, please contact SGLang team and run the [release workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepgemm.yml) under SGLang repo

For each major version release (0.X.Y -> 0.(X+1).0), a new branch should be created (release/v0.(X+1).0) for stability purpose.

For any incoming pull requests, it should be rebased upon `dev` branch. Any newly added or modified tests should be put under `sgl_deep_gemm/tests`
