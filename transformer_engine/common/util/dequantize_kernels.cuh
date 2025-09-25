/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file dequantize_kernels.cuh
 *  \brief CUDA kernels to cast from MXFP8.
 */

#ifndef TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
#define TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>
#include <transformer_engine/cast.h>

#include <cfloat>
#include <limits>

#include "../common.h"
#include "../transpose/cast_transpose.h"
#include "../util/vectorized_pointwise.h"
#include "../utils.cuh"
#include "math.h"
#include "ptx.cuh"
#include "transformer_engine/activation.h"
#include "transformer_engine/transpose.h"

namespace transformer_engine {

namespace dequantization {

constexpr size_t CHUNK_DIM_Y = 128;
constexpr size_t CHUNK_DIM_X = 128;
constexpr size_t THREADS_PER_CHUNK = 128;
constexpr size_t BUFFERS_NUM = 2;

constexpr size_t ELEMS_PER_THREAD = 16;
constexpr size_t BUFFER_DIM_Y = 16;           // only 32 is supported
constexpr size_t BUFFER_DIM_X = CHUNK_DIM_X;  // 128
constexpr size_t SHMEM_DIM_Y = BUFFER_DIM_Y;  // 16
constexpr size_t SHMEM_DIM_X = BUFFER_DIM_X;  // 128

constexpr size_t THREADS_PER_CHUNK_X_ROWWISE = CHUNK_DIM_X / ELEMS_PER_THREAD;  //  8 = 128 / 16
constexpr size_t THREADS_PER_CHUNK_X_COLWISE = CHUNK_DIM_X;                     //  128
constexpr size_t ITERATIONS = CHUNK_DIM_Y / BUFFER_DIM_Y;                       //    8 = 128 / 16
static_assert(ITERATIONS >= 1);

template <typename IType, typename OType, size_t SCALE_DIM_Y, size_t SCALE_DIM_X, typename ScaleType>
__global__ void __launch_bounds__(THREADS_PER_CHUNK)
    dequantize_mxnv_kernel(const __grid_constant__ CUtensorMap tensor_map_input,
                           const __grid_constant__ CUtensorMap tensor_map_output,
                           const ScaleType *const scales_ptr, const size_t rows, const size_t cols,
                           const size_t scales_stride) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const size_t packing = [&] {
    if constexpr (std::is_same_v<IType, fp8e4m3> || std::is_same_v<IType, fp8e5m2>) {
      return 1;
    } else if constexpr (std::is_same_v<IType, fp4e2m1>) {
      return 2;
    } else {
      static_assert(!std::is_same_v<IType, IType>, "Unsupported OType");
      return 0;
    }
  }();
  constexpr bool USE_ROWWISE_SCALING = SCALE_DIM_X > 1;

  constexpr size_t SCALES_ROWWISE_PER_CHUNK_Y = CHUNK_DIM_Y;                //  128
  constexpr size_t SCALES_ROWWISE_PER_CHUNK_X = CHUNK_DIM_X / SCALE_DIM_X;  //  mx:4, nv:8

  constexpr size_t SCALES_COLWISE_PER_CHUNK_Y = CHUNK_DIM_Y / SCALE_DIM_Y;  //  mx:4, nv:8
  constexpr size_t SCALES_COLWISE_PER_CHUNK_X = CHUNK_DIM_X;                //  128

  constexpr size_t THREADS_PER_SCALE_X_ROWWISE =
      DIVUP(SCALE_DIM_X, ELEMS_PER_THREAD);  //  mx:2, nv:1

  const int chunk_offset_Y = blockIdx.y * CHUNK_DIM_Y;
  const int chunk_offset_X = blockIdx.x * CHUNK_DIM_X / packing;

  const int scales_rowwise_chunk_offset_Y = blockIdx.y * SCALES_ROWWISE_PER_CHUNK_Y;
  const int scales_rowwise_chunk_offset_X = blockIdx.x * SCALES_ROWWISE_PER_CHUNK_X;
  const int scales_colwise_chunk_offset_Y = blockIdx.y * SCALES_COLWISE_PER_CHUNK_Y;
  const int scales_colwise_chunk_offset_X = blockIdx.x * SCALES_COLWISE_PER_CHUNK_X;

  const int tid_rowwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_ROWWISE;
  const int tid_rowwise_X = threadIdx.x % THREADS_PER_CHUNK_X_ROWWISE;
  // const int tid_colwise_Y = threadIdx.x / THREADS_PER_CHUNK_X_COLWISE;
  const int tid_colwise_X = threadIdx.x % THREADS_PER_CHUNK_X_COLWISE;

  const int thread_offset_Y = tid_rowwise_Y;
  const int thread_offset_X_rowwise = tid_rowwise_X * ELEMS_PER_THREAD;
  // const int thread_offset_X_colwise = tid_colwise_X;

  // The destination shared memory buffer of a bulk tensor operation should be 128 e8m0_t aligned
  __shared__ alignas(128) IType in_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X/packing];
  __shared__ alignas(128) OType out_sh[BUFFERS_NUM][SHMEM_DIM_Y][SHMEM_DIM_X];

  constexpr int shmem_buff_size = sizeof(in_sh) / BUFFERS_NUM;
  constexpr int transaction_size = shmem_buff_size;

  const bool is_master_thread = (threadIdx.x == 0);

// Initialize shared memory barrier with the number of threads participating in the barrier.
#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ alignas(8) uint64_t mbar[ITERATIONS];

  if (is_master_thread) {
// Initialize barrier. All `blockDim.x * blockDim.y` threads in block participate.
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_init(&mbar[iter], THREADS_PER_CHUNK);
    }
    ptx::fence_proxy_async_shared_cta();
  }
  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  int parity = 0;
  constexpr int iteration_zero = 0;
  constexpr int buffer_zero = 0;
  if (is_master_thread) {
    const int chunk_stage_offset_Y = chunk_offset_Y;
    const int chunk_stage_offset_X = chunk_offset_X;
    // Initiate bulk tensor copy
    ptx::cp_async_bulk_tensor_2d_global_to_shared(
        reinterpret_cast<uint64_t *>(&in_sh[buffer_zero]),
        reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_stage_offset_X,
        chunk_stage_offset_Y, &mbar[iteration_zero]);

    // Arrive on the barrier and tell how many bytes are expected to come in.
    ptx::mbarrier_arrive_expect_tx(&mbar[iteration_zero], transaction_size);

  } else {
    // Other threads just arrive
    ptx::mbarrier_arrive(&mbar[iteration_zero]);
  }

#pragma unroll
  for (int iter = 0; iter < ITERATIONS; ++iter) {
    const int buff = iter % BUFFERS_NUM;
    const int next_iter = iter + 1;
    if (next_iter < ITERATIONS) {
      if (is_master_thread) {
        const int next_buff = next_iter % BUFFERS_NUM;
        const int chunk_it_offset_y = chunk_offset_Y + next_iter * BUFFER_DIM_Y;
        const int chunk_it_offset_x = chunk_offset_X;
        // Initiate bulk tensor copy
        ptx::cp_async_bulk_tensor_2d_global_to_shared(
            reinterpret_cast<uint64_t *>(&in_sh[next_buff]),
            reinterpret_cast<const uint64_t *>(&tensor_map_input), chunk_it_offset_x,
            chunk_it_offset_y, &mbar[next_iter]);

        // Arrive on the barrier and tell how many bytes are expected to come in.
        ptx::mbarrier_arrive_expect_tx(&mbar[next_iter], transaction_size);
      } else {
        // Other threads just arrive
        ptx::mbarrier_arrive(&mbar[next_iter]);
      }
    }

    ptx::fence_proxy_async_shared_cta();

    // Wait for the data to have arrived
    ptx::mbarrier_wait_parity(&mbar[iter], parity);

    const int scale_offset_Y =
        USE_ROWWISE_SCALING ? (scales_rowwise_chunk_offset_Y + iter * BUFFER_DIM_Y + tid_rowwise_Y)
                            : (scales_colwise_chunk_offset_Y + (iter * BUFFER_DIM_Y) / SCALE_DIM_Y);

    const int scale_offset_X =
        USE_ROWWISE_SCALING
            ? (scales_rowwise_chunk_offset_X + tid_rowwise_X / THREADS_PER_SCALE_X_ROWWISE)
            : (scales_colwise_chunk_offset_X + tid_colwise_X);

    const int scale_idx = scale_offset_Y * scales_stride + scale_offset_X;

    const float block_scale = [&] {
      if constexpr (std::is_same_v<ScaleType, e8m0_t>) {
        const e8m0_t biased_exponent = scales_ptr[scale_idx];
        return exp2f(static_cast<float>(biased_exponent) - FP32_EXPONENT_BIAS);
      } else if constexpr (std::is_same_v<ScaleType, fp8e4m3>) {
        return static_cast<float>(scales_ptr[scale_idx]);
      } else {
        static_assert(!std::is_same_v<ScaleType, ScaleType>, "Unsupported ScaleType");
        return 0.0f; 
      }
    }();

    if constexpr (USE_ROWWISE_SCALING) {
      Vec<IType, ELEMS_PER_THREAD> in;
      Vec<OType, ELEMS_PER_THREAD> out;

      const int shmem_offset_y = thread_offset_Y;
      const int shmem_offset_x = thread_offset_X_rowwise;
      in.load_from(&in_sh[buff][shmem_offset_y][shmem_offset_x]);

#pragma unroll
      for (int j = 0; j < ELEMS_PER_THREAD/packing; ++j) {
        if constexpr (std::is_same_v<IType, fp8e4m3> || std::is_same_v<IType, fp8e5m2>) {
          out.data.elt[j] = static_cast<OType>(block_scale * static_cast<float>(in.data.elt[j]));
        } else if constexpr (std::is_same_v<IType, fp4e2m1>) {
          // fp4(y), fp4(x) -> fp16.x, fp16.y (no need special handling, just reversing the convention of how we pack)
          __half2_raw hfraw2 = __nv_cvt_fp4x2_to_halfraw2(in.data.elt[j].__x, __NV_E2M1);
          __half2 h2;
          memcpy(&h2, &hfraw2, sizeof(h2));
          out.data.elt[j*2]   = static_cast<OType>(block_scale * static_cast<float>(h2.x));
          out.data.elt[j*2+1] = static_cast<OType>(block_scale * static_cast<float>(h2.y));
        } else {
          static_assert(!std::is_same_v<IType, IType>, "Unsupported IType");
        }
      }
      out.store_to(&out_sh[buff][shmem_offset_y][shmem_offset_x]);
    } else {
#pragma unroll
      for (int i = 0; i < BUFFER_DIM_Y/packing; ++i) {
        if constexpr (std::is_same_v<IType, fp8e4m3> || std::is_same_v<IType, fp8e5m2>) {
          const float elt = static_cast<float>(in_sh[buff][i][tid_colwise_X]);
          out_sh[buff][i][tid_colwise_X] = static_cast<OType>(block_scale * elt);
        } else if constexpr (std::is_same_v<IType, fp4e2m1>) {
          // fp4(y), fp4(x) -> fp16.x, fp16.y (no need special handling, just reversing the convention of how we pack)
          __half2_raw hfraw2 = __nv_cvt_fp4x2_to_halfraw2(in_sh[buff][i][tid_colwise_X].__x, __NV_E2M1);
          __half2 h2;
          memcpy(&h2, &hfraw2, sizeof(h2));
          out_sh[buff][i*2][tid_colwise_X]   = static_cast<OType>(block_scale * static_cast<float>(h2.x));
          out_sh[buff][i*2+1][tid_colwise_X] = static_cast<OType>(block_scale * static_cast<float>(h2.y));
        } else {
          static_assert(!std::is_same_v<IType, IType>, "Unsupported IType");
        }
      }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    ptx::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (is_master_thread) {
      const int chunk_it_offset_y = chunk_offset_Y + iter * BUFFER_DIM_Y;
      const int chunk_it_offset_x = chunk_offset_X;
      ptx::cp_async_bulk_tensor_2d_shared_to_global(
          reinterpret_cast<const uint64_t *>(&tensor_map_output), chunk_it_offset_x,
          chunk_it_offset_y, reinterpret_cast<uint64_t *>(&out_sh[buff]));

      // Create a "bulk async-group" out of the previous bulk copy operation.
      ptx::cp_async_bulk_commit_group();

      // Wait for TMA transfer to have finished reading shared memory.
      ptx::cp_async_bulk_wait_group_read<1>();
    }
  }
  ptx::cp_async_bulk_wait_group_read<0>();
  __syncthreads();

  parity ^= 1;

  // Destroy barrier. This invalidates the memory region of the barrier. If
  // further computations were to take place in the kernel, this allows the
  // memory location of the shared memory barrier to be reused.
  if (is_master_thread) {
#pragma unroll
    for (int iter = 0; iter < ITERATIONS; ++iter) {
      ptx::mbarrier_invalid(&mbar[iter]);
    }
  }
#endif  // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
}

static void fp8_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  NVTE_CHECK(is_fp8_dtype(input.data.dtype), "Input must have FP8 type.");
  NVTE_CHECK(!is_fp8_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input.data.shape, "Input and output shapes need to match.");

  const size_t N = product(input.data.shape);
  TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
      input.data.dtype, IType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
          output->data.dtype, OType,

          constexpr int nvec = 32 / sizeof(OType);
          detail::DequantizeParam p;
          p.scale_inv = reinterpret_cast<const fp32 *>(input.scale_inv.dptr);
          VectorizedUnaryKernelLauncher<nvec, detail::DequantizeParam, detail::dequantize_func>(
              reinterpret_cast<const IType *>(input.data.dptr), nullptr,
              reinterpret_cast<OType *>(output->data.dptr), nullptr, nullptr, nullptr, N, p,
              stream););  // NOLINT(*)
  );                      // NOLINT(*)
}

template <const size_t ScaleDim, typename ScaleType>
static void mxnv_dequantize(const Tensor &input, Tensor *output, cudaStream_t stream) {
  bool use_rowwise_scaling = input.has_data();
  bool use_colwise_scaling = input.has_columnwise_data();
  checkCuDriverContext(stream);

  auto input_shape = input.data.shape;
  NVTE_CHECK(input_shape.size() >= 2, "Input must have at least 2 dimensions.");
  if (input.scaling_mode == NVTE_NVFP4_1D_SCALING) {
    input_shape[1] *= 2; // tensor.shape() use rowwise data shape.
  }

  if (use_rowwise_scaling) {
    NVTE_CHECK(input.has_data(), "Cannot dequantize tensor without rowwise data.");
    NVTE_CHECK(is_narrow_dtype(input.data.dtype), "Input must have FP8 type.");
  }

  if (use_colwise_scaling) {
    NVTE_CHECK(input.has_columnwise_data(), "Cannot dequantize tensor without columnwise data.");
    NVTE_CHECK(is_narrow_dtype(input.columnwise_data.dtype), "Input must have FP8 type.");
  }

  NVTE_CHECK(!is_narrow_dtype(output->data.dtype), "Output must be in higher precision.");
  NVTE_CHECK(output->data.shape == input_shape, "Input and output shapes need to match.");

  // TODO: Make more general
  const size_t scale_dim_X_rowwise = use_rowwise_scaling ? ScaleDim : 1;
  const size_t scale_dim_Y_colwise = use_colwise_scaling ? ScaleDim : 1;

  // t.flat_first_dim()/.flat_first_dim() depends on t.shape() which has a rather odd design
  // t.shape returns rowwise data shape when it exists and
  // shape of colwise data shape iff only rowwise data does not exist
  // when both exist, rowwise data shape gets returned.
  // rows, cols are logical dim.
  size_t rows, cols;
  if (input.has_data()) {
    rows = input.flat_first_dim();
    cols = input.scaling_mode == NVTE_NVFP4_1D_SCALING ? input.flat_last_dim() * 2 : input.flat_last_dim();
  } else if (input.has_columnwise_data()) {
    rows = input.scaling_mode == NVTE_NVFP4_1D_SCALING ? input.flat_first_dim() * 2 : input.flat_first_dim();
    cols = input.flat_last_dim();
  }

  const size_t chunks_Y = DIVUP(rows, CHUNK_DIM_Y);
  const size_t chunks_X = DIVUP(cols, CHUNK_DIM_X);

  const size_t unpadded_scales_Y_rowwise = rows;
  const size_t unpadded_scales_X_rowwise = DIVUP(cols, scale_dim_X_rowwise);
  const size_t unpadded_scales_Y_colwise = DIVUP(rows, scale_dim_Y_colwise);
  const size_t unpadded_scales_X_colwise = cols;

  const size_t scales_Y_rowwise =
      DIVUP(unpadded_scales_Y_rowwise, scale_tensor_alignment_Y_rowwise) *
      scale_tensor_alignment_Y_rowwise;
  const size_t scales_X_rowwise =
      DIVUP(unpadded_scales_X_rowwise, scale_tensor_alignment_X_rowwise) *
      scale_tensor_alignment_X_rowwise;
  const size_t scales_Y_colwise =
      DIVUP(unpadded_scales_Y_colwise, scale_tensor_alignment_Y_colwise) *
      scale_tensor_alignment_Y_colwise;
  const size_t scales_X_colwise =
      DIVUP(unpadded_scales_X_colwise, scale_tensor_alignment_X_colwise) *
      scale_tensor_alignment_X_colwise;

  const ScaleType *const scales_ptr =
      use_rowwise_scaling ? reinterpret_cast<ScaleType *>(input.scale_inv.dptr)
                          : reinterpret_cast<ScaleType *>(input.columnwise_scale_inv.dptr);

  const size_t scales_stride = use_rowwise_scaling ? scales_X_rowwise : scales_X_colwise;

  const SimpleTensor &input_data = use_rowwise_scaling ? input.data : input.columnwise_data;

  const dim3 block(THREADS_PER_CHUNK);
  const dim3 grid(chunks_X, chunks_Y);

  TRANSFORMER_ENGINE_MXNV_SCALE_DIM_SWITCH(
      scale_dim_Y_colwise, SCALE_DIM_Y,
      TRANSFORMER_ENGINE_MXNV_SCALE_DIM_SWITCH(
          scale_dim_X_rowwise, SCALE_DIM_X,
          TRANSFORMER_ENGINE_TYPE_SWITCH_FP8FP4ONLY(
              input.dtype(), IType,
              TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(
                  output->dtype(), OType,

                  alignas(64) CUtensorMap tensor_map_input{};
                  alignas(64) CUtensorMap tensor_map_output{};

                  create_2D_tensor_map(tensor_map_input, input_data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(input.dtype()));
                  create_2D_tensor_map(tensor_map_output, output->data, rows, cols, SHMEM_DIM_Y,
                                       SHMEM_DIM_X, cols, 0, typeToNumBits(output->dtype()));

                  dequantize_mxnv_kernel<IType, OType, SCALE_DIM_Y, SCALE_DIM_X, ScaleType>
                  <<<grid, block, 0, stream>>>(tensor_map_input, tensor_map_output, scales_ptr,
                                               rows, cols, scales_stride););  // NOLINT(*)
          );                                                                  // NOLINT(*)
      );                                                                      // NOLINT(*)
  );                                                                          // NOLINT(*)
}
}  // namespace dequantization

namespace detail {

void dequantize_helper(const Tensor &input, Tensor *output, cudaStream_t stream) {
  CheckInputTensor(input, "cast_input");
  CheckOutputTensor(*output, "cast_output");

  if (is_tensor_scaling(input.scaling_mode)) {
    dequantization::fp8_dequantize(input, output, stream);
  } else if (is_mxfp_scaling(input.scaling_mode) || is_nvfp_scaling(input.scaling_mode)) {
    if (is_supported_by_CC_100()) {
      switch (input.scaling_mode) {
        case NVTE_MXFP8_1D_SCALING:
          dequantization::mxnv_dequantize<32, e8m0_t>(input, output, stream);
          break;
        case NVTE_NVFP4_1D_SCALING:
          dequantization::mxnv_dequantize<16, fp8e4m3>(input, output, stream);
          break;
        default:
          NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
      }
    } else {
      NVTE_ERROR("MXFP8/NVFP4 Dequantization is NOT supported by architectures < 10.0");
    }
  } else {
    // TODO(kwyss): Move dequantization code from torch to C++ for NVTE_BLOCK_SCALING
    NVTE_ERROR("Not implemented scaling mode: " + to_string(input.scaling_mode) + ".");
  }
}

}  // namespace detail

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_DEQUANTIZE_KERNELS_CUH_
