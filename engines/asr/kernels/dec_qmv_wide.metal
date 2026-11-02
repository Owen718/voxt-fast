// Small-M (2..12) quantized matmul: MLX 0.32.3 affine_qmv_wide, the kernel MLX
// dispatches for M < get_qmv_batch_limit (13 on this device) and M >= 2.
// Ported verbatim; verified against an MLX 0.32.3 reference dump.
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"

#define DEC_QMV_WIDE_B(name, bits, nv)                                      \
  template [[host_name(name)]] [[kernel]] decltype(                         \
      affine_qmv_wide<bfloat16_t, 64, bits, nv, 8, false>)                  \
      affine_qmv_wide<bfloat16_t, 64, bits, nv, 8, false>;
#define DEC_QMV_WIDE(name, nv) DEC_QMV_WIDE_B(name, 6, nv)

DEC_QMV_WIDE("dec_qmv_wide_nv2", 2)
DEC_QMV_WIDE("dec_qmv_wide_nv3", 3)
DEC_QMV_WIDE("dec_qmv_wide_nv4", 4)
DEC_QMV_WIDE("dec_qmv_wide_nv5", 5)
DEC_QMV_WIDE_B("dec_qmv_wide_b4_nv2", 4, 2)
DEC_QMV_WIDE_B("dec_qmv_wide_b4_nv3", 4, 3)
DEC_QMV_WIDE_B("dec_qmv_wide_b4_nv4", 4, 4)
DEC_QMV_WIDE_B("dec_qmv_wide_b4_nv5", 4, 5)
