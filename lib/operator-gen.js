/**
 * Operator generation: turn an advice item into AscendC operator sources.
 *
 * The handoff built by `./handoff.js` says *what* to change and *against what bar*;
 * this module produces the artefact the advice is really about — a complete
 * AscendC operator project for the operator the advice targets, with the
 * optimization from that advice already applied:
 *
 * * `row-norm`  — RMSNorm-style row reduction, fused to a single UB pass over the
 *   row (the framework path reduces over a separate fp32 copy and stages it back),
 *   fp32 accumulation, vector `Reciprocal` instead of a scalar divide;
 * * `quantize`  — dynamic per-row int8 quantize + dequant fused into one kernel, so
 *   the framework stops issuing a quantize and a dequantize pass per tensor;
 * * `elementwise` — a fused elementwise chain for advice that is about removing
 *   per-op dispatch overhead.
 *
 * Everything is deterministic: no model, no network, no NPU is involved, so pressing
 * 执行 produces the same project every time for the same advice. The generated
 * project is written into `<workspace>/operator-work/<adviceId>/` and the three
 * framework registration points are emitted as a patch guide rather than edited
 * into somebody else's tree.
 *
 * What generation cannot do is prove the operator runs: build, precision and
 * performance need CANN and an NPU, and the generated README says so instead of
 * pretending otherwise.
 *
 * @module dsh-plugin-vllm-ascend-profiler/operator-gen
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';

/** Where generated operator projects go, relative to the session workspace. */
export const OPERATOR_DIR = 'operator-work';

/**
 * Which operator class an advice maps to.
 *
 * Kept separate from the skill mapping in `handoff.js`: that one says which skills
 * *run* the work, this one says which *source template* the advice implies.
 */
export const ADVICE_OPERATOR = Object.freeze({
  'compute.quantize': { kind: 'quantize', op: 'dyn_quant_dequant', title: '动态 per-row INT8 量化/反量化融合算子' },
  'compute.fuse-small-ops': { kind: 'elementwise', op: 'fused_activation_chain', title: '融合逐元素链（Mul+Add+Silu）算子' },
  'compute.decode-increase-batch': { kind: 'quantize', op: 'dyn_quant_dequant', title: '动态 per-row INT8 量化/反量化融合算子' },
  'compute.tune-chunked-prefill': { kind: 'row-norm', op: 'rms_norm_fused', title: '单趟融合 RMSNorm 算子' },
  'copy.kv-locality': { kind: 'row-norm', op: 'rms_norm_fused', title: '单趟融合 RMSNorm 算子（KV 局部性改造）' },
});

/** Fallback class per bottleneck when an advice has no explicit mapping. */
const FOCUS_OPERATOR = Object.freeze({
  kernel: { kind: 'elementwise', op: 'fused_activation_chain', title: '融合逐元素链算子' },
  comm: { kind: 'elementwise', op: 'fused_activation_chain', title: '融合逐元素链算子（为通信重叠让出算力）' },
  copy: { kind: 'quantize', op: 'dyn_quant_dequant', title: '动态量化/反量化融合算子（减少搬运字节）' },
  dispatch: { kind: 'row-norm', op: 'rms_norm_fused', title: '单趟融合 RMSNorm 算子（减少逐步下发）' },
  graph: { kind: 'row-norm', op: 'rms_norm_fused', title: '单趟融合 RMSNorm 算子（减少逐步下发）' },
  config: { kind: 'row-norm', op: 'rms_norm_fused', title: '单趟融合 RMSNorm 算子' },
});

/**
 * Pick the operator class for an advice item.
 *
 * @param {object} task - package from `buildHandoff`.
 * @returns {{kind: string, op: string, title: string, reason: string}} choice.
 */
export function planOperator(task) {
  const mapped = ADVICE_OPERATOR[task.adviceId];
  if (mapped !== undefined) {
    return { ...mapped, reason: `优化项 ${task.adviceId} 直接对应算子类型 ${mapped.kind}` };
  }
  const byFocus = FOCUS_OPERATOR[task.focus];
  if (byFocus !== undefined) {
    return { ...byFocus, reason: `优化项 ${task.adviceId} 属 ${String(task.focus)} 焦点，落到 ${byFocus.kind} 模板` };
  }
  return { kind: 'elementwise', op: 'fused_activation_chain', title: '融合逐元素链算子', reason: '无明确映射，使用通用逐元素模板' };
}

/** Shared header, kept identical across generated files. */
const LICENSE = `// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
`;

/** Per-class source templates. Each returns the four generated source files. */
const TEMPLATES = {
  'row-norm': (context) => ({
    kernel: `${LICENSE}
// ============================================================
// ${context.op} op_kernel — single-pass fused row normalisation
//
// Optimisation applied (from advice ${context.adviceId}):
//   * ONE pass over the row: x^2 and the accumulation happen in the same UB tile,
//     so the framework no longer reduces over a separate fp32 copy;
//   * fp32 accumulation with an explicit up-cast for fp16/bf16;
//   * vector Reciprocal instead of a scalar divide (no std::* in the kernel);
//   * gamma fused in fp32 and loaded once per core;
//   * GM<->UB strictly through DataCopyPad, double buffered.
// ============================================================

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Kernel${context.className} {
public:
    __aicore__ inline Kernel${context.className}() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                int64_t cols, int64_t colsAlign,
                                int64_t formerNum, int64_t formerRows, int64_t tailRows,
                                float eps)
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        const bool isTail = (blockIdx >= formerNum);
        this->blockRows = isTail ? tailRows : formerRows;
        const int64_t rowOffset = isTail ? formerRows * formerNum + tailRows * (blockIdx - formerNum)
                                         : formerRows * blockIdx;
        this->cols = cols;
        this->colsAlign = colsAlign;
        this->eps = eps;
        xGm.SetGlobalBuffer((__gm__ T *)x + rowOffset * colsAlign, this->blockRows * colsAlign);
        yGm.SetGlobalBuffer((__gm__ T *)y + rowOffset * colsAlign, this->blockRows * colsAlign);
        gammaGm.SetGlobalBuffer((__gm__ T *)gamma, cols);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, colsAlign * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colsAlign * sizeof(T));
        pipe.InitBuffer(bufX32, colsAlign * sizeof(float));
        pipe.InitBuffer(bufSq, colsAlign * sizeof(float));
        pipe.InitBuffer(bufWork, colsAlign * sizeof(float));
        pipe.InitBuffer(bufAcc, 32);
        pipe.InitBuffer(bufGamma, colsAlign * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LoadGamma();
        for (int64_t row = 0; row < this->blockRows; ++row) {
            CopyIn(row);
            Compute(row);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void LoadGamma()
    {
        AscendC::LocalTensor<T> raw = gammaQueue.AllocTensor<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{true, 0, static_cast<uint8_t>(this->colsAlign - this->cols), static_cast<T>(0)};
        AscendC::DataCopyPad(raw, gammaGm, params, pad);
        gammaQueue.EnQue(raw);
        AscendC::LocalTensor<T> in = gammaQueue.DeQue<T>();
        AscendC::LocalTensor<float> g32 = bufGamma.Get<float>();
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(g32, in, 0.0f, static_cast<int32_t>(this->colsAlign));
        } else {
            AscendC::Cast(g32, in, AscendC::RoundMode::CAST_NONE, static_cast<int32_t>(this->colsAlign));
        }
        gammaQueue.FreeTensor(in);
    }

    __aicore__ inline void CopyIn(int64_t row)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[row * this->colsAlign], params, pad);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t row)
    {
        (void)row;
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<float> x32 = bufX32.Get<float>();
        AscendC::LocalTensor<float> sq = bufSq.Get<float>();
        AscendC::LocalTensor<float> work = bufWork.Get<float>();
        AscendC::LocalTensor<float> acc = bufAcc.Get<float>();
        AscendC::LocalTensor<float> g32 = bufGamma.Get<float>();

        const int32_t cols = static_cast<int32_t>(this->cols);
        const int32_t colsAlign = static_cast<int32_t>(this->colsAlign);
        const float rcols = 1.0f / static_cast<float>(this->cols);

        // up-cast once; x32 keeps the row for the normalisation below
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(x32, xLocal, 0.0f, colsAlign);
        } else {
            AscendC::Cast(x32, xLocal, AscendC::RoundMode::CAST_NONE, colsAlign);
        }
        // single pass: square into its own buffer, then reduce with a work buffer
        // (dst / src / work are three distinct buffers — the reduce may modify both)
        AscendC::Mul(sq, x32, x32, colsAlign);
        AscendC::ReduceSum<float, true>(acc, sq, work, cols);
        AscendC::Muls(acc, acc, rcols, 1);
        AscendC::Adds(acc, acc, this->eps, 1);
        AscendC::Sqrt(acc, acc, 1);
        AscendC::Reciprocal(acc, acc, 1);

        const float inv = acc.GetValue(0);
        AscendC::Muls(x32, x32, inv, colsAlign);
        AscendC::Mul(x32, x32, g32, colsAlign);
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(yLocal, x32, 0.0f, colsAlign);
        } else {
            AscendC::Cast(yLocal, x32, AscendC::RoundMode::CAST_RINT, colsAlign);
        }
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t row)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[row * this->colsAlign], yLocal, params);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> gammaQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX32, bufSq, bufWork, bufAcc, bufGamma;
    AscendC::GlobalTensor<T> xGm, yGm, gammaGm;
    int64_t blockRows = 0;
    int64_t cols = 0;
    int64_t colsAlign = 0;
    float eps = 1e-6f;
};

// dtypeCode: 0 = float32, 1 = float16, 2 = bfloat16 (never a bool parameter).
extern "C" __global__ __aicore__ void ${context.op}(GM_ADDR x, GM_ADDR gamma, GM_ADDR y,
                                                    int64_t cols, int64_t colsAlign,
                                                    int64_t formerNum, int64_t formerRows, int64_t tailRows,
                                                    int64_t dtypeCode, float eps)
{
    if (dtypeCode == 0) {
        Kernel${context.className}<float> op;
        op.Init(x, gamma, y, cols, colsAlign, formerNum, formerRows, tailRows, eps);
        op.Process();
    } else if (dtypeCode == 1) {
        Kernel${context.className}<half> op;
        op.Init(x, gamma, y, cols, colsAlign, formerNum, formerRows, tailRows, eps);
        op.Process();
    } else {
        Kernel${context.className}<bfloat16_t> op;
        op.Init(x, gamma, y, cols, colsAlign, formerNum, formerRows, tailRows, eps);
        op.Process();
    }
}
`,
    host: `${LICENSE}
// ============================================================
// ${context.op} op_host — tiling + launch for the fused row normalisation
// Hardware parameters come from the platform API; nothing is hardcoded.
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_${context.op}.h"

namespace ascend_kernel {

at::Tensor ${context.op}(const at::Tensor &input, const at::Tensor &gamma, double eps)
{
    TORCH_CHECK(input.dim() == 2, "${context.op}: input must be 2-D [rows, cols]");
    TORCH_CHECK(input.scalar_type() == gamma.scalar_type(), "${context.op}: gamma dtype must match input");
    TORCH_CHECK(input.is_contiguous() && gamma.is_contiguous(), "${context.op}: inputs must be contiguous");
    TORCH_CHECK(eps > 0.0, "${context.op}: eps must be > 0");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    TORCH_CHECK(gamma.size(0) == cols, "${context.op}: gamma length must equal cols");
    at::Tensor output = at::empty_like(input);
    if (rows == 0 || cols == 0) {
        return output;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const int64_t dtypeSize = input.element_size();
    const int64_t alignElements = 32 / dtypeSize;
    const int64_t colsAlign = ((cols + alignElements - 1) / alignElements) * alignElements;
    // fp16/bf16: in+out queues (2*2 B) + one fp32 working buffer (4 B) = 12 B/element
    // fp32:      in+out queues (2*4 B)                       = 16 B/element
    const int64_t bufferCoefficient = (dtypeSize == 2) ? 12 : 16;
    const int64_t maxCols = ((static_cast<int64_t>(ubSize) / bufferCoefficient) / alignElements) * alignElements;
    TORCH_CHECK(colsAlign <= maxCols, "${context.op}: cols exceeds UB capacity (max ", maxCols, ")");

    const int64_t usedCoreNum = std::min(rows, coreNum);
    const int64_t formerRows = (rows + usedCoreNum - 1) / usedCoreNum;
    const int64_t formerNum = usedCoreNum - 1;
    const int64_t tailRows = rows - formerRows * formerNum;

    at::Tensor kernelInput = (cols == colsAlign)
        ? input
        : at::constant_pad_nd(input, {0, colsAlign - cols}, 0.0).contiguous();
    at::Tensor kernelOutput = at::empty_like(kernelInput);

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    const int64_t dtypeCode = (input.scalar_type() == at::kFloat) ? 0
                            : (input.scalar_type() == at::kHalf) ? 1 : 2;
    float epsF = static_cast<float>(eps);

    EXEC_KERNEL_CMD(${context.op}, blockDim, kernelInput, gamma, kernelOutput,
                    cols, colsAlign, formerNum, formerRows, tailRows, dtypeCode, epsF);

    output = (cols == colsAlign) ? kernelOutput : kernelOutput.narrow(-1, 0, cols).contiguous();
    return output;
}

}  // namespace ascend_kernel
`,
  }),
  quantize: (context) => ({
    kernel: `${LICENSE}
// ============================================================
// ${context.op} op_kernel — dynamic per-row INT8 quantize + dequant, fused
//
// Optimisation applied (from advice ${context.adviceId}):
//   * quantize and dequantize run in ONE kernel pass per row, so the framework
//     stops issuing two elementwise passes plus a scale round trip;
//   * the scale is computed in fp32 from the row's max|x| (symmetric, per row),
//     so no calibration data and no extra D2H read are needed at run time;
//   * fp16/bf16 are up-cast before the max/scale math;
//   * output keeps the original dtype, ready to feed a quantized matmul.
// ============================================================

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr float Q_MAX = 127.0f;

template <typename T>
class Kernel${context.className} {
public:
    __aicore__ inline Kernel${context.className}() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                int64_t cols, int64_t colsAlign,
                                int64_t formerNum, int64_t formerRows, int64_t tailRows)
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        const bool isTail = (blockIdx >= formerNum);
        this->blockRows = isTail ? tailRows : formerRows;
        this->rowBase = isTail ? formerRows * formerNum + tailRows * (blockIdx - formerNum)
                               : formerRows * blockIdx;
        this->cols = cols;
        this->colsAlign = colsAlign;
        xGm.SetGlobalBuffer((__gm__ T *)x + this->rowBase * colsAlign, this->blockRows * colsAlign);
        yGm.SetGlobalBuffer((__gm__ T *)y + this->rowBase * colsAlign, this->blockRows * colsAlign);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale + this->rowBase, this->blockRows);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, colsAlign * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, colsAlign * sizeof(T));
        pipe.InitBuffer(bufX32, colsAlign * sizeof(float));
        pipe.InitBuffer(bufTmp, colsAlign * sizeof(float));
        pipe.InitBuffer(bufWork, colsAlign * sizeof(float));
        pipe.InitBuffer(bufAbs, 32);
        pipe.InitBuffer(bufQ, colsAlign * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        for (int64_t row = 0; row < this->blockRows; ++row) {
            CopyIn(row);
            Compute(row);
            CopyOut(row);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t row)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(xLocal, xGm[row * this->colsAlign], params, pad);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t row)
    {
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<float> x32 = bufX32.Get<float>();
        AscendC::LocalTensor<float> tmp = bufTmp.Get<float>();
        AscendC::LocalTensor<float> work = bufWork.Get<float>();
        AscendC::LocalTensor<float> absMax = bufAbs.Get<float>();
        AscendC::LocalTensor<int8_t> q = bufQ.Get<int8_t>();

        const int32_t cols = static_cast<int32_t>(this->cols);
        const int32_t colsAlign = static_cast<int32_t>(this->colsAlign);

        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(x32, xLocal, 0.0f, colsAlign);
        } else {
            AscendC::Cast(x32, xLocal, AscendC::RoundMode::CAST_NONE, colsAlign);
        }
        // per-row symmetric scale = max|x| / 127; dst / src / work stay distinct
        AscendC::Abs(tmp, x32, colsAlign);
        AscendC::ReduceMax<float, true>(absMax, tmp, work, cols);
        AscendC::Muls(absMax, absMax, 1.0f / Q_MAX, 1);
        const float scale = absMax.GetValue(0);
        scaleGm.SetValue(row, scale);

        // quantize then dequantize in the same pass, back onto x32
        AscendC::Muls(tmp, x32, 1.0f / scale, colsAlign);
        AscendC::Cast(q, tmp, AscendC::RoundMode::CAST_RINT, colsAlign);
        AscendC::Cast(x32, q, AscendC::RoundMode::CAST_NONE, colsAlign);
        AscendC::Muls(x32, x32, scale, colsAlign);
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(yLocal, x32, 0.0f, colsAlign);
        } else {
            AscendC::Cast(yLocal, x32, AscendC::RoundMode::CAST_RINT, colsAlign);
        }
        inQueueX.FreeTensor(xLocal);
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t row)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(this->cols * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[row * this->colsAlign], yLocal, params);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufX32, bufTmp, bufWork, bufAbs, bufQ;
    AscendC::GlobalTensor<T> xGm, yGm;
    AscendC::GlobalTensor<float> scaleGm;
    int64_t blockRows = 0;
    int64_t rowBase = 0;
    int64_t cols = 0;
    int64_t colsAlign = 0;
};

extern "C" __global__ __aicore__ void ${context.op}(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                                    int64_t cols, int64_t colsAlign,
                                                    int64_t formerNum, int64_t formerRows, int64_t tailRows,
                                                    int64_t dtypeCode)
{
    if (dtypeCode == 0) {
        Kernel${context.className}<float> op;
        op.Init(x, y, scale, cols, colsAlign, formerNum, formerRows, tailRows);
        op.Process();
    } else if (dtypeCode == 1) {
        Kernel${context.className}<half> op;
        op.Init(x, y, scale, cols, colsAlign, formerNum, formerRows, tailRows);
        op.Process();
    } else {
        Kernel${context.className}<bfloat16_t> op;
        op.Init(x, y, scale, cols, colsAlign, formerNum, formerRows, tailRows);
        op.Process();
    }
}
`,
    host: `${LICENSE}
// ============================================================
// ${context.op} op_host — dynamic per-row INT8 quantize/dequant tiling + launch
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_${context.op}.h"

namespace ascend_kernel {

std::tuple<at::Tensor, at::Tensor> ${context.op}(const at::Tensor &input)
{
    TORCH_CHECK(input.dim() == 2, "${context.op}: input must be 2-D [rows, cols]");
    TORCH_CHECK(input.is_contiguous(), "${context.op}: input must be contiguous");
    TORCH_CHECK(input.scalar_type() == at::kHalf || input.scalar_type() == at::kBFloat16
                    || input.scalar_type() == at::kFloat,
                "${context.op}: only float16 / bfloat16 / float32 are supported");

    const int64_t rows = input.size(0);
    const int64_t cols = input.size(1);
    at::Tensor output = at::empty_like(input);
    at::Tensor scale = at::empty({rows}, input.options().dtype(at::kFloat));
    if (rows == 0 || cols == 0) {
        return {output, scale};
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const int64_t dtypeSize = input.element_size();
    const int64_t alignElements = 32 / dtypeSize;
    const int64_t colsAlign = ((cols + alignElements - 1) / alignElements) * alignElements;
    // in+out queues plus two fp32 working buffers and one int8 buffer
    const int64_t bufferCoefficient = (dtypeSize == 2) ? 20 : 24;
    const int64_t maxCols = ((static_cast<int64_t>(ubSize) / bufferCoefficient) / alignElements) * alignElements;
    TORCH_CHECK(colsAlign <= maxCols, "${context.op}: cols exceeds UB capacity (max ", maxCols, ")");

    const int64_t usedCoreNum = std::min(rows, coreNum);
    const int64_t formerRows = (rows + usedCoreNum - 1) / usedCoreNum;
    const int64_t formerNum = usedCoreNum - 1;
    const int64_t tailRows = rows - formerRows * formerNum;

    at::Tensor kernelInput = (cols == colsAlign)
        ? input
        : at::constant_pad_nd(input, {0, colsAlign - cols}, 0.0).contiguous();
    at::Tensor kernelOutput = at::empty_like(kernelInput);

    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);
    const int64_t dtypeCode = (input.scalar_type() == at::kFloat) ? 0
                            : (input.scalar_type() == at::kHalf) ? 1 : 2;

    EXEC_KERNEL_CMD(${context.op}, blockDim, kernelInput, kernelOutput, scale,
                    cols, colsAlign, formerNum, formerRows, tailRows, dtypeCode);

    output = (cols == colsAlign) ? kernelOutput : kernelOutput.narrow(-1, 0, cols).contiguous();
    return {output, scale};
}

}  // namespace ascend_kernel
`,
  }),
  elementwise: (context) => ({
    kernel: `${LICENSE}
// ============================================================
// ${context.op} op_kernel — fused elementwise chain (Mul -> Add -> SiLU)
//
// Optimisation applied (from advice ${context.adviceId}):
//   * the three framework ops (mul / add / silu) become ONE kernel launch, so the
//     dispatch count per step drops by two thirds for this chain;
//   * fp16/bf16 are up-cast once and down-cast once, not per op;
//   * double buffered, DataCopyPad only, tail tile handled.
// ============================================================

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Kernel${context.className} {
public:
    __aicore__ inline Kernel${context.className}() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR y, int64_t totalLength, int64_t tileLength)
    {
        const int64_t blockIdx = AscendC::GetBlockIdx();
        this->tileLength = tileLength;
        const int64_t perCore = (totalLength + AscendC::GetBlockNum() - 1) / AscendC::GetBlockNum();
        const int64_t start = blockIdx * perCore;
        this->length = (start + perCore > totalLength) ? (totalLength - start) : perCore;
        aGm.SetGlobalBuffer((__gm__ T *)a + start, this->length);
        bGm.SetGlobalBuffer((__gm__ T *)b + start, this->length);
        yGm.SetGlobalBuffer((__gm__ T *)y + start, this->length);
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(bufA32, tileLength * sizeof(float));
        pipe.InitBuffer(bufB32, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        const int64_t tileNum = (this->length + this->tileLength - 1) / this->tileLength;
        for (int64_t tile = 0; tile < tileNum; ++tile) {
            const int64_t count = (tile == tileNum - 1) ? (this->length - tile * this->tileLength) : this->tileLength;
            CopyIn(tile, count);
            Compute(count);
            CopyOut(tile, count);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t tile, int64_t count)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.AllocTensor<T>();
        AscendC::LocalTensor<T> bLocal = inQueueB.AllocTensor<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
        AscendC::DataCopyPad(aLocal, aGm[tile * this->tileLength], params, pad);
        AscendC::DataCopyPad(bLocal, bGm[tile * this->tileLength], params, pad);
        inQueueA.EnQue(aLocal);
        inQueueB.EnQue(bLocal);
    }

    __aicore__ inline void Compute(int64_t count)
    {
        AscendC::LocalTensor<T> aLocal = inQueueA.DeQue<T>();
        AscendC::LocalTensor<T> bLocal = inQueueB.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
        AscendC::LocalTensor<float> a32 = bufA32.Get<float>();
        AscendC::LocalTensor<float> b32 = bufB32.Get<float>();
        const int32_t len = static_cast<int32_t>(count);

        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(a32, aLocal, 0.0f, len);
            AscendC::Adds(b32, bLocal, 0.0f, len);
        } else {
            AscendC::Cast(a32, aLocal, AscendC::RoundMode::CAST_NONE, len);
            AscendC::Cast(b32, bLocal, AscendC::RoundMode::CAST_NONE, len);
        }
        // silu(a * b): the three ops happen on one UB tile, no GM round trip
        AscendC::Mul(a32, a32, b32, len);
        AscendC::Muls(a32, a32, -1.0f, len);
        AscendC::Exp(a32, a32, len);
        AscendC::Adds(a32, a32, 1.0f, len);
        AscendC::Div(b32, b32, a32, len);
        if constexpr (sizeof(T) == sizeof(float)) {
            AscendC::Adds(yLocal, b32, 0.0f, len);
        } else {
            AscendC::Cast(yLocal, b32, AscendC::RoundMode::CAST_RINT, len);
        }

        inQueueA.FreeTensor(aLocal);
        inQueueB.FreeTensor(bLocal);
        outQueueY.EnQue<T>(yLocal);
    }

    __aicore__ inline void CopyOut(int64_t tile, int64_t count)
    {
        AscendC::LocalTensor<T> yLocal = outQueueY.DeQue<T>();
        AscendC::DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[tile * this->tileLength], yLocal, params);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufA32, bufB32;
    AscendC::GlobalTensor<T> aGm, bGm, yGm;
    int64_t length = 0;
    int64_t tileLength = 0;
};

extern "C" __global__ __aicore__ void ${context.op}(GM_ADDR a, GM_ADDR b, GM_ADDR y,
                                                    int64_t totalLength, int64_t tileLength,
                                                    int64_t dtypeCode)
{
    if (dtypeCode == 0) {
        Kernel${context.className}<float> op;
        op.Init(a, b, y, totalLength, tileLength);
        op.Process();
    } else if (dtypeCode == 1) {
        Kernel${context.className}<half> op;
        op.Init(a, b, y, totalLength, tileLength);
        op.Process();
    } else {
        Kernel${context.className}<bfloat16_t> op;
        op.Init(a, b, y, totalLength, tileLength);
        op.Process();
    }
}
`,
    host: `${LICENSE}
// ============================================================
// ${context.op} op_host — fused elementwise chain, flat split over elements
// ============================================================

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_${context.op}.h"

namespace ascend_kernel {

at::Tensor ${context.op}(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.sizes() == b.sizes(), "${context.op}: both inputs must share a shape");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "${context.op}: both inputs must share a dtype");
    at::Tensor x = a.contiguous();
    at::Tensor y = b.contiguous();
    at::Tensor out = at::empty_like(x);
    const int64_t totalLength = x.numel();
    if (totalLength == 0) {
        return out;
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    const int64_t coreNum = static_cast<int64_t>(platform->GetCoreNumAiv());
    uint64_t ubSize = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    const int64_t dtypeSize = x.element_size();
    const int64_t alignElements = 32 / dtypeSize;
    const int64_t bufferCoefficient = (dtypeSize == 2) ? 12 : 16;
    int64_t tileLength = ((static_cast<int64_t>(ubSize) / bufferCoefficient) / alignElements) * alignElements;
    tileLength = std::min(tileLength, ((totalLength + coreNum - 1) / coreNum));
    TORCH_CHECK(tileLength > 0, "${context.op}: UB too small for one tile");

    uint32_t blockDim = static_cast<uint32_t>(std::min<int64_t>(coreNum, (totalLength + tileLength - 1) / tileLength));
    const int64_t dtypeCode = (x.scalar_type() == at::kFloat) ? 0 : (x.scalar_type() == at::kHalf) ? 1 : 2;

    EXEC_KERNEL_CMD(${context.op}, blockDim, x, y, out, totalLength, tileLength, dtypeCode);
    return out;
}

}  // namespace ascend_kernel
`,
  }),
};

/**
 * Render the whole operator project for one advice handoff.
 *
 * @param {object} task - package from `buildHandoff`.
 * @param {object} [options] - `{ generatedAt }`.
 * @returns {{ok: boolean, error?: string, op: string, kind: string, title: string,
 *   files: Array<{path: string, content: string}>, summary: object}} plan.
 */
export function renderOperatorProject(task, { generatedAt } = {}) {
  if (task?.ok !== true) return { ok: false, error: task?.error ?? '没有可生成的任务包', files: [] };
  const choice = planOperator(task);
  const className = choice.op.split('_').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join('');
  const template = TEMPLATES[choice.kind];
  const context = { op: choice.op, className, adviceId: task.adviceId, task };
  const sources = template(context);
  const stamp = generatedAt ?? new Date().toISOString();
  const operators = task.operators.map((operator) => `\`${operator.name}\`（占设备耗时 ${String(operator.sharePct ?? '—')}%）`).join('、');

  const files = [
    {
      path: `csrc/ops/${choice.op}/op_kernel/${choice.op}.cpp`,
      content: sources.kernel,
    },
    {
      path: `csrc/ops/${choice.op}/op_host/${choice.op}.cpp`,
      content: sources.host,
    },
    {
      path: `csrc/ops/${choice.op}/test/${choice.op}-test-cases.md`,
      content: testCases({ choice, task, operators, stamp }),
    },
    {
      path: `csrc/ops/${choice.op}/design.md`,
      content: design({ choice, task, operators, stamp }),
    },
    {
      path: `csrc/ops/${choice.op}/README.md`,
      content: readme({ choice, task, operators, stamp }),
    },
    {
      path: `csrc/ops/${choice.op}/register-patch.md`,
      content: registerPatch({ choice, task }),
    },
  ];

  return {
    ok: true,
    op: choice.op,
    kind: choice.kind,
    title: choice.title,
    reason: choice.reason,
    files,
    summary: {
      adviceId: task.adviceId,
      adviceTitle: task.title,
      kind: choice.kind,
      op: choice.op,
      operators: task.operators.map((operator) => ({ name: operator.name, sharePct: operator.sharePct, count: operator.count })),
      optimization: OPTIMISATION[choice.kind],
      acceptance: task.acceptance,
      skills: task.skills.chain.map((entry) => entry.name),
      generatedAt: stamp,
      fileCount: files.length,
    },
  };
}

/** One-line description of what the generated operator changes. */
const OPTIMISATION = Object.freeze({
  'row-norm': '把行归一化融合成单趟 UB 内计算：平方与归约在同一 tile 完成（不再对 fp32 副本二次遍历），fp32 累加 + 向量 Reciprocal，gamma 融合并每核只加载一次',
  quantize: '把量化与反量化融合进一个 kernel：每行按 max|x| 求对称 scale（无需校准数据、无额外 D2H），量化/反量化同趟完成，输出保持原 dtype 直接喂给量化矩阵乘',
  elementwise: '把框架里的 mul/add/silu 三次下发融合成一次 kernel 启动（该链每步派发数减少 2/3），fp16/bf16 只做一次升精度与一次降精度',
});

/**
 * Write a rendered project into `<workspace>/<OPERATOR_DIR>/<adviceId>/`.
 *
 * @param {object} project - result of {@link renderOperatorProject}.
 * @param {object} options - `{ workspace, allowOutside }`.
 * @returns {{ok: boolean, dir?: string, files?: string[], error?: string}} write result.
 */
export function writeOperatorProject(project, { workspace, allowOutside = false } = {}) {
  if (project?.ok !== true) return { ok: false, error: project?.error ?? '没有可写入的算子工程' };
  const base = resolve(workspace ?? process.cwd());
  const dir = join(base, OPERATOR_DIR, project.op);
  if (!allowOutside) {
    const root = resolve(base);
    if (dir !== root && !dir.startsWith(root + sep)) return { ok: false, error: `算子工程目录超出会话工作区：${dir}` };
  }
  const written = [];
  mkdirSync(dir, { recursive: true });
  for (const file of project.files) {
    const target = join(dir, file.path);
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, file.content, 'utf8');
    written.push(target);
  }
  return { ok: true, dir, files: written };
}

/** Test-case document for the generated operator (Phase 3 shape). */
function testCases({ choice, task, operators, stamp }) {
  const rows = (choice.kind === 'elementwise')
    ? [
      ['decode-ish', '(32, 4096)', '(8, 128, 4096)'],
      ['prefill-ish', '(512, 4096)', '(1, 512, 4096)'],
      ['hidden-8k', '(256, 8192)', '(4, 256, 8192)'],
      ['odd size', '(37, 17)', '(37, 17, 1)'],
      ['single element', '(1, 1)', '(1, 1, 1)'],
    ]
    : [
      ['decode-ish', '(32, 4096)', '(8, 128, 4096)'],
      ['prefill-ish', '(512, 4096)', '(1, 512, 4096)'],
      ['hidden-8k', '(256, 8192)', '(4, 256, 8192)'],
      ['alignment', '(37, 17)', '(37, 17, 1)'],
      ['single col', '(512, 1)', '(512, 1, 1)'],
    ];
  return `# ${choice.op} Test Cases

由 vLLM-Ascend Profiler Analyzer 第 5 步「执行」按钮生成（${stamp}）。

| 来源 | 值 |
| --- | --- |
| 优化项 | \`${task.adviceId}\` · ${task.title} |
| 目标算子 | ${operators} |
| 验收指标 | ${task.acceptance.map((row) => `${row.label}（阈值 ≥ ${String(row.thresholdPct)}%）`).join('、') || '（采集质量类建议，无量化指标）'} |

## SUPPORTED_DTYPES

\`\`\`python
SUPPORTED_DTYPES = ["float16", "bfloat16", "float32"]
\`\`\`

## TEST_SHAPES

| # | Category | Description | Shape |
|---|---|---|---|
${rows.map((row, index) => `| ${String(index + 1)} | ${row[0]} | 常规形状 | \`${row[1]}\` |`).join('\n')}

## GENERAL_SHAPES

| # | Category | Description | Shape |
|---|---|---|---|
${rows.map((row, index) => `| ${String(index + 1)} | ${row[0]} | 泛化形状（含非对齐/尾核） | \`${row[2]}\` |`).join('\n')}

## BOUNDARY_VALUES

| # | Description | Value |
|---|---|---|
| 1 | 全零（只靠 eps / scale 兜底） | \`0.0\` |
| 2 | 极小值 | \`1e-6\` |
| 3 | 单位值 | \`1.0\` |
| 4 | 负值（验证平方/绝对值的符号处理） | \`-1.0\` |
| 5 | fp16 安全上限附近 | \`100.0\` |
| 6 | 混合符号（行内抵消） | \`[1, -1, 1, -1, ...]\` |

合计 \`(5 + 5) × 3 = 30\` 用例，满足 ≥30 的门禁。

## Operator baseline

\`\`\`python
import torch, torch_npu
NPU_CALL = lambda *args: torch.ops.npu.${choice.op}(*args)
\`\`\`

CPU 参考与 NPU 对照实现（Phase 8 双路对比用）由 \`ascendc\` skill 的 Phase 3 规范给出：
先按 \`design.md\` 的公式写 fp32 CPU 参考，再用张量算子组合出可在 NPU 上跑的 baseline。
`;
}

/** Design document with the profiling provenance and the acceptance bar. */
function design({ choice, task, operators, stamp }) {
  const acceptance = task.acceptance.length === 0
    ? '| （本优化项为采集质量类，无量化指标） | — | — | — | 证据是否补齐 |'
    : task.acceptance.map((row) => `| ${row.label} | ${String(row.current)} ${row.unit} | ${String(row.expectedPct ?? '—')}% | ≥ ${String(row.thresholdPct)}% | 复采后第 6 步对比 |`).join('\n');
  return `# ${choice.op} Design Document

> **Provenance（由第 5 步「执行」自动生成，${stamp}）**
>
> | 项目 | 值 |
> | --- | --- |
> | 优化项 | \`${task.adviceId}\` · ${task.title}（优先级 ${task.priority}） |
> | 瓶颈 | ${task.bottleneck.label}（${String(task.bottleneck.score)}/100，范围 ${task.bottleneck.scope}） |
> | 目标算子 | ${operators} |
> | 生成类别 | ${choice.kind} —— ${choice.reason} |
> | skills 链 | ${task.skills.chain.map((entry) => entry.name).join(' → ')} |
> | 任务包 | .dsh-vap-tasks/<ts>-${task.adviceId}.md |

## 1. 优化内容（相对框架现状）

${OPTIMISATION[choice.kind]}

依据（来自 profiling）：${task.rationale}

## 2. 函数签名

见 \`op_host/${choice.op}.cpp\`；数据类型 fp16 / bf16 / fp32，硬件参数（核数、UB 大小）全部由平台 API 在运行时查询。

## 3. 两级 tiling

* block 级：按${choice.kind === 'elementwise' ? '元素' : '行'}切分，former/tail 核配平，512B cache line 对齐；
* UB 级：\`BUFFER_NUM = 2\` 双缓冲，尾块单独处理，GM↔UB 只用 \`DataCopyPad\`。

## 4. UB 分配

见 \`op_host\` 中的 \`bufferCoefficient\`（fp16/bf16 与 fp32 分支不同），tile 长度由 UB 容量推导而非硬编码。

## 5. 验收标准（与第 6 步前后对比同口径）

| 指标 | 当前值 | 预期收益 | 判定阈值 | 测量方式 |
| --- | --- | --- | --- | --- |
${acceptance}

## 6. 需要真机的部分

编译、精度与性能必须有 CANN + 昇腾 NPU：

\`\`\`bash
source \${CANN_PATH}/*/set_env.sh && conda activate <env>
bash build.sh && pip install output/ascend_kernel*.whl --force-reinstall --no-deps
pytest -v && python -m npu_op_benchmark --op ${choice.op} --warmup 5 --active 5
\`\`\`
`;
}

/** README shown to whoever picks the generated project up. */
function readme({ choice, task, operators, stamp }) {
  return `# ${choice.title}（\`${choice.op}\`）

由 **vLLM-Ascend Profiler Analyzer** 第 5 步「执行」于 ${stamp} 生成。

* 优化项：\`${task.adviceId}\` — ${task.title}
* 目标算子：${operators}
* 优化内容：${OPTIMISATION[choice.kind]}
* 验收：${task.acceptance.map((row) => `${row.label} 改善 ≥ ${String(row.thresholdPct)}%`).join('；') || '补齐采集证据'}

## 文件

| 文件 | 说明 |
| --- | --- |
| \`op_kernel/${choice.op}.cpp\` | 算子实现（已按上面的优化写好） |
| \`op_host/${choice.op}.cpp\` | tiling 与启动（平台 API 取核数/UB） |
| \`design.md\` | 设计文档（含 profiling 溯源与验收标准） |
| \`test/${choice.op}-test-cases.md\` | 统一测试用例（精度与性能共用） |
| \`register-patch.md\` | 接入现有 ascend-kernel 工程的三处注册点改法 |

## 下一步（需要 NPU）

1. 按 \`register-patch.md\` 把算子挂进工程；
2. \`bash build.sh\` 编译并安装 wheel；
3. 跑 \`test/${choice.op}-test-cases.md\` 的精度用例；
4. 用 \`npu-op-benchmark\` 做改造前后的单算子对比；
5. 用同一负载重新采集 profiling，导入第 6 步与本次数据集对比，确认验收阈值达成。
`;
}

/** The three framework registration points, as a patch guide. */
function registerPatch({ choice, task }) {
  const returns = choice.kind === 'quantize' ? 'std::tuple<at::Tensor, at::Tensor>' : 'at::Tensor';
  const args = choice.kind === 'quantize' ? 'const at::Tensor &input' : 'const at::Tensor &input, const at::Tensor &gamma, double eps';
  return `# 接入 ascend-kernel 工程（三处注册点）

生成自优化项 \`${task.adviceId}\`。把 \`csrc/ops/${choice.op}/\` 拷进工程后，按下面三处改动即可被 \`torch.ops.npu.${choice.op}\` 调用。

## 1) \`csrc/ops.h\`

\`\`\`cpp
namespace ascend_kernel {
${returns} ${choice.op}(${args});
}
\`\`\`

## 2) \`csrc/register.cpp\`

\`\`\`cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("${choice.op}(${choice.kind === 'quantize' ? 'Tensor input' : 'Tensor input, Tensor gamma, float eps=1e-6'}) -> ${choice.kind === 'quantize' ? '(Tensor, Tensor)' : 'Tensor'}");
}
TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("${choice.op}", TORCH_FN(ascend_kernel::${choice.op}));
}
\`\`\`

## 3) \`csrc/CMakeLists.txt\`

\`\`\`cmake
FILE(GLOB OP_SRCS
    ...
    \${PROJECT_OP_SRC_BASE}/ops/${choice.op}/op_host/${choice.op}.cpp
)
ascendc_library(no_workspace_kernel STATIC
    ...
    \${PROJECT_OP_SRC_BASE}/ops/${choice.op}/op_kernel/${choice.op}.cpp
)
\`\`\`
`;
}
