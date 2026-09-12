/**
 * Steps ④ and ⑤ of the reasoning chain: turn a located bottleneck and its
 * inferred root causes into prioritised, actionable optimisations with an
 * expected benefit.
 *
 * Every recommendation carries:
 *
 * * `priority` — 高 / 中 / 低, derived from the candidate's score and the phase
 *   it applies to (not hand-assigned);
 * * `actions` — concrete configuration, code, or measurement steps;
 * * `rationale` — the evidence (metric values) that made this the right move;
 * * `expectedGain` — a quantified projection **computed from this dataset**
 *   where the arithmetic is sound (host time, exposed communication, copy
 *   time), and an explicitly labelled experience range where it is not (for
 *   example quantisation speedups, which depend on the model and the CANN
 *   version);
 * * `verification` — how to confirm the gain on the machine, and `risk` — what
 *   can go wrong.
 *
 * Optimization options that only exist in some vLLM-Ascend releases are marked
 * `confirmInVersion: true` and phrased as "confirm support in your version"
 * rather than asserted as available.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis/recommend
 */

import { percentOf, round } from '../model/stats.js';

/** Priority ordering used by the report and the UI. */
const PRIORITY_RANK = { 高: 3, 中: 2, 低: 1 };

/**
 * Whether a template written for `required` applies to the scope's phase.
 *
 * The overall scope has no single phase, so it matches any phase — the caller
 * resolves the overall scope against the dominant phase before calling, and
 * this tolerance keeps phase-specific advice from disappearing there.
 *
 * @param {string} scopePhase - the scope's phase id (`overall` or a phase).
 * @param {string} required - the template's phase (`prefill`/`decode`/`both`).
 * @returns {boolean} whether it applies.
 */
function phaseMatches(scopePhase, required) {
  if (required === 'both') return true;
  if (scopePhase === 'overall' || scopePhase === undefined) return true;
  return scopePhase === required;
}

/**
 * Recommendation templates. Each entry describes when it applies and what it
 * projects; `estimate` receives the indicator block and returns the expected
 * gain object.
 */
const TEMPLATES = [
  // ── Host scheduling ──────────────────────────────────────────────────────
  {
    id: 'host.enable-graph-mode',
    title: '对 Decode 启用图模式（ACL Graph / npugraph_ex），把逐步下发变成一次回放',
    bottleneck: 'host',
    phase: 'decode',
    priority: '高',
    applies: (indicators, context) => context.phase === 'decode'
      && indicators.hostOnlyPct >= 15
      && indicators.dispatchPerStep >= 300,
    actions: [
      { type: 'config', text: '在 vLLM-Ascend 启动参数中启用图模式/编译配置（不同版本的开关名不同，请以当前版本文档为准），并确认 torch_npu 的 ACL Graph 支持已开启（如 torch.npu.set_compile_mode / npugraph_ex 相关配置）。' },
      { type: 'verify', text: '先用单请求固定 batch 验证图捕获成功，再逐步放开并发；图模式对动态 shape 敏感。' },
      { type: 'code', text: '若存在逐 token 的动态控制流（如 early-exit、动态 stop 判定），把它们移出被捕获的图之外。' },
    ],
    rationale: (indicators) => `Host 每步下发 ${indicators.dispatchPerStep.toFixed(0)} 个算子、独占 ${indicators.hostExclusivePerStepUs.toFixed(0)}µs，说明每步的 CPU 派发时间是固定开销；图模式把 N 次下发压成一次回放，理论上可去掉大部分派发成本。`,
    estimate: (indicators) => {
      const stepUs = indicators.avgStepUs;
      const removable = indicators.hostExclusivePerStepUs * 0.75;
      const ratio = stepUs > 0 ? Math.min(0.8, removable / stepUs) : 0;
      return {
        metric: 'Decode 单步延迟（TPOT）',
        estimatePct: round(ratio * 100, 1),
        rangePct: [round(ratio * 60, 1), round(ratio * 100, 1)],
        basis: `每步 host 独占 ${indicators.hostExclusivePerStepUs.toFixed(0)}µs ÷ 平均步长 ${stepUs.toFixed(0)}µs，按可消除 60%–100% 计算`,
        confidence: 'medium',
        assumption: '假设图模式能消除 60%–100% 的 host 独占时间；实际取决于图捕获覆盖率与动态 shape 比例。',
      };
    },
    verification: '同一并发、同一输入长度下对比开启前后 TPOT 与吞吐；同时确认 Host 独占/步 指标下降。',
    risk: '图模式对动态 shape 与多流场景支持有限，可能出现回退或精度差异；需在小流量环境先行验证。',
  },
  {
    id: 'host.async-sampling',
    title: '消除采样与统计路径上的同步点（.item()/tolist/synchronize）',
    bottleneck: 'host',
    phase: 'both',
    priority: '高',
    applies: (indicators) => indicators.hostSyncCount > 0 && (indicators.hostSyncPerStep >= 0.5 || indicators.hostSyncUs >= 50),
    actions: [
      { type: 'code', text: '把需要 host 判定的逻辑（stop string、结构化输出校验、指标统计）改为异步/后置处理，避免每步同步。' },
      { type: 'code', text: '用批量 D2H 替代逐项 .item()：一次拷贝整块结果再在 host 侧解析。' },
      { type: 'config', text: '关闭或降频 profiling/日志中的逐算子统计开关（如 memory/FLOPS 采集），它们会加同步。' },
    ],
    rationale: (indicators) => `检测到 ${indicators.hostSyncCount} 次同步类算子（累计 ${indicators.hostSyncUs.toFixed(0)}µs，约 ${indicators.hostSyncPerStep.toFixed(2)} 次/步）：每次同步都会排空设备队列，把异步流水打断。`,
    estimate: (indicators) => {
      const ratio = indicators.avgStepUs > 0 ? Math.min(0.5, (indicators.hostSyncUs / Math.max(1, indicators.stepCount)) / indicators.avgStepUs) : 0;
      return {
        metric: 'Decode 单步延迟（TPOT）',
        estimatePct: round(ratio * 100 * 0.8, 1),
        rangePct: [round(ratio * 50, 1), round(ratio * 100, 1)],
        basis: `同步开销约 ${(indicators.hostSyncUs / Math.max(1, indicators.stepCount)).toFixed(0)}µs/步，占平均步长 ${indicators.avgStepUs.toFixed(0)}µs 的比例`,
        confidence: 'medium',
        assumption: '假设同步开销可被完全移出关键路径；部分同步是语义必需的。',
      };
    },
    verification: '统计同步算子次数是否降为 0（或仅在请求结束出现），并观察 TPOT 抖动是否收敛。',
    risk: '把校验逻辑后置可能改变异常语义（例如提前停止的判定时点），需要回归测试。',
  },
  {
    id: 'host.reduce-dispatch',
    title: '减少每步算子数量：算子融合与多流并行',
    bottleneck: 'host',
    phase: 'both',
    priority: '中',
    applies: (indicators) => indicators.dispatchPerStep >= 800,
    actions: [
      { type: 'config', text: '启用昇腾侧的融合算子能力（例如融合 add+rmsnorm、融合 rope、融合 MoE/MC2 等；具体开关按当前 vLLM-Ascend 版本确认）。' },
      { type: 'code', text: '合并可并行的独立算子到不同 stream，提升下发并行度。' },
      { type: 'config', text: '在 host 侧减少每步重复的元数据构造（缓存 block table/slot mapping 的构建结果）。' },
    ],
    rationale: (indicators) => `每步下发 ${indicators.dispatchPerStep.toFixed(0)} 个算子，超过 800 个后派发开销成为可见成本；融合与多流可直接减少下发次数。`,
    estimate: (indicators) => ({
      metric: 'Host 派发开销',
      estimatePct: round(Math.min(40, indicators.hostBusyPct * 0.35), 1),
      rangePct: [round(Math.min(40, indicators.hostBusyPct * 0.15), 1), round(Math.min(40, indicators.hostBusyPct * 0.45), 1)],
      basis: `Host 忙占比 ${indicators.hostBusyPct.toFixed(1)}%，按融合可削减 15%–45% 的派发次数估算`,
      confidence: 'low',
      assumption: '经验区间：实际收益取决于可融合算子的比例。',
    }),
    verification: '对比融合开关开启前后 Host 派发算子数/步 与 Host 独占时间。',
    risk: '融合算子可能与特定数据类型/布局绑定，存在回退路径；需确认精度一致性。',
  },

  // ── NPU compute ──────────────────────────────────────────────────────────
  {
    id: 'compute.quantize',
    title: '启用/校准量化（W8A8、FP8 等）降低 Prefill 矩阵乘与注意力开销',
    bottleneck: 'compute',
    phase: 'prefill',
    priority: '高',
    applies: (indicators, context) => context.phase === 'prefill' && indicators.computePct >= 50,
    actions: [
      { type: 'config', text: '为线性层与注意力启用昇腾量化（quantization=ascend 或对应量化方案），KV Cache 视精度要求选择低比特缓存。' },
      { type: 'verify', text: '量化需要校准数据；先用离线精度评测（如 lm-eval / 业务集）确认精度损失可接受。' },
      { type: 'code', text: '确认量化后的算子确实落到 Cube 上（查看 kernel_details 中算子名与 MAC 利用率变化）。' },
    ],
    rationale: (indicators, context) => `Prefill 设备侧计算占墙钟 ${indicators.computePct.toFixed(1)}%、NPU 忙碌率 ${indicators.deviceBusyPct.toFixed(1)}%，属于算力受限；降低计算精度是提升吞吐最直接的手段。${context.macRatio === undefined ? '' : `当前 MAC 利用率 ${context.macRatio.toFixed(3)}。`}`,
    estimate: () => ({
      metric: 'Prefill 计算耗时（TTFT 的主要部分）',
      estimatePct: 30,
      rangePct: [20, 45],
      basis: '经验区间：W8A8/FP8 在昇腾 Cube 上的矩阵乘加速典型值，实际取决于模型结构与算子覆盖率',
      confidence: 'low',
      assumption: '经验区间而非本数据集推导；请以实际测量为准。',
    }),
    verification: '对比量化前后 Prefill 阶段 computeUs、MAC 利用率与 TTFT。',
    risk: '精度损失、部分算子无量化实现导致回退、量化本身增加显存/转换开销。',
  },
  {
    id: 'compute.decode-increase-batch',
    title: '提高 Decode 并发/批大小，提升算术强度以摊薄权重读取',
    bottleneck: 'compute',
    phase: 'decode',
    priority: '高',
    applies: (indicators, context) => context.phase === 'decode'
      && indicators.deviceBusyPct >= 70
      && (context.macRatio === undefined || context.macRatio < 0.4),
    actions: [
      { type: 'config', text: '提高 max-num-seqs / 增大 continuous batching 的并发上限，使每步处理更多 token。' },
      { type: 'config', text: '在显存允许的前提下增大 KV Cache 容量，避免因显存不足限制并发。' },
      { type: 'verify', text: '绘制吞吐-并发曲线，确认当前是否处于未饱和区间。' },
    ],
    rationale: (indicators, context) => `Decode 阶段 NPU 忙碌率 ${indicators.deviceBusyPct.toFixed(1)}% 但 MAC 利用率${context.macRatio === undefined ? '未知（未采集流水利用率）' : `仅 ${context.macRatio.toFixed(3)}`}，属于"忙而不算"：瓶颈是每步权重读取带宽，提高 batch 能让同一份权重服务更多 token。`,
    estimate: (indicators) => {
      const potential = Math.max(0, 100 - indicators.deviceBusyPct) + 15;
      return {
        metric: 'Decode 吞吐（tokens/s）',
        estimatePct: round(Math.min(60, potential), 1),
        rangePct: [round(Math.min(60, potential) * 0.5, 1), round(Math.min(60, potential), 1)],
        basis: `以设备忙碌率 ${indicators.deviceBusyPct.toFixed(1)}% 与权重读取受限为前提估算的吞吐提升空间`,
        confidence: 'medium',
        assumption: '假设显存可容纳更大 KV Cache 且请求队列足够长；时延可能会上升。',
      };
    },
    verification: '固定输入输出长度，逐步提高并发至吞吐拐点；同时观察 TPOT 是否仍在可接受范围。',
    risk: '并发提高会拉长单请求时延；显存不足会触发抢占/重算。',
  },
  {
    id: 'compute.fuse-small-ops',
    title: '融合碎片化小算子，减少 Vector/Scalar 段与 kernel 切换',
    bottleneck: 'compute',
    phase: 'both',
    priority: '中',
    applies: (indicators, context) => indicators.deviceBusyPct >= 70
      && (context.medianComputeDurUs ?? 0) > 0
      && (context.medianComputeDurUs ?? 0) < 30,
    actions: [
      { type: 'config', text: '启用可用的融合算子集合（融合归一化、融合激活、融合 rope 等），按版本确认开关。' },
      { type: 'code', text: '把逐元素链（add/mul/cast）合并为一个算子，减少中间张量写回。' },
    ],
    rationale: (indicators, context) => `设备计算算子时长中位数仅 ${(context.medianComputeDurUs ?? 0).toFixed(1)}µs，说明算力被切碎在大量小算子上，启动与调度开销摊薄了 Cube 利用率。`,
    estimate: () => ({
      metric: '设备侧小算子总耗时',
      estimatePct: 20,
      rangePct: [10, 30],
      basis: '经验区间：算子融合对小算子总耗时的典型压缩比例',
      confidence: 'low',
      assumption: '经验区间；收益取决于融合覆盖率。',
    }),
    verification: '对比融合前后小算子数量与 computeUs。',
    risk: '融合可能引入额外的内存布局转换。',
  },
  {
    id: 'compute.tune-chunked-prefill',
    title: '调整 chunked prefill 的 chunk 大小，匹配序列长度分布',
    bottleneck: 'compute',
    phase: 'prefill',
    priority: '中',
    applies: (indicators, context) => context.phase === 'prefill' && indicators.stepCount >= 2,
    actions: [
      { type: 'config', text: '调整 max-num-batched-tokens（chunk 大小）与 max-num-seqs，使 Prefill 步的计算密度与显存占用平衡。' },
      { type: 'verify', text: '以 TTFT 与 Prefill 吞吐为双目标做小范围扫描（例如 2k/4k/8k token）。' },
    ],
    rationale: (indicators) => `Prefill 共 ${indicators.stepCount} 步、平均 ${indicators.avgStepUs.toFixed(0)}µs/步：chunk 过小会让每步固定开销占比过高，过大则会拉长 TTFT 并挤占 Decode。`,
    estimate: () => ({
      metric: 'Prefill 步耗时 / TTFT',
      estimatePct: 12,
      rangePct: [5, 25],
      basis: '经验区间：chunk 大小调优的典型收益',
      confidence: 'low',
      assumption: '经验区间；需结合业务序列长度分布实测。',
    }),
    verification: '对比不同 chunk 大小下的 Prefill 步耗时与 TTFT 分布。',
    risk: 'chunk 过大可能造成显存峰值上升与 Decode 饥饿。',
  },

  // ── Communication ────────────────────────────────────────────────────────
  {
    id: 'comm.enable-overlap-fusion',
    title: '启用通信计算融合/重叠能力，把集合通信藏到计算之后',
    bottleneck: 'comm',
    phase: 'both',
    priority: '高',
    applies: (indicators) => indicators.commPctOfDevice >= 10 && indicators.commOverlapPct < 60,
    actions: [
      { type: 'config', text: '启用昇腾侧的通信计算融合能力（MC2 类融合算子、AllGather+MatMul / MatMul+ReduceScatter 融合；具体开关名与可用性请按当前 vLLM-Ascend 版本确认，标记为需版本确认）。', confirmInVersion: true },
      { type: 'config', text: '启用通信切分（把一次大集合通信拆成多段与计算流水交错），提升重叠率。', confirmInVersion: true },
      { type: 'code', text: '检查是否有多余的 stream 同步阻断了重叠；把与通信无关的同步点移除。' },
    ],
    rationale: (indicators) => `通信占设备耗时 ${indicators.commPctOfDevice.toFixed(1)}%，但只有 ${indicators.commOverlapPct.toFixed(1)}% 与计算重叠（未掩盖 ${indicators.commExposedPct.toFixed(1)}% 墙钟）：把未掩盖部分藏进计算即可直接缩短关键路径。`,
    estimate: (indicators) => {
      const stepUs = indicators.avgStepUs;
      const exposedPerStep = indicators.commExposedUs / Math.max(1, indicators.stepCount);
      const ratio = stepUs > 0 ? Math.min(0.6, (exposedPerStep * 0.6) / stepUs) : 0;
      return {
        metric: '单步延迟（TPOT / Prefill 步耗时）',
        estimatePct: round(ratio * 100, 1),
        rangePct: [round(ratio * 60, 1), round(ratio * 100, 1)],
        basis: `未掩盖通信 ${exposedPerStep.toFixed(0)}µs/步 ÷ 平均步长 ${stepUs.toFixed(0)}µs，按可再重叠 60% 计算`,
        confidence: 'medium',
        assumption: '假设融合/切分能把 60% 的未掩盖通信藏进计算；受算子依赖关系限制。',
      };
    },
    verification: '观察通信重叠率是否上升、未掩盖通信占比是否下降，以及单步延迟变化。',
    risk: '融合算子对 shape 与并行度有约束；部分版本可能不支持特定组合。',
  },
  {
    id: 'comm.batch-amortize',
    title: '提高并发以摊薄小消息集合通信的固定开销',
    bottleneck: 'comm',
    phase: 'decode',
    priority: '高',
    applies: (indicators, context) => context.phase === 'decode' && indicators.commLatencyMedianUs <= 120,
    actions: [
      { type: 'config', text: '提高并发请求数，使每步处理更多 token：集合通信的消息量随之上升，固定开销被摊薄。' },
      { type: 'verify', text: '观察单步内通信时间是否随 batch 增长而次线性增长。' },
    ],
    rationale: (indicators) => `单次通信时长中位数仅 ${indicators.commLatencyMedianUs.toFixed(0)}µs，处于 HCCL 固定启动开销主导区间；每步通信 ${indicators.commPerStep.toFixed(1)} 次、${indicators.commUsPerStep.toFixed(0)}µs，增大 batch 是唯一能摊薄固定开销的方向。`,
    estimate: (indicators) => {
      const share = indicators.avgStepUs > 0 ? indicators.commUsPerStep / indicators.avgStepUs : 0;
      return {
        metric: 'Decode 吞吐（tokens/s）',
        estimatePct: round(Math.min(45, share * 100 * 0.7), 1),
        rangePct: [round(Math.min(45, share * 100 * 0.3), 1), round(Math.min(45, share * 100 * 0.8), 1)],
        basis: `通信耗时占单步 ${(share * 100).toFixed(1)}%，按 batch 增大后固定开销占比下降估算`,
        confidence: 'medium',
        assumption: '假设通信量为小消息、并发可提升且显存充足。',
      };
    },
    verification: '在并发扫描中记录通信时长/步与吞吐，确认拐点。',
    risk: '并发过大会使单请求时延升高，也可能引发显存压力。',
  },
  {
    id: 'comm.parallel-strategy',
    title: '复核并行策略：TP/DP/EP 组合与通信组划分是否匹配负载',
    bottleneck: 'comm',
    phase: 'both',
    priority: '中',
    applies: (indicators) => indicators.commPctOfDevice >= 20,
    actions: [
      { type: 'config', text: '评估降低 TP 度数、改用 EP/DP 组合是否能在保持吞吐的前提下减少每层集合通信次数。' },
      { type: 'verify', text: '对比不同并行配置下的通信占比与端到端吞吐（同一并发、同一序列长度）。' },
      { type: 'config', text: '确认通信组（HCCL group）划分与拓扑一致，避免跨平面通信。' },
    ],
    rationale: (indicators) => `通信占设备耗时已达 ${indicators.commPctOfDevice.toFixed(1)}%，并行策略直接决定通信次数与消息量；TP 每层都要集合通信，是 Decode 的主要来源。`,
    estimate: () => ({
      metric: '端到端吞吐',
      estimatePct: 15,
      rangePct: [5, 30],
      basis: '经验区间：并行策略调整的典型收益，强依赖模型结构与卡数',
      confidence: 'low',
      assumption: '经验区间；需实测不同配置。',
    }),
    verification: '固定负载对比 TP=2/4/8 等配置的通信占比与吞吐。',
    risk: '并行策略改变会影响显存占用与精度行为，需要重新验证。',
  },

  // ── Copy ─────────────────────────────────────────────────────────────────
  {
    id: 'copy.async-d2h',
    title: 'D2H 拷贝异步化并使用 pinned memory，避免同步往返',
    bottleneck: 'copy',
    phase: 'both',
    priority: '高',
    applies: (indicators) => indicators.d2hPerStepUs >= 100 || (indicators.hostCopyPctOfHost >= 5 && indicators.hostSyncCount > 0),
    actions: [
      { type: 'code', text: '把 logits/统计量回传改为异步拷贝（aclrtMemcpyAsync + 独立 stream + event 等待），避免阻塞计算流。' },
      { type: 'code', text: '使用 pinned（页锁定）主机内存作为拷贝目标，降低单次拷贝与页锁定开销。' },
      { type: 'code', text: '合并多次小 D2H 为一次批量拷贝。' },
    ],
    rationale: (indicators) => `每步 D2H ${indicators.d2hPerStepUs.toFixed(0)}µs、Host 侧拷贝占 Host 忙时间 ${indicators.hostCopyPctOfHost.toFixed(1)}%，并伴随 ${indicators.hostSyncCount} 次同步：拷贝本身不大，但同步往返代价高。`,
    estimate: (indicators) => {
      const ratio = indicators.avgStepUs > 0 ? Math.min(0.3, (indicators.d2hPerStepUs * 0.7) / indicators.avgStepUs) : 0;
      return {
        metric: 'Decode 单步延迟（TPOT）',
        estimatePct: round(ratio * 100, 1),
        rangePct: [round(ratio * 50, 1), round(ratio * 100, 1)],
        basis: `D2H ${indicators.d2hPerStepUs.toFixed(0)}µs/步 ÷ 平均步长 ${indicators.avgStepUs.toFixed(0)}µs，按可异步化 70% 计算`,
        confidence: 'medium',
        assumption: '假设 host 侧后续逻辑不依赖同步结果。',
      };
    },
    verification: '确认 D2H 不再阻塞计算流：同步算子次数下降、单步延迟抖动收敛。',
    risk: '异步化后若 host 侧时序处理不当，可能出现读到旧数据的竞态。',
  },
  {
    id: 'copy.kv-locality',
    title: '检查 KV Cache 局部性：避免每步 H2D（PD 传输/前缀缓存换入）',
    bottleneck: 'copy',
    phase: 'both',
    priority: '中',
    applies: (indicators) => indicators.h2dPerStepUs >= 150,
    actions: [
      { type: 'config', text: '检查 PD 分离场景下的 KV 传输路径，尽量让 KV 直接落在设备侧（先分配后传输，避免经主机内存中转）。' },
      { type: 'config', text: '调整前缀缓存策略（block 大小、命中率），减少重复 prefill 与 KV 换入。' },
      { type: 'verify', text: '确认稳态步中 H2D 是否接近 0；持续 H2D 通常意味着有东西每步都在重载。' },
    ],
    rationale: (indicators) => `每步 H2D 达 ${indicators.h2dPerStepUs.toFixed(0)}µs：推理稳态下权重与 KV 都应常驻显存，持续的 H2D 说明主机内存被放进了关键路径。`,
    estimate: (indicators) => {
      const ratio = indicators.avgStepUs > 0 ? Math.min(0.25, (indicators.h2dPerStepUs * 0.8) / indicators.avgStepUs) : 0;
      return {
        metric: '单步延迟',
        estimatePct: round(ratio * 100, 1),
        rangePct: [round(ratio * 50, 1), round(ratio * 100, 1)],
        basis: `H2D ${indicators.h2dPerStepUs.toFixed(0)}µs/步 ÷ 平均步长 ${indicators.avgStepUs.toFixed(0)}µs`,
        confidence: 'medium',
        assumption: '假设 H2D 可通过缓存/传输路径调整消除。',
      };
    },
    verification: '统计稳态步的 H2D 时长是否降为接近 0。',
    risk: '消除中转可能要求改架构（PD 传输路径、缓存策略），改动面较大。',
  },
  {
    id: 'copy.reduce-profiling-overhead',
    title: '复核 profiling/日志自身带来的拷贝与同步开销',
    bottleneck: 'copy',
    phase: 'both',
    priority: '低',
    applies: (indicators) => indicators.hostSyncCount > 0 && indicators.hostSyncUs >= indicators.wallUs * 0.05,
    actions: [
      { type: 'config', text: '采集时降低采样级别（如 Level0/Level1）、关闭不需要的采集项（内存、FLOPS、算子级流水），避免 profiling 本身成为瓶颈。' },
      { type: 'verify', text: '对比开启/关闭 profiling 时的单步延迟与同步次数，确认测量本身没有改变结论。' },
    ],
    rationale: (indicators) => `同步类算子累计 ${indicators.hostSyncUs.toFixed(0)}µs，占墙钟 ${((indicators.hostSyncUs / Math.max(1, indicators.wallUs)) * 100).toFixed(1)}%：其中一部分可能来自 profiling 采集本身。`,
    estimate: () => ({
      metric: '测量噪声/同步开销',
      estimatePct: 5,
      rangePct: [2, 10],
      basis: '经验区间：降低采集级别通常可减少的同步与拷贝开销',
      confidence: 'low',
      assumption: '经验区间；用于排除测量偏差，而非提升真实性能。',
    }),
    verification: '同负载下对比不同采集级别的单步延迟，确认差异。',
    risk: '降低采集级别会损失分析所需的指标，需要在分析完成后单独做验证运行。',
  },

  // ── Cross-cutting ────────────────────────────────────────────────────────
  {
    id: 'common.phase-isolation',
    title: '把 Prefill 与 Decode 分开采集/分池，避免混部掩盖各自的瓶颈',
    bottleneck: 'balanced',
    phase: 'both',
    priority: '中',
    applies: (indicators, context) => context.phaseMix === true && context.phaseConfidence === 'low',
    actions: [
      { type: 'measure', text: '分别采集 prefill-only 与 decode-only 窗口（例如先只发长 prompt 请求，再只发纯生成请求），profile 阶段用 /start_profile 与 /stop_profile 精确框定。' },
      { type: 'config', text: '生产环境考虑 PD 分离部署，让两类负载各自达到最优配置。' },
      { type: 'measure', text: '开启 mstx 阶段标记（mstx_torch_plugin 的 forward/step 域）以获得逐步阶段标签。' },
    ],
    rationale: () => '本数据集未能可靠划分 Prefill/Decode（缺少 mstx 阶段标记，且步长分布无双峰特征）。混部采集会让两类瓶颈互相掩盖：Prefill 的长步拉高平均，Decode 的逐步开销又被稀释。',
    estimate: () => ({
      metric: '分析置信度',
      estimatePct: 0,
      rangePct: [0, 0],
      basis: '该建议不直接产生性能收益，而是让后续优化有的放矢（避免在错误阶段上做优化）',
      confidence: 'high',
      assumption: '无性能假设；目的是提高结论可信度。',
    }),
    verification: '分开采集后重新运行本分析，确认 Prefill/Decode 阶段指标独立可得。',
    risk: '分开采集需要控制流量，可能影响线上压测结果。',
  },
  {
    id: 'common.collect-richer-profile',
    title: '补齐 profiling 采集项（算子级流水利用率与通信统计）以提高结论可信度',
    bottleneck: 'balanced',
    phase: 'both',
    priority: '中',
    applies: (indicators, context) => context.hasUtilization === false || context.missingTables === true,
    actions: [
      { type: 'measure', text: '开启 AI Core 流水/算力指标采集（PipeUtilization 级别），以导出 mac_ratio / mte*_ratio 等列到 kernel_details.csv。' },
      { type: 'measure', text: '导出 op_statistic.csv / operator_details.csv / step_trace_time.csv，让累计耗时与阶段指标有 CANN 侧交叉校验。' },
      { type: 'measure', text: '多卡场景补充通信统计产物（communication.json / communication_matrix.json）。' },
    ],
    rationale: (indicators, context) => `当前产物${context.hasUtilization ? '' : '缺少算子级流水利用率'}${context.missingTables ? '且缺少 CANN 统计表' : ''}，导致"算力受限 vs 访存受限"这类判断只能给出区间而非定论。`,
    estimate: () => ({
      metric: '分析置信度',
      estimatePct: 0,
      rangePct: [0, 0],
      basis: '不直接产生性能收益；把经验区间替换为实测结论',
      confidence: 'high',
      assumption: '无性能假设。',
    }),
    verification: '补齐后重新运行分析，确认利用率与阶段指标出现且瓶颈判定更明确。',
    risk: '更高采集级别会带来额外开销（见上一条建议），建议与分析运行分开。',
  },
];

/**
 * Build the recommendation list for one candidate and its hypotheses.
 *
 * @param {object} input - recommendation input.
 * @param {object} input.candidate - scored bottleneck candidate.
 * @param {object[]} input.hypotheses - root causes for the candidate.
 * @param {object} input.indicators - indicator block for the phase.
 * @param {object} input.context - dataset summary (`phase`, `macRatio`, `mte2Ratio`, `medianComputeDurUs`, `hasUtilization`, `missingTables`, `phaseMix`, `phaseConfidence`).
 * @returns {object[]} recommendations, highest priority first.
 */
export function buildRecommendations({ candidate, hypotheses, indicators, context }) {
  const out = [];
  for (const template of TEMPLATES) {
    const matchesBottleneck = template.bottleneck === candidate.id
      || (template.bottleneck === 'balanced' && candidate.score < 60);
    if (!matchesBottleneck && candidate.id !== 'balanced') continue;
    if (candidate.id === 'balanced' && template.bottleneck !== 'balanced') continue;
    if (!phaseMatches(context.phase, template.phase)) continue;
    let applies = false;
    try {
      applies = template.applies(indicators, context) === true;
    } catch {
      applies = false;
    }
    if (!applies) continue;
    const gain = safeEstimate(template, indicators, context);
    out.push({
      id: template.id,
      title: template.title,
      phase: template.phase,
      priority: template.priority,
      confirmInVersion: template.confirmInVersion === true,
      bottleneck: candidate.id,
      actions: template.actions,
      rationale: safeCall(template.rationale, indicators, context),
      expectedGain: gain,
      verification: template.verification,
      risk: template.risk,
      linkedRootCauses: hypotheses.map((hypothesis) => hypothesis.id),
      priorityScore: priorityScore(template.priority, candidate.score, gain),
    });
  }
  return out.sort((left, right) => right.priorityScore - left.priorityScore);
}

/** Priority is a function of the template's own level and the measured severity. */
function priorityScore(priority, candidateScore, gain) {
  const base = (PRIORITY_RANK[priority] ?? 1) * 20;
  const measured = Math.min(40, candidateScore * 0.4);
  const projected = Math.min(20, (gain?.estimatePct ?? 0) * 0.4);
  return round(base + measured + projected, 2);
}

function safeCall(fn, indicators, context) {
  try {
    return typeof fn === 'function' ? fn(indicators, context) : String(fn ?? '');
  } catch (error) {
    return `（推断说明生成失败：${error instanceof Error ? error.message : String(error)}）`;
  }
}

function safeEstimate(template, indicators, context) {
  if (typeof template.estimate !== 'function') return undefined;
  try {
    const gain = template.estimate(indicators, context);
    if (gain === undefined) return undefined;
    // Long-run sanity: never project more than the measured share could allow.
    if (Number.isFinite(gain.estimatePct)) gain.estimatePct = Math.max(0, Math.min(90, gain.estimatePct));
    return { ...gain, scope: describeScope(gain.metric, indicators) };
  } catch {
    return undefined;
  }
}

function describeScope(metric, indicators) {
  return `${metric}；基准：平均步长 ${indicators.avgStepUs.toFixed(0)}µs、步数 ${indicators.stepCount}、墙钟 ${(indicators.wallUs / 1000).toFixed(1)}ms`;
}

/**
 * Rank recommendations across every phase, dropping duplicates that appear in
 * both phases (they are merged, with the phases listed).
 *
 * @param {object[][]} groups - recommendation lists per phase.
 * @returns {object[]} merged recommendations.
 */
export function mergeRecommendations(groups) {
  const byId = new Map();
  for (const list of groups) {
    for (const item of list) {
      const existing = byId.get(item.id);
      if (existing === undefined) {
        byId.set(item.id, { ...item, phases: [item.phase] });
        continue;
      }
      if (!existing.phases.includes(item.phase)) existing.phases.push(item.phase);
      if (item.priorityScore > existing.priorityScore) {
        existing.priorityScore = item.priorityScore;
        existing.expectedGain = item.expectedGain;
        existing.rationale = item.rationale;
      }
      existing.linkedRootCauses = [...new Set([...existing.linkedRootCauses, ...item.linkedRootCauses])];
    }
  }
  return [...byId.values()].sort((left, right) => right.priorityScore - left.priorityScore);
}

/**
 * Group recommendations by priority for the report's action plan.
 * @param {object[]} recommendations - merged recommendations.
 * @returns {{ 高: object[], 中: object[], 低: object[] }} grouped list.
 */
export function groupByPriority(recommendations) {
  return {
    高: recommendations.filter((item) => item.priority === '高'),
    中: recommendations.filter((item) => item.priority === '中'),
    低: recommendations.filter((item) => item.priority === '低'),
  };
}

/** Percent helper re-exported for report templates. */
export { percentOf };
