/**
 * Step ③ of the reasoning chain: infer *why* the located bottleneck exists, in
 * terms of how vLLM runs a model on Ascend NPUs.
 *
 * Each hypothesis is a record with a mechanism explanation, the vLLM-Ascend
 * behaviour behind it, the observable checks that confirm or reject it, and a
 * selector that decides whether it applies to this dataset at all. Hypotheses
 * are never emitted for a phase they cannot apply to, so a Decode-only capture
 * does not receive prefill advice and vice versa.
 *
 * Scope note: everything here is about the *inference engine and the NPU*. No
 * hypothesis is emitted without at least one measurable trigger in the parsed
 * data, and each carries `triggers` naming the numbers that selected it.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis/rootcause
 */

import { round } from '../model/stats.js';

/**
 * @typedef {object} Hypothesis
 * @property {string} id - stable id.
 * @property {string} title - one-line cause.
 * @property {'host'|'compute'|'comm'|'copy'} bottleneck - the bottleneck it explains.
 * @property {'prefill'|'decode'|'both'} phase
 * @property {string} mechanism - why the numbers look this way.
 * @property {string} vllmBehaviour - the vLLM-Ascend implementation detail.
 * @property {string[]} checks - how to confirm on the machine.
 * @property {Array<{metric: string, value: number|undefined, unit: string, note: string}>} triggers
 */

/**
 * Candidate root causes, ordered by specificity.
 *
 * Selectors receive `(indicators, context)` where `context` carries the dataset
 * summary, the bottleneck candidate that was selected, and the phase id.
 */
const HYPOTHESES = [
  // ── Host scheduling ──────────────────────────────────────────────────────
  {
    id: 'host.eager-python-per-token',
    title: 'Decode 逐步 eager 执行：Python 调度与算子派发成本无法被摊薄',
    bottleneck: 'host',
    phase: 'decode',
    select: (indicators, context) => context.phase === 'decode'
      && indicators.dispatchPerStep >= 400
      && indicators.deviceBusyPct < 95,
    mechanism: 'Decode 每步只生成 1 个 token，单步 NPU 计算量很小（毫秒级），因此每步固定的 CPU 成本（调度器选块、构造 attention metadata、逐算子派发、采样）占比被放大。当 host 每步派发的算子数量达到数百至上千个时，CPU 成为关键路径，NPU 在两次下发之间出现成片空闲。',
    vllmBehaviour: 'vLLM V1 引擎每个 decode step 都要跑 execute_model → 准备 slot mapping / block table / attention metadata → 逐个算子下发 → Sampler 取 logits。昇腾侧每次 aclnn 算子下发还包含 torch_npu 的算子适配层开销；未启用图模式时这些开销按步重复，无法被流水掩盖。',
    checks: [
      '确认是否启用图模式：torch_npu 的 ACL Graph / npugraph_ex 或 vLLM 的 compilation-config',
      '统计单步 host 算子数量与 host 独占时间（本报告的 Host 独占/步 指标）',
      '对比开启图模式前后同一并发下的 TPOT',
    ],
  },
  {
    id: 'host.sampler-sync',
    title: '采样/统计路径存在设备到主机的同步点（.item()/tolist/synchronize）',
    bottleneck: 'host',
    phase: 'both',
    select: (indicators) => indicators.hostSyncCount > 0
      && (indicators.hostSyncPerStep >= 0.5 || indicators.hostSyncUs >= indicators.wallUs * 0.03),
    mechanism: '同步类算子会强制等待设备队列排空。它们把“异步下发、设备连续执行”的流水打断，使每一处同步都变成一次串行往返，在 Decode 中直接叠加到 TPOT 上。',
    vllmBehaviour: '采样器、惩罚项（repetition/presence penalty）、stop-string 判定、日志与指标采集都可能触发 D2H 拷贝 + 同步；昇腾上 aten::item / _local_scalar_dense 会等待对应的 device-to-host 拷贝完成。',
    checks: [
      '按耗时排序查看 host 侧 item/tolist/_local_scalar_dense/synchronize 算子出现频次',
      '确认采样是否可异步化（异步输出处理 / 结构化输出后置）',
    ],
  },
  {
    id: 'host.prefill-metadata',
    title: 'Prefill 步内 host 侧元数据准备与长序列 attention metadata 构造开销过大',
    bottleneck: 'host',
    phase: 'prefill',
    select: (indicators, context) => context.phase === 'prefill'
      && indicators.hostOnlyPct >= 15
      && indicators.dispatchPerStep >= 300,
    mechanism: 'Prefill 单步承载大量 token，host 需要在步内构造变长序列的 block table、slot mapping 与 attention mask，再做大量小算子的派发；当序列长度分布不均匀时，这部分 CPU 工作量随批内序列数量增长。',
    vllmBehaviour: 'vLLM 的 chunked prefill 会把长 prompt 切成多个 chunk，每个 chunk 都是一次完整的 host 元数据构造；昇腾侧 attention 算子的 metadata 需按 batch 逐个准备，容易出现 host 与 device 交替等待。',
    checks: [
      '查看 Prefill 阶段 host 独占时间与派发算子数',
      '对比不同 max-num-batched-tokens 下的 Prefill 步耗时',
      '确认是否开启前缀缓存（prefix caching）以减少重复 prefill',
    ],
  },
  {
    id: 'host.launch-gaps',
    title: '设备忙于启动间隙：算子粒度太碎导致下发速率跟不上执行速率',
    bottleneck: 'host',
    phase: 'both',
    select: (indicators) => indicators.gapCount >= 5
      && indicators.meanGapUs >= 80
      && indicators.idlePct >= 5,
    mechanism: '设备空闲间隙的次数多、平均长度在百微秒量级，说明不是"某个算子慢"，而是"设备经常在等下一个算子"。这通常是算子粒度过碎或 stream 并行度不足造成的下发吞吐问题。',
    vllmBehaviour: '昇腾上每个算子任务需要经过 host 侧队列 → runtime → device 调度；当单算子执行时间与下发开销同量级（几十微秒）时，下发速率就成为吞吐上限。多 stream 并行与算子融合可缓解。',
    checks: [
      '查看设备时间线上空闲间隙的分布与前后算子',
      '检查是否可启用算子融合（如 FlashComm、融合 MoE、融合采样）',
      '检查是否存在不必要的 stream 同步',
    ],
  },

  // ── NPU 计算 ─────────────────────────────────────────────────────────────
  {
    id: 'compute.prefill-attention-quadratic',
    title: 'Prefill 计算量集中在长序列注意力（O(n²)）与未量化的矩阵乘',
    bottleneck: 'compute',
    phase: 'prefill',
    select: (indicators, context) => context.phase === 'prefill'
      && indicators.computePct >= 55
      && (context.topComputeSubtypes ?? []).some((entry) => entry.subtype === 'attention'),
    mechanism: 'Prefill 的算力消耗随提示长度平方增长（注意力）与线性增长（投影矩阵乘）。当 NPU 忙碌率高且 MAC 利用率高时，说明算力已被充分使用，进一步提速只能靠减少计算量（量化、稀疏、更高效的注意力实现）或改变切分策略。',
    vllmBehaviour: 'vLLM-Ascend 的 Prefill 走 FlashAttention 类算子；长序列下 attention 的算力占比显著上升。张量并行（TP）会把矩阵乘切分到多卡，Attention 则按 head 切分，二者对算力利用率的影响不同。',
    checks: [
      '查看 attention 类算子的累计耗时与输入 shape（序列长度）',
      '确认量化配置（W8A8/FP8）是否生效，比较量化前后 MatMul 耗时',
      '确认 chunked prefill 的 chunk 大小是否与序列长度匹配',
    ],
  },
  {
    id: 'compute.decode-memory-bound',
    title: 'Decode 受权重读取带宽限制：小 batch 下算术强度不足（忙而不算）',
    bottleneck: 'compute',
    phase: 'decode',
    select: (indicators, context) => context.phase === 'decode'
      && indicators.deviceBusyPct >= 70
      && (context.macRatio === undefined || context.macRatio < 0.35)
      && (context.mte2Ratio === undefined || context.mte2Ratio >= 0.25),
    mechanism: 'Decode 每步只处理 1 个 token，矩阵乘退化为矩阵-向量乘，每个权重只被使用一次，算力单元大量空转，真正的瓶颈是权重从 HBM 读取的带宽。此时 NPU 看起来"很忙"，但 MAC 利用率很低。',
    vllmBehaviour: 'vLLM-Ascend 在 Decode 阶段用 paged attention 读取 KV Cache，同时每步都要把全部模型权重读一遍。提升 batch（continuous batching 的并发请求数）是提高算术强度最直接的手段；MoE 模型在小 batch 下专家并行度不足，更加明显。',
    checks: [
      '查看 MAC/MTE2 利用率对比（mac_ratio vs mte2_ratio）',
      '提高并发请求数（max-num-seqs / batch）后观察 TPOT 与吞吐变化',
      '检查是否启用了权重压缩/量化以减少每步权重读取量',
    ],
  },
  {
    id: 'compute.fragmented-small-ops',
    title: '大量小算子碎片化执行，Cube 利用率被启动开销摊薄',
    bottleneck: 'compute',
    phase: 'both',
    select: (indicators, context) => indicators.deviceBusyPct >= 70
      && indicators.computePct >= 40
      && (context.medianComputeDurUs ?? 0) > 0
      && (context.medianComputeDurUs ?? 0) < 30,
    mechanism: '设备侧计算算子时长中位数只有几十微秒甚至更低，说明算力消耗分散在大量小算子上，每个算子的启动/同步开销占比过高，Cube 难以形成长流水。',
    vllmBehaviour: '昇腾上小算子（elementwise、cast、reshape 等）由 Vector/Scalar 单元执行，频繁的 kernel 切换会打断 Cube 流水；算子融合（如融合 add+rmsnorm、融合 rope）能显著减少这类碎片。',
    checks: [
      '统计设备侧计算算子时长分布（本报告的 p50/p95 与算子数量）',
      '查看单算子耗时 <30µs 的算子总占比',
      '确认可用融合算子是否已启用',
    ],
  },

  // ── 跨卡通信 ─────────────────────────────────────────────────────────────
  {
    id: 'comm.decode-tp-small-message',
    title: 'Decode 张量并行集合通信为小消息延迟受限，且未与计算重叠',
    bottleneck: 'comm',
    phase: 'decode',
    select: (indicators, context) => context.phase === 'decode'
      && indicators.commPctOfDevice >= 12
      && indicators.commLatencyMedianUs <= 120,
    mechanism: 'Decode 每层都要做一次 AllReduce（张量并行），每步消息量只有 KB 级，通信时间由 HCCL 的固定启动开销（数十微秒）决定而非带宽。这类通信无法通过调大 buffer 改善，只能减少次数、提高重叠度或改变并行策略。',
    vllmBehaviour: 'vLLM-Ascend 的 TP 推理在每层 attention/MLP 之后插入 AllReduce；由于每一步计算量小，通信无法被计算掩盖，未掩盖部分直接叠加到 TPOT。常见的缓解手段是通信计算融合（MC2/FlashComm 类）与更大 batch 摊薄固定开销。',
    checks: [
      '查看通信算子单次时长中位数与每步通信次数',
      '确认是否启用通信计算融合能力（需确认当前 vLLM-Ascend 版本支持的具体开关）',
      '增大并发后观察通信占比是否被摊薄',
    ],
  },
  {
    id: 'comm.prefill-bandwidth',
    title: 'Prefill 集合通信为带宽受限，且与计算的流水重叠不足',
    bottleneck: 'comm',
    phase: 'prefill',
    select: (indicators, context) => context.phase === 'prefill'
      && indicators.commPctOfDevice >= 12
      && (indicators.commSizeMedianMb === undefined || indicators.commSizeMedianMb >= 1),
    mechanism: 'Prefill 的消息量大（激活值可达数十 MB），通信时间由互联带宽决定。若未与计算重叠，则每层的 AllReduce/AllGather 都是串行段，直接拉长 Prefill 首 token 延迟。',
    vllmBehaviour: '昇腾多卡通过 HCCL 在 RoCE/HCCS 上做集合通信；流水并行与专家并行会引入 AlltoAll（MoE dispatch/combine），消息模式更碎，对重叠更敏感。',
    checks: [
      '查看通信消息量与单次时长，判断带宽受限还是延迟受限',
      '检查是否存在 DP/EP 带来的额外 AllGather/AlltoAll',
      '确认通信与计算的重叠率（本报告的通信重叠率指标）',
    ],
  },
  {
    id: 'comm.overlap-disabled',
    title: '通信与计算重叠率低：所有集合通信都在关键路径上串行执行',
    bottleneck: 'comm',
    phase: 'both',
    select: (indicators) => indicators.commPctOfDevice >= 8 && indicators.commOverlapPct < 40,
    mechanism: '通信总时长中只有不到 40% 与计算并行发生，其余都在关键路径上。这说明通信没有被放进独立的流并让计算先行，或者存在的依赖使 overlap 无法实现。',
    vllmBehaviour: 'vLLM-Ascend 的张量并行通信默认在独立流上执行，但如果没有算子级的切分（把一次大 AllReduce 拆成按 chunk 的多段）或缺少通信计算融合，计算与通信就会互相等待。',
    checks: [
      '查看通信与计算在时间线上的重叠情况',
      '确认是否有可用的通信切分/融合配置（需按版本确认）',
      '检查是否因为同步点（.item()/barrier）打断了重叠',
    ],
  },

  // ── 数据拷贝 ─────────────────────────────────────────────────────────────
  {
    id: 'copy.d2h-sync-sampling',
    title: '采样/统计路径的 D2H 拷贝与同步拖慢每步',
    bottleneck: 'copy',
    phase: 'both',
    select: (indicators) => indicators.d2hPerStepUs >= 150 || indicators.hostSyncCount > 0,
    mechanism: 'D2H 拷贝本身耗时不大，但它必须等待设备侧产生数据，且常在拷贝后立即被 host 读取，形成一次完整的同步往返。在 Decode 中，这类往返每步都会发生。',
    vllmBehaviour: '采样输出、stop 判定、指标统计、结构化输出校验都可能触发 logits/统计量的 D2H 回传；昇腾上通过 aclrtMemcpy（device to host）实现，未使用 pinned memory 时还会额外付出一次页锁定开销。',
    checks: [
      '统计每步 D2H 时长与次数',
      '确认采样能否异步化、能否把需要 host 判定的逻辑后置',
      '确认使用 pinned memory 与异步拷贝接口',
    ],
  },
  {
    id: 'copy.h2d-kv-or-weight',
    title: 'H2D 拷贝出现在稳态步中：KV Cache 传输或权重重载',
    bottleneck: 'copy',
    phase: 'both',
    select: (indicators) => indicators.h2dPerStepUs >= 200,
    mechanism: '推理稳态下权重应常驻显存、KV Cache 应留在设备侧，因此持续的 H2D 流量通常来自：PD 分离的 KV 传输、前缀缓存换入、或权重分片重载（LoRA 切换、专家换入换出）。',
    vllmBehaviour: 'vLLM-Ascend 的 PD 分离与 prefix caching 会把 KV block 在主机与设备之间搬运；如果每步都有 H2D，说明缓存命中策略或 PD 传输路径把主机内存放进了关键路径。',
    checks: [
      '统计每步 H2D 时长与拷贝次数',
      '确认前缀缓存命中率与 PD 分离的 KV 传输路径',
      '确认 KV Cache 是否可直接在设备侧预分配',
    ],
  },
];

/**
 * Select the hypotheses that apply to one bottleneck candidate.
 *
 * @param {object} candidate - a scored candidate from `scoreBottlenecks`.
 * @param {object} indicators - indicator block for the same phase.
 * @param {object} context - dataset summary: `topComputeSubtypes`, `macRatio`, `mte2Ratio`, `medianComputeDurUs`, `phase`.
 * @returns {Hypothesis[]} applicable hypotheses.
 */
export function inferRootCauses(candidate, indicators, context) {
  const out = [];
  for (const hypothesis of HYPOTHESES) {
    if (hypothesis.bottleneck !== candidate.id) continue;
    if (hypothesis.phase !== 'both' && hypothesis.phase !== context.phase) continue;
    let applicable = false;
    try {
      applicable = hypothesis.select(indicators, context) === true;
    } catch {
      applicable = false;
    }
    if (!applicable) continue;
    out.push({
      id: hypothesis.id,
      title: hypothesis.title,
      bottleneck: hypothesis.bottleneck,
      phase: hypothesis.phase,
      mechanism: hypothesis.mechanism,
      vllmBehaviour: hypothesis.vllmBehaviour,
      checks: hypothesis.checks,
      triggers: buildTriggers(hypothesis.id, indicators, context),
    });
  }
  // A bottleneck with no specialised hypothesis still gets the generic one.
  if (out.length === 0 && candidate.score >= 40) {
    out.push({
      id: `${candidate.id}.generic`,
      title: `${candidate.label}：证据充分但需要现场确认根因`,
      bottleneck: candidate.id,
      phase: 'both',
      mechanism: candidate.summary,
      vllmBehaviour: '该瓶颈的量化证据已满足门限，但现有 profiling 产物不足以进一步区分具体成因（缺少算子级流水利用率、通信报文大小或阶段标记）。',
      checks: [
        '补充采集 kernel_details.csv / op_summary.csv 以获得流水与算力利用率',
        '若问题集中在 Decode，补充开启通信算子级 profiling（mstx 或通信统计）',
        '对同一模型做变量对照：并发数、chunk 大小、并行策略各改一项',
      ],
      triggers: [{
        metric: '瓶颈得分',
        value: candidate.score,
        unit: '',
        note: '候选得分来自门限打分，未细分到具体成因',
      }],
    });
  }
  return out;
}

/** Quantified triggers that made a hypothesis applicable. */
function buildTriggers(id, indicators, context) {
  const triggers = [];
  const add = (metric, value, unit, note) => triggers.push({ metric, value: Number.isFinite(value) ? round(value, 2) : undefined, unit, note });
  if (id.startsWith('host.')) {
    add('Host 独占占比', indicators.hostOnlyPct, '%', '设备等待主机的时间比例');
    add('NPU 空闲占比', indicators.idlePct, '%', '设备区间未覆盖的墙钟比例');
    add('Host 派发算子数/步', indicators.dispatchPerStep, '个', '每步 CPU 侧下发算子数量');
    add('Host 独占时间/步', indicators.hostExclusivePerStepUs, 'µs', '每步设备空等主机的时间');
    if (indicators.hostSyncCount > 0) add('同步类算子次数', indicators.hostSyncCount, '次', `累计 ${indicators.hostSyncUs.toFixed(0)}µs`);
    if (indicators.gapCount > 0) add('设备空闲间隙', indicators.gapCount, '个', `平均 ${indicators.meanGapUs.toFixed(0)}µs，最大 ${indicators.maxGapUs.toFixed(0)}µs`);
  } else if (id.startsWith('compute.')) {
    add('NPU 忙碌率', indicators.deviceBusyPct, '%', '设备区间并集/墙钟');
    add('计算算子占墙钟', indicators.computePct, '%', 'compute 类设备算子耗时');
    add('MAC 利用率', context.macRatio, '', 'CANN mac_ratio 加权值');
    add('MTE2 利用率', context.mte2Ratio, '', 'CANN mte2_ratio 加权值');
    add('通信未掩盖', indicators.commExposedPct, '%', '通信未与计算重叠的部分');
  } else if (id.startsWith('comm.')) {
    add('通信占设备耗时', indicators.commPctOfDevice, '%', '通信区间并集/设备忙碌');
    add('通信未掩盖', indicators.commExposedPct, '%', '通信未与计算重叠');
    add('通信重叠率', indicators.commOverlapPct, '%', '通信区间 ∩ 计算区间');
    add('单次通信时长中位数', indicators.commLatencyMedianUs, 'µs', '判断延迟受限/带宽受限');
    add('每步通信次数', indicators.commPerStep, '次', '通信算子计数/步数');
    if (indicators.commSizeMedianMb !== undefined) add('报文大小中位数', indicators.commSizeMedianMb, 'MB', '来自算子报文大小');
  } else if (id.startsWith('copy.')) {
    add('设备侧拷贝占墙钟', indicators.copyPct, '%', 'device copy 算子耗时');
    add('Host 侧拷贝占 Host 忙', indicators.hostCopyPctOfHost, '%', 'host copy 算子耗时');
    add('D2H 时长/步', indicators.d2hPerStepUs, 'µs', 'device→host 拷贝');
    add('H2D 时长/步', indicators.h2dPerStepUs, 'µs', 'host→device 拷贝');
    if (indicators.hostSyncCount > 0) add('同步类算子次数', indicators.hostSyncCount, '次', 'D2H 常伴随同步');
  }
  return triggers;
}

/** Exported for the docs page and tests. */
export const ROOT_CAUSE_CATALOG = HYPOTHESES.map((hypothesis) => ({
  id: hypothesis.id,
  title: hypothesis.title,
  bottleneck: hypothesis.bottleneck,
  phase: hypothesis.phase,
}));
