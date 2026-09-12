/**
 * Every numeric threshold the advice engine uses, with the rationale for the
 * value and the direction of the comparison.
 *
 * Thresholds are named constants rather than inline literals so that the report
 * can quote the rule it applied ("NPU 忙碌率 71% < 85% 门限") and so a
 * deployment can retune them through plugin config without touching logic.
 * They are engineering defaults for Ascend NPU inference: a 100% busy NPU means
 * ~1–2% launch gaps, and anything below ~85% on the device side while the host
 * is over half busy is a scheduling problem, not a math problem.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis/thresholds
 */

/**
 * @typedef {object} Threshold
 * @property {number} value - the threshold.
 * @property {'>='|'<='|'>'|'<'} relation - comparison that counts as "triggered".
 * @property {string} unit - display unit.
 * @property {string} rationale - why this value.
 */

/** @type {Record<string, Threshold>} */
export const THRESHOLDS = {
  // ── host scheduling ──────────────────────────────────────────────────────
  hostBusyGate: {
    value: 45,
    relation: '>=',
    unit: '%',
    rationale: 'Host 侧忙占比超过 45% 时，CPU 已经从“发起者”变成“瓶颈候选”；正常图模式/流水线充分的推理负载 host 占比通常在 10%–30%。',
  },
  hostOnlyGate: {
    value: 25,
    relation: '>=',
    unit: '%',
    rationale: 'Host 独占（未被设备工作覆盖）的时间超过墙钟 25%，说明设备在等 CPU，属于典型的调度瓶颈。',
  },
  deviceIdleSupport: {
    value: 8,
    relation: '>=',
    unit: '%',
    rationale: 'NPU 空闲超过 8% 才值得归因于 host 调度；更小的空隙属于正常 launch 延迟与流切换。',
  },
  dispatchPerStep: {
    value: 1200,
    relation: '>=',
    unit: '个/step',
    rationale: '单个推理步内 host 侧算子超过 ~1200 个时，逐 op 派发开销（Python + ATen + aclnn 包装）会成为每步的固定成本。',
  },
  hostExclusivePerStepUs: {
    value: 800,
    relation: '>=',
    unit: 'µs',
    rationale: '每步 host 独占时间超过 800µs，在 Decode（每步 5–30ms）中已占总延迟的显著比例。',
  },

  // ── NPU 计算 ──────────────────────────────────────────────────────────────
  deviceBusyGate: {
    value: 82,
    relation: '>=',
    unit: '%',
    rationale: 'NPU 忙碌率 ≥82% 表示设备几乎全程有任务，瓶颈在设备内部；此时优化方向是“算得更快”而不是“喂得更快”。',
  },
  computeShareGate: {
    value: 55,
    relation: '>=',
    unit: '%',
    rationale: '设备侧计算类算子占墙钟 55% 以上，才能判定为计算瓶颈。',
  },
  commExposedLow: {
    value: 12,
    relation: '<',
    unit: '%',
    rationale: '通信未掩盖时间低于墙钟 12% 时，通信不是主瓶颈（可作为次要因素）。',
  },
  macRatioComputeBound: {
    value: 0.45,
    relation: '>=',
    unit: '',
    rationale: 'MAC（Cube）利用率 ≥0.45 表明计算单元是主要受限资源，属于算力受限。',
  },
  mte2RatioMemoryBound: {
    value: 0.45,
    relation: '>=',
    unit: '',
    rationale: 'MTE2（GM→L1）利用率 ≥0.45 而 MAC 偏低，说明算子受显存/带宽限制，属于访存受限。',
  },

  // ── 跨卡通信 ──────────────────────────────────────────────────────────────
  commShareGate: {
    value: 12,
    relation: '>=',
    unit: '%',
    rationale: '通信算子在设备侧总耗时中占比 ≥12% 时进入通信瓶颈判定（张量并行 8 卡典型值）。',
  },
  commExposedGate: {
    value: 5,
    relation: '>=',
    unit: '%',
    rationale: '只有“未被计算掩盖”的通信时间才真正进入关键路径；未掩盖占比 ≥5% 时才建议动通信优化。',
  },
  commLatencyBoundUs: {
    value: 60,
    relation: '<=',
    unit: 'µs',
    rationale: '单次集合通信时长 ≤60µs 属于小消息延迟受限区间（HCCL 固定开销主导），增大 batch 比调算法更有效。',
  },
  commBandwidthBoundMb: {
    value: 16,
    relation: '>=',
    unit: 'MB',
    rationale: '单次通信消息量 ≥16MB 时进入带宽受限区间（8 卡 200GB/s 级互联的交叉点附近）。',
  },

  // ── 数据拷贝 ──────────────────────────────────────────────────────────────
  copyShareGate: {
    value: 4,
    relation: '>=',
    unit: '%',
    rationale: '设备侧拷贝占墙钟 ≥4% 时进入拷贝瓶颈判定；推理负载中拷贝本应接近 0（权重常驻显存）。',
  },
  hostCopyShareGate: {
    value: 8,
    relation: '>=',
    unit: '%',
    rationale: 'Host 侧发起的拷贝占 host 忙时间 ≥8% 时，通常意味着同步式 D2H（采样、日志、指标）在拖慢每步。',
  },
  d2hPerStepUs: {
    value: 200,
    relation: '>=',
    unit: 'µs',
    rationale: '每步 D2H 超过 200µs 通常来自 logits/统计量回传，且往往伴随流同步。',
  },

  // ── 空闲与长尾 ────────────────────────────────────────────────────────────
  gapShareGate: {
    value: 10,
    relation: '>=',
    unit: '%',
    rationale: '最大单次空闲占墙钟 ≥10% 说明存在一次性停顿（编译、权重加载、KV 传输、首次执行），必须单独解释。',
  },
  p95OverP50Gate: {
    value: 2,
    relation: '>=',
    unit: '倍',
    rationale: '步骤 p95/p50 ≥2 表示步长长尾明显，通常来自 chunked prefill 与 decode 混排或 PD 混部。',
  },
  lowUtilizationGate: {
    value: 0.3,
    relation: '<',
    unit: '',
    rationale: '设备忙碌但 MAC 利用率 <0.3，属于“忙而不算”，指向访存受限或小算子碎片化。',
  },
};

/**
 * Read a threshold, allowing a deployment override.
 * @param {string} key - threshold name.
 * @param {Record<string, number>} [overrides] - plugin config overrides.
 * @returns {Threshold} effective threshold.
 */
export function threshold(key, overrides = {}) {
  const base = THRESHOLDS[key];
  if (base === undefined) throw new Error(`unknown threshold: ${key}`);
  const override = overrides[key];
  return override === undefined ? base : { ...base, value: override };
}

/**
 * Test a value against a threshold.
 * @param {number} value - measured value.
 * @param {Threshold} rule - threshold rule.
 * @returns {boolean} whether the rule is satisfied.
 */
export function passes(value, rule) {
  if (!Number.isFinite(value)) return false;
  switch (rule.relation) {
    case '>=': return value >= rule.value;
    case '>': return value > rule.value;
    case '<=': return value <= rule.value;
    case '<': return value < rule.value;
    default: return false;
  }
}

/**
 * Map a measured value onto 0–1 severity for scoring.
 *
 * Below the threshold the score is 0 (the rule is a gate, not a gradient); at
 * or above `saturation` it is 1. `saturation` defaults to twice the threshold,
 * which is intentionally generous: a metric at 2× the trigger level is treated
 * as maximally indicative rather than unbounded.
 *
 * @param {number} value - measured value.
 * @param {Threshold} rule - threshold rule.
 * @param {number} [saturation] - value at which severity saturates.
 * @returns {number} severity in 0–1.
 */
export function severity(value, rule, saturation) {
  if (!Number.isFinite(value) || !passes(value, rule)) return 0;
  const top = saturation ?? (rule.relation.startsWith('<') ? rule.value / 2 : rule.value * 2);
  if (rule.relation === '>=' || rule.relation === '>') {
    return clamp01((value - rule.value) / Math.max(1e-6, top - rule.value)) * 0.5 + 0.5;
  }
  return clamp01((rule.value - value) / Math.max(1e-6, rule.value - top)) * 0.5 + 0.5;
}

/**
 * Describe a comparison for the evidence table.
 * @param {number} value - measured value.
 * @param {Threshold} rule - threshold rule.
 * @param {number} [digits] - decimals.
 * @returns {string} human-readable comparison.
 */
export function describe(value, rule, digits = 1) {
  const shown = Number.isFinite(value) ? value.toFixed(digits) : 'N/A';
  return `${shown}${rule.unit} ${rule.relation} ${rule.value}${rule.unit}（门限依据：${rule.rationale}）`;
}

function clamp01(value) {
  return Math.min(1, Math.max(0, value));
}
