/**
 * Step ① and ② of the reasoning chain: locate the bottleneck type and quantify
 * it with evidence taken from the parsed profiling data.
 *
 * Four bottleneck candidates are always evaluated — host scheduling, NPU
 * compute, cross-card communication, and H2D/D2H data copy — for the workload
 * as a whole and again per phase (prefill / decode), because the same capture
 * usually has two different bottlenecks. Every candidate carries:
 *
 * * a `score` (0–100) built from gated, saturated severity terms;
 * * the gate results, so a reader can see which rule was *not* met;
 * * an `evidence` list where each entry names the metric, its measured value,
 *   the threshold it was compared against, the comparison itself, and which
 *   artifact the number came from.
 *
 * The engine never invents a bottleneck: when nothing clears its gate the
 * verdict is `balanced` and the closest candidate is reported as a weak signal
 * with its shortfall spelled out.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis/bottleneck
 */

import { overlapLength, percentOf, round, unionIntervals } from '../model/stats.js';
import { describe, passes, severity, threshold } from './thresholds.js';

/** Bottleneck identifiers, in report order. */
export const BOTTLENECKS = Object.freeze({
  host: { id: 'host', label: 'Host 调度瓶颈', short: 'Host 调度', color: '#9b59b6' },
  compute: { id: 'compute', label: 'NPU 计算瓶颈', short: 'NPU 计算', color: '#4f7cff' },
  comm: { id: 'comm', label: '跨卡通信瓶颈', short: '跨卡通信', color: '#f2994a' },
  copy: { id: 'copy', label: 'H2D/D2H 数据拷贝瓶颈', short: '数据拷贝', color: '#27ae60' },
  balanced: { id: 'balanced', label: '未发现单一主导瓶颈', short: '相对均衡', color: '#7f8c9b' },
});

/**
 * Compute every indicator the analysis and the report refer to.
 *
 * @param {object} dataset - dataset from `buildDataset`.
 * @param {object} [options] - `thresholds` overrides.
 * @returns {object} indicator set, overall and per phase.
 */
export function computeIndicators(dataset, options = {}) {
  const overrides = options.thresholds ?? {};
  const overall = indicatorsForWindow(dataset.events, dataset.meta.window, dataset, overrides);
  const phases = {};
  for (const phase of dataset.phases.phases ?? []) {
    const memberEvents = selectPhaseEvents(dataset, phase.stepIndexes);
    if (memberEvents.length === 0) continue;
    const window = {
      start: Math.min(...memberEvents.map((event) => event.relTsUs)),
      end: Math.max(...memberEvents.map((event) => event.relTsUs + event.durUs)),
    };
    phases[phase.id] = {
      ...indicatorsForWindow(memberEvents, window, dataset, overrides),
      stepCount: phase.stepCount,
      avgStepUs: phase.avgStepUs,
      label: phase.label,
      fromCsvStepTable: (dataset.phases.steps ?? []).some((step) => step.phase === phase.id && step.metrics.hasCsvMetrics),
    };
  }
  return { overall, phases, thresholds: overrides };
}

/** Events belonging to the given step indexes. */
function selectPhaseEvents(dataset, stepIndexes) {
  const steps = (dataset.phases.steps ?? []).filter((step) => stepIndexes.includes(step.index));
  if (steps.length === 0) return [];
  return dataset.events.filter((event) => steps.some((step) => event.relTsUs >= step.startUs && event.relTsUs < step.endUs));
}

/**
 * Indicator block for one set of events over one window.
 * @param {object[]} events - events in the window.
 * @param {{start: number, end: number}} window - window.
 * @param {object} dataset - dataset (for gaps, utilization, phases).
 * @param {Record<string, number>} overrides - threshold overrides.
 * @returns {object} indicators.
 */
function indicatorsForWindow(events, window, dataset, overrides) {
  const wallUs = Math.max(1, window.end - window.start);
  const inWindow = (event) => event.relTsUs >= window.start && event.relTsUs < window.end;
  const scoped = events.filter(inWindow);
  const device = scoped.filter((event) => event.device === 'device');
  const host = scoped.filter((event) => event.device === 'host');

  const deviceBusyUs = unionIntervals(device.map(toInterval)).length;
  const hostBusyUs = unionIntervals(host.map(toInterval)).length;
  const commEvents = device.filter((event) => event.category === 'comm');
  const computeEvents = device.filter((event) => event.category === 'compute');
  const copyEvents = device.filter((event) => event.category === 'copy');
  const commUnionUs = unionIntervals(commEvents.map(toInterval)).length;
  const commOverlapUs = overlap(commEvents, computeEvents);
  const commExposedUs = Math.max(0, commUnionUs - commOverlapUs);
  const deviceCopyUs = copyEvents.reduce((sum, event) => sum + event.durUs, 0);
  const hostCopyUs = host.filter((event) => event.category === 'copy').reduce((sum, event) => sum + event.durUs, 0);
  const deviceComputeUs = computeEvents.reduce((sum, event) => sum + event.durUs, 0);
  const hostScheduleUs = host.reduce((sum, event) => sum + event.durUs, 0);

  const hostSyncEvents = host.filter((event) => /item|tolist|to_list|local_scalar_dense|synchronize|numpy/i.test(event.normalizedName));
  const hostSyncUs = hostSyncEvents.reduce((sum, event) => sum + event.durUs, 0);
  const dispatchEvents = host.filter((event) => event.role === 'framework' || event.category === 'schedule');

  const gaps = gapsWithin(scoped, window, overrides);
  const idleUs = gaps.totalUs;
  const maxGapUs = gaps.maxUs;
  const stepCount = estimateStepCount(dataset, window);

  const commLatency = medianOf(commEvents.map((event) => event.durUs));
  const commSizeMedian = medianOf(commEvents.map((event) => event.messageBytes).filter((value) => Number.isFinite(value)));
  const commPerStep = stepCount === 0 ? 0 : commEvents.length / stepCount;
  const commUsPerStep = stepCount === 0 ? 0 : commUnionUs / stepCount;
  const hostOnlyUs = hostBusyUs - overlap(host, device);
  const deviceOnlyUs = deviceBusyUs - overlap(host, device);

  const utilization = utilizationForWindow(dataset, scoped);

  return {
    wallUs: round(wallUs, 2),
    stepCount,
    eventCount: scoped.length,
    deviceEventCount: device.length,
    hostEventCount: host.length,
    deviceBusyUs: round(deviceBusyUs, 2),
    deviceBusyPct: round(percentOf(deviceBusyUs, wallUs), 2),
    hostBusyUs: round(hostBusyUs, 2),
    hostBusyPct: round(percentOf(hostBusyUs, wallUs), 2),
    idleUs: round(idleUs, 2),
    idlePct: round(percentOf(idleUs, wallUs), 2),
    hostOnlyUs: round(Math.max(0, hostOnlyUs), 2),
    hostOnlyPct: round(percentOf(Math.max(0, hostOnlyUs), wallUs), 2),
    deviceOnlyUs: round(Math.max(0, deviceOnlyUs), 2),
    deviceOnlyPct: round(percentOf(Math.max(0, deviceOnlyUs), wallUs), 2),
    commUs: round(commUnionUs, 2),
    commPct: round(percentOf(commUnionUs, wallUs), 2),
    commPctOfDevice: round(percentOf(commUnionUs, Math.max(1, deviceBusyUs)), 2),
    commExposedUs: round(commExposedUs, 2),
    commExposedPct: round(percentOf(commExposedUs, wallUs), 2),
    commOverlapPct: round(percentOf(commOverlapUs, Math.max(1, commUnionUs)), 2),
    commEventCount: commEvents.length,
    commPerStep: round(commPerStep, 2),
    commUsPerStep: round(commUsPerStep, 2),
    commLatencyMedianUs: round(commLatency, 2),
    commSizeMedianMb: Number.isFinite(commSizeMedian) ? round(commSizeMedian / (1024 * 1024), 3) : undefined,
    computeUs: round(deviceComputeUs, 2),
    computePct: round(percentOf(deviceComputeUs, wallUs), 2),
    computePctOfDevice: round(percentOf(deviceComputeUs, Math.max(1, deviceBusyUs)), 2),
    copyUs: round(deviceCopyUs, 2),
    copyPct: round(percentOf(deviceCopyUs, wallUs), 2),
    hostCopyUs: round(hostCopyUs, 2),
    hostCopyPctOfHost: round(percentOf(hostCopyUs, Math.max(1, hostBusyUs)), 2),
    hostCopyPct: round(percentOf(hostCopyUs, wallUs), 2),
    d2hPerStepUs: round(stepCount === 0 ? 0 : copyEvents.filter((event) => event.subtype === 'd2h').reduce((sum, event) => sum + event.durUs, 0) / stepCount, 2),
    h2dPerStepUs: round(stepCount === 0 ? 0 : copyEvents.filter((event) => event.subtype === 'h2d').reduce((sum, event) => sum + event.durUs, 0) / stepCount, 2),
    hostScheduleUs: round(hostScheduleUs, 2),
    hostSchedulePctOfHost: round(percentOf(hostScheduleUs, Math.max(1, hostBusyUs)), 2),
    dispatchEventCount: dispatchEvents.length,
    dispatchPerStep: round(stepCount === 0 ? 0 : dispatchEvents.length / stepCount, 1),
    hostSyncUs: round(hostSyncUs, 2),
    hostSyncCount: hostSyncEvents.length,
    hostSyncPerStep: round(stepCount === 0 ? 0 : hostSyncEvents.length / stepCount, 2),
    hostExclusivePerStepUs: round(stepCount === 0 ? hostOnlyUs : hostOnlyUs / stepCount, 2),
    maxGapUs: round(maxGapUs, 2),
    maxGapPct: round(percentOf(maxGapUs, wallUs), 2),
    meanGapUs: round(gaps.count === 0 ? 0 : idleUs / gaps.count, 2),
    gapCount: gaps.count,
    utilization,
    avgStepUs: round(stepCount === 0 ? wallUs : wallUs / stepCount, 2),
    sources: {
      commPct: '设备侧通信算子区间并集 / 窗口时长',
      commExposedPct: '通信区间并集 − 与计算区间重叠部分',
      hostBusyPct: 'Host 事件区间并集 / 窗口时长',
      utilization: utilization.source,
    },
  };
}

function toInterval(event) {
  return { start: event.relTsUs, end: event.relTsUs + event.durUs };
}

/** Interval overlap that merges both sides first (see `stats.overlapLength`). */
function overlap(a, b) {
  return overlapLength(a.map(toInterval), b.map(toInterval));
}

/** Device idle gaps inside a window, with a floor to ignore launch jitter. */
function gapsWithin(events, window, overrides) {
  const device = events.filter((event) => event.device === 'device');
  if (device.length === 0) {
    return { totalUs: window.end - window.start, maxUs: window.end - window.start, count: 1 };
  }
  const floorUs = overrides.minGapUs ?? 50;
  const { merged } = unionIntervals(device.map(toInterval));
  let cursor = window.start;
  let totalUs = 0;
  let maxUs = 0;
  let count = 0;
  for (const interval of merged) {
    const length = interval.start - cursor;
    if (length > floorUs) {
      totalUs += length;
      maxUs = Math.max(maxUs, length);
      count += 1;
    }
    cursor = Math.max(cursor, interval.end);
  }
  const tail = window.end - cursor;
  if (tail > floorUs) {
    totalUs += tail;
    maxUs = Math.max(maxUs, tail);
    count += 1;
  }
  return { totalUs, maxUs, count };
}

/** Weighted utilization metrics restricted to the window. */
function utilizationForWindow(dataset, events) {
  const names = new Set(events.map((event) => event.normalizedName));
  const rows = (dataset.utilization?.perOperator ?? []).filter((row) => names.has(row.name));
  if (rows.length === 0) {
    return {
      available: false,
      metrics: dataset.utilization?.metrics ?? [],
      source: dataset.utilization?.note ?? '无算子级利用率数据',
      scope: 'dataset',
    };
  }
  const weights = new Map();
  for (const row of rows) {
    const weight = row.durUs ?? 1;
    for (const [metric, value] of Object.entries(row.utilization)) {
      const entry = weights.get(metric) ?? { weighted: 0, weight: 0, label: row.labels?.[metric] };
      entry.weighted += value * weight;
      entry.weight += weight;
      weights.set(metric, entry);
    }
  }
  const metrics = [...weights.entries()].map(([metric, entry]) => ({
    metric,
    label: entry.label ?? metric,
    value: round(entry.weight === 0 ? 0 : entry.weighted / entry.weight, 4),
    valuePct: round((entry.weight === 0 ? 0 : entry.weighted / entry.weight) * 100, 1),
    weightUs: round(entry.weight, 2),
  })).sort((left, right) => right.weightUs - left.weightUs);
  return { available: metrics.length > 0, metrics, source: 'CANN 算子级流水/算力指标（按耗时加权）', scope: 'window' };
}

function estimateStepCount(dataset, window) {
  const steps = (dataset.phases.steps ?? []).filter((step) => step.startUs < window.end && step.endUs > window.start);
  return steps.length;
}

function medianOf(values) {
  const numbers = values.filter((value) => Number.isFinite(value));
  if (numbers.length === 0) return 0;
  const sorted = [...numbers].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

/**
 * Score the four bottleneck candidates for one indicator block.
 *
 * @param {object} indicators - from {@link computeIndicators}.
 * @param {object} [options] - `thresholds` overrides.
 * @param {string} [phase] - phase id this block belongs to.
 * @returns {{ candidates: object[], primary: object, secondary: object[], verdict: string }} scored candidates.
 */
export function scoreBottlenecks(indicators, options = {}, phase = 'overall') {
  const overrides = options.thresholds ?? {};
  const candidates = [
    scoreHost(indicators, overrides, phase),
    scoreCompute(indicators, overrides, phase),
    scoreComm(indicators, overrides, phase),
    scoreCopy(indicators, overrides, phase),
  ].sort((left, right) => right.score - left.score);
  const primary = candidates[0];
  const triggered = candidates.filter((candidate) => candidate.score >= 40);
  const verdict = triggered.length === 0 ? 'balanced' : triggered[0].id;
  return {
    phase,
    candidates,
    primary: verdict === 'balanced' ? { ...BOTTLENECKS.balanced, score: primary.score, evidence: primary.evidence, missing: primary.missing } : primary,
    secondary: triggered.slice(1),
    verdict,
  };
}

function scoreHost(indicators, overrides, phase) {
  const evidence = [];
  const missing = [];
  const gates = {
    hostOnly: threshold('hostOnlyGate', overrides),
    idle: threshold('deviceIdleSupport', overrides),
    dispatch: threshold('dispatchPerStep', overrides),
    exclusive: threshold('hostExclusivePerStepUs', overrides),
  };
  push(evidence, missing, 'Host 独占时间占比', indicators.hostOnlyPct, gates.hostOnly, '区间并集：host 忙且设备空闲的时间 / 墙钟');
  push(evidence, missing, 'NPU 空闲占比', indicators.idlePct, gates.idle, '设备事件区间未覆盖的时间 / 墙钟');
  push(evidence, missing, 'Host 派发算子数/步', indicators.dispatchPerStep, gates.dispatch, 'host 侧 schedule/framework 类事件计数 ÷ 步数');
  push(evidence, missing, 'Host 每步独占时间', indicators.hostExclusivePerStepUs, gates.exclusive, 'host 独占总时长 ÷ 步数');

  const primaryGate = passes(indicators.hostOnlyPct, gates.hostOnly) || passes(indicators.idlePct, gates.idle);
  const supportGate = passes(indicators.dispatchPerStep, gates.dispatch) || passes(indicators.hostExclusivePerStepUs, gates.exclusive);
  let score = 0;
  if (primaryGate) {
    score = 45 * Math.max(severity(indicators.hostOnlyPct, gates.hostOnly), severity(indicators.idlePct, gates.idle));
    if (supportGate) score += 35 * Math.max(severity(indicators.dispatchPerStep, gates.dispatch), severity(indicators.hostExclusivePerStepUs, gates.exclusive));
    if (indicators.hostBusyPct > 0) score += 20 * severity(indicators.hostBusyPct, threshold('hostBusyGate', overrides));
  }
  evidence.push({
    metric: 'Host 忙占比',
    value: indicators.hostBusyPct,
    unit: '%',
    threshold: threshold('hostBusyGate', overrides).value,
    comparison: describe(indicators.hostBusyPct, threshold('hostBusyGate', overrides)),
    source: 'host 事件区间并集 / 墙钟（仅供参考，不单独构成门限）',
  });
  if (indicators.hostSyncCount > 0) {
    evidence.push({
      metric: 'Host 同步类算子（.item()/tolist/synchronize）',
      value: indicators.hostSyncCount,
      unit: '次',
      threshold: 0,
      comparison: `每步 ${indicators.hostSyncPerStep.toFixed(2)} 次，累计 ${indicators.hostSyncUs.toFixed(1)}µs`,
      source: 'host 事件名匹配 item/tolist/synchronize',
    });
  }
  return {
    ...BOTTLENECKS.host,
    phase,
    score: round(clamp(score), 1),
    gates,
    evidence,
    missing,
    summary: primaryGate
      ? `Host 独占 ${indicators.hostOnlyPct.toFixed(1)}% 墙钟、设备空闲 ${indicators.idlePct.toFixed(1)}%，每个推理步 host 派发 ${indicators.dispatchPerStep.toFixed(0)} 个算子、独占 ${indicators.hostExclusivePerStepUs.toFixed(0)}µs。`
      : `未达到 Host 调度瓶颈门限：Host 独占占比 ${indicators.hostOnlyPct.toFixed(1)}% 与设备空闲 ${indicators.idlePct.toFixed(1)}% 均低于触发线。`,
  };
}

function scoreCompute(indicators, overrides, phase) {
  const evidence = [];
  const missing = [];
  const gates = {
    busy: threshold('deviceBusyGate', overrides),
    share: threshold('computeShareGate', overrides),
    commLow: threshold('commExposedLow', overrides),
    mac: threshold('macRatioComputeBound', overrides),
    mte2: threshold('mte2RatioMemoryBound', overrides),
    lowUtil: threshold('lowUtilizationGate', overrides),
  };
  push(evidence, missing, 'NPU 忙碌率', indicators.deviceBusyPct, gates.busy, '设备事件区间并集 / 墙钟');
  push(evidence, missing, '设备侧计算算子占墙钟', indicators.computePct, gates.share, 'compute 类设备算子耗时 / 墙钟');
  push(evidence, missing, '通信未掩盖占比', indicators.commExposedPct, gates.commLow, '通信区间并集 − 与计算重叠');

  const mac = metricValue(indicators, 'mac');
  const mte2 = metricValue(indicators, 'mte2');
  const vec = metricValue(indicators, 'vec');
  if (mac !== undefined) push(evidence, missing, 'MAC（Cube）利用率', mac, gates.mac, 'CANN 算子级 mac_ratio 按耗时加权');
  if (mte2 !== undefined) push(evidence, missing, 'MTE2（GM→L1）利用率', mte2, gates.mte2, 'CANN 算子级 mte2_ratio 按耗时加权');
  if (vec !== undefined) {
    evidence.push({
      metric: 'Vector 利用率',
      value: vec,
      unit: '',
      threshold: undefined,
      comparison: `实测 ${vec.toFixed(3)}（用于区分 Cube 受限与 Vector 受限）`,
      source: 'CANN 算子级 vec_ratio 按耗时加权',
    });
  }

  const gated = passes(indicators.deviceBusyPct, gates.busy) && passes(indicators.computePct, gates.share);
  let score = 0;
  let subtype;
  if (gated) {
    score = 55 * Math.max(severity(indicators.deviceBusyPct, gates.busy), severity(indicators.computePct, gates.share));
    if (passes(indicators.commExposedPct, gates.commLow)) score += 25;
    if (mac !== undefined && passes(mac, gates.mac)) {
      score += 20;
      subtype = 'compute-bound';
    } else if (mte2 !== undefined && passes(mte2, gates.mte2)) {
      score += 12;
      subtype = 'memory-bound';
    } else if (mac !== undefined && passes(mac, gates.lowUtil)) {
      subtype = 'low-efficiency';
      evidence.push({
        metric: '算力利用率偏低',
        value: mac,
        unit: '',
        threshold: gates.lowUtil.value,
        comparison: describe(mac, gates.lowUtil),
        source: 'CANN mac_ratio（设备忙但 MAC 利用率低 → 忙而不算）',
      });
    }
  }
  return {
    ...BOTTLENECKS.compute,
    phase,
    score: round(clamp(score), 1),
    subtype,
    gates,
    evidence,
    missing,
    summary: gated
      ? `NPU 忙碌率 ${indicators.deviceBusyPct.toFixed(1)}%、计算算子占墙钟 ${indicators.computePct.toFixed(1)}%，通信仅 ${indicators.commExposedPct.toFixed(1)}% 未被掩盖${subtype === undefined ? '' : `，判定为${subtype === 'compute-bound' ? '算力受限' : subtype === 'memory-bound' ? '访存受限' : '低效忙碌'}型`}。`
      : `未达到计算瓶颈门限：NPU 忙碌率 ${indicators.deviceBusyPct.toFixed(1)}%（门限 ${gates.busy.value}%）或计算占比 ${indicators.computePct.toFixed(1)}%（门限 ${gates.share.value}%）不足。`,
  };
}

function scoreComm(indicators, overrides, phase) {
  const evidence = [];
  const missing = [];
  const gates = {
    share: threshold('commShareGate', overrides),
    exposed: threshold('commExposedGate', overrides),
    latency: threshold('commLatencyBoundUs', overrides),
    bandwidth: threshold('commBandwidthBoundMb', overrides),
  };
  push(evidence, missing, '通信算子占设备耗时', indicators.commPctOfDevice, gates.share, '通信区间并集 / 设备忙碌时间');
  push(evidence, missing, '通信未掩盖占墙钟', indicators.commExposedPct, gates.exposed, '通信区间并集 − 与计算区间重叠');
  evidence.push({
    metric: '通信与计算重叠率',
    value: indicators.commOverlapPct,
    unit: '%',
    threshold: undefined,
    comparison: `已重叠 ${indicators.commOverlapPct.toFixed(1)}%，未掩盖 ${indicators.commExposedPct.toFixed(1)}%`,
    source: '通信区间 ∩ 计算区间 / 通信区间并集',
  });
  evidence.push({
    metric: '单次通信时长中位数',
    value: indicators.commLatencyMedianUs,
    unit: 'µs',
    threshold: gates.latency.value,
    comparison: describe(indicators.commLatencyMedianUs, gates.latency, 1) + '（≤ 门限属小消息延迟受限）',
    source: '通信算子 durUs 中位数',
  });
  if (indicators.commSizeMedianMb !== undefined) {
    evidence.push({
      metric: '单次通信消息量中位数',
      value: indicators.commSizeMedianMb,
      unit: 'MB',
      threshold: gates.bandwidth.value,
      comparison: describe(indicators.commSizeMedianMb, gates.bandwidth, 3),
      source: '通信算子报文大小（来自 messageBytes/算子参数，缺失时为 N/A）',
    });
  }
  evidence.push({
    metric: '每步通信次数 / 通信时长',
    value: indicators.commPerStep,
    unit: '次/step',
    threshold: undefined,
    comparison: `${indicators.commPerStep.toFixed(1)} 次、${indicators.commUsPerStep.toFixed(0)}µs/步`,
    source: '通信算子计数与并集时长 ÷ 步数',
  });

  const gated = passes(indicators.commPctOfDevice, gates.share) && passes(indicators.commExposedPct, gates.exposed);
  let score = 0;
  if (gated) {
    score = 55 * Math.max(severity(indicators.commPctOfDevice, gates.share), severity(indicators.commExposedPct, gates.exposed));
    if (passes(indicators.commLatencyMedianUs, gates.latency)) score += 20;
    else if (indicators.commSizeMedianMb !== undefined && passes(indicators.commSizeMedianMb, gates.bandwidth)) score += 15;
    if (indicators.commOverlapPct < 40) score += 15;
  }
  const bound = passes(indicators.commLatencyMedianUs, gates.latency) ? 'latency'
    : indicators.commSizeMedianMb !== undefined && passes(indicators.commSizeMedianMb, gates.bandwidth) ? 'bandwidth'
      : undefined;
  return {
    ...BOTTLENECKS.comm,
    phase,
    score: round(clamp(score), 1),
    subtype: bound,
    gates,
    evidence,
    missing,
    summary: gated
      ? `通信占设备耗时 ${indicators.commPctOfDevice.toFixed(1)}%、未掩盖 ${indicators.commExposedPct.toFixed(1)}%，单次时长中位数 ${indicators.commLatencyMedianUs.toFixed(0)}µs（${bound === 'latency' ? '延迟受限' : bound === 'bandwidth' ? '带宽受限' : '未定'}），每步 ${indicators.commPerStep.toFixed(1)} 次。`
      : `未达到通信瓶颈门限：通信占设备耗时 ${indicators.commPctOfDevice.toFixed(1)}%（门限 ${gates.share.value}%）或未掩盖占比 ${indicators.commExposedPct.toFixed(1)}%（门限 ${gates.exposed.value}%）不足。`,
  };
}

function scoreCopy(indicators, overrides, phase) {
  const evidence = [];
  const missing = [];
  const gates = {
    device: threshold('copyShareGate', overrides),
    host: threshold('hostCopyShareGate', overrides),
    d2h: threshold('d2hPerStepUs', overrides),
  };
  push(evidence, missing, '设备侧拷贝占墙钟', indicators.copyPct, gates.device, 'device copy 类算子耗时 / 墙钟');
  push(evidence, missing, 'Host 侧拷贝占 Host 忙时间', indicators.hostCopyPctOfHost, gates.host, 'host copy 类算子耗时 / host 忙碌时间');
  push(evidence, missing, '每步 D2H 时长', indicators.d2hPerStepUs, gates.d2h, 'device→host 拷贝耗时 ÷ 步数');
  evidence.push({
    metric: '每步 H2D 时长',
    value: indicators.h2dPerStepUs,
    unit: 'µs',
    threshold: undefined,
    comparison: `${indicators.h2dPerStepUs.toFixed(1)}µs/步`,
    source: 'host→device 拷贝耗时 ÷ 步数',
  });

  const gated = passes(indicators.copyPct, gates.device)
    || passes(indicators.hostCopyPctOfHost, gates.host)
    || passes(indicators.d2hPerStepUs, gates.d2h);
  let score = 0;
  if (gated) {
    // Weighted by how far past its gate each signal is, so a marginal D2H that
    // merely crosses the line cannot outrank a genuine host or communication
    // bottleneck. Synchronisation evidence adds to this bottleneck because a
    // synchronous D2H is what turns a copy into a stall.
    score = 45 * Math.max(
      severity(indicators.copyPct, gates.device),
      severity(indicators.hostCopyPctOfHost, gates.host),
      severity(indicators.d2hPerStepUs, gates.d2h),
    );
    if (indicators.hostSyncCount > 0) score += 12 * Math.min(1, indicators.hostSyncPerStep);
    if (passes(indicators.h2dPerStepUs, gates.d2h)) score += 10 * severity(indicators.h2dPerStepUs, gates.d2h);
  }
  return {
    ...BOTTLENECKS.copy,
    phase,
    score: round(clamp(score), 1),
    gates,
    evidence,
    missing,
    summary: gated
      ? `设备侧拷贝占墙钟 ${indicators.copyPct.toFixed(1)}%、Host 侧拷贝占 Host 忙时间 ${indicators.hostCopyPctOfHost.toFixed(1)}%、D2H ${indicators.d2hPerStepUs.toFixed(0)}µs/步（同步类算子 ${indicators.hostSyncCount} 次）。`
      : `未达到拷贝瓶颈门限：设备侧拷贝 ${indicators.copyPct.toFixed(1)}%（门限 ${gates.device.value}%）、Host 侧拷贝 ${indicators.hostCopyPctOfHost.toFixed(1)}%（门限 ${gates.host.value}%）。`,
  };
}

/** Push an evidence row and record the shortfall when the gate fails. */
function push(evidence, missing, metric, value, rule, source) {
  const ok = passes(value, rule);
  evidence.push({
    metric,
    value: Number.isFinite(value) ? round(value, 2) : undefined,
    unit: rule.unit,
    threshold: rule.value,
    relation: rule.relation,
    passed: ok,
    comparison: describe(value, rule, rule.unit === '' ? 3 : 1),
    source,
  });
  if (!ok) missing.push(`${metric} ${Number.isFinite(value) ? value.toFixed(1) : 'N/A'}${rule.unit} 未达到门限 ${rule.value}${rule.unit}`);
  return ok;
}

function metricValue(indicators, metric) {
  const found = indicators.utilization?.metrics?.find((entry) => entry.metric === metric);
  return found?.value;
}

function clamp(value) {
  return Math.min(100, Math.max(0, value));
}
