/**
 * Before/after comparison of two captures of the same workload.
 *
 * The optimization loop needs one thing the single-capture analysis cannot give:
 * proof that a change actually moved the numbers. This module compares the
 * "before" capture with the "after" capture and produces:
 *
 * * **headline deltas** — per-step wall time, device busy, host-exclusive time,
 *   dispatch count: the numbers a graph-mode / batching change is supposed to move;
 * * **per-metric rows** with an explicit better/worse direction, so a decrease in
 *   a "lower is better" metric is not reported the same way as a decrease in a
 *   "higher is better" one;
 * * **category and operator deltas** — where the time actually went and came from;
 * * **recommendation verification** — every advice item from the before-analysis
 *   names the metric it targets, so the comparison can say 达成 / 部分达成 / 未达成
 *   instead of leaving the reader to eyeball two tables;
 * * **comparability warnings** — two captures only compare if they measure the
 *   same thing: different step counts, a different prefill/decode mix, sampling,
 *   different artifact sets or a very different wall window are all reported.
 *
 * Nothing here re-parses anything: both sides are `{ dataset, analysis, datasetId,
 * label }` pairs already produced by the pipeline.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis/compare
 */

import { round } from '../model/stats.js';

/** Relative change below which a metric is reported as "unchanged" (percent). */
const NOISE_PCT = 1;

/**
 * Metrics compared, in report order.
 *
 * `direction` is what makes the comparison readable: `lower` / `higher` mark the
 * improvement direction, `neutral` marks context (a ratio whose denominator is the
 * wall time moves even when nothing improved).
 */
const METRICS = Object.freeze([
  { key: 'avgStepUs', label: '每步墙钟（平均步长）', unit: 'µs', direction: 'lower', group: '总览' },
  { key: 'wallUs', label: '采集窗口', unit: 'µs', direction: 'lower', group: '总览' },
  { key: 'deviceBusyPct', label: 'NPU 忙碌率', unit: '%', direction: 'higher', group: '总览' },
  { key: 'idlePct', label: 'NPU 空闲占比', unit: '%', direction: 'lower', group: '总览' },
  { key: 'hostExclusivePerStepUs', label: 'Host 独占（设备空等）/步', unit: 'µs', direction: 'lower', group: 'Host 侧' },
  { key: 'hostOnlyPct', label: 'Host 独占占比', unit: '%', direction: 'lower', group: 'Host 侧' },
  { key: 'hostBusyPct', label: 'Host 忙占比', unit: '%', direction: 'lower', group: 'Host 侧' },
  { key: 'hostScheduleUs', label: 'Host 调度算子耗时', unit: 'µs', direction: 'lower', group: 'Host 侧' },
  { key: 'dispatchPerStep', label: 'Host 派发算子数/步', unit: '个', direction: 'lower', group: 'Host 侧' },
  { key: 'hostSyncPerStep', label: '同步类算子/步', unit: '次', direction: 'lower', group: 'Host 侧' },
  { key: 'commExposedPct', label: '通信未掩盖', unit: '%', direction: 'lower', group: '通信/拷贝' },
  { key: 'commOverlapPct', label: '通信与计算重叠率', unit: '%', direction: 'higher', group: '通信/拷贝' },
  { key: 'commUsPerStep', label: '通信时长/步', unit: 'µs', direction: 'lower', group: '通信/拷贝' },
  { key: 'commPerStep', label: '通信次数/步', unit: '次', direction: 'neutral', group: '通信/拷贝' },
  { key: 'copyUs', label: '设备拷贝耗时', unit: 'µs', direction: 'lower', group: '通信/拷贝' },
  { key: 'd2hPerStepUs', label: 'D2H 拷贝/步', unit: 'µs', direction: 'lower', group: '通信/拷贝' },
  { key: 'h2dPerStepUs', label: 'H2D 拷贝/步', unit: 'µs', direction: 'lower', group: '通信/拷贝' },
  { key: 'computeUs', label: '设备计算耗时', unit: 'µs', direction: 'lower', group: '设备侧' },
  { key: 'computePct', label: '计算占墙钟', unit: '%', direction: 'neutral', group: '设备侧' },
  { key: 'meanGapUs', label: '平均空闲段长度', unit: 'µs', direction: 'lower', group: '设备侧' },
  { key: 'gapCount', label: '空闲段数量', unit: '个', direction: 'lower', group: '设备侧' },
  { key: 'eventCount', label: '事件总数', unit: '个', direction: 'neutral', group: '数据规模' },
  { key: 'deviceEventCount', label: '设备算子条数', unit: '个', direction: 'neutral', group: '数据规模' },
]);

/**
 * Which metric each recommendation is supposed to move.
 *
 * Keeping the mapping next to the comparison (instead of inside the advice text)
 * makes it testable: `compare.test` asserts that every recommendation the three
 * shipped scenarios can produce is either mapped here or explicitly marked as
 * "no comparable metric" (the confidence-only advice items).
 */
const ADVICE_TARGETS = Object.freeze({
  'host.enable-graph-mode': ['dispatchPerStep', 'hostExclusivePerStepUs', 'avgStepUs'],
  'host.async-sampling': ['hostSyncPerStep', 'avgStepUs'],
  'host.reduce-dispatch': ['dispatchPerStep', 'hostScheduleUs'],
  'host.smooth-sampling-path': ['hostSyncPerStep'],
  'compute.quantize': ['computeUs'],
  'compute.decode-increase-batch': ['avgStepUs', 'commPerStep'],
  'compute.fuse-small-ops': ['computeUs', 'dispatchPerStep'],
  'compute.tune-chunked-prefill': ['avgStepUs'],
  'comm.enable-overlap-fusion': ['commExposedPct', 'commOverlapPct'],
  'comm.batch-amortize': ['commUsPerStep', 'commPerStep'],
  'comm.parallel-strategy': ['commExposedPct', 'commUsPerStep'],
  'copy.async-d2h': ['d2hPerStepUs'],
  'copy.kv-locality': ['d2hPerStepUs', 'copyUs'],
  'copy.reduce-profiling-overhead': ['avgStepUs', 'hostBusyPct'],
  'common.phase-isolation': [],
  'common.collect-richer-profile': [],
});

/** Recommendation ids that only ask for a better measurement, not a faster run. */
const CONFIDENCE_ONLY_ADVICE = new Set(['common.phase-isolation', 'common.collect-richer-profile']);

/** Better/worse for one relative delta, honouring the metric's direction. */
function judge(deltaPct, direction) {
  // Context metrics carry no verdict at all — a ratio that moves because its
  // denominator moved is not an improvement.
  if (direction !== 'lower' && direction !== 'higher') return undefined;
  if (!Number.isFinite(deltaPct)) return undefined;
  if (Math.abs(deltaPct) < NOISE_PCT) return false;
  return direction === 'lower' ? deltaPct < 0 : deltaPct > 0;
}

/** Relative change in percent, tolerant of a zero (or absent) baseline. */
function relPct(before, after) {
  if (!Number.isFinite(before) || !Number.isFinite(after)) return undefined;
  if (before === 0) return after === 0 ? 0 : undefined;
  return ((after - before) / Math.abs(before)) * 100;
}

/** Improvement in percent, always positive when things got better. */
function improvementPct(deltaPct, direction) {
  if (!Number.isFinite(deltaPct)) return undefined;
  if (direction === 'higher') return deltaPct;
  return -deltaPct;
}

function sideOf(id, label, dataset, analysis) {
  return {
    datasetId: id,
    label: label ?? dataset?.meta?.label ?? id,
    eventCount: analysis.indicators.eventCount,
    deviceEventCount: analysis.indicators.deviceEventCount,
    wallUs: round(analysis.indicators.wallUs, 2),
    stepCount: analysis.indicators.stepCount,
    bottleneck: {
      id: analysis.bottleneck.id,
      label: analysis.bottleneck.label,
      short: analysis.bottleneck.short,
      score: round(analysis.bottleneck.score, 1),
      scope: analysis.bottleneck.scope,
    },
    phaseOverride: analysis.options?.phaseOverride ?? 'auto',
    sampling: dataset?.meta?.sampling?.applied === true,
    totalsSource: dataset?.meta?.totalsSource,
    files: (dataset?.meta?.files ?? []).map((file) => (typeof file === 'string' ? file : file.name)),
  };
}

/** Phase wall-time mix (share of the window), used for the comparability check. */
function phaseMix(analysis) {
  const phases = analysis.phaseIndicators ?? {};
  const total = Object.values(phases).reduce((sum, phase) => sum + (phase.wallUs ?? 0), 0);
  const out = {};
  for (const [id, phase] of Object.entries(phases)) {
    out[id] = total === 0 ? 0 : round(((phase.wallUs ?? 0) / total) * 100, 1);
  }
  return out;
}

/**
 * Compare two captures.
 *
 * @param {object} input - `{ before, after }`, each `{ dataset, analysis, datasetId, label }`.
 * @returns {object} comparison payload consumed by the page and the reports.
 */
export function compareCaptures({ before, after }) {
  const beforeIndicators = before.analysis.indicators;
  const afterIndicators = after.analysis.indicators;
  const metrics = METRICS.map((metric) => {
    const from = beforeIndicators[metric.key];
    const to = afterIndicators[metric.key];
    const deltaPct = relPct(from, to);
    return {
      key: metric.key,
      label: metric.label,
      unit: metric.unit,
      group: metric.group,
      direction: metric.direction,
      before: Number.isFinite(from) ? round(from, 2) : undefined,
      after: Number.isFinite(to) ? round(to, 2) : undefined,
      deltaAbs: Number.isFinite(from) && Number.isFinite(to) ? round(to - from, 2) : undefined,
      deltaPct: deltaPct === undefined ? undefined : round(deltaPct, 1),
      improvementPct: deltaPct === undefined ? undefined : round(improvementPct(deltaPct, metric.direction), 1),
      improved: judge(deltaPct, metric.direction),
    };
  });

  const categories = compareCategories(before.dataset, after.dataset);
  const operators = compareOperators(before.dataset, after.dataset);
  const recommendations = verifyRecommendations(before.analysis, after.analysis, metrics);
  const comparability = assessComparability(before, after);

  const headlineMetric = (key) => metrics.find((metric) => metric.key === key);
  const step = headlineMetric('avgStepUs');
  const busy = headlineMetric('deviceBusyPct');
  const hostExclusive = headlineMetric('hostExclusivePerStepUs');
  const dispatch = headlineMetric('dispatchPerStep');
  const changed = before.analysis.bottleneck.id !== after.analysis.bottleneck.id;

  const headline = {
    stepBeforeUs: step?.before,
    stepAfterUs: step?.after,
    stepPct: step?.deltaPct,
    stepImproved: step?.improved,
    busyBeforePct: busy?.before,
    busyAfterPct: busy?.after,
    busyDeltaPts: busy === undefined || busy.before === undefined || busy.after === undefined ? undefined : round(busy.after - busy.before, 2),
    hostExclusivePct: hostExclusive?.improvementPct,
    dispatchPct: dispatch?.improvementPct,
    bottleneckChanged: changed,
    achieved: recommendations.filter((item) => item.verdict === 'achieved').length,
    missed: recommendations.filter((item) => item.verdict === 'missed').length,
    unknown: recommendations.filter((item) => item.verdict === 'unknown').length,
    summary: headlineSummary({ step, busy, hostExclusive, dispatch, before, after, changed, recommendations }),
  };

  return {
    generatedAt: new Date().toISOString(),
    sides: {
      before: sideOf(before.datasetId, before.label, before.dataset, before.analysis),
      after: sideOf(after.datasetId, after.label, after.dataset, after.analysis),
    },
    comparability,
    metrics,
    categories,
    operators,
    bottleneck: {
      before: { id: before.analysis.bottleneck.id, label: before.analysis.bottleneck.label, score: round(before.analysis.bottleneck.score, 1) },
      after: { id: after.analysis.bottleneck.id, label: after.analysis.bottleneck.label, score: round(after.analysis.bottleneck.score, 1) },
      changed,
      scoreDelta: round(after.analysis.bottleneck.score - before.analysis.bottleneck.score, 1),
    },
    recommendations,
    headline,
    warnings: buildWarnings(comparability, metrics, recommendations),
  };
}

/** Where the device time moved, per operator category. */
function compareCategories(beforeDataset, afterDataset) {
  const byId = new Map();
  for (const item of beforeDataset.categories.items) byId.set(item.id, { item, beforeUs: item.totalUs, afterUs: 0, beforeCount: item.count, afterCount: 0 });
  for (const item of afterDataset.categories.items) {
    const entry = byId.get(item.id);
    if (entry === undefined) byId.set(item.id, { item, beforeUs: 0, afterUs: item.totalUs, beforeCount: 0, afterCount: item.count });
    else {
      entry.afterUs = item.totalUs;
      entry.afterCount = item.count;
    }
  }
  const beforeTotal = [...byId.values()].reduce((sum, entry) => sum + entry.beforeUs, 0);
  const afterTotal = [...byId.values()].reduce((sum, entry) => sum + entry.afterUs, 0);
  return [...byId.values()]
    .map((entry) => ({
      id: entry.item.id,
      label: entry.item.label,
      color: entry.item.color,
      beforeUs: round(entry.beforeUs, 1),
      afterUs: round(entry.afterUs, 1),
      deltaUs: round(entry.afterUs - entry.beforeUs, 1),
      deltaPct: round(relPct(entry.beforeUs, entry.afterUs) ?? 0, 1),
      beforeSharePct: beforeTotal === 0 ? 0 : round((entry.beforeUs / beforeTotal) * 100, 2),
      afterSharePct: afterTotal === 0 ? 0 : round((entry.afterUs / afterTotal) * 100, 2),
      beforeCount: entry.beforeCount,
      afterCount: entry.afterCount,
    }))
    .sort((left, right) => Math.abs(right.deltaUs) - Math.abs(left.deltaUs));
}

/** Which operators saved (or cost) the most device time. */
function compareOperators(beforeDataset, afterDataset) {
  const beforeOps = new Map(beforeDataset.operators.map((row) => [row.name, row]));
  const afterOps = new Map(afterDataset.operators.map((row) => [row.name, row]));
  const names = new Set([...beforeOps.keys(), ...afterOps.keys()]);
  const rows = [];
  for (const name of names) {
    const from = beforeOps.get(name);
    const to = afterOps.get(name);
    const beforeUs = from?.totalUs ?? 0;
    const afterUs = to?.totalUs ?? 0;
    if (beforeUs === 0 && afterUs === 0) continue;
    rows.push({
      name,
      group: (to ?? from).group,
      category: (to ?? from).category,
      beforeUs: round(beforeUs, 1),
      afterUs: round(afterUs, 1),
      beforeCount: from?.count ?? 0,
      afterCount: to?.count ?? 0,
      deltaUs: round(afterUs - beforeUs, 1),
      deltaPct: round(relPct(beforeUs, afterUs) ?? 0, 1),
      beforeAvgUs: from?.avgUs,
      afterAvgUs: to?.avgUs,
    });
  }
  const saved = rows.filter((row) => row.deltaUs < -1).sort((left, right) => left.deltaUs - right.deltaUs);
  const added = rows.filter((row) => row.deltaUs > 1).sort((left, right) => right.deltaUs - left.deltaUs);
  return {
    improved: saved.slice(0, 8).map((row) => ({ ...row, improvementUs: Math.abs(row.deltaUs) })),
    regressed: added.slice(0, 8),
    unchangedCount: rows.length - saved.length - added.length,
    comparedCount: rows.length,
  };
}

/** Did the before-capture's advice actually move its target metric? */
function verifyRecommendations(beforeAnalysis, afterAnalysis, metrics) {
  const stillRecommended = new Set(afterAnalysis.steps.actions.items.map((item) => item.id));
  const metricOf = (key) => metrics.find((metric) => metric.key === key);
  return beforeAnalysis.steps.actions.items.map((item) => {
    const targets = (ADVICE_TARGETS[item.id] ?? []).map(metricOf).filter((metric) => metric !== undefined);
    const primary = targets[0];
    const observedPct = primary?.improvementPct;
    const expectedPct = item.expectedGain?.estimatePct;
    // A recommendation counts as achieved when at least half of its projected
    // gain (and at least 1%) shows up in the target metric; the threshold is part
    // of the payload so the report never has to guess what "achieved" meant.
    const thresholdPct = Number.isFinite(expectedPct) ? round(Math.max(1, expectedPct * 0.5), 1) : undefined;
    let verdict;
    if (CONFIDENCE_ONLY_ADVICE.has(item.id) || targets.length === 0) verdict = 'unknown';
    else if (!Number.isFinite(observedPct)) verdict = 'unknown';
    else if (observedPct >= (thresholdPct ?? 1)) verdict = 'achieved';
    else if (observedPct > 0) verdict = 'partial';
    else verdict = 'missed';
    return {
      id: item.id,
      title: item.title,
      priority: item.priority,
      phase: item.phases ?? [item.phase],
      expectedPct,
      expectedMetric: item.expectedGain?.metric,
      confidence: item.expectedGain?.confidence,
      observedPct: Number.isFinite(observedPct) ? round(observedPct, 1) : undefined,
      thresholdPct,
      verdict,
      stillRecommended: stillRecommended.has(item.id),
      targets: targets.map((metric) => ({
        key: metric.key,
        label: metric.label,
        unit: metric.unit,
        before: metric.before,
        after: metric.after,
        deltaPct: metric.deltaPct,
        improvementPct: metric.improvementPct,
      })),
      note: verdictNote({ verdict, observedPct, thresholdPct, primary, expectedPct, stillRecommended: stillRecommended.has(item.id) }),
    };
  });
}

function verdictNote({ verdict, observedPct, thresholdPct, primary, expectedPct, stillRecommended }) {
  if (verdict === 'unknown' && primary === undefined) {
    return '该建议针对采集质量而非运行时性能，没有可直接对比的指标；请对比两次采集的告警与置信度。';
  }
  if (verdict === 'unknown') return `目标指标「${String(primary?.label)}」在本次对比中缺少可比数值。`;
  const observed = Number.isFinite(observedPct) ? `${observedPct >= 0 ? '' : ''}${observedPct.toFixed(1)}%` : '—';
  const expected = Number.isFinite(expectedPct) ? `${expectedPct.toFixed(1)}%` : '—';
  const target = String(primary?.label);
  const tail = stillRecommended ? '该建议在优化后仍然被列出，说明目标指标仍未达到门限。' : '该建议在优化后不再被列出。';
  if (verdict === 'achieved') return `目标指标「${target}」改善 ${observed}，达到判定阈值 ${String(thresholdPct)}%（预期 ${expected}）。${tail}`;
  if (verdict === 'partial') return `目标指标「${target}」改善 ${observed}，未达到判定阈值 ${String(thresholdPct)}%（预期 ${expected}），方向正确但幅度有限。${tail}`;
  return `目标指标「${target}」未改善（${observed}），判定阈值 ${String(thresholdPct)}%（预期 ${expected}）。${tail}`;
}

/** Do the two captures measure the same thing? */
function assessComparability(before, after) {
  const notes = [];
  const beforeSide = sideOf(before.datasetId, before.label, before.dataset, before.analysis);
  const afterSide = sideOf(after.datasetId, after.label, after.dataset, after.analysis);

  const stepDelta = relPct(beforeSide.stepCount, afterSide.stepCount);
  if (Number.isFinite(stepDelta) && Math.abs(stepDelta) > 15) {
    notes.push({
      level: 'warn',
      text: `两次采集的推理步数相差 ${Math.abs(stepDelta).toFixed(0)}%（${String(beforeSide.stepCount)} → ${String(afterSide.stepCount)} 步）：逐步指标已按步归一化，但总窗口与占比类指标的可比性下降。`,
    });
  }
  const mixBefore = phaseMix(before.analysis);
  const mixAfter = phaseMix(after.analysis);
  const mixIds = new Set([...Object.keys(mixBefore), ...Object.keys(mixAfter)]);
  const mixShift = [...mixIds].filter((id) => Math.abs((mixBefore[id] ?? 0) - (mixAfter[id] ?? 0)) > 10);
  if (mixShift.length > 0) {
    notes.push({
      level: 'warn',
      text: `两次采集的 Prefill/Decode 时间构成不同（${mixShift.map((id) => `${id} ${String(mixBefore[id] ?? 0)}% → ${String(mixAfter[id] ?? 0)}%`).join('、')}）：混合负载下结论可能来自负载变化而非优化。建议分开采集同一阶段对比。`,
    });
  }
  if (beforeSide.phaseOverride !== afterSide.phaseOverride) {
    notes.push({ level: 'warn', text: `两侧的阶段口径不同（${beforeSide.phaseOverride} vs ${afterSide.phaseOverride}）：请用同一口径重新分析后再比较。` });
  }
  if (beforeSide.sampling || afterSide.sampling) {
    notes.push({ level: 'warn', text: '至少一侧的 trace 已采样：累计耗时取自 CANN 统计表，泳道图为抽样结果。' });
  }
  if (beforeSide.totalsSource !== afterSide.totalsSource) {
    notes.push({ level: 'info', text: `两侧的耗时口径来源不同（${String(beforeSide.totalsSource)} vs ${String(afterSide.totalsSource)}）。` });
  }
  const beforeFiles = new Set(beforeSide.files);
  const afterFiles = new Set(afterSide.files);
  const fileDiff = [...new Set([...beforeSide.files, ...afterSide.files])].filter((name) => !beforeFiles.has(name) || !afterFiles.has(name));
  if (fileDiff.length > 0) {
    notes.push({ level: 'warn', text: `两次采集的产物文件不同（${fileDiff.join('、')}）：缺失的表会让对应指标不可比。` });
  }
  if (beforeSide.bottleneck.id !== afterSide.bottleneck.id) {
    notes.push({
      level: 'info',
      text: `两侧的主导瓶颈类型不同（${beforeSide.bottleneck.label} → ${afterSide.bottleneck.label}）：这既可能是优化把瓶颈推到了下一步，也可能是两次采集的负载不同——请先确认采集命令、并发与输入长度一致。`,
    });
  }
  const wallDelta = relPct(beforeSide.wallUs, afterSide.wallUs);
  if (Number.isFinite(wallDelta) && Math.abs(wallDelta) > 20) {
    notes.push({
      level: 'info',
      text: `采集窗口变化 ${wallDelta.toFixed(0)}%（${(beforeSide.wallUs / 1000).toFixed(1)}ms → ${(afterSide.wallUs / 1000).toFixed(1)}ms）：占比类指标的分母随之变化，判断"变好/变差"请以 µs 绝对值为准。`,
    });
  }
  const blocking = notes.filter((note) => note.level === 'warn').length;
  return {
    level: blocking === 0 ? 'high' : blocking === 1 ? 'medium' : 'low',
    comparable: blocking === 0,
    notes,
  };
}

function buildWarnings(comparability, metrics, recommendations) {
  const warnings = comparability.notes.filter((note) => note.level !== 'info').map((note) => note.text);
  for (const note of comparability.notes.filter((entry) => entry.level === 'info' && /瓶颈类型不同/.test(entry.text))) {
    warnings.push(note.text);
  }
  const ratioMoved = metrics.filter((metric) => metric.unit === '%'
    && metric.direction !== 'neutral'
    && Number.isFinite(metric.improvementPct)
    && metric.improvementPct < 0);
  if (ratioMoved.length > 0) {
    warnings.push(`存在占比指标反向变化（${ratioMoved.map((metric) => `${metric.label} ${metric.before}% → ${metric.after}%`).join('、')}）：优化缩短了墙钟，占比的分母同时变小，请结合绝对值判断。`);
  }
  const missed = recommendations.filter((item) => item.verdict === 'missed');
  if (missed.length > 0) {
    warnings.push(`${String(missed.length)} 条建议的目标指标在本次对比中未改善：${missed.map((item) => item.title).join('；')}。若两次采集负载不同，请先排除负载变化。`);
  }
  return warnings;
}

function headlineSummary({ step, busy, hostExclusive, dispatch, before, after, changed, recommendations }) {
  const parts = [];
  if (Number.isFinite(step?.deltaPct)) {
    const verb = step.deltaPct < 0 ? '下降' : '上升';
    parts.push(`单步墙钟${verb} ${Math.abs(step.deltaPct).toFixed(1)}%（${(step.before / 1000).toFixed(2)}ms → ${(step.after / 1000).toFixed(2)}ms）`);
  }
  if (Number.isFinite(busy?.deltaAbs) || (Number.isFinite(busy?.before) && Number.isFinite(busy?.after))) {
    parts.push(`NPU 忙碌率 ${busy.before}% → ${busy.after}%`);
  }
  if (Number.isFinite(hostExclusive?.improvementPct) && Math.abs(hostExclusive.improvementPct) > NOISE_PCT) {
    parts.push(`Host 独占/步${hostExclusive.improvementPct > 0 ? '下降' : '上升'} ${Math.abs(hostExclusive.improvementPct).toFixed(1)}%`);
  }
  if (Number.isFinite(dispatch?.improvementPct) && Math.abs(dispatch.improvementPct) > NOISE_PCT) {
    parts.push(`派发算子数/步${dispatch.improvementPct > 0 ? '下降' : '上升'} ${Math.abs(dispatch.improvementPct).toFixed(1)}%`);
  }
  const achieved = recommendations.filter((item) => item.verdict === 'achieved').length;
  if (achieved > 0) parts.push(`${String(achieved)} 条建议的目标指标达成`);
  parts.push(changed
    ? `瓶颈类型由 ${before.analysis.bottleneck.label} 转为 ${after.analysis.bottleneck.label}`
    : `瓶颈类型未变（${after.analysis.bottleneck.label} ${before.analysis.bottleneck.score.toFixed(0)} → ${after.analysis.bottleneck.score.toFixed(0)} 分）`);
  return `${parts.join('；')}。`;
}

/** Every advice id the mapper knows about, for the coverage test. */
export const COMPARE_ADVICE_IDS = Object.freeze(Object.keys(ADVICE_TARGETS));
