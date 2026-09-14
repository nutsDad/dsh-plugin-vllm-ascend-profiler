/**
 * The analysis entry point: runs the fixed five-step reasoning chain over a
 * dataset and returns a result whose structure mirrors the chain, so the UI and
 * the report cannot present a conclusion without its evidence.
 *
 * ```
 * ① locate   — score the four bottleneck candidates (overall + per phase)
 * ② evidence — quantified indicators with thresholds and sources
 * ③ cause    — vLLM-Ascend root-cause hypotheses per bottleneck
 * ④ actions  — prioritised, actionable optimisations
 * ⑤ benefit  — expected gain per optimisation, computed or explicitly ranged
 * ```
 *
 * The chain is deterministic: the same dataset and options always produce the
 * same report.
 *
 * @module dsh-plugin-vllm-ascend-profiler/analysis
 */

import { percentile, round } from '../model/stats.js';
import { BOTTLENECKS, computeIndicators, scoreBottlenecks } from './bottleneck.js';
import { buildRecommendations, groupByPriority, mergeRecommendations } from './recommend.js';
import { inferRootCauses } from './rootcause.js';

export { COMPARE_ADVICE_IDS, compareCaptures } from './compare.js';

/** Analysis defaults. */
export const ANALYSIS_DEFAULTS = Object.freeze({
  /** Threshold overrides keyed by name from `thresholds.js`. */
  thresholds: {},
  /**
   * Phase override. `auto` uses the derived phase model; `prefill` / `decode`
   * assert that the whole capture belongs to one phase, which is what an
   * operator should select when the capture was taken with a phase-specific
   * workload (the reliable practice, since Ascend traces carry no phase label).
   */
  phaseOverride: 'auto',
  /** Minimum candidate score that counts as a located bottleneck. */
  triggerScore: 40,
});

/**
 * Run the analysis.
 *
 * @param {object} dataset - dataset from `buildDataset`.
 * @param {object} [options] - {@link ANALYSIS_DEFAULTS} overrides.
 * @returns {object} analysis result with the five chain steps.
 */
export function analyzeDataset(dataset, options = {}) {
  const config = { ...ANALYSIS_DEFAULTS, ...options };
  const indicators = computeIndicators(dataset, { thresholds: config.thresholds });
  const context = datasetContext(dataset);

  const scopeIds = resolveScopes(dataset, config);
  const perScope = scopeIds.map((scope) => analyzeScope(scope, indicators, dataset, context, config));
  const primary = pickPrimary(perScope);

  const recommendations = mergeRecommendations(perScope.map((scope) => scope.recommendations));
  const grouped = groupByPriority(recommendations);

  return {
    generatedAt: new Date().toISOString(),
    options: {
      phaseOverride: config.phaseOverride,
      triggerScore: config.triggerScore,
      thresholds: config.thresholds,
    },
    context,
    indicators: indicators.overall,
    phaseIndicators: indicators.phases,
    steps: {
      locate: {
        id: 'locate',
        title: '① 瓶颈类型定位',
        summary: primary.summary,
        perScope,
        primary: primary.headline,
      },
      evidence: {
        id: 'evidence',
        title: '② 量化证据',
        overall: evidenceRows(indicators.overall),
        phases: Object.fromEntries(Object.entries(indicators.phases).map(([id, phase]) => [id, evidenceRows(phase)])),
      },
      cause: {
        id: 'cause',
        title: '③ 根因推断',
        items: dedupeCauses(perScope),
      },
      actions: {
        id: 'actions',
        title: '④ 可落地优化方案（按优先级）',
        items: recommendations,
        grouped,
      },
      benefit: {
        id: 'benefit',
        title: '⑤ 预期收益',
        items: recommendations
          .filter((item) => item.expectedGain !== undefined && item.expectedGain.estimatePct > 0)
          .map((item) => ({
            id: item.id,
            title: item.title,
            phase: item.phases ?? [item.phase],
            priority: item.priority,
            ...item.expectedGain,
          })),
        combined: combineGains(recommendations, perScope),
        note: '收益为本数据集推导或标注的经验区间；区间下限代表保守估计。经验区间（confidence=low）不得当作承诺值。',
      },
    },
    bottleneck: primary.headline,
    recommendations,
    warnings: buildAnalysisWarnings(dataset, perScope, indicators),
  };
}

/** Scopes analyzed: overall plus every phase that has enough data. */
function resolveScopes(dataset, config) {
  if (config.phaseOverride !== 'auto') {
    return [{ id: config.phaseOverride, label: `全量（人工指定为 ${config.phaseOverride}）`, forced: true }];
  }
  const scopes = [{ id: 'overall', label: '全量采集窗口' }];
  for (const phase of dataset.phases.phases ?? []) {
    if (phase.id === 'unknown') continue;
    scopes.push({ id: phase.id, label: phase.label });
  }
  return scopes;
}

/** Analyze one scope (overall or a phase). */
function analyzeScope(scope, indicators, dataset, context, config) {
  const block = scope.id === 'overall' ? indicators.overall : (indicators.phases[scope.id] ?? indicators.overall);
  // The overall scope has no phase of its own, so it is evaluated against the
  // dominant phase by wall time. Without this, phase-specific advice (graph
  // mode for decode, quantisation for prefill) would silently vanish from the
  // overall verdict — and reporting that assumption is mandatory.
  const assumedPhase = scope.id === 'overall' ? dominantPhase(dataset) : undefined;
  const evaluationPhase = assumedPhase ?? scope.id;
  const phaseContext = {
    ...contextForPhase(context, evaluationPhase),
    phase: evaluationPhase,
    phaseMix: (dataset.phases.phases ?? []).filter((phase) => phase.id !== 'unknown').length > 1,
    phaseConfidence: dataset.phases.confidence,
  };
  const scored = scoreBottlenecks(block, { thresholds: config.thresholds }, scope.id);
  const located = scored.candidates.filter((candidate) => candidate.score >= config.triggerScore);
  const hypotheses = located.flatMap((candidate) => inferRootCauses(candidate, block, phaseContext));
  const recommendations = buildRecommendations({
    candidate: scored.primary.id === 'balanced' ? scored.primary : (located[0] ?? scored.primary),
    hypotheses,
    indicators: block,
    context: phaseContext,
  });
  return {
    scope: scope.id,
    scopeLabel: scope.label,
    forced: scope.forced === true,
    assumedPhase,
    block,
    candidates: scored.candidates,
    verdict: scored.verdict,
    primaryCandidate: scored.primary,
    hypotheses,
    recommendations,
    summary: summarizeScope(scope, scored, hypotheses, assumedPhase),
  };
}

/**
 * Merge hypotheses by id across scopes.
 *
 * The same root cause legitimately matches several scopes (whole window, the
 * prefill scope, the decode scope); the report must list it once, naming every
 * scope it applies to, rather than repeating the same mechanism three times.
 */
function dedupeCauses(perScope) {
  const byId = new Map();
  for (const scope of perScope) {
    for (const hypothesis of scope.hypotheses) {
      const existing = byId.get(hypothesis.id);
      if (existing === undefined) {
        byId.set(hypothesis.id, { ...hypothesis, scope: scope.scope, scopes: [scope.scope] });
        continue;
      }
      if (!existing.scopes.includes(scope.scope)) existing.scopes.push(scope.scope);
      // Keep the strongest evidence set (most trigger rows) for the merged row.
      if (hypothesis.triggers.length > existing.triggers.length) {
        existing.triggers = hypothesis.triggers;
        existing.mechanism = hypothesis.mechanism;
      }
    }
  }
  return [...byId.values()];
}

/** Phase with the largest wall time, used as the overall scope's phase context. */
function dominantPhase(dataset) {
  const phases = (dataset.phases.phases ?? []).filter((phase) => phase.id !== 'unknown');
  if (phases.length === 0) return undefined;
  if (phases.length === 1) return phases[0].id;
  return [...phases].sort((left, right) => right.wallUs - left.wallUs)[0].id;
}

function contextForPhase(context, phase) {
  return { ...context, phase };
}

/** Dataset-level facts the hypotheses and templates consult. */
function datasetContext(dataset) {
  const computeEvents = dataset.events.filter((event) => event.device === 'device' && event.category === 'compute');
  const subtypeTotals = new Map();
  for (const event of computeEvents) {
    const entry = subtypeTotals.get(event.subtype) ?? { subtype: event.subtype, totalUs: 0, count: 0 };
    entry.totalUs += event.durUs;
    entry.count += 1;
    subtypeTotals.set(event.subtype, entry);
  }
  const utilMetrics = dataset.utilization?.metrics ?? [];
  const metric = (name) => utilMetrics.find((entry) => entry.metric === name)?.value;
  return {
    hasUtilization: dataset.utilization?.available === true,
    missingTables: (dataset.meta.tableSummaries ?? []).length === 0,
    macRatio: metric('mac'),
    mte2Ratio: metric('mte2'),
    vecRatio: metric('vec'),
    medianComputeDurUs: computeEvents.length === 0 ? 0 : percentile(computeEvents.map((event) => event.durUs), 0.5),
    topComputeSubtypes: [...subtypeTotals.values()].sort((left, right) => right.totalUs - left.totalUs).slice(0, 5)
      .map((entry) => ({ ...entry, totalUs: round(entry.totalUs, 2) })),
    sampling: dataset.meta.sampling,
    totalsSource: dataset.meta.totalsSource,
    ranks: dataset.meta.counts?.ranks ?? 0,
    stepSource: dataset.phases.source,
    stepSourceLabel: dataset.phases.sourceLabel,
    phaseCount: (dataset.phases.phases ?? []).length,
  };
}

/** Pick the most severe located bottleneck across scopes as the headline. */
function pickPrimary(perScope) {
  const ranked = perScope
    .flatMap((scope) => scope.candidates.map((candidate) => ({ ...candidate, scope: scope.scope, scopeLabel: scope.scopeLabel })))
    .sort((left, right) => right.score - left.score);
  const top = ranked[0];
  if (top === undefined || top.score < 40) {
    return {
      headline: {
        ...BOTTLENECKS.balanced,
        score: top?.score ?? 0,
        scope: top?.scope,
        summary: '四个候选瓶颈均未达到触发门限：负载在这四个维度上相对均衡，或当前采集粒度不足以定位单一主导瓶颈。',
        candidates: ranked.slice(0, 4),
      },
      summary: '未定位到单一主导瓶颈',
    };
  }
  const scopeLabel = perScope.find((scope) => scope.scope === top.scope)?.scopeLabel ?? top.scope;
  return {
    headline: {
      id: top.id,
      label: top.label,
      short: top.short,
      color: top.color,
      score: top.score,
      subtype: top.subtype,
      scope: top.scope,
      scopeLabel,
      summary: top.summary,
      candidates: ranked.slice(0, 4),
    },
    summary: `${scopeLabel}的主导瓶颈为${top.label}（得分 ${top.score.toFixed(0)}/100）：${top.summary}`,
  };
}

function summarizeScope(scope, scored, hypotheses, assumedPhase) {
  const assumed = assumedPhase === undefined ? '' : `（按主导阶段 ${assumedPhase === 'prefill' ? 'Prefill' : 'Decode'} 评估）`;
  if (scored.verdict === 'balanced') {
    return `${scope.label}${assumed}：未发现达到门限的主导瓶颈（最高候选 ${scored.primary.label} 得分 ${scored.primary.score.toFixed(0)}/100）。`;
  }
  return `${scope.label}${assumed}：主导瓶颈为 ${scored.primary.label}（得分 ${scored.primary.score.toFixed(0)}/100），匹配 ${String(hypotheses.length)} 条根因假设。`;
}

/** Flatten an indicator block into evidence rows for the report tables. */
function evidenceRows(block) {
  const rows = [
    { group: '时间分布', metric: '采集窗口', value: round(block.wallUs / 1000, 2), unit: 'ms', note: `包含 ${String(block.stepCount)} 个推理步` },
    { group: '时间分布', metric: 'NPU 忙碌率', value: block.deviceBusyPct, unit: '%', note: '设备事件区间并集 / 墙钟' },
    { group: '时间分布', metric: 'Host 忙占比', value: block.hostBusyPct, unit: '%', note: 'Host 事件区间并集 / 墙钟' },
    { group: '时间分布', metric: 'NPU 空闲', value: block.idlePct, unit: '%', note: `空闲段 ${String(block.gapCount)} 个，最大 ${block.maxGapUs.toFixed(0)}µs` },
    { group: '时间分布', metric: 'Host 独占（设备空等）', value: block.hostOnlyPct, unit: '%', note: `每步 ${block.hostExclusivePerStepUs.toFixed(0)}µs` },
    { group: '算子构成', metric: '设备计算算子', value: block.computePct, unit: '%', note: '占墙钟比例' },
    { group: '算子构成', metric: '设备通信算子', value: block.commPct, unit: '%', note: `占设备忙碌 ${block.commPctOfDevice.toFixed(1)}%` },
    { group: '算子构成', metric: '通信未掩盖', value: block.commExposedPct, unit: '%', note: `已重叠 ${block.commOverlapPct.toFixed(1)}%` },
    { group: '算子构成', metric: '设备拷贝算子', value: block.copyPct, unit: '%', note: `H2D ${block.h2dPerStepUs.toFixed(0)}µs/步、D2H ${block.d2hPerStepUs.toFixed(0)}µs/步` },
    { group: '算子构成', metric: 'Host 调度算子', value: block.hostScheduleUs, unit: 'µs', note: `占 Host 忙时间 ${block.hostSchedulePctOfHost.toFixed(1)}%` },
    { group: '步粒度', metric: '平均步长', value: round(block.avgStepUs / 1000, 3), unit: 'ms', note: '由阶段模型的步骤边界推导' },
    { group: '步粒度', metric: 'Host 派发算子数/步', value: block.dispatchPerStep, unit: '个', note: `同步类 ${block.hostSyncPerStep.toFixed(2)} 次/步` },
  ];
  if (block.utilization?.available) {
    for (const metric of block.utilization.metrics.slice(0, 6)) {
      rows.push({
        group: '算力/流水利用率',
        metric: metric.label,
        value: metric.valuePct,
        unit: '%',
        note: `按耗时加权（权重 ${(metric.weightUs / 1000).toFixed(2)}ms）`,
      });
    }
  } else {
    rows.push({ group: '算力/流水利用率', metric: '不可用', value: undefined, unit: '', note: '当前产物缺少算子级流水/算力指标列' });
  }
  return rows;
}

/** Conservative combination of the projected gains. */
function combineGains(recommendations, perScope) {
  const items = recommendations.filter((item) => item.expectedGain !== undefined && item.expectedGain.estimatePct > 0);
  const optimistic = items.reduce((total, item) => total + item.expectedGain.estimatePct, 0);
  const conservative = items.reduce((total, item) => total + (item.expectedGain.rangePct?.[0] ?? item.expectedGain.estimatePct * 0.5), 0);
  const byPhase = {};
  for (const scope of perScope) {
    const phaseItems = scope.recommendations.filter((item) => item.expectedGain !== undefined);
    byPhase[scope.scope] = round(phaseItems.reduce((total, item) => total + (item.expectedGain?.estimatePct ?? 0), 0), 1);
  }
  return {
    itemCount: items.length,
    optimisticPct: round(Math.min(70, optimistic), 1),
    conservativePct: round(Math.min(50, conservative), 1),
    byPhase,
    note: items.length === 0
      ? '当前没有可量化的收益项；先按“补齐采集”建议获取更完整的 profiling 数据。'
      : '多项优化并行时收益不可简单相加（存在重叠与相互制约）：上限按 70% 截断，下限为各项保守估计的累加。',
  };
}

function buildAnalysisWarnings(dataset, perScope, indicators) {
  const warnings = [];
  if (dataset.phases.confidence === 'low') {
    warnings.push(`Prefill/Decode 阶段划分为低置信度：${dataset.phases.confidenceReason}`);
  }
  if (dataset.meta.sampling?.applied === true) {
    warnings.push(`trace 已采样（${dataset.meta.sampling.strategy}，步长 ${String(dataset.meta.sampling.stride)}）：泳道图为抽样结果，累计耗时优先取自 CANN 统计表。`);
  }
  if (indicators.overall.stepCount <= 1) {
    warnings.push('仅识别到 1 个推理步：逐步指标与阶段对比的可信度有限，建议采集覆盖多个请求的窗口。');
  }
  if (!dataset.utilization?.available) {
    warnings.push('缺少算子级流水/算力利用率数据，"算力受限 vs 访存受限"的判断只能给出条件性结论。');
  }
  const noEvidence = perScope.filter((scope) => scope.candidates.every((candidate) => candidate.score < 40));
  if (noEvidence.length > 0) {
    warnings.push(`${noEvidence.map((scope) => scope.scopeLabel).join('、')}：未定位到主导瓶颈，报告仅给出各候选的接近程度与缺失证据。`);
  }
  return [...new Set(warnings)];
}
