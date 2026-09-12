/**
 * Step extraction and prefill/decode phase attribution.
 *
 * vLLM-Ascend serves two very different workloads: prefill steps are
 * compute-bound, long, and dominated by large GEMMs and attention over the
 * prompt; decode steps are short, latency-bound, dominated by per-token
 * bookkeeping and communication that cannot be hidden. Advice that does not
 * distinguish them is worthless, so phase attribution is a first-class step
 * here with an explicit source and confidence rather than a guess.
 *
 * Sources, in priority order:
 *
 * 1. `step_trace_time.csv` (torch_npu / msprof): per-step computing,
 *    communication, non-overlapped communication, and free time. The most
 *    authoritative source, in its own coordinate system.
 * 2. mstx / profiler markers naming prefill or decode in the trace.
 * 3. Repeating engine-level host events (`execute_model`, `model.forward`, …).
 * 4. Idle-gap segmentation of the device timeline.
 * 5. A single step covering the whole window.
 *
 * The phase split then uses explicit marker labels when present, and otherwise
 * a two-means split of step durations in log space — long steps are prefill,
 * short steps are decode — which is only applied when the two clusters are
 * statistically separated. Otherwise the phase stays `unknown`, and the report
 * says why.
 *
 * @module dsh-plugin-vllm-ascend-profiler/model/phase
 */

import { bimodalSplit, overlapLength, percentOf, round, unionIntervals } from './stats.js';

/** Event-name patterns that look like one inference step on the host side. */
const STEP_EVENT_PATTERNS = [
  /^(?:execute_model|model_runner\.execute_model)$/i,
  /^model\.forward|^forward$/i,
  /^execute_model_?v1$/i,
  /^step$/i,
  /^iteration/i,
];

/** Marker names that name a phase explicitly. */
const PREFILL_PATTERN = /prefill|prompt|chunked_?prefill|context_?phase/i;
/** Marker names that name decode explicitly. */
const DECODE_PATTERN = /decode|generation|generating|token_?phase/i;

/**
 * Extract steps and phases.
 *
 * @param {object} input - extraction input.
 * @param {object[]} input.events - normalized timeline events.
 * @param {object[]} [input.tables] - parsed CSV tables.
 * @param {object[]} [input.aux] - auxiliary JSON artifacts.
 * @param {{start: number, end: number}} input.window - timeline window.
 * @param {object} [input.options] - options (`idleGapUs`).
 * @returns {object} phase model.
 */
export function buildPhases({ events, tables = [], aux = [], window, options = {} }) {
  const stepTable = findStepTable(tables);
  const markers = findPhaseMarkers(events, aux);
  const engineSteps = findEngineSteps(events);
  const gapSegments = segmentByGaps(events, window, options);

  const candidates = [
    stepTable === undefined ? undefined : fromStepTable(stepTable, options),
    markers.steps.length === 0 ? undefined : fromMarkers(markers),
    engineSteps.length === 0 ? undefined : fromEngineEvents(engineSteps),
    gapSegments.length === 0 ? undefined : fromGapSegments(gapSegments),
    fromWholeWindow(window),
  ].filter((candidate) => candidate !== undefined);

  const chosen = candidates[0];
  const steps = chosen.steps.map((step, index) => ({ ...step, index }));
  const classification = classifySteps(steps, markers, options);
  // Events are sorted once and sliced by binary search: a profile can hold
  // hundreds of thousands of events and thousands of steps, and a per-step
  // linear scan would be quadratic.
  const sortedEvents = [...events].sort((left, right) => left.tsUs - right.tsUs);
  const startTimes = sortedEvents.map((event) => event.tsUs);
  const enrich = (step) => {
    const from = lowerBound(startTimes, step.startUs);
    const to = lowerBound(startTimes, step.endUs);
    return eventsInRange(sortedEvents, from, to);
  };
  const enriched = steps.map((step) => ({
    ...step,
    phase: classification.labels[step.index] ?? 'unknown',
    metrics: metricsForStep(enrich(step), step),
  }));
  enrichTokens(enriched, sortedEvents, enrich);
  const phases = summarizePhases(enriched);

  return {
    source: chosen.source,
    sourceLabel: chosen.sourceLabel,
    confidence: classification.confidence,
    confidenceReason: classification.reason,
    classificationMethod: classification.method,
    markers: markers.markers,
    steps: enriched,
    phases,
    alternatives: candidates.slice(1).map((candidate) => ({
      source: candidate.source,
      sourceLabel: candidate.sourceLabel,
      steps: candidate.steps.length,
    })),
  };
}

/** Find the step-level table among parsed CSVs. */
function findStepTable(tables) {
  return tables.find((table) => table.kind === 'step_trace_time' && table.rows.some((row) => row.isStepRow === true))
    ?? tables.find((table) => table.kind === 'step_trace_time');
}

/**
 * Steps coming from `step_trace_time.csv`.
 *
 * The table's own columns are authoritative for computing / communication /
 * non-overlapped communication / free time; absolute boundaries are rebuilt by
 * accumulating the step totals when no start timestamp is present.
 */
function fromStepTable(table, options) {
  const rows = table.rows.filter((row) => row.isStepRow === true || row.stepTotalUs !== undefined || row.durUs !== undefined);
  if (rows.length === 0) return undefined;
  const hasAbsolute = rows.every((row) => Number.isFinite(row.startRawUs));
  const hasStage = rows.some((row) => typeof row.stage === 'string' && row.stage !== '');
  let cursor = hasAbsolute ? Math.min(...rows.map((row) => row.startRawUs)) : 0;
  const steps = rows.map((row, index) => {
    const durUs = stepDurationUs(row);
    const startUs = hasAbsolute ? row.startRawUs : cursor;
    const endUs = startUs + durUs;
    if (!hasAbsolute) cursor = endUs;
    return {
      label: row.step === undefined ? `Step ${String(index + 1)}` : `Step ${String(row.step)}`,
      stepNumber: row.step ?? index + 1,
      startUs,
      endUs,
      durUs,
      tokens: row.extra?.Batch ?? undefined,
      stage: row.stage,
      declaredPhase: stageToPhase(row.stage),
      csvMetrics: {
        computingUs: row.computingUs,
        commUs: row.commUs,
        commNotOverlappedUs: row.commNotOverlappedUs,
        commOverlappedUs: row.commOverlappedUs,
        freeUs: row.freeUs,
      },
    };
  });
  void options;
  return {
    source: 'step_trace_time',
    sourceLabel: hasAbsolute
      ? `step_trace_time.csv（含绝对起始时间${hasStage ? '与 Stage 列' : ''}）`
      : `step_trace_time.csv（按步长累加的相对时间轴，与 trace 时间轴不做对齐${hasStage ? '；含 Stage 列' : ''}）`,
    steps,
  };
}

/**
 * Step duration from a `step_trace_time` row.
 *
 * The table usually has no total column: its step time is the decomposition
 * `Computing + Communication(Not Overlapped) + Free`, so the total is rebuilt
 * from the parts when no explicit total exists. Returning 0 here would collapse
 * every step onto the origin and silently disable step-based analysis.
 */
function stepDurationUs(row) {
  const explicit = row.stepTotalUs ?? row.durUs ?? row.totalUs;
  if (Number.isFinite(explicit) && explicit > 0) return explicit;
  const parts = [row.computingUs, row.commNotOverlappedUs ?? row.commUs, row.freeUs]
    .filter((value) => Number.isFinite(value));
  if (parts.length === 0) return 0;
  return parts.reduce((sum, value) => sum + value, 0);
}

/** Map the `Stage` column's spellings onto a phase id. */
function stageToPhase(stage) {
  if (typeof stage !== 'string') return undefined;
  if (PREFILL_PATTERN.test(stage)) return 'prefill';
  if (DECODE_PATTERN.test(stage)) return 'decode';
  return undefined;
}

/** Phase markers from mstx ranges or explicitly named trace events. */
function findPhaseMarkers(events, aux) {
  const markers = [];
  for (const event of events) {
    const text = `${event.name} ${event.message ?? ''}`;
    const isMarker = /mstx|marker|profiler_?step/i.test(event.name) || /mstx/i.test(String(event.cat ?? ''));
    if (!isMarker) continue;
    const prefill = PREFILL_PATTERN.test(text);
    const decode = DECODE_PATTERN.test(text);
    if (!prefill && !decode) continue;
    markers.push({
      tsUs: event.tsUs,
      endUs: event.tsUs + event.durUs,
      phase: prefill ? 'prefill' : 'decode',
      label: event.message ?? event.name,
    });
  }
  for (const artifact of aux) {
    if (artifact.kind !== 'mstx' || !Array.isArray(artifact.data)) continue;
    for (const item of artifact.data) {
      if (typeof item !== 'object' || item === null) continue;
      const record = /** @type {Record<string, unknown>} */ (item);
      const text = `${String(record.name ?? record.message ?? '')} ${String(record.domain ?? '')}`;
      const ts = Number(record.ts ?? record.start ?? record.timestamp);
      if (!Number.isFinite(ts)) continue;
      const prefill = PREFILL_PATTERN.test(text);
      const decode = DECODE_PATTERN.test(text);
      if (!prefill && !decode) continue;
      const dur = Number(record.dur ?? 0);
      markers.push({
        tsUs: ts,
        endUs: ts + (Number.isFinite(dur) ? dur : 0),
        phase: prefill ? 'prefill' : 'decode',
        label: text.trim(),
      });
    }
  }
  markers.sort((left, right) => left.tsUs - right.tsUs);
  if (markers.length === 0) return { markers, steps: [] };
  const steps = markers.map((marker, index) => {
    const next = markers[index + 1];
    const endUs = next === undefined ? Math.max(marker.endUs, marker.tsUs) : next.tsUs;
    return {
      label: marker.label,
      startUs: marker.tsUs,
      endUs: Math.max(endUs, marker.tsUs),
      durUs: Math.max(0, endUs - marker.tsUs),
      declaredPhase: marker.phase,
    };
  }).filter((step) => step.durUs > 0);
  return { markers, steps };
}

function fromMarkers(markers) {
  return {
    source: 'mstx-markers',
    sourceLabel: 'mstx / profiler 阶段标记',
    steps: markers.steps,
  };
}

/** Repeating engine-level host events identify step boundaries. */
function findEngineSteps(events) {
  const hostEvents = events.filter((event) => event.device === 'host');
  const buckets = new Map();
  for (const event of hostEvents) {
    for (const pattern of STEP_EVENT_PATTERNS) {
      if (!pattern.test(event.normalizedName)) continue;
      const key = event.normalizedName;
      const bucket = buckets.get(key);
      if (bucket === undefined) buckets.set(key, [event]);
      else bucket.push(event);
      break;
    }
  }
  let best;
  for (const [name, bucket] of buckets) {
    // Prefer the name that repeats most; ties go to the longer total duration.
    const total = bucket.reduce((sum, event) => sum + event.durUs, 0);
    if (best === undefined || bucket.length > best.bucket.length
      || (bucket.length === best.bucket.length && total > best.total)) {
      best = { name, bucket, total };
    }
  }
  if (best === undefined || best.bucket.length < 3) return [];
  const sorted = [...best.bucket].sort((left, right) => left.tsUs - right.tsUs);
  // Keep the outermost occurrences: nested calls of the same name inside one
  // another are implementation detail, not separate steps.
  const outermost = [];
  for (const event of sorted) {
    const last = outermost[outermost.length - 1];
    if (last !== undefined && event.tsUs < last.tsUs + last.durUs) {
      if (event.tsUs + event.durUs > last.tsUs + last.durUs) last.endUs = event.tsUs + event.durUs;
      continue;
    }
    outermost.push({ tsUs: event.tsUs, endUs: event.tsUs + event.durUs, name: event.normalizedName });
  }
  return outermost.map((entry, index) => ({
    label: `${entry.name} #${String(index + 1)}`,
    startUs: entry.tsUs,
    endUs: entry.endUs,
    durUs: entry.endUs - entry.tsUs,
    driver: entry.name,
  }));
}

function fromEngineEvents(steps) {
  return {
    source: 'engine-events',
    sourceLabel: `引擎算子重复出现（${steps[0]?.driver ?? 'execute_model'}）`,
    steps,
  };
}

/** Fall back to segmenting the device timeline at long idle gaps. */
function segmentByGaps(events, window, options) {
  const deviceEvents = events.filter((event) => event.device === 'device');
  if (deviceEvents.length < 4) return [];
  const sorted = [...deviceEvents].sort((left, right) => left.tsUs - right.tsUs);
  const gaps = [];
  let cursor = sorted[0].tsUs;
  for (const event of sorted) {
    if (event.tsUs > cursor) gaps.push(event.tsUs - cursor);
    cursor = Math.max(cursor, event.tsUs + event.durUs);
  }
  if (gaps.length === 0) return [];
  const sortedGaps = [...gaps].sort((left, right) => left - right);
  const median = sortedGaps[Math.floor(sortedGaps.length / 2)];
  const threshold = Math.max(options.idleGapUs ?? 0, median * 4, 200);
  const boundaries = [window.start];
  cursor = sorted[0].tsUs;
  for (const event of sorted) {
    if (event.tsUs - cursor > threshold) boundaries.push(event.tsUs);
    cursor = Math.max(cursor, event.tsUs + event.durUs);
  }
  boundaries.push(window.end);
  const steps = [];
  for (let index = 0; index + 1 < boundaries.length; index += 1) {
    const startUs = boundaries[index];
    const endUs = boundaries[index + 1];
    if (endUs - startUs <= 0) continue;
    steps.push({
      label: `区间 ${String(index + 1)}`,
      startUs,
      endUs,
      durUs: endUs - startUs,
      driver: 'idle-gap',
    });
  }
  if (steps.length < 2) return [];
  return steps;
}

function fromGapSegments(steps) {
  return {
    source: 'idle-gaps',
    sourceLabel: '设备时间线空闲间隔切分',
    steps,
  };
}

function fromWholeWindow(window) {
  return {
    source: 'whole-window',
    sourceLabel: '整个采样窗口作为单步',
    steps: [{
      label: '整个采样窗口',
      startUs: window.start,
      endUs: window.end,
      durUs: Math.max(0, window.end - window.start),
      driver: 'window',
    }],
  };
}

/**
 * Label every step with a phase.
 * @param {object[]} steps - steps.
 * @param {{markers: object[]}} markers - marker model.
 * @param {object} options - options.
 * @returns {{ labels: Record<number, string>, method: string, confidence: string, reason: string }} labels.
 */
function classifySteps(steps, markers, options) {
  const labels = {};
  const declared = steps.map((step) => step.declaredPhase);
  if (declared.some((phase) => phase !== undefined)) {
    steps.forEach((step, index) => {
      labels[index] = declared[index] ?? 'unknown';
    });
    const fromStepTable = steps.some((step) => step.stage !== undefined);
    return {
      labels,
      method: fromStepTable ? 'step-table-stage' : 'marker-labels',
      confidence: 'high',
      reason: fromStepTable
        ? 'prefill/decode 直接取自 step_trace_time.csv 的 Stage 列（CANN 侧给出的阶段标签），无需推断。'
        : 'prefill/decode 由 profiler 阶段标记（mstx）直接给出，无需推断。',
    };
  }
  const markerPhaseByTime = markers.markers;
  if (markerPhaseByTime.length > 0) {
    steps.forEach((step, index) => {
      const covering = markerPhaseByTime.filter((marker) => marker.tsUs >= step.startUs && marker.tsUs < step.endUs);
      labels[index] = covering[0]?.phase ?? 'unknown';
    });
  }

  const durations = steps.map((step) => step.durUs).filter((value) => value > 0);
  const split = bimodalSplit(durations, options.minPhaseSeparation ?? 0.35);
  if (split === undefined) {
    const uniform = durations.length >= 8 && isUniform(durations);
    if (uniform && Object.keys(labels).length === 0) {
      steps.forEach((step, index) => {
        labels[index] = 'decode';
      });
      return {
        labels,
        method: 'uniform-decode',
        confidence: 'low',
        reason: `共 ${String(durations.length)} 个步长高度一致（离散度低），按纯 Decode 负载处理；若本次也包含 Prefill，请开启 mstx 阶段标记以获得准确划分。`,
      };
    }
    if (Object.keys(labels).length > 0) {
      return {
        labels,
        method: 'marker-windows',
        confidence: 'medium',
        reason: '部分步骤由标记覆盖，其余步骤时长分布不具备双峰特征，保持未定。',
      };
    }
    return {
      labels,
      method: 'undetermined',
      confidence: 'low',
      reason: `步长分布未呈现 Prefill/Decode 双峰特征（${String(durations.length)} 个步骤），阶段无法可靠划分；建议在 torch_npu profiling 中开启 mstx 标记，或提供 step_trace_time.csv。`,
    };
  }
  steps.forEach((step, index) => {
    if (labels[index] !== undefined) return;
    labels[index] = step.durUs >= split.threshold ? 'prefill' : 'decode';
  });
  const prefillCount = Object.values(labels).filter((label) => label === 'prefill').length;
  return {
    labels,
    method: 'duration-bimodal',
    confidence: prefillCount > 0 && prefillCount < steps.length ? 'medium' : 'low',
    reason: `步长在 log 空间呈双峰分布（长步均值 ${round(split.highMean, 1)}µs / 短步均值 ${round(split.lowMean, 1)}µs，分离度 ${round(split.separation, 2)}），据此把长步判为 Prefill、短步判为 Decode。`,
  };
}

function isUniform(values) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (mean <= 0) return false;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance) / mean < 0.25;
}

/**
 * Compute the per-step metric block from the events that fall inside the step.
 * @param {object[]} inside - events inside the step window.
 * @param {object} step - step with `startUs`/`endUs`.
 * @returns {object} metrics.
 */
export function metricsForStep(inside, step) {
  const deviceIntervals = inside.filter((event) => event.device === 'device').map(toInterval);
  const hostIntervals = inside.filter((event) => event.device === 'host').map(toInterval);
  const deviceUnion = unionIntervals(deviceIntervals).length;
  const hostUnion = unionIntervals(hostIntervals).length;
  const wallUs = Math.max(1, step.endUs - step.startUs);
  const totals = { compute: 0, comm: 0, copy: 0, schedule: 0, other: 0 };
  const deviceTotals = { compute: 0, comm: 0, copy: 0, schedule: 0, other: 0 };
  for (const event of inside) {
    totals[event.category] = (totals[event.category] ?? 0) + event.durUs;
    if (event.device === 'device') deviceTotals[event.category] = (deviceTotals[event.category] ?? 0) + event.durUs;
  }
  const commIntervals = inside.filter((event) => event.category === 'comm' && event.device === 'device').map(toInterval);
  const computeIntervals = inside.filter((event) => event.category === 'compute' && event.device === 'device').map(toInterval);
  const commOverlap = overlapLength(commIntervals, computeIntervals);
  const csvMetrics = step.csvMetrics ?? {};
  return {
    wallUs,
    eventCount: inside.length,
    deviceBusyUs: deviceUnion,
    hostBusyUs: hostUnion,
    deviceIdleUs: Math.max(0, wallUs - deviceUnion),
    npuUtilPct: round(percentOf(deviceUnion, wallUs), 1),
    hostUtilPct: round(percentOf(hostUnion, wallUs), 1),
    categoryUs: totals,
    deviceCategoryUs: deviceTotals,
    commUs: csvMetrics.commUs ?? totals.comm,
    commNotOverlappedUs: csvMetrics.commNotOverlappedUs ?? Math.max(0, unionIntervals(commIntervals).length - commOverlap),
    commOverlapUs: commOverlap,
    computingUs: csvMetrics.computingUs ?? deviceTotals.compute,
    freeUs: csvMetrics.freeUs ?? Math.max(0, wallUs - deviceUnion),
    commPct: round(percentOf(csvMetrics.commUs ?? totals.comm, wallUs), 1),
    computePct: round(percentOf(deviceTotals.compute, wallUs), 1),
    copyPct: round(percentOf(totals.copy, wallUs), 1),
    schedulePct: round(percentOf(totals.schedule, wallUs), 1),
    hostSchedulePct: round(percentOf(deviceTotals.schedule, wallUs), 1),
    hasCsvMetrics: Object.keys(csvMetrics).length > 0,
  };
}

function toInterval(event) {
  return { start: event.tsUs, end: event.tsUs + event.durUs };
}

/** Attach token counts to steps when the trace exposes them. */
function enrichTokens(steps, sortedEvents, enrich) {
  const withTokens = sortedEvents.filter((event) => Number.isFinite(event.tokenCount));
  if (withTokens.length === 0) return;
  for (const step of steps) {
    if (Number.isFinite(step.tokens)) continue;
    const inside = enrich(step);
    const hit = inside.find((event) => Number.isFinite(event.tokenCount));
    if (hit !== undefined) step.tokens = hit.tokenCount;
  }
}

/** First index whose value is >= target (array must be sorted ascending). */
function lowerBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >> 1;
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

/** Copy the `[from, to)` slice of an event array. */
function eventsInRange(events, from, to) {
  if (to <= from) return [];
  return events.slice(from, to);
}

/**
 * Aggregate steps into one record per phase.
 * @param {object[]} steps - enriched steps.
 * @returns {object[]} phase summaries.
 */
function summarizePhases(steps) {
  const ids = ['prefill', 'decode', 'unknown'];
  const labels = { prefill: 'Prefill（首 token / 提示处理）', decode: 'Decode（逐 token 生成）', unknown: '未划分' };
  const out = [];
  for (const id of ids) {
    const members = steps.filter((step) => step.phase === id);
    if (members.length === 0) continue;
    const wallUs = members.reduce((sum, step) => sum + step.metrics.wallUs, 0);
    const sum = (pick) => members.reduce((total, step) => total + pick(step.metrics), 0);
    out.push({
      id,
      label: labels[id],
      stepCount: members.length,
      stepIndexes: members.map((step) => step.index),
      wallUs,
      avgStepUs: round(wallUs / members.length, 1),
      minStepUs: round(Math.min(...members.map((step) => step.metrics.wallUs)), 1),
      maxStepUs: round(Math.max(...members.map((step) => step.metrics.wallUs)), 1),
      deviceBusyUs: sum((metrics) => metrics.deviceBusyUs),
      hostBusyUs: sum((metrics) => metrics.hostBusyUs),
      deviceIdleUs: sum((metrics) => metrics.deviceIdleUs),
      npuUtilPct: round(percentOf(sum((metrics) => metrics.deviceBusyUs), wallUs), 1),
      hostUtilPct: round(percentOf(sum((metrics) => metrics.hostBusyUs), wallUs), 1),
      commUs: sum((metrics) => metrics.commUs),
      commPct: round(percentOf(sum((metrics) => metrics.commUs), wallUs), 1),
      commNotOverlappedUs: sum((metrics) => metrics.commNotOverlappedUs),
      commNotOverlappedPct: round(percentOf(sum((metrics) => metrics.commNotOverlappedUs), wallUs), 1),
      computeUs: sum((metrics) => metrics.deviceCategoryUs.compute),
      computePct: round(percentOf(sum((metrics) => metrics.deviceCategoryUs.compute), wallUs), 1),
      copyUs: sum((metrics) => metrics.categoryUs.copy),
      copyPct: round(percentOf(sum((metrics) => metrics.categoryUs.copy), wallUs), 1),
      scheduleUs: sum((metrics) => metrics.categoryUs.schedule),
      schedulePct: round(percentOf(sum((metrics) => metrics.categoryUs.schedule), wallUs), 1),
      freeUs: sum((metrics) => metrics.freeUs),
      avgTokensPerStep: average(members.map((step) => step.tokens)),
    });
  }
  return out;
}

function average(values) {
  const numbers = values.filter((value) => Number.isFinite(value));
  if (numbers.length === 0) return undefined;
  return round(numbers.reduce((sum, value) => sum + value, 0) / numbers.length, 1);
}
