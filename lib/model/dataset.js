/**
 * Dataset construction: the single normalized model every view and every
 * analysis step reads.
 *
 * Responsibilities:
 *
 * * classify every event, give it a lane, and rebase multi-rank timelines;
 * * aggregate per-operator statistics from trace events **and** from the CSV
 *   tables, keeping both and declaring which one the totals come from (a
 *   sampled trace must never be presented as the source of truth when the
 *   CANN tables are available);
 * * build the swimlane row model, cost-share category totals, Top-N rankings,
 *   device idle gaps, host/device overlap, and hardware utilization;
 * * attach the prefill/decode phase model.
 *
 * @module dsh-plugin-vllm-ascend-profiler/model/dataset
 */

import { CATEGORIES, CATEGORY_ORDER, classifyOperator, normalizeOperatorName } from './classify.js';
import { buildPhases } from './phase.js';
import { findGaps, groupBy, mean, overlapLength, percentOf, percentile, round, unionIntervals } from './stats.js';

/** Defaults for dataset construction. */
export const DATASET_DEFAULTS = Object.freeze({
  /** Swimlane rows kept per device group; the remainder is folded into one row. */
  maxLanesPerGroup: 120,
  /** Gaps shorter than this are ignored when reporting NPU idle time. */
  minGapUs: 50,
  /** Rows listed in the Top-N ranking table. */
  topN: 20,
  /** Peak AI Core throughput used to turn FLOPs into a utilization figure (TFLOP/s, fp16). */
  peakTflops: 0,
  /** How to place multiple ranks on one time axis. */
  multiRankMode: 'rebase',
});

/**
 * Build the dataset.
 *
 * @param {object} parse - result of `parseProfileSet`.
 * @param {object} [options] - {@link DATASET_DEFAULTS} overrides.
 * @returns {object} the dataset.
 */
export function buildDataset(parse, options = {}) {
  const config = { ...DATASET_DEFAULTS, ...options };
  const warnings = [...parse.warnings];
  const events = parse.events.map((event, index) => normalizeEvent(event, index));

  rebaseRanks(events, config, warnings);
  const window = computeWindow(events);
  const laneModel = buildLanes(events, config, warnings);
  const csvStats = aggregateTables(parse.tables ?? [], warnings);
  const operatorRows = mergeOperatorStats(laneModel.aggregated, csvStats, parse.sampling, config);
  const categories = aggregateCategories(operatorRows, parse.sampling);
  const overlap = computeOverlap(events, window);
  const gaps = computeGaps(events, window, config);
  const utilization = aggregateUtilization(parse.tables ?? [], events, config);
  const phases = buildPhases({ events, tables: parse.tables ?? [], aux: parse.aux ?? [], window, options: config });
  const ranking = buildRanking(operatorRows, config);

  const wallUs = Math.max(1, window.end - window.start);
  const deviceBusyUs = overlap.deviceBusyUs;
  const hostBusyUs = overlap.hostBusyUs;

  return {
    meta: {
      generatedAt: new Date().toISOString(),
      window,
      wallUs,
      eventCount: events.length,
      sampling: parse.sampling,
      validation: parse.validation,
      files: parse.files ?? [],
      encodingNotes: parse.encodingNotes ?? [],
      tableSummaries: summarizeTables(parse.tables ?? []),
      auxSummaries: (parse.aux ?? []).map((artifact) => ({ kind: artifact.kind, name: artifact.name, ...artifact.summary })),
      warnings: dedupe(warnings),
      totalsSource: categories.totalsSource,
      options: {
        maxLanesPerGroup: config.maxLanesPerGroup,
        topN: config.topN,
        multiRankMode: config.multiRankMode,
        peakTflops: config.peakTflops,
      },
      counts: {
        hostEvents: events.filter((event) => event.device === 'host').length,
        deviceEvents: events.filter((event) => event.device === 'device').length,
        operators: operatorRows.length,
        ranks: [...new Set(events.map((event) => event.rank).filter((rank) => rank !== undefined))].length,
      },
    },
    events,
    lanes: laneModel.lanes,
    /** Non-empty device groups in display order, each with its swimlane rows. */
    laneGroups: laneModel.lanes,
    /** Static group descriptors (labels, tooltips) independent of membership. */
    laneDescriptors: laneModel.groups,
    operators: operatorRows,
    categories,
    ranking,
    overlap: {
      ...overlap,
      wallUs,
      deviceBusyPct: round(percentOf(deviceBusyUs, wallUs), 1),
      hostBusyPct: round(percentOf(hostBusyUs, wallUs), 1),
      overlapPct: round(percentOf(overlap.overlapUs, wallUs), 1),
      deviceOnlyPct: round(percentOf(overlap.deviceOnlyUs, wallUs), 1),
      hostOnlyPct: round(percentOf(overlap.hostOnlyUs, wallUs), 1),
      idlePct: round(percentOf(Math.max(0, wallUs - deviceBusyUs), wallUs), 1),
    },
    gaps,
    utilization,
    phases,
    aux: parse.aux ?? [],
  };
}

/** Copy an event into the normalized shape and classify it. */
function normalizeEvent(event, index) {
  const classification = classifyOperator({
    name: event.name,
    device: event.device,
    opType: event.opType,
    taskType: event.taskType,
    coreType: event.coreType,
    cat: event.cat,
  });
  const normalizedName = event.normalizedName ?? normalizeOperatorName(event.name);
  return {
    ...event,
    index,
    normalizedName,
    category: classification.category,
    subtype: classification.subtype,
    role: classification.role,
    rank: event.rank === undefined ? undefined : String(event.rank),
    endUs: event.tsUs + event.durUs,
  };
}

/**
 * Put every event on one axis.
 *
 * Ascend exports absolute timestamps (Unix microseconds on the host side) or
 * per-rank device clocks whose origins differ by tens of milliseconds. The
 * dataset therefore always rebases: a single rank is rebased onto its own first
 * event so the axis starts at zero, and multiple ranks are each rebased onto
 * their own origin so per-rank shape is preserved. The removed offsets are
 * reported, because they are the only place absolute skew is visible.
 */
function rebaseRanks(events, config, warnings) {
  const ranks = [...new Set(events.map((event) => event.rank).filter((rank) => rank !== undefined))];
  if (ranks.length === 0) {
    const origin = events.length === 0 ? 0 : Math.min(...events.map((event) => event.tsUs));
    for (const event of events) {
      event.relTsUs = event.tsUs - origin;
      event.rankOffsetUs = origin;
    }
    return;
  }
  if (config.multiRankMode === 'raw') {
    warnings.push(`检测到 ${String(ranks.length)} 个 rank，泳道图按原始时钟叠加（multiRankMode=raw），不同卡的时间原点可能不同。`);
    for (const event of events) event.relTsUs = event.tsUs;
    return;
  }
  const offsets = new Map();
  for (const rank of ranks) {
    const first = Math.min(...events.filter((event) => event.rank === rank).map((event) => event.tsUs));
    offsets.set(rank, first);
  }
  for (const event of events) {
    const offset = event.rank === undefined ? Math.min(...offsets.values()) : (offsets.get(event.rank) ?? 0);
    event.relTsUs = event.tsUs - offset;
    event.rankOffsetUs = offset;
  }
  if (ranks.length > 1) {
    warnings.push(
      `检测到 ${String(ranks.length)} 个 rank，泳道图已按各 rank 首个事件对齐（multiRankMode=rebase）；`
      + `各 rank 原始起点：${[...offsets.entries()].map(([rank, offset]) => `rank${rank} ${round(offset / 1000, 2)}ms`).join('，')}。`
      + '设备时钟起点在不同卡之间存在漂移属正常现象，多卡对齐请以同一通信算子的结束时刻为基准。',
    );
  }
}

/** Compute the timeline window over the rebased events. */
function computeWindow(events) {
  if (events.length === 0) return { start: 0, end: 0 };
  let start = Infinity;
  let end = -Infinity;
  for (const event of events) {
    start = Math.min(start, event.relTsUs);
    end = Math.max(end, event.relTsUs + event.durUs);
  }
  return Number.isFinite(start) ? { start, end } : { start: 0, end: 0 };
}

/**
 * Build swimlane rows: one row per operator inside each device group.
 *
 * Rows are ordered by total duration so the eye lands on what matters, and the
 * tail beyond `maxLanesPerGroup` is folded into a single explicit overflow row
 * rather than being silently dropped.
 */
function buildLanes(events, config, warnings) {
  const groups = [
    { id: 'host', label: 'Host（CPU）', description: '主机侧算子：引擎调度、Python/ATen 派发、采样、H2D/D2H 发起' },
    { id: 'device', label: 'Device（昇腾 NPU）', description: '设备侧算子：AI Core/AI Vector 计算、HCCL 通信、数据搬运' },
  ];
  const lanes = [];
  const aggregated = [];
  let overflowed = 0;

  for (const group of groups) {
    const members = events.filter((event) => event.device === group.id);
    if (members.length === 0) continue;
    const byName = groupBy(members, (event) => event.normalizedName);
    const rows = [...byName.entries()].map(([name, bucket]) => {
      const durations = bucket.map((event) => event.durUs);
      const totalUs = durations.reduce((sum, value) => sum + value, 0);
      const categories = new Set(bucket.map((event) => event.category));
      const subtypes = new Set(bucket.map((event) => event.subtype));
      const waitUs = bucket.reduce((sum, event) => sum + (event.waitUs ?? 0), 0);
      return {
        key: `${group.id}:${name}`,
        name,
        group: group.id,
        label: name,
        category: dominantCategory(bucket),
        subtypes: [...subtypes],
        ambiguous: categories.size > 1,
        count: bucket.length,
        totalUs: round(totalUs, 2),
        avgUs: round(totalUs / bucket.length, 2),
        maxUs: round(Math.max(...durations), 2),
        p50Us: round(percentile(durations, 0.5), 2),
        p95Us: round(percentile(durations, 0.95), 2),
        waitUs: round(waitUs, 2),
        streams: [...new Set(bucket.map((event) => event.stream).filter((stream) => stream !== undefined))].slice(0, 16),
        ranks: [...new Set(bucket.map((event) => event.rank).filter((rank) => rank !== undefined))],
        events: bucket.map((event) => ({ i: event.index, start: round(event.relTsUs, 2), dur: round(event.durUs, 2) }))
          .sort((left, right) => left.start - right.start),
        sample: sampleDetail(bucket),
      };
    }).sort((left, right) => right.totalUs - left.totalUs);

    const kept = rows.slice(0, config.maxLanesPerGroup);
    const rest = rows.slice(config.maxLanesPerGroup);
    if (rest.length > 0) {
      overflowed += rest.length;
      const restEvents = rest.flatMap((row) => row.events);
      kept.push({
        key: `${group.id}:__overflow__`,
        name: `其他 ${String(rest.length)} 个算子`,
        group: group.id,
        label: `其他 ${String(rest.length)} 个算子（已折叠）`,
        category: 'other',
        subtypes: [...new Set(rest.flatMap((row) => row.subtypes))],
        ambiguous: true,
        overflow: true,
        foldedOperators: rest.map((row) => ({ name: row.name, totalUs: row.totalUs, count: row.count })),
        count: restEvents.length,
        totalUs: round(rest.reduce((sum, row) => sum + row.totalUs, 0), 2),
        avgUs: round(mean(rest.map((row) => row.avgUs)), 2),
        maxUs: round(Math.max(...rest.map((row) => row.maxUs)), 2),
        p50Us: round(percentile(rest.map((row) => row.p50Us), 0.5), 2),
        p95Us: round(percentile(rest.map((row) => row.p95Us), 0.95), 2),
        waitUs: round(rest.reduce((sum, row) => sum + row.waitUs, 0), 2),
        streams: [],
        ranks: [...new Set(rest.flatMap((row) => row.ranks))],
        events: restEvents.sort((left, right) => left.start - right.start).slice(0, 20000),
        sample: undefined,
      });
    }
    lanes.push({
      id: group.id,
      group: group.id,
      label: group.label,
      description: group.description,
      rows: kept,
      hiddenRows: rest.length,
      eventCount: members.length,
      totalUs: round(members.reduce((sum, event) => sum + event.durUs, 0), 2),
    });
    for (const row of rows) aggregated.push(row);
  }

  if (overflowed > 0) {
    warnings.push(`泳道图每组仅展示耗时最高的 ${String(config.maxLanesPerGroup)} 个算子，其余 ${String(overflowed)} 个已折叠为“其他”一行（完整清单见算子耗时排行表）。`);
  }
  return { lanes, groups, aggregated };
}

/** Category with the largest total duration inside one operator bucket. */
function dominantCategory(bucket) {
  const totals = new Map();
  for (const event of bucket) totals.set(event.category, (totals.get(event.category) ?? 0) + event.durUs);
  let best = 'other';
  let bestValue = -1;
  for (const [category, value] of totals) {
    if (value > bestValue) {
      best = category;
      bestValue = value;
    }
  }
  return best;
}

/** Keep one representative event for the hover card. */
function sampleDetail(bucket) {
  const longest = [...bucket].sort((left, right) => right.durUs - left.durUs)[0];
  return {
    name: longest.name,
    opType: longest.opType,
    taskType: longest.taskType,
    coreType: longest.coreType,
    shapesIn: longest.shapesIn,
    shapesOut: longest.shapesOut,
    callStack: longest.callStack,
    category: longest.category,
    subtype: longest.subtype,
    device: longest.device,
    maxDurUs: round(longest.durUs, 2),
    rank: longest.rank,
    stream: longest.stream,
  };
}

/**
 * Aggregate the CSV tables into per-operator statistics.
 * @param {object[]} tables - parsed tables.
 * @param {string[]} warnings - warning sink.
 * @returns {{ byName: Map<string, object>, statTables: string[], instanceTables: string[], pipeUtilization: object[] }} aggregated table statistics.
 */
function aggregateTables(tables, warnings) {
  const byName = new Map();
  const statTables = [];
  const instanceTables = [];
  const pipeUtilization = [];
  for (const table of tables) {
    if (table.kind === 'pipe_utilization') {
      pipeUtilization.push(...table.rows);
      continue;
    }
    if (!['op_statistic', 'api_statistic', 'op_summary', 'kernel_details', 'operator_details', 'communication_statistic'].includes(table.kind)) {
      continue;
    }
    if (['op_statistic', 'api_statistic'].includes(table.kind)) statTables.push(table.name);
    else instanceTables.push(table.name);
    for (const row of table.rows) {
      const classification = classifyOperator({
        name: row.name,
        opType: row.opType,
        taskType: row.taskType,
        coreType: row.coreType,
        device: table.kind === 'api_statistic' ? 'host' : 'device',
      });
      const name = normalizeOperatorName(row.name);
      const key = `${classification.category === 'comm' ? 'comm' : classification.category}:${name}`;
      const bucket = byName.get(name) ?? {
        name,
        category: classification.category,
        subtype: classification.subtype,
        role: classification.role,
        count: 0,
        totalUs: 0,
        waitUs: 0,
        durations: [],
        sources: new Set(),
        deviceIds: new Set(),
        utilization: new Map(),
        utilizationSamples: 0,
      };
      const count = row.count ?? 1;
      bucket.count += count;
      bucket.totalUs += row.totalUs ?? row.durUs ?? 0;
      bucket.waitUs += row.waitUs ?? 0;
      if (row.durUs !== undefined) bucket.durations.push(row.durUs);
      bucket.sources.add(table.kind);
      if (row.deviceId !== undefined) bucket.deviceIds.add(row.deviceId);
      if (row.utilization !== undefined && Object.keys(row.utilization).length > 0) {
        const weight = row.durUs ?? row.totalUs ?? 1;
        for (const [metric, value] of Object.entries(row.utilization)) {
          const entry = bucket.utilization.get(metric) ?? { weighted: 0, weight: 0 };
          entry.weighted += value * weight;
          entry.weight += weight;
          bucket.utilization.set(metric, entry);
        }
        bucket.utilizationSamples += 1;
      }
      byName.set(name, bucket);
      void key;
    }
  }
  // API statistics are host-side rows; note that in the caller through sources.
  const hostApiTables = tables.filter((table) => table.kind === 'api_statistic');
  if (hostApiTables.length > 0) {
    warnings.push(`api_statistic.csv 计入 Host 调度开销（${hostApiTables.map((table) => table.name).join('，')}），其耗时为 host API 调用时间，与设备算子时间不可直接相加。`);
  }
  return { byName, statTables, instanceTables, pipeUtilization };
}

/**
 * Merge trace-derived operator statistics with CSV-derived statistics.
 *
 * The trace wins when it is complete; the CSV wins when the trace was sampled,
 * because sampling biases totals. Both numbers are kept so the report can show
 * the cross-check delta, and CSV-only operators (present when trace sampling
 * dropped them, or when no trace was uploaded) are added rather than lost.
 */
function mergeOperatorStats(traceRows, csvStats, sampling, config) {
  const sampled = sampling?.applied === true;
  const rows = [];
  const seen = new Set();
  for (const row of traceRows) {
    const csv = csvStats.byName.get(row.name);
    const traceTotal = row.totalUs;
    const csvTotal = csv === undefined ? undefined : round(csv.totalUs, 2);
    const useCsv = sampled && csvTotal !== undefined;
    rows.push({
      name: row.name,
      group: row.group,
      label: row.label,
      category: csv?.category ?? row.category,
      subtype: csv?.subtype ?? row.subtypes[0],
      ambiguous: row.ambiguous,
      count: useCsv ? csv.count : row.count,
      traceCount: row.count,
      csvCount: csv?.count,
      totalUs: useCsv ? csvTotal : traceTotal,
      traceTotalUs: traceTotal,
      csvTotalUs: csvTotal,
      avgUs: round((useCsv ? csvTotal : traceTotal) / Math.max(1, useCsv ? csv.count : row.count), 2),
      traceAvgUs: row.avgUs,
      csvAvgUs: csv === undefined || csv.count === 0 ? undefined : round(csv.totalUs / csv.count, 2),
      maxUs: row.maxUs,
      p50Us: row.p50Us,
      p95Us: row.p95Us,
      waitUs: csv !== undefined && csv.waitUs > 0 ? round(csv.waitUs, 2) : row.waitUs,
      streams: row.streams,
      ranks: row.ranks,
      utilization: csv === undefined ? {} : utilizationOf(csv),
      utilizationSamples: csv?.utilizationSamples ?? 0,
      sources: {
        trace: true,
        csv: csv !== undefined,
        csvTables: csv === undefined ? [] : [...csv.sources],
      },
      totalsSource: useCsv ? 'csv' : 'trace',
      crossCheckPct: csvTotal === undefined || traceTotal === 0
        ? undefined
        : round(((useCsv ? traceTotal - csvTotal : csvTotal - traceTotal) / Math.max(1, useCsv ? csvTotal : traceTotal)) * 100, 1),
      sample: row.sample,
      foldedOperators: row.foldedOperators,
      overflow: row.overflow === true,
    });
    seen.add(row.name);
  }
  for (const [name, csv] of csvStats.byName) {
    if (seen.has(name)) continue;
    rows.push({
      name,
      group: csv.role === 'framework' || csv.role === 'cpu-kernel' ? 'host' : 'device',
      label: name,
      category: csv.category,
      subtype: csv.subtype,
      count: csv.count,
      traceCount: 0,
      csvCount: csv.count,
      totalUs: round(csv.totalUs, 2),
      traceTotalUs: 0,
      csvTotalUs: round(csv.totalUs, 2),
      avgUs: round(csv.totalUs / Math.max(1, csv.count), 2),
      maxUs: csv.durations.length === 0 ? undefined : round(Math.max(...csv.durations), 2),
      p50Us: csv.durations.length === 0 ? undefined : round(percentile(csv.durations, 0.5), 2),
      p95Us: csv.durations.length === 0 ? undefined : round(percentile(csv.durations, 0.95), 2),
      waitUs: round(csv.waitUs, 2),
      streams: [],
      ranks: [...csv.deviceIds],
      utilization: utilizationOf(csv),
      utilizationSamples: csv.utilizationSamples,
      sources: { trace: false, csv: true, csvTables: [...csv.sources] },
      totalsSource: 'csv',
      csvOnly: true,
    });
  }
  rows.sort((left, right) => right.totalUs - left.totalUs);
  void config;
  return rows;
}

function utilizationOf(bucket) {
  const out = {};
  for (const [metric, entry] of bucket.utilization) {
    out[metric] = entry.weight === 0 ? 0 : round(entry.weighted / entry.weight, 4);
  }
  return out;
}

/** Category totals plus their share of total operator time. */
function aggregateCategories(operatorRows, sampling) {
  const totals = {};
  const counts = {};
  const perDevice = { host: {}, device: {} };
  for (const category of CATEGORY_ORDER) {
    totals[category] = 0;
    counts[category] = 0;
    perDevice.host[category] = 0;
    perDevice.device[category] = 0;
  }
  for (const row of operatorRows) {
    totals[row.category] = (totals[row.category] ?? 0) + row.totalUs;
    counts[row.category] = (counts[row.category] ?? 0) + row.count;
    if (row.group === 'host' || row.group === 'device') perDevice[row.group][row.category] += row.totalUs;
  }
  const sum = CATEGORY_ORDER.reduce((total, category) => total + totals[category], 0);
  const items = CATEGORY_ORDER.map((category) => ({
    id: category,
    label: CATEGORIES[category].label,
    color: CATEGORIES[category].color,
    totalUs: round(totals[category], 2),
    count: counts[category],
    sharePct: round(percentOf(totals[category], sum), 2),
    hostUs: round(perDevice.host[category], 2),
    deviceUs: round(perDevice.device[category], 2),
  }));
  return {
    items,
    totalUs: round(sum, 2),
    totalsSource: sampling?.applied === true ? '混合（trace 已采样，累计耗时优先取 CSV 统计）' : 'trace 事件累计',
  };
}

/** Top-N ranking by both cumulative and per-call duration. */
function buildRanking(operatorRows, config) {
  const byTotal = [...operatorRows].filter((row) => row.overflow !== true).sort((left, right) => right.totalUs - left.totalUs);
  const byAverage = [...operatorRows]
    .filter((row) => row.overflow !== true && row.count >= 1)
    .sort((left, right) => right.avgUs - left.avgUs);
  const sum = operatorRows.reduce((total, row) => total + row.totalUs, 0);
  const decorate = (rows, dimension) => rows.slice(0, config.topN).map((row, index) => ({
    rank: index + 1,
    name: row.name,
    group: row.group,
    category: row.category,
    subtype: row.subtype,
    count: row.count,
    totalUs: row.totalUs,
    avgUs: row.avgUs,
    maxUs: row.maxUs,
    p95Us: row.p95Us,
    waitUs: row.waitUs,
    shareOfOpsPct: round(percentOf(row.totalUs, sum), 2),
    sources: row.sources,
    totalsSource: row.totalsSource,
    crossCheckPct: row.crossCheckPct,
    utilization: row.utilization,
    dimension,
  }));
  return {
    byTotal: decorate(byTotal, 'total'),
    byAverage: decorate(byAverage, 'average'),
    totalOperatorUs: round(sum, 2),
  };
}

/** Host/device busy time and how much of the device work is hidden under host work. */
function computeOverlap(events, window) {
  const deviceIntervals = events.filter((event) => event.device === 'device').map((event) => ({ start: event.relTsUs, end: event.relTsUs + event.durUs }));
  const hostIntervals = events.filter((event) => event.device === 'host').map((event) => ({ start: event.relTsUs, end: event.relTsUs + event.durUs }));
  const deviceBusyUs = unionIntervals(deviceIntervals).length;
  const hostBusyUs = unionIntervals(hostIntervals).length;
  const overlapUs = overlapLength(deviceIntervals, hostIntervals);
  const commIntervals = events.filter((event) => event.device === 'device' && event.category === 'comm').map((event) => ({ start: event.relTsUs, end: event.relTsUs + event.durUs }));
  const computeIntervals = events.filter((event) => event.device === 'device' && event.category === 'compute').map((event) => ({ start: event.relTsUs, end: event.relTsUs + event.durUs }));
  const commUnionUs = unionIntervals(commIntervals).length;
  const commOverlapUs = overlapLength(commIntervals, computeIntervals);
  void window;
  return {
    deviceBusyUs: round(deviceBusyUs, 2),
    hostBusyUs: round(hostBusyUs, 2),
    overlapUs: round(overlapUs, 2),
    deviceOnlyUs: round(Math.max(0, deviceBusyUs - overlapUs), 2),
    hostOnlyUs: round(Math.max(0, hostBusyUs - overlapUs), 2),
    commUnionUs: round(commUnionUs, 2),
    commOverlapUs: round(commOverlapUs, 2),
    commExposedUs: round(Math.max(0, commUnionUs - commOverlapUs), 2),
    commOverlapPct: round(percentOf(commOverlapUs, commUnionUs), 1),
  };
}

/** NPU idle gaps, each annotated with the operators that bracket it. */
function computeGaps(events, window, config) {
  const deviceEvents = events.filter((event) => event.device === 'device').sort((left, right) => left.relTsUs - right.relTsUs);
  if (deviceEvents.length === 0) return [];
  const intervals = deviceEvents.map((event) => ({ start: event.relTsUs, end: event.relTsUs + event.durUs }));
  const raw = findGaps(intervals, window.start, window.end, config.minGapUs);
  const totalIdleUs = raw.reduce((sum, gap) => sum + gap.length, 0);
  return raw.slice(0, 40).map((gap) => {
    const before = [...deviceEvents].reverse().find((event) => event.relTsUs + event.durUs <= gap.start + 1);
    const after = deviceEvents.find((event) => event.relTsUs >= gap.end - 1);
    const hostDuring = events.filter((event) => event.device === 'host' && event.relTsUs < gap.end && event.relTsUs + event.durUs > gap.start);
    return {
      startUs: round(gap.start, 2),
      endUs: round(gap.end, 2),
      lengthUs: round(gap.length, 2),
      sharePct: round(percentOf(gap.length, Math.max(1, window.end - window.start)), 2),
      before: before?.normalizedName,
      beforeCategory: before?.category,
      after: after?.normalizedName,
      afterCategory: after?.category,
      hostOperatorCount: hostDuring.length,
      hostTop: topNames(hostDuring, 3),
    };
  }).concat(totalIdleUs > 0 ? [] : []).sort((left, right) => right.lengthUs - left.lengthUs);
}

function topNames(events, limit) {
  const totals = new Map();
  for (const event of events) totals.set(event.normalizedName, (totals.get(event.normalizedName) ?? 0) + event.durUs);
  return [...totals.entries()].sort((left, right) => right[1] - left[1]).slice(0, limit).map(([name, totalUs]) => ({ name, totalUs: round(totalUs, 2) }));
}

/** Weighted hardware utilization from the CANN tables, plus FLOPs-derived throughput. */
function aggregateUtilization(tables, events, config) {
  const metricWeights = new Map();
  const perOperator = [];
  for (const table of tables) {
    for (const row of table.rows) {
      if (row.utilization === undefined || Object.keys(row.utilization).length === 0) continue;
      const weight = row.durUs ?? row.totalUs ?? 1;
      for (const [metric, value] of Object.entries(row.utilization)) {
        const entry = metricWeights.get(metric) ?? { weighted: 0, weight: 0, label: row.utilizationLabels?.[metric] };
        entry.weighted += value * weight;
        entry.weight += weight;
        if (entry.label === undefined) entry.label = row.utilizationLabels?.[metric];
        metricWeights.set(metric, entry);
      }
      perOperator.push({
        name: normalizeOperatorName(row.name),
        opType: row.opType,
        durUs: round(weight, 2),
        utilization: row.utilization,
        labels: row.utilizationLabels,
      });
    }
  }
  const metrics = [...metricWeights.entries()].map(([metric, entry]) => ({
    metric,
    label: entry.label ?? metric,
    value: entry.weight === 0 ? 0 : round(entry.weighted / entry.weight, 4),
    valuePct: round((entry.weight === 0 ? 0 : entry.weighted / entry.weight) * 100, 1),
    weightUs: round(entry.weight, 2),
  })).sort((left, right) => right.weightUs - left.weightUs);

  const flopsEvents = events.filter((event) => Number.isFinite(event.flops) && event.flops > 0 && event.durUs > 0);
  const totalFlops = flopsEvents.reduce((sum, event) => sum + event.flops, 0);
  const totalDurUs = flopsEvents.reduce((sum, event) => sum + event.durUs, 0);
  const achievedTflops = totalDurUs === 0 ? undefined : round((totalFlops / (totalDurUs / 1e6)) / 1e12, 2);
  const peak = Number(config.peakTflops) > 0 ? Number(config.peakTflops) : undefined;
  return {
    available: metrics.length > 0,
    metrics,
    perOperator: perOperator.sort((left, right) => right.durUs - left.durUs).slice(0, 40),
    flops: flopsEvents.length === 0 ? undefined : {
      events: flopsEvents.length,
      totalFlops,
      achievedTflops,
      peakTflops: peak,
      utilizationPct: peak === undefined || achievedTflops === undefined ? undefined : round((achievedTflops / peak) * 100, 1),
      note: peak === undefined
        ? '未配置 peakTflops（单卡峰值算力），仅给出实测 TFLOPS；在插件配置中填入芯片峰值可换算算力利用率。'
        : `按配置的峰值算力 ${String(peak)} TFLOPS 换算。`,
    },
    note: metrics.length === 0
      ? '未在 CSV 产物中发现流水/算力利用率列；请在 profiling 中开启 AI Core 指标（kernel_details.csv / op_summary.csv 的 mac_ratio、mte*_ratio 等列）。'
      : '利用率为按算子耗时加权的平均值，来源：CANN 导出的算子级流水/算力指标列。',
  };
}

function summarizeTables(tables) {
  return tables.map((table) => ({
    name: table.name,
    kind: table.kind,
    rows: table.rows.length,
    truncated: table.truncated,
    encoding: table.encoding,
    unmatchedColumns: table.unmatchedColumns.slice(0, 12),
    header: table.header.slice(0, 24),
  }));
}

function dedupe(values) {
  return [...new Set(values.filter((value) => typeof value === 'string' && value !== ''))];
}
