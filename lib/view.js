/**
 * View-model projection: the exact JSON contract the browser page consumes.
 *
 * The dataset in memory is richer than any page needs (and a 400k-event
 * timeline would be a 40 MB response), so the view model is projected
 * explicitly:
 *
 * * swimlane rows keep their statistics in full, but their event lists are
 *   budgeted — the budget is shared across rows in proportion to their duration
 *   share, and within a row the longest events are always kept, because those
 *   are the ones a profile is read for;
 * * every dropped count is reported, so the page can say "显示 N/M 个算子条"
 *   instead of quietly lying;
 * * numbers are rounded once, here, so the browser never re-derives a metric.
 *
 * @module dsh-plugin-vllm-ascend-profiler/view
 */

import { CATEGORIES, subtypeLabel } from './model/classify.js';
import { round } from './model/stats.js';

/** Defaults for the projection. */
export const VIEW_DEFAULTS = Object.freeze({
  /** Total timeline events shipped to the browser. */
  eventBudget: 150000,
  /** Operator rows in each ranking table. */
  rankingLimit: 60,
  /** Gaps listed in the idle table. */
  gapLimit: 30,
});

/**
 * Project a dataset + analysis into the browser view model.
 *
 * @param {object} input - projection input.
 * @param {object} input.dataset - dataset.
 * @param {object} input.analysis - analysis result.
 * @param {string} [input.datasetId] - dataset id.
 * @param {string} [input.label] - human label for the dataset.
 * @param {object} [input.options] - {@link VIEW_DEFAULTS} overrides.
 * @returns {object} the view model.
 */
export function buildViewModel({ dataset, analysis, datasetId, label, options = {} }) {
  const config = { ...VIEW_DEFAULTS, ...options };
  const budget = allocateEventBudget(dataset, config.eventBudget);
  const lanes = dataset.laneGroups.map((group) => ({
    id: group.id,
    label: group.label,
    description: group.description,
    eventCount: group.eventCount,
    totalUs: group.totalUs,
    hiddenRows: group.hiddenRows,
    rows: group.rows.map((row) => projectRow(row, budget.get(row.key) ?? 0)),
  }));

  return {
    datasetId,
    label,
    generatedAt: new Date().toISOString(),
    meta: {
      window: dataset.meta.window,
      wallUs: dataset.meta.wallUs,
      eventCount: dataset.meta.eventCount,
      counts: dataset.meta.counts,
      sampling: dataset.meta.sampling,
      totalsSource: dataset.meta.totalsSource,
      files: dataset.meta.files,
      tableSummaries: dataset.meta.tableSummaries,
      auxSummaries: dataset.meta.auxSummaries,
      warnings: dataset.meta.warnings,
      validation: {
        ok: dataset.meta.validation?.ok,
        profileType: dataset.meta.validation?.profileType,
        weight: dataset.meta.validation?.weight,
        confidence: dataset.meta.validation?.confidence,
        markers: (dataset.meta.validation?.markers ?? []).slice(0, 24),
      },
      options: dataset.meta.options,
    },
    timeline: {
      eventBudget: config.eventBudget,
      shippedEvents: lanes.reduce((sum, group) => sum + group.rows.reduce((inner, row) => inner + row.events.length, 0), 0),
      truncated: lanes.some((group) => group.rows.some((row) => row.eventsTruncated === true)),
      lanes,
    },
    categories: dataset.categories,
    operators: dataset.operators.slice(0, 400).map((row) => ({
      name: row.name,
      group: row.group,
      category: row.category,
      subtype: row.subtype,
      subtypeLabel: subtypeLabel(row.category, row.subtype),
      count: row.count,
      totalUs: row.totalUs,
      avgUs: row.avgUs,
      maxUs: row.maxUs,
      p50Us: row.p50Us,
      p95Us: row.p95Us,
      waitUs: row.waitUs,
      traceTotalUs: row.traceTotalUs,
      csvTotalUs: row.csvTotalUs,
      totalsSource: row.totalsSource,
      crossCheckPct: row.crossCheckPct,
      utilization: row.utilization,
      sources: row.sources,
      overflow: row.overflow === true,
      sample: row.sample,
    })),
    ranking: {
      byTotal: dataset.ranking.byTotal.slice(0, config.rankingLimit),
      byAverage: dataset.ranking.byAverage.slice(0, config.rankingLimit),
      totalOperatorUs: dataset.ranking.totalOperatorUs,
    },
    overlap: dataset.overlap,
    gaps: dataset.gaps.slice(0, config.gapLimit),
    utilization: dataset.utilization,
    phases: {
      source: dataset.phases.source,
      sourceLabel: dataset.phases.sourceLabel,
      confidence: dataset.phases.confidence,
      confidenceReason: dataset.phases.confidenceReason,
      classificationMethod: dataset.phases.classificationMethod,
      steps: dataset.phases.steps.map((step) => ({
        index: step.index,
        label: step.label,
        phase: step.phase,
        startUs: round(step.startUs, 2),
        endUs: round(step.endUs, 2),
        durUs: round(step.durUs, 2),
        tokens: step.tokens,
        metrics: step.metrics,
      })),
      phases: dataset.phases.phases,
    },
    analysis,
    legend: Object.values(CATEGORIES).map((category) => ({
      id: category.id,
      label: category.label,
      color: category.color,
    })),
  };
}

/**
 * Share the timeline event budget across swimlane rows.
 *
 * Rows are budgeted by duration share with a floor, so a row holding one long
 * kernel still gets bars while the heaviest rows get the detail. Within a row
 * the longest events win.
 *
 * @param {object} dataset - dataset.
 * @param {number} budget - total events to ship.
 * @returns {Map<string, number>} per-row allocation.
 */
function allocateEventBudget(dataset, budget) {
  const rows = dataset.laneGroups.flatMap((group) => group.rows);
  const total = rows.reduce((sum, row) => sum + row.events.length, 0);
  const allocation = new Map();
  if (total <= budget) {
    for (const row of rows) allocation.set(row.key, row.events.length);
    return allocation;
  }
  const totalUs = rows.reduce((sum, row) => sum + row.totalUs, 0);
  const floor = Math.max(8, Math.floor(budget / Math.max(1, rows.length) / 4));
  let assigned = 0;
  for (const row of rows) {
    const share = totalUs > 0 ? row.totalUs / totalUs : 1 / rows.length;
    const quota = Math.max(floor, Math.floor(budget * share));
    const value = Math.min(row.events.length, quota);
    allocation.set(row.key, value);
    assigned += value;
  }
  // Hand any remainder to the heaviest rows so the budget is actually used.
  let leftover = budget - assigned;
  if (leftover > 0) {
    for (const row of [...rows].sort((left, right) => right.totalUs - left.totalUs)) {
      if (leftover <= 0) break;
      const current = allocation.get(row.key) ?? 0;
      const headroom = row.events.length - current;
      if (headroom <= 0) continue;
      const add = Math.min(headroom, leftover);
      allocation.set(row.key, current + add);
      leftover -= add;
    }
  }
  return allocation;
}

/** Project one swimlane row, applying its event budget. */
function projectRow(row, quota) {
  const events = row.events;
  let kept = events;
  let truncated = false;
  if (quota > 0 && events.length > quota) {
    truncated = true;
    // Keep the longest events, then restore time order for rendering.
    const longest = [...events].sort((left, right) => right.dur - left.dur).slice(0, quota);
    kept = longest.sort((left, right) => left.start - right.start);
  } else if (quota === 0) {
    kept = [];
    truncated = events.length > 0;
  }
  return {
    key: row.key,
    name: row.name,
    label: row.label,
    group: row.group,
    category: row.category,
    subtypes: row.subtypes,
    ambiguous: row.ambiguous === true,
    overflow: row.overflow === true,
    foldedOperators: row.foldedOperators,
    count: row.count,
    totalUs: row.totalUs,
    avgUs: row.avgUs,
    maxUs: row.maxUs,
    p50Us: row.p50Us,
    p95Us: row.p95Us,
    waitUs: row.waitUs,
    streams: row.streams,
    ranks: row.ranks,
    sample: row.sample,
    events: kept,
    eventsShipped: kept.length,
    eventsTotal: events.length,
    eventsTruncated: truncated,
  };
}
