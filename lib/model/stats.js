/**
 * Small numeric and interval helpers shared by the aggregation and analysis
 * layers. Everything here is deterministic and side-effect free so the same
 * inputs always produce the same report.
 *
 * @module dsh-plugin-vllm-ascend-profiler/model/stats
 */

/**
 * Merge overlapping intervals and return the covered time.
 *
 * @param {{start: number, end: number}[]} intervals - intervals to merge (any order).
 * @returns {{ length: number, merged: {start: number, end: number}[] }} covered time and merged intervals.
 */
export function unionIntervals(intervals) {
  if (intervals.length === 0) return { length: 0, merged: [] };
  const sorted = [...intervals].sort((left, right) => left.start - right.start);
  const merged = [];
  let current = { start: sorted[0].start, end: sorted[0].end };
  for (const interval of sorted.slice(1)) {
    if (interval.start <= current.end) {
      current.end = Math.max(current.end, interval.end);
      continue;
    }
    merged.push(current);
    current = { start: interval.start, end: interval.end };
  }
  merged.push(current);
  return { length: merged.reduce((sum, interval) => sum + (interval.end - interval.start), 0), merged };
}

/**
 * Time covered by both interval sets — how much of `a` is hidden under `b`.
 *
 * Both sides are merged first. The classic two-pointer sweep over raw interval
 * lists double counts as soon as either list contains overlapping intervals
 * (a hundred mutually overlapping host operators against one long kernel would
 * report more overlap than the host was ever busy for), so merging is not an
 * optimisation here, it is a correctness requirement.
 *
 * @param {{start: number, end: number}[]} a - first intervals.
 * @param {{start: number, end: number}[]} b - second intervals.
 * @returns {number} overlap length.
 */
export function overlapLength(a, b) {
  if (a.length === 0 || b.length === 0) return 0;
  const left = unionIntervals(a).merged;
  const right = unionIntervals(b).merged;
  let i = 0;
  let j = 0;
  let total = 0;
  while (i < left.length && j < right.length) {
    const start = Math.max(left[i].start, right[j].start);
    const end = Math.min(left[i].end, right[j].end);
    if (end > start) total += end - start;
    if (left[i].end < right[j].end) i += 1;
    else j += 1;
  }
  return total;
}

/**
 * Find uncovered spans inside a window given a set of busy intervals.
 *
 * @param {{start: number, end: number}[]} busy - busy intervals.
 * @param {number} windowStart - window start.
 * @param {number} windowEnd - window end.
 * @param {number} [minLength] - discard gaps shorter than this.
 * @returns {{start: number, end: number, length: number}[]} idle gaps, longest first.
 */
export function findGaps(busy, windowStart, windowEnd, minLength = 0) {
  const { merged } = unionIntervals(busy);
  const gaps = [];
  let cursor = windowStart;
  for (const interval of merged) {
    if (interval.start - cursor > minLength) {
      gaps.push({ start: cursor, end: interval.start, length: interval.start - cursor });
    }
    cursor = Math.max(cursor, interval.end);
  }
  if (windowEnd - cursor > minLength) gaps.push({ start: cursor, end: windowEnd, length: windowEnd - cursor });
  return gaps.sort((left, right) => right.length - left.length);
}

/**
 * Percentile of a numeric list (nearest-rank, no interpolation surprises).
 * @param {number[]} values - samples.
 * @param {number} percentile - 0–1.
 * @returns {number} percentile value, 0 for an empty list.
 */
export function percentile(values, percentile) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const rank = Math.min(sorted.length - 1, Math.max(0, Math.ceil(percentile * sorted.length) - 1));
  return sorted[rank];
}

/**
 * Arithmetic mean.
 * @param {number[]} values - samples.
 * @returns {number} mean, 0 for an empty list.
 */
export function mean(values) {
  if (values.length === 0) return 0;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/**
 * Group values by a key function.
 * @template T
 * @param {T[]} values - items.
 * @param {(item: T) => string} keyOf - key extractor.
 * @returns {Map<string, T[]>} grouped items.
 */
export function groupBy(values, keyOf) {
  const map = new Map();
  for (const value of values) {
    const key = keyOf(value);
    const bucket = map.get(key);
    if (bucket === undefined) map.set(key, [value]);
    else bucket.push(value);
  }
  return map;
}

/**
 * Safe percentage.
 * @param {number} part - numerator.
 * @param {number} whole - denominator.
 * @returns {number} percentage in 0–100, or 0 when the denominator is 0.
 */
export function percentOf(part, whole) {
  if (!Number.isFinite(part) || !Number.isFinite(whole) || whole <= 0) return 0;
  return (part / whole) * 100;
}

/**
 * Round to a fixed number of decimals, keeping the value numeric.
 * @param {number} value - value.
 * @param {number} [digits] - decimals.
 * @returns {number} rounded value.
 */
export function round(value, digits = 2) {
  if (!Number.isFinite(value)) return 0;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

/**
 * One-dimensional two-means split used for the prefill/decode duration split.
 *
 * Values are compared in log space so a long prefill step and a short decode
 * step separate even when their ratio is large. The split is accepted only when
 * the two clusters are far enough apart; otherwise `undefined` is returned and
 * the caller keeps the phase undetermined instead of inventing one.
 *
 * @param {number[]} values - positive durations.
 * @param {number} [minSeparation] - minimum log10 ratio between centroids.
 * @returns {{ threshold: number, lowMean: number, highMean: number, separation: number }|undefined} split.
 */
export function bimodalSplit(values, minSeparation = 0.35) {
  const positive = values.filter((value) => value > 0);
  if (positive.length < 4) return undefined;
  const logs = positive.map((value) => Math.log10(value));
  let low = Math.min(...logs);
  let high = Math.max(...logs);
  if (high - low < minSeparation) return undefined;
  let lowCentroid = low;
  let highCentroid = high;
  for (let iteration = 0; iteration < 64; iteration += 1) {
    const lowGroup = [];
    const highGroup = [];
    for (const value of logs) {
      const midpoint = (lowCentroid + highCentroid) / 2;
      (value <= midpoint ? lowGroup : highGroup).push(value);
    }
    if (lowGroup.length === 0 || highGroup.length === 0) return undefined;
    const nextLow = lowGroup.reduce((sum, value) => sum + value, 0) / lowGroup.length;
    const nextHigh = highGroup.reduce((sum, value) => sum + value, 0) / highGroup.length;
    if (Math.abs(nextLow - lowCentroid) < 1e-6 && Math.abs(nextHigh - highCentroid) < 1e-6) {
      lowCentroid = nextLow;
      highCentroid = nextHigh;
      break;
    }
    lowCentroid = nextLow;
    highCentroid = nextHigh;
  }
  const separation = highCentroid - lowCentroid;
  if (separation < minSeparation) return undefined;
  return {
    threshold: 10 ** ((lowCentroid + highCentroid) / 2),
    lowMean: 10 ** lowCentroid,
    highMean: 10 ** highCentroid,
    separation,
  };
}
