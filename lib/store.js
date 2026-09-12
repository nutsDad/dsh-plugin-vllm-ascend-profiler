/**
 * Job and dataset store.
 *
 * Uploads are parsed asynchronously so the page can show progress, which means
 * parse state must live somewhere between requests. It lives here: a small
 * bounded in-memory table with an idle TTL, newest-first eviction, and an
 * explicit hard cap on how many datasets are retained.
 *
 * Nothing is written to disk. A profiling upload is parsed, the parsed model is
 * kept in memory for as long as someone is looking at it, and the bytes are
 * dropped as soon as parsing finishes — which is also why a huge trace should be
 * ingested by path rather than uploaded.
 *
 * @module dsh-plugin-vllm-ascend-profiler/store
 */

import { randomUUID } from 'node:crypto';

/** Store defaults. */
export const STORE_DEFAULTS = Object.freeze({
  maxDatasets: 6,
  datasetTtlMs: 60 * 60 * 1000,
  maxJobs: 16,
  jobTtlMs: 15 * 60 * 1000,
});

export class JobStore {
  /**
   * @param {object} [options] - store limits.
   * @param {number} [options.maxDatasets] - parsed datasets retained.
   * @param {number} [options.datasetTtlMs] - idle lifetime of a dataset.
   * @param {number} [options.maxJobs] - concurrent/recent jobs retained.
   * @param {number} [options.jobTtlMs] - idle lifetime of a finished job.
   */
  constructor(options = {}) {
    this.config = { ...STORE_DEFAULTS, ...options };
    /** @type {Map<string, object>} */
    this.jobs = new Map();
    /** @type {Map<string, object>} */
    this.datasets = new Map();
  }

  /**
   * Create a job in the `queued` state.
   * @param {object} init - initial fields (`label`, `inputs`).
   * @returns {object} the created job record.
   */
  createJob(init = {}) {
    const job = {
      id: randomUUID(),
      state: 'queued',
      progress: 0,
      phase: 'queued',
      detail: '等待开始',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      warnings: [],
      errors: [],
      datasetId: undefined,
      ...init,
    };
    this.jobs.set(job.id, job);
    this.#evictJobs();
    return job;
  }

  /**
   * Update a job.
   * @param {string} id - job id.
   * @param {object} patch - fields to merge.
   * @returns {object|undefined} the updated job.
   */
  updateJob(id, patch) {
    const job = this.jobs.get(id);
    if (job === undefined) return undefined;
    Object.assign(job, patch, { updatedAt: Date.now() });
    return job;
  }

  /**
   * Read a job.
   * @param {string} id - job id.
   * @returns {object|undefined} the job.
   */
  getJob(id) {
    return this.jobs.get(id);
  }

  /**
   * Store a parsed dataset together with its analysis.
   * @param {object} entry - `{ label, dataset, analysis, options, files }`.
   * @returns {object} the stored record.
   */
  putDataset(entry) {
    const record = {
      id: randomUUID(),
      createdAt: Date.now(),
      lastAccess: Date.now(),
      ...entry,
    };
    this.datasets.set(record.id, record);
    this.#evictDatasets();
    return record;
  }

  /**
   * Read a dataset and mark it as recently used.
   * @param {string} id - dataset id.
   * @returns {object|undefined} the record.
   */
  getDataset(id) {
    const record = this.datasets.get(id);
    if (record === undefined) return undefined;
    record.lastAccess = Date.now();
    return record;
  }

  /**
   * Replace the analysis of a stored dataset (used by the re-analyze endpoint).
   * @param {string} id - dataset id.
   * @param {object} analysis - new analysis.
   * @returns {object|undefined} the record.
   */
  setAnalysis(id, analysis) {
    const record = this.getDataset(id);
    if (record === undefined) return undefined;
    record.analysis = analysis;
    record.lastAccess = Date.now();
    return record;
  }

  /**
   * Remove a dataset.
   * @param {string} id - dataset id.
   * @returns {boolean} whether it existed.
   */
  deleteDataset(id) {
    return this.datasets.delete(id);
  }

  /** @returns {object[]} dataset summaries, newest first. */
  listDatasets() {
    return [...this.datasets.values()]
      .sort((left, right) => right.createdAt - left.createdAt)
      .map((record) => ({
        id: record.id,
        label: record.label,
        createdAt: record.createdAt,
        files: record.dataset?.meta?.files?.map((file) => file.name) ?? [],
        eventCount: record.dataset?.meta?.eventCount ?? 0,
        bottleneck: record.analysis?.bottleneck?.short,
        windowMs: record.dataset?.meta?.wallUs === undefined ? undefined : record.dataset.meta.wallUs / 1000,
      }));
  }

  /** Drop expired datasets and jobs; safe to call on every request. */
  sweep() {
    const now = Date.now();
    for (const [id, record] of this.datasets) {
      if (now - record.lastAccess > this.config.datasetTtlMs) this.datasets.delete(id);
    }
    for (const [id, job] of this.jobs) {
      if (now - job.updatedAt > this.config.jobTtlMs) this.jobs.delete(id);
    }
    this.#evictDatasets();
    this.#evictJobs();
  }

  /** @returns {object} store statistics for the health endpoint. */
  stats() {
    return {
      datasets: this.datasets.size,
      jobs: this.jobs.size,
      maxDatasets: this.config.maxDatasets,
      datasetTtlMs: this.config.datasetTtlMs,
    };
  }

  #evictDatasets() {
    while (this.datasets.size > this.config.maxDatasets) {
      const oldest = [...this.datasets.values()].sort((left, right) => left.lastAccess - right.lastAccess)[0];
      if (oldest === undefined) break;
      this.datasets.delete(oldest.id);
    }
  }

  #evictJobs() {
    while (this.jobs.size > this.config.maxJobs) {
      const oldest = [...this.jobs.values()].sort((left, right) => left.updatedAt - right.updatedAt)[0];
      if (oldest === undefined) break;
      this.jobs.delete(oldest.id);
    }
  }
}
