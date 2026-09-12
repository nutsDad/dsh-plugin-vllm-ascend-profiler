/**
 * Chrome-trace parsing for Ascend / torch_npu profiling output.
 *
 * Handles the shapes `trace_view.json` actually takes in the wild:
 *
 * * `{"traceEvents": [...]}` (the torch_npu / Ascend export) and bare arrays;
 * * `ph: "X"` complete events, `ph: "B"/"E"` begin/end pairs, `ph: "i"` instant
 *   markers (mstx ranges), `ph: "M"` metadata (`process_name`, `thread_name`),
 *   `ph: "C"` counters and flow events, which are counted but produce no lanes;
 * * Host (CPU) versus Device (NPU) separation from process metadata, process
 *   ids, or the `device id` / `stream id` argument pair;
 * * adaptive stride sampling, so a multi-GB trace cannot exhaust memory, with
 *   communication, copy, and long-running operators always retained.
 *
 * Time unit: chrome traces are conventionally microseconds, and torch_npu
 * follows that convention, but some exporters write milliseconds while still
 * declaring `displayTimeUnit: "ms"`. The unit is therefore inferred from the
 * data (provisionally after the first {@link UNIT_SAMPLE_SIZE} durations, then
 * finalized over the whole stream) and can be pinned through configuration.
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/trace
 */

import { looksImportantOperator, normalizeOperatorName } from '../model/classify.js';
import { scanJsonText } from './jsonstream.js';

/** Durations inspected before a provisional time unit is fixed. */
export const UNIT_SAMPLE_SIZE = 2000;

/** Argument keys that carry input shapes, in the spellings CANN emits. */
const INPUT_SHAPE_KEYS = [
  'input dims', 'input_dims', 'input shapes', 'input_shapes', 'inputshape', 'input shape',
  'args.input dims', 'args.input shapes',
];
/** Argument keys that carry output shapes. */
const OUTPUT_SHAPE_KEYS = ['output shapes', 'output_shapes', 'outputshape', 'output shape', 'args.output shapes'];
/** Argument keys that carry a host call stack. */
const CALL_STACK_KEYS = ['call stack', 'call_stack', 'callstack', 'stack', 'args.call stack'];
/** Argument keys that carry the operator type. */
const OP_TYPE_KEYS = ['op type', 'op_type', 'optype', 'operator type', 'args.op type'];
/** Argument keys that carry the task type. */
const TASK_TYPE_KEYS = ['task type', 'task_type', 'tasktype', 'args.task type'];
/** Argument keys that carry the AI core type. */
const CORE_TYPE_KEYS = ['core type', 'core_type', 'coretype'];
/** Argument keys that carry FLOPs. */
const FLOPS_KEYS = ['flops', 'flop', 'args.flops'];
/** Argument keys that carry a device-side wait time. */
const WAIT_KEYS = ['wait time', 'wait_time', 'wait time(us)', 'args.wait time'];
/** Argument keys that carry the device id. */
const DEVICE_ID_KEYS = ['device id', 'device_id', 'deviceid'];
/** Argument keys that carry the stream id. */
const STREAM_ID_KEYS = ['stream id', 'stream_id', 'streamid'];
/** Argument keys that carry the rank. */
const RANK_KEYS = ['rank', 'rank id', 'rank_id', 'args.rank'];
/** Argument keys that carry an mstx message or domain. */
const MESSAGE_KEYS = ['message', 'msg', 'domain', 'args.message'];
/** Argument keys that carry per-step token counts. */
const TOKEN_KEYS = ['num_tokens', 'num tokens', 'num_prompt_tokens', 'num_generated_tokens', 'batch_size'];

/**
 * Process-name patterns that identify the device side.
 *
 * torch_npu labels the device lane through `process_name` metadata: the
 * `Ascend Hardware` process carries kernel/communication/memcpy events, while
 * `Python` and `CANN` are host processes. This is the authoritative signal —
 * `pid` itself is heterogeneous (OS pids, packed integers, or the literal
 * string `"HCCL"`), so it is never used as the primary discriminator.
 */
const DEVICE_LABEL = /ascend hardware|ascend_hardware|aicore|ai core|ai_core|aicpu|ai_cpu|device|npu/i;
/** Process-name patterns that identify the host side. */
const HOST_LABEL = /^(?:python|cann|communication|hccl|overlap analysis|python gc|gc|host|cpu|main|worker|dequeue|enqueue|fwdbwd)/i;

/**
 * @typedef {object} RawTraceEvent
 * @property {string} name - event name as exported.
 * @property {string} normalizedName - name with instance suffixes and scope prefixes removed.
 * @property {number} tsUs - start timestamp in microseconds.
 * @property {number} durUs - duration in microseconds.
 * @property {'host'|'device'} device - owning side.
 * @property {string} lane - lane key used by the swimlane view.
 * @property {string} pid
 * @property {string} tid
 * @property {string|undefined} pidLabel
 * @property {string|undefined} tidLabel
 * @property {string|undefined} stream
 * @property {string|undefined} rank
 * @property {string|undefined} opType
 * @property {string|undefined} taskType
 * @property {string|undefined} coreType
 * @property {string|undefined} cat
 * @property {string|undefined} message
 * @property {number|undefined} flops
 * @property {number|undefined} waitUs
 * @property {unknown} shapesIn
 * @property {unknown} shapesOut
 * @property {unknown} callStack
 * @property {Record<string, unknown>} args
 */

/**
 * Collects trace events from a streamed chrome trace.
 */
export class TraceCollector {
  /**
   * @param {object} [options] - collector options.
   * @param {number} [options.maxEvents] - retained-event budget; sampling engages past it.
   * @param {number} [options.hardCap] - absolute ceiling on retained events.
   * @param {'auto'|'us'|'ms'} [options.timeUnit] - pin the timestamp unit.
   * @param {number} [options.longEventUs] - duration above which events are always kept, in microseconds.
   */
  constructor({ maxEvents = 400000, hardCap = 1200000, timeUnit = 'auto', longEventUs = 20000 } = {}) {
    this.maxEvents = Math.max(1000, maxEvents);
    this.hardCap = Math.max(this.maxEvents, hardCap);
    this.timeUnitOption = timeUnit;
    this.longEventUs = longEventUs;

    /** @type {RawTraceEvent[]} */
    this.events = [];
    this.seen = 0;
    this.stride = 1;
    this.truncated = false;
    this.droppedApprox = 0;
    this.timeUnit = timeUnit === 'auto' ? 'us' : timeUnit;
    this.unitResolved = timeUnit !== 'auto';
    this.timeUnitEvidence = [];
    this.warnings = [];
    this.stats = {
      total: 0,
      complete: 0,
      beginEnd: 0,
      instant: 0,
      counter: 0,
      flow: 0,
      metadata: 0,
      other: 0,
      unmatchedBegin: 0,
      unmatchedEnd: 0,
    };
    this.processNames = new Map();
    this.threadNames = new Map();
    this.processLabels = new Map();
    this.devicePids = new Set();
    this.hostPids = new Set();
    this.openBegin = new Map();
    this.displayTimeUnit = undefined;
    this.traceVersion = undefined;
    this.spanRaw = { min: Infinity, max: -Infinity };
    this.durationSamples = [];
    this.tokenCounts = [];
  }

  /**
   * Feed one parsed trace event or metadata record.
   * @param {unknown} value - a `traceEvents` element.
   */
  accept(value) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return;
    const record = /** @type {Record<string, unknown>} */ (value);
    this.stats.total += 1;
    const phase = typeof record.ph === 'string' ? record.ph : undefined;
    if (phase === 'M') {
      this.stats.metadata += 1;
      this.#acceptMetadata(record);
      return;
    }
    if (phase === 'C') {
      this.stats.counter += 1;
      return;
    }
    if (phase === 's' || phase === 'f' || phase === 't') {
      this.stats.flow += 1;
      return;
    }
    if (phase === 'B') {
      this.stats.beginEnd += 1;
      this.#acceptBegin(record);
      return;
    }
    if (phase === 'E') {
      this.stats.beginEnd += 1;
      this.#acceptEnd(record);
      return;
    }
    if (phase === 'X') this.stats.complete += 1;
    else if (phase === 'i') this.stats.instant += 1;
    else if (phase === undefined) this.stats.other += 1;
    else {
      this.stats.other += 1;
      return;
    }
    const ts = toNumber(record.ts);
    if (ts === undefined) return;
    const dur = toNumber(record.dur) ?? 0;
    this.#record(record, ts, dur);
  }

  /**
   * Read document-level fields (`displayTimeUnit`, `version`) and per-step token counts.
   * @param {unknown} value - a non-array top-level object.
   */
  acceptDocument(value) {
    if (typeof value !== 'object' || value === null || Array.isArray(value)) return;
    const record = /** @type {Record<string, unknown>} */ (value);
    if (typeof record.displayTimeUnit === 'string') this.displayTimeUnit = record.displayTimeUnit;
    if (typeof record.version === 'string') this.traceVersion = record.version;
    else if (typeof record.version === 'number') this.traceVersion = String(record.version);
    const tokens = firstNumeric(record, ['num_tokens', 'numTokens', 'batch_size', 'batchSize']);
    if (tokens !== undefined) this.tokenCounts.push(tokens);
  }

  /** Flush unmatched begin events and finalize the time unit. */
  finish() {
    if (this.openBegin.size > 0) {
      this.stats.unmatchedBegin += this.openBegin.size;
      this.warnings.push(`${String(this.openBegin.size)} 个 "ph":"B" 事件没有配对的 "E"，已丢弃。`);
      this.openBegin.clear();
    }
    this.#finalizeTimeUnit();
    return this;
  }

  /** @returns {{ events: RawTraceEvent[], meta: object, stats: object, sampling: object, warnings: string[] }} collected result. */
  result() {
    return {
      events: this.events,
      meta: {
        timeUnit: this.timeUnit,
        timeUnitEvidence: this.timeUnitEvidence,
        displayTimeUnit: this.displayTimeUnit,
        traceVersion: this.traceVersion,
        processNames: Object.fromEntries(this.processNames),
        threadNames: Object.fromEntries(this.threadNames),
        processLabels: Object.fromEntries(this.processLabels),
        devicePids: [...this.devicePids],
        hostPids: [...this.hostPids],
        spanRaw: Number.isFinite(this.spanRaw.min)
          ? { min: this.spanRaw.min, max: this.spanRaw.max }
          : undefined,
        tokenCounts: this.tokenCounts.slice(0, 10),
      },
      stats: this.stats,
      sampling: {
        applied: this.stride > 1 || this.truncated,
        strategy: this.stride > 1
          ? '自适应等距采样（通信/拷贝/长耗时算子全量保留）'
          : '全量保留',
        stride: this.stride,
        seen: this.seen,
        kept: this.events.length,
        droppedApprox: this.droppedApprox,
        truncated: this.truncated,
        budget: this.maxEvents,
      },
      warnings: this.warnings,
    };
  }

  // ── internals ────────────────────────────────────────────────────────────

  #acceptMetadata(record) {
    const name = String(record.name ?? '');
    const pid = String(record.pid ?? '');
    const tid = String(record.tid ?? '');
    const args = asRecord(record.args);
    if (name === 'process_name') {
      const label = String(args?.name ?? '');
      this.processNames.set(pid, label);
      this.#classifyProcess(pid, label);
    } else if (name === 'thread_name') {
      this.threadNames.set(`${pid}:${tid}`, String(args?.name ?? ''));
    } else if (name === 'process_labels') {
      const label = String(args?.labels ?? args?.name ?? '');
      this.processLabels.set(pid, label);
      this.#classifyProcess(pid, label);
    }
  }

  #classifyProcess(pid, label) {
    if (label === '') return;
    if (DEVICE_LABEL.test(label)) {
      this.devicePids.add(pid);
      this.hostPids.delete(pid);
      return;
    }
    if (HOST_LABEL.test(label)) {
      this.hostPids.add(pid);
      this.devicePids.delete(pid);
    }
  }

  #acceptBegin(record) {
    const ts = toNumber(record.ts);
    if (ts === undefined) return;
    const key = beginKey(record);
    this.openBegin.set(key, { record, ts });
  }

  #acceptEnd(record) {
    const key = beginKey(record);
    const open = this.openBegin.get(key);
    if (open === undefined) {
      this.stats.unmatchedEnd += 1;
      return;
    }
    this.openBegin.delete(key);
    const ts = toNumber(record.ts);
    if (ts === undefined) return;
    this.#record(open.record, open.ts, Math.max(0, ts - open.ts));
  }

  #record(record, ts, dur) {
    this.seen += 1;
    this.spanRaw.min = Math.min(this.spanRaw.min, ts);
    this.spanRaw.max = Math.max(this.spanRaw.max, ts + dur);
    // Unit detection must cost O(1) per event. The provisional decision is taken
    // once, after UNIT_SAMPLE_SIZE durations, and is final for the stream; a
    // sparse reservoir afterwards gives the final report more evidence without
    // re-running the decision on every event (which would be quadratic).
    if (!this.unitResolved) {
      this.durationSamples.push(dur);
      if (this.durationSamples.length >= UNIT_SAMPLE_SIZE) this.#resolveProvisionalUnit();
    } else if (this.durationSamples.length < UNIT_SAMPLE_SIZE * 10 && this.seen % 50 === 0) {
      this.durationSamples.push(dur);
    }

    const name = String(record.name ?? '');
    const important = looksImportantOperator(name);
    this.stride = Math.max(1, Math.ceil(this.seen / this.maxEvents));
    const durUs = this.timeUnit === 'ms' ? dur * 1000 : dur;
    if (!important && this.seen % this.stride !== 0 && durUs < this.longEventUs) {
      this.droppedApprox += this.stride;
      return;
    }
    if (this.events.length >= this.hardCap) {
      this.droppedApprox += this.stride;
      this.truncated = true;
      return;
    }

    const rawArgs = asRecord(record.args) ?? {};
    const args = flattenArgs(rawArgs);
    const device = this.#deviceOf(record, args);
    const tid = String(record.tid ?? '');
    const stream = pickString(args, STREAM_ID_KEYS) ?? (device === 'device' ? tid : undefined);
    const pidLabel = this.processNames.get(String(record.pid ?? ''));
    const tidLabel = this.threadNames.get(`${String(record.pid ?? '')}:${tid}`);

    this.events.push({
      name,
      normalizedName: normalizeOperatorName(name),
      tsUs: ts,
      durUs,
      device,
      lane: device === 'device' ? `device:${stream ?? tid}` : `host:${tidLabel ?? tid}`,
      pid: String(record.pid ?? ''),
      tid,
      pidLabel,
      tidLabel,
      stream,
      rank: pickString(args, RANK_KEYS),
      opType: pickString(args, OP_TYPE_KEYS),
      taskType: pickString(args, TASK_TYPE_KEYS),
      coreType: pickString(args, CORE_TYPE_KEYS),
      cat: typeof record.cat === 'string' ? record.cat : undefined,
      message: pickString(args, MESSAGE_KEYS),
      flops: pickNumber(args, FLOPS_KEYS),
      waitUs: pickNumber(args, WAIT_KEYS),
      tokenCount: pickNumber(args, TOKEN_KEYS),
      shapesIn: pickValue(args, INPUT_SHAPE_KEYS),
      shapesOut: pickValue(args, OUTPUT_SHAPE_KEYS),
      callStack: pickValue(args, CALL_STACK_KEYS),
      args,
    });
  }

  #deviceOf(record, args) {
    const pid = String(record.pid ?? '');
    if (this.devicePids.has(pid)) return 'device';
    if (this.hostPids.has(pid)) return 'host';
    const pidLabel = this.processNames.get(pid) ?? this.processLabels.get(pid) ?? '';
    if (DEVICE_LABEL.test(pidLabel) || DEVICE_LABEL.test(pid)) return 'device';
    if (HOST_LABEL.test(pidLabel) || HOST_LABEL.test(pid)) return 'host';
    if (pickValue(args, DEVICE_ID_KEYS) !== undefined || pickValue(args, STREAM_ID_KEYS) !== undefined) return 'device';
    const cat = String(record.cat ?? '');
    if (/kernel|hccl|ascend|acl|npu|mstx/i.test(cat)) return 'device';
    if (/python|cpu_op|host|operator|user_annotation|ac2g/i.test(cat)) return 'host';
    return 'host';
  }

  /**
   * Take the provisional time-unit decision once the sample window is full.
   *
   * The decision is final for the rest of the stream whether or not it was
   * conclusive: leaving `unitResolved` false when the evidence is ambiguous
   * would re-run the decision on every later event.
   */
  #resolveProvisionalUnit() {
    this.unitResolved = true;
    const decision = decideUnit(this.durationSamples, this.spanRaw);
    if (decision.unit === undefined || decision.unit === this.timeUnit) return;
    this.timeUnit = decision.unit;
    this.timeUnitEvidence.push(`前 ${String(this.durationSamples.length)} 个事件推断时间单位：${decision.reason}`);
  }

  #finalizeTimeUnit() {
    if (this.timeUnitOption !== 'auto') {
      this.timeUnitEvidence.push(`时间单位由插件配置指定为 ${this.timeUnitOption}`);
      if (this.timeUnitOption === 'ms') this.#rescaleToMilliseconds();
      return;
    }
    const decision = decideUnit(this.durationSamples, this.spanRaw, true);
    const unit = decision.unit ?? 'us';
    if (unit === 'ms' && this.timeUnit !== 'ms') this.#rescaleToMilliseconds();
    this.timeUnit = unit;
    this.timeUnitEvidence.push(decision.reason);
  }

  /** Convert already-collected microsecond values back to a millisecond source. */
  #rescaleToMilliseconds() {
    for (const event of this.events) {
      event.tsUs *= 1000;
      event.durUs *= 1000;
    }
    if (Number.isFinite(this.spanRaw.min)) {
      this.spanRaw.min *= 1000;
      this.spanRaw.max *= 1000;
    }
  }
}

/**
 * Decide the timestamp unit from observed durations and span.
 *
 * @param {number[]} samples - raw `dur` values.
 * @param {{ min: number, max: number }} span - raw timestamp span.
 * @param {boolean} [final] - whether this is the final decision (allows the span rule).
 * @returns {{ unit: 'us'|'ms'|undefined, reason: string }} decision.
 */
export function decideUnit(samples, span, final = false) {
  const median = medianOf(samples);
  const positive = samples.filter((value) => value > 0);
  const subMicro = positive.length === 0
    ? 0
    : positive.filter((value) => value < 0.05).length / positive.length;
  const rawSpan = Number.isFinite(span.min) ? span.max - span.min : 0;
  const microSpanImplausible = rawSpan > 6 * 3600 * 1e6;
  if (subMicro > 0.3) {
    return {
      unit: 'ms',
      reason: `${(subMicro * 100).toFixed(0)}% 的事件时长小于 0.05，毫秒单位下才合理（中位数 ${median.toFixed(3)}）`,
    };
  }
  if (microSpanImplausible) {
    return {
      unit: 'ms',
      reason: `按微秒解释时间窗为 ${(rawSpan / 1e6).toFixed(1)}s，超出合理范围，改按毫秒解释`,
    };
  }
  if (final) {
    return {
      unit: 'us',
      reason: `时间窗 ${(rawSpan / 1000).toFixed(2)}ms、时长中位数 ${median.toFixed(2)}µs，微秒单位成立`,
    };
  }
  return { unit: undefined, reason: '' };
}

/**
 * Parse an in-memory trace text.
 * @param {string} text - decoded trace JSON.
 * @param {object} [options] - collector options plus `arrayKeys` and `maxElements`.
 * @returns {{ events: RawTraceEvent[], meta: object, stats: object, sampling: object, warnings: string[], scan: object }} parsed trace.
 */
export function parseTraceText(text, options = {}) {
  const collector = new TraceCollector(options);
  const scan = scanTraceInto(text, collector, {
    arrayKeys: options.arrayKeys ?? ['traceEvents'],
    maxElements: options.maxElements,
  });
  if (scan.truncated) collector.warnings.push(`trace 事件扫描提前结束：${String(scan.reason ?? '达到解析预算')}`);
  if (scan.error !== undefined) collector.warnings.push(scan.error);
  // torch_npu writes the trace array incrementally and only closes it when the
  // export finishes, so a killed job leaves a file with no closing bracket.
  // A streaming parse keeps every complete element, and this says so.
  const tail = text.slice(-64).trimEnd();
  if (tail !== '' && !/[\]}]$/.test(tail)) {
    collector.warnings.push('trace 文件末尾没有收尾括号，文件可能在导出过程中被截断；已解析到最后一个完整事件。');
  }
  const result = collector.finish().result();
  return { ...result, scan };
}

/**
 * Whether a parsed JSON object is a trace event rather than document metadata.
 *
 * Ascend exports a bare top-level array, so an element whose array key is
 * unknown must be routed by shape: real events carry `ph`, or a numeric/string
 * `ts` together with `name`/`pid`. Anything else (a document object that held
 * `traceEvents`, `displayTimeUnit`, `version`) is metadata.
 *
 * @param {unknown} value - parsed JSON value.
 * @returns {boolean} whether the value is a trace event.
 */
export function isTraceEvent(value) {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false;
  const record = /** @type {Record<string, unknown>} */ (value);
  if (typeof record.ph === 'string') return true;
  const hasTime = typeof record.ts === 'number' || (typeof record.ts === 'string' && record.ts.trim() !== '');
  if (!hasTime) return false;
  return record.name !== undefined || record.pid !== undefined;
}

/**
 * Stream a trace document into a collector.
 *
 * @param {string} text - decoded JSON text (whole document or a chunk).
 * @param {TraceCollector} collector - event sink.
 * @param {object} [options] - scan options.
 * @param {string[]} [options.arrayKeys] - array keys treated as event lists.
 * @param {number} [options.maxElements] - element budget for this call.
 * @returns {{ elements: number, delivered: number, truncated: boolean, reason: string|undefined, error: string|undefined, scanError?: string }} scan statistics.
 */
export function scanTraceInto(text, collector, options = {}) {
  const arrayKeys = options.arrayKeys ?? ['traceEvents'];
  const route = (value, key) => {
    if (Array.isArray(value)) {
      for (const item of value) collector.accept(item);
      return;
    }
    if (isTraceEvent(value)) {
      collector.accept(value);
      return;
    }
    // Only a document-level object is metadata; a stray scalar is ignored.
    if (key === undefined && typeof value === 'object' && value !== null) collector.acceptDocument(value);
  };
  try {
    const stats = scanJsonText(text, {
      arrayKeys,
      maxElements: options.maxElements,
      onValue: (value, meta) => route(value, meta.key),
    });
    return stats;
  } catch (error) {
    return {
      elements: 0,
      delivered: 0,
      truncated: true,
      reason: 'JSON 解析异常',
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/** Stable key for pairing `ph:"B"` with `ph:"E"`. */
function beginKey(record) {
  return `${String(record.pid)}:${String(record.tid)}:${String(record.name)}`;
}

/**
 * Normalize a nested `args` object into a lowercase lookup map.
 *
 * torch_npu nests user arguments under `Args`, so that one level is flattened
 * with an `args.` prefix.
 *
 * @param {Record<string, unknown>} raw - raw event args.
 * @returns {Record<string, unknown>} flattened lookup map.
 */
function flattenArgs(raw) {
  const flat = {};
  for (const [key, value] of Object.entries(raw)) flat[key.toLowerCase()] = value;
  const nested = raw.Args ?? raw.args;
  if (typeof nested === 'object' && nested !== null && !Array.isArray(nested)) {
    for (const [key, value] of Object.entries(nested)) flat[`args.${key.toLowerCase()}`] = value;
  }
  return flat;
}

function asRecord(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? /** @type {Record<string, unknown>} */ (value)
    : undefined;
}

function toNumber(value) {
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined;
  if (typeof value === 'string' && value.trim() !== '') {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function pickValue(args, keys) {
  for (const key of keys) {
    const value = args[key];
    if (value !== undefined && value !== null && value !== '') return value;
  }
  return undefined;
}

function pickString(args, keys) {
  const value = pickValue(args, keys);
  if (value === undefined) return undefined;
  if (typeof value === 'string') return value;
  if (typeof value === 'number' || typeof value === 'boolean') return String(value);
  return JSON.stringify(value);
}

function pickNumber(args, keys) {
  const value = pickValue(args, keys);
  if (value === undefined) return undefined;
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined;
  return toNumber(String(value));
}

function firstNumeric(record, keys) {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) return value;
  }
  return undefined;
}

function medianOf(values) {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}
