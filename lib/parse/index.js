/**
 * Parse orchestration: turn uploaded artifacts into raw, validated profiling
 * collections.
 *
 * Flow: expand archives → sniff every member → validate the set → parse each
 * artifact according to its kind → hand back the raw pieces (trace events,
 * tables, auxiliary JSON) plus everything the UI must disclose about the parse
 * (sampling, provenance, warnings, encoding).
 *
 * Coordinate systems are deliberately kept apart: trace events carry timestamps
 * relative to the profiling session, while CSV `Start Time` columns carry
 * absolute device timestamps. They are never mixed on one axis; the timeline
 * module is built from trace events (or, when no trace exists, from CSV
 * instance rows rebased to their own minimum), and the cost-share module uses
 * durations only.
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse
 */

import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { basename, extname, join, relative, resolve, sep } from 'node:path';

import { expandArchive, formatBytes } from './archive.js';
import { INSTANCE_KINDS, parseAscendCsv } from './ascendcsv.js';
import { decodeText } from './csv.js';
import { JsonStreamScanner } from './jsonstream.js';
import { extractProtoEvents } from './protobuf.js';
import { sniffArtifact, validateProfileSet } from './sniff.js';
import { TraceCollector, isTraceEvent, parseTraceText } from './trace.js';

/** Default budgets, overridable through plugin config. */
export const PARSE_DEFAULTS = Object.freeze({
  maxUploadBytes: 2 * 1024 * 1024 * 1024,
  maxInMemoryBytes: 256 * 1024 * 1024,
  maxTimelineEvents: 400000,
  maxTableRows: 400000,
  maxFiles: 64,
  /** Exact protobuf field numbers, when the deployment knows its schema. */
  protoFieldMap: undefined,
});

/**
 * @typedef {object} ParseInput
 * @property {string} name - artifact name (may include a relative path).
 * @property {Buffer|Uint8Array} [buffer] - bytes, when uploaded.
 * @property {string} [path] - absolute path, when ingested from disk.
 * @property {number} [size] - byte size when known.
 */

/**
 * Parse a set of artifacts.
 *
 * @param {object} request - parse request.
 * @param {ParseInput[]} request.inputs - artifacts to parse.
 * @param {object} [request.options] - budgets and overrides.
 * @param {(progress: {phase: string, percent: number, detail: string}) => void} [request.onProgress] - progress sink.
 * @returns {Promise<object>} parse result.
 */
export async function parseProfileSet({ inputs, options = {}, onProgress }) {
  const budget = { ...PARSE_DEFAULTS, ...options };
  const report = (phase, percent, detail) => onProgress?.({ phase, percent, detail });
  const warnings = [];
  const errors = [];

  report('inspect', 2, `收到 ${String(inputs.length)} 个输入`);
  const expanded = await expandInputs(inputs, budget, warnings, report);
  if (expanded.length === 0) {
    return {
      ok: false,
      errors: ['没有可解析的文件。请上传 vLLM-Ascend profiling 产物（trace_view.json / *.csv / *_ascend_pt 目录压缩包）。'],
      warnings,
    };
  }

  report('validate', 12, `校验 ${String(expanded.length)} 个文件`);
  const artifacts = expanded.map((file) => sniffArtifact({ name: file.name, buffer: file.buffer }));
  const verdict = validateProfileSet(artifacts);
  warnings.push(...verdict.warnings);
  if (!verdict.ok) return { ok: false, errors: verdict.errors, warnings, validation: verdict };

  const traceFiles = artifacts.filter((artifact) => artifact.kind === 'trace-json');
  const tableFiles = artifacts.filter((artifact) => TABLE_KINDS.has(artifact.kind));
  const auxFiles = artifacts.filter((artifact) => artifact.kind === 'json-generic' || artifact.kind === 'csv-unknown');
  const protoFiles = artifacts.filter((artifact) => artifact.kind === 'proto');

  const tables = [];
  const aux = [];
  const traceResults = [];
  const protoResults = [];
  let step = 0;
  const totalSteps = Math.max(1, traceFiles.length + tableFiles.length + auxFiles.length + protoFiles.length);

  for (let index = 0; index < traceFiles.length; index += 1) {
    const artifact = traceFiles[index];
    const file = expanded[artifacts.indexOf(artifact)];
    step += 1;
    report('parse', 15 + (step / totalSteps) * 65, `解析 trace：${artifact.name}`);
    const result = await parseTraceArtifact(file, budget, report, artifact.name);
    traceResults.push(result);
    warnings.push(...result.warnings.map((warning) => `${artifact.name}：${warning}`));
  }

  for (const artifact of tableFiles) {
    const file = expanded[artifacts.indexOf(artifact)];
    step += 1;
    report('parse', 15 + (step / totalSteps) * 65, `解析表格：${artifact.name}`);
    const table = parseAscendCsv({ name: artifact.name, buffer: file.buffer, maxRows: budget.maxTableRows, kind: artifact.kind });
    tables.push(table);
    warnings.push(...table.warnings.map((warning) => `${artifact.name}：${warning}`));
  }

  for (const artifact of auxFiles) {
    const file = expanded[artifacts.indexOf(artifact)];
    step += 1;
    report('parse', 15 + (step / totalSteps) * 65, `解析附属产物：${artifact.name}`);
    const parsed = parseAuxJson(artifact.name, file.buffer);
    if (parsed !== undefined) aux.push(parsed);
    warnings.push(...(parsed?.warnings ?? []).map((warning) => `${artifact.name}：${warning}`));
  }

  for (const artifact of protoFiles) {
    const file = expanded[artifacts.indexOf(artifact)];
    step += 1;
    report('parse', 15 + (step / totalSteps) * 65, `解析 proto/binary：${artifact.name}`);
    const result = extractProtoEvents({
      name: artifact.name,
      buffer: file.buffer,
      fieldMap: budget.protoFieldMap,
      limit: budget.maxTimelineEvents,
    });
    protoResults.push(result);
    warnings.push(...result.warnings.map((warning) => `${artifact.name}：${warning}`));
  }

  report('assemble', 86, '汇总解析结果');
  const events = [];
  let sampling;
  for (const result of traceResults) {
    for (const event of result.events) {
      if (result.rank !== undefined) event.rank ??= result.rank;
      events.push(event);
    }
    if (result.sampling?.applied === true) sampling = result.sampling;
  }
  // Proto-derived events join the same timeline but keep their low-confidence
  // marker, so the report can disclose that part of it was inferred.
  for (const result of protoResults) {
    for (const event of result.events) {
      event.heuristic = result.confidence === 'low';
      events.push(event);
    }
  }

  const files = artifacts.map((artifact, index) => ({
    name: artifact.name,
    kind: artifact.kind,
    bytes: expanded[index]?.buffer?.length ?? artifact.bytes,
    confidence: artifact.confidence,
    markers: artifact.markers,
    rows: tables.find((table) => table.name === artifact.name)?.rows.length,
    events: traceResults.find((result) => result.file === artifact.name)?.events.length
      ?? protoResults.find((result) => result.file === artifact.name)?.events.length,
  }));

  report('assemble', 94, '解析完成');
  return {
    ok: true,
    validation: verdict,
    files,
    events,
    tables,
    aux,
    sampling,
    warnings: dedupe(warnings),
    errors,
    encodingNotes: tables.filter((table) => table.encoding !== 'utf-8').map((table) => `${table.name}：${table.encoding}`),
    instanceKinds: [...INSTANCE_KINDS],
    proto: protoResults.map((result) => ({
      file: result.file,
      confidence: result.confidence,
      messages: result.messages,
      events: result.events.length,
    })),
  };
}

/** Artifact kinds that map onto an Ascend CSV table. */
const TABLE_KINDS = new Set([
  'op_statistic', 'api_statistic', 'op_summary', 'kernel_details', 'operator_details',
  'step_trace_time', 'communication_statistic', 'pipe_utilization',
]);

/**
 * Expand inputs into member files: archives are unpacked, directory paths are
 * walked, and single large files are streamed rather than buffered.
 *
 * @param {ParseInput[]} inputs - raw inputs.
 * @param {object} budget - size budgets.
 * @param {string[]} warnings - warning sink.
 * @param {(phase: string, percent: number, detail: string) => void} report - progress sink.
 * @returns {Promise<{name: string, buffer: Buffer, path?: string}[]>} member files.
 */
async function expandInputs(inputs, budget, warnings, report) {
  const out = [];
  for (const input of inputs) {
    let buffer = input.buffer;
    let path = input.path;
    if (buffer === undefined && path !== undefined) {
      const info = await stat(path).catch(() => undefined);
      if (info === undefined) {
        warnings.push(`路径不存在：${path}`);
        continue;
      }
      if (info.isDirectory()) {
        const members = await walkDirectory(path, budget, warnings);
        for (const member of members) out.push(member);
        continue;
      }
      if (info.size > budget.maxInMemoryBytes) {
        // Large single files stay on disk and are streamed by the trace parser.
        out.push({ name: basename(path), buffer: Buffer.alloc(0), path, size: info.size });
        continue;
      }
      buffer = await readFile(path);
    }
    if (buffer === undefined) continue;
    if (input.size !== undefined && input.size > budget.maxUploadBytes) {
      warnings.push(`${input.name}：文件 ${formatBytes(input.size)} 超过单文件上限 ${formatBytes(budget.maxUploadBytes)}，已跳过。`);
      continue;
    }
    const name = input.name.replaceAll('\\', '/');
    const expanded = expandArchive({ name, buffer, entryLimit: budget.maxInMemoryBytes });
    warnings.push(...expanded.warnings.map((warning) => `${name}：${warning}`));
    if (expanded.skipped.length > 0) {
      warnings.push(`${name}：跳过了 ${String(expanded.skipped.length)} 个成员（${expanded.skipped.slice(0, 3).join('；')}${expanded.skipped.length > 3 ? ' …' : ''}）。`);
    }
    for (const member of expanded.files) {
      out.push({ name: member.name, buffer: member.buffer });
    }
    report('inspect', 5, `展开 ${name}`);
    void path;
  }
  if (out.length > budget.maxFiles) {
    warnings.push(`文件数量 ${String(out.length)} 超过上限 ${String(budget.maxFiles)}，仅解析前 ${String(budget.maxFiles)} 个。`);
    return out.slice(0, budget.maxFiles);
  }
  return out;
}

/**
 * Walk a profiling directory, choosing the files the analyzer can read.
 * @param {string} root - directory.
 * @param {object} budget - budgets.
 * @param {string[]} warnings - warning sink.
 * @returns {Promise<{name: string, buffer: Buffer, path: string, size: number}[]>} members.
 */
async function walkDirectory(root, budget, warnings) {
  const { readdir } = await import('node:fs/promises');
  const out = [];
  const queue = [root];
  const base = resolve(root);
  while (queue.length > 0 && out.length < budget.maxFiles) {
    const current = queue.shift();
    const entries = await readdir(current, { withFileTypes: true }).catch(() => []);
    for (const entry of entries) {
      if (out.length >= budget.maxFiles) break;
      const full = join(current, entry.name);
      if (entry.isDirectory()) {
        queue.push(full);
        continue;
      }
      if (!entry.isFile()) continue;
      const extension = extname(entry.name).toLowerCase();
      if (!/\.(?:json|csv|txt|log|proto|pb|bin)$/.test(extension)) continue;
      const info = await stat(full).catch(() => undefined);
      if (info === undefined) continue;
      if (/\.db$/i.test(entry.name)) {
        warnings.push(`${relative(base, full)}：msprof 数据库不直接解析，请先用 msprof-analyze 导出 CSV。`);
        continue;
      }
      const relativeName = relative(base, full).replaceAll(sep, '/');
      if (info.size > budget.maxInMemoryBytes) {
        out.push({ name: relativeName, buffer: Buffer.alloc(0), path: full, size: info.size });
        continue;
      }
      out.push({ name: relativeName, buffer: await readFile(full), path: full, size: info.size });
    }
  }
  if (out.length === 0) warnings.push(`目录 ${root} 内没有可解析的 profiling 产物。`);
  return out;
}

/**
 * Parse one trace artifact, streaming when the file is on disk.
 * @param {{name: string, buffer: Buffer, path?: string, size?: number}} file - artifact.
 * @param {object} budget - budgets.
 * @param {(phase: string, percent: number, detail: string) => void} report - progress sink.
 * @param {string} displayName - name used in progress messages.
 * @returns {Promise<object>} trace parse result.
 */
async function parseTraceArtifact(file, budget, report, displayName) {
  const rank = rankFromName(displayName);
  if (file.path !== undefined && (file.size ?? 0) > 0) {
    const collector = new TraceCollector({ maxEvents: budget.maxTimelineEvents });
    const total = file.size ?? 0;
    let seenBytes = 0;
    const scanner = new JsonStreamScanner({
      onElement: (element) => {
        if (element.kind === 'oversized') return;
        let value;
        try {
          value = JSON.parse(element.text);
        } catch {
          return;
        }
        routeElement(value, element, collector);
      },
    });
    const decoder = new TextDecoder('utf-8', { fatal: false });
    const stream = createReadStream(file.path, { highWaterMark: 4 * 1024 * 1024 });
    for await (const chunk of stream) {
      scanner.push(decoder.decode(chunk, { stream: true }));
      seenBytes += chunk.length;
      if (total > 0 && seenBytes % (16 * 1024 * 1024) < chunk.length) {
        report('parse', 15 + (seenBytes / total) * 60, `流式解析 ${displayName}（${formatBytes(seenBytes)}/${formatBytes(total)}）`);
      }
      if (scanner.truncated) break;
    }
    scanner.push(decoder.decode());
    scanner.end();
    const result = collector.finish().result();
    if (scanner.truncated) {
      result.warnings.push(`扫描提前结束：${String(scanner.truncateReason ?? '达到事件预算')}`);
    }
    return { ...result, file: displayName, rank, streamed: true };
  }

  const decoded = decodeText(file.buffer);
  const result = parseTraceText(decoded.text, { maxEvents: budget.maxTimelineEvents });
  if (decoded.warning !== undefined) result.warnings.push(decoded.warning);
  return { ...result, file: displayName, rank, streamed: false };
}

/** Route one streamed JSON element either to the collector or to document metadata. */
function routeElement(value, element, collector) {
  if (Array.isArray(value)) {
    for (const item of value) collector.accept(item);
    return;
  }
  if (isTraceEvent(value)) collector.accept(value);
  else collector.acceptDocument(value);
  void element;
}

/**
 * Parse an auxiliary JSON artifact (`profiler_info.json`, `communication_matrix.json`,
 * `mstx.json`, …).
 * @param {string} name - artifact name.
 * @param {Buffer} buffer - artifact bytes.
 * @returns {{kind: string, name: string, data: unknown, warnings: string[], summary: object}|undefined} parsed artifact.
 */
export function parseAuxJson(name, buffer) {
  const decoded = decodeText(buffer);
  const warnings = [];
  let data;
  try {
    data = JSON.parse(decoded.text);
  } catch (error) {
    return {
      kind: 'json-invalid',
      name,
      data: undefined,
      warnings: [`JSON 解析失败：${error instanceof Error ? error.message : String(error)}`],
      summary: {},
    };
  }
  const kind = auxKindOf(name, data);
  if (decoded.warning !== undefined) warnings.push(decoded.warning);
  return { kind, name, data, warnings, summary: summarizeAux(kind, data) };
}

function auxKindOf(name, data) {
  if (/mstx/i.test(name) || (Array.isArray(data) && data.some((item) => JSON.stringify(item).includes('mstx')))) return 'mstx';
  if (/communication_?matrix/i.test(name)) return 'communication_matrix';
  if (/communication/i.test(name)) return 'communication';
  if (/memory/i.test(name)) return 'memory';
  if (/profiler_?info|profiler_?metadata/i.test(name)) return 'profiler_info';
  return 'metadata';
}

function summarizeAux(kind, data) {
  const summary = { kind };
  if (typeof data !== 'object' || data === null) return summary;
  if (Array.isArray(data)) {
    summary.items = data.length;
    return summary;
  }
  const record = /** @type {Record<string, unknown>} */ (data);
  summary.keys = Object.keys(record).slice(0, 32);
  if (kind === 'profiler_info') {
    summary.entries = Object.fromEntries(
      Object.entries(record)
        .filter(([, value]) => typeof value === 'string' || typeof value === 'number')
        .slice(0, 32),
    );
  }
  if (kind === 'communication_matrix' && Array.isArray(record.matrix)) summary.matrixRows = record.matrix.length;
  return summary;
}

/** Extract a rank id from a trace file name when one is encoded there. */
export function rankFromName(name) {
  // `RegExp.exec` returns `null` (not `undefined`) on a miss, so both arms of
  // the fallback and the final check must tolerate null.
  const match = /(?:rank|device|dev|card|npu)[_-]?(\d+)/i.exec(name) ?? /[-_](\d+)\.(?:json|pt)$/i.exec(name);
  if (match === null || match === undefined) return undefined;
  const value = Number(match[1]);
  return Number.isFinite(value) ? value : undefined;
}

function dedupe(values) {
  const seen = new Set();
  const out = [];
  for (const value of values) {
    if (typeof value !== 'string' || value === '' || seen.has(value)) continue;
    seen.add(value);
    out.push(value);
  }
  return out;
}

/** Artifact kinds that contribute operator statistics. */
export const SUPPORTED_TABLE_KINDS = [...TABLE_KINDS];
