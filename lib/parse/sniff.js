/**
 * Artifact identification and upload validation.
 *
 * Before anything is parsed, each artifact is sniffed: what kind of file is
 * this, and does the set as a whole look like a vLLM-Ascend / Ascend NPU
 * profiling export? The answer drives a precise, actionable error for
 * incompatible files instead of a mysterious empty chart.
 *
 * Evidence is collected as *markers* with weights, so the verdict carries its
 * own justification: which file names matched Ascend conventions, which trace
 * arguments mention the NPU, which CSV headers are CANN headers, and whether
 * anything ties the profile to vLLM specifically (engine operators, an mstx
 * marker, a `profiler_info.json` that names vllm).
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/sniff
 */

import { archiveFormatOf } from './archive.js';
import { tableKindOf } from './ascendcsv.js';

/** Ascend profiling directory and file conventions. */
export const ASCEND_NAME_MARKERS = [
  { pattern: /_ascend_pt($|\/)/i, weight: 4, label: 'torch_npu 导出目录（*_ascend_pt）' },
  { pattern: /ASCEND_PROFILER_OUTPUT/i, weight: 4, label: 'ASCEND_PROFILER_OUTPUT 目录' },
  { pattern: /mindstudio_profiler_output/i, weight: 3, label: 'MindStudio Profiler 输出目录' },
  { pattern: /(?:^|\/)PROF_\d+/i, weight: 3, label: 'msprof 结果目录（PROF_*）' },
  { pattern: /trace_view\.json$/i, weight: 2, label: 'trace_view.json' },
  { pattern: /kernel_details.*\.csv$/i, weight: 3, label: 'kernel_details.csv' },
  { pattern: /op_statistic.*\.csv$/i, weight: 3, label: 'op_statistic.csv' },
  { pattern: /op_summary.*\.csv$/i, weight: 3, label: 'op_summary.csv' },
  { pattern: /operator_details.*\.csv$/i, weight: 3, label: 'operator_details.csv' },
  { pattern: /step_trace_time.*\.csv$/i, weight: 2, label: 'step_trace_time.csv' },
  { pattern: /communication_(?:statistic|matrix).*\.(?:csv|json)$/i, weight: 2, label: '通信统计产物' },
  { pattern: /(?:profiler_info|profiler_metadata).*\.json$/i, weight: 2, label: 'profiler_info.json' },
  { pattern: /mstx.*\.json$/i, weight: 2, label: 'mstx 标记产物' },
  { pattern: /api_statistic.*\.csv$/i, weight: 2, label: 'api_statistic.csv' },
  { pattern: /(?:memory_record|npu_module_mem).*\.csv$/i, weight: 1, label: '内存记录产物' },
  { pattern: /aicore_freq.*\.csv$/i, weight: 1, label: 'AI Core 频率记录' },
  { pattern: /analysis\.db$|\.db$/i, weight: 1, label: 'msprof 数据库（不支持直接解析）' },
];

/** Ascend vocabulary found in trace arguments or JSON payloads. */
export const ASCEND_CONTENT_MARKERS = [
  { pattern: /"ascend"|ascend_profiler|torch_npu|cann/i, weight: 3, label: '内容包含 Ascend/CANN/torch_npu 标识' },
  { pattern: /"device id"|\\?"stream id\\?"/i, weight: 2, label: '事件参数包含 device id / stream id' },
  { pattern: /aicore|aicpu|mte1|mte2|mte3|fixpipe/i, weight: 3, label: '内容包含 AI Core/AI CPU/MTE 流水标识' },
  { pattern: /hccl|hcom_|allreduce|allgather|reducescatter/i, weight: 2, label: '内容包含 HCCL 通信算子' },
  { pattern: /aclrt|aclnn|acl_?memcpy/i, weight: 2, label: '内容包含 ACL 运行时接口' },
  { pattern: /npu|昇腾|华为/i, weight: 1, label: '内容包含 NPU 字样' },
];

/** vLLM-specific vocabulary that upgrades the verdict from Ascend to vLLM-Ascend. */
export const VLLM_MARKERS = [
  { pattern: /vllm/i, weight: 4, label: '内容包含 vllm 标识' },
  { pattern: /execute_model|model_runner|model\.forward/i, weight: 3, label: '包含 vLLM 引擎执行算子（execute_model）' },
  { pattern: /(?:^|[^a-z])(?:prefill|decode|chunked_prefill)(?:[^a-z]|$)/i, weight: 3, label: '包含 prefill/decode 阶段标记' },
  { pattern: /kv[_-]?cache|block_?table|paged_attention|unified_attention/i, weight: 3, label: '包含 KV Cache / PagedAttention 算子' },
  { pattern: /tensor_parallel|tp_?group|world_?size|allreduce/i, weight: 2, label: '包含张量并行/通信组信息' },
  { pattern: /sampler|logits_processor|rejection_sampler/i, weight: 1, label: '包含采样器算子' },
  { pattern: /quant|w8a8|int8|fp8|ascend_quant/i, weight: 1, label: '包含量化算子' },
];

/** File extensions the analyzer can read. */
const SUPPORTED_EXTENSIONS = /\.(?:json|json\.gz|csv|txt|log|proto|pb|bin|gz|zip|tar)$/i;

/** Head bytes inspected for content markers. */
const HEAD_BYTES = 256 * 1024;

/**
 * @typedef {object} SniffedArtifact
 * @property {string} name - artifact name (relative path inside an archive).
 * @property {string} kind - detected artifact kind.
 * @property {number} bytes - artifact size.
 * @property {number} confidence - 0–1 identification confidence.
 * @property {number} weight - accumulated Ascend evidence weight.
 * @property {string[]} markers - matched evidence labels.
 * @property {string[]} warnings - per-artifact warnings.
 * @property {boolean} supported - whether the parser can read it.
 */

/**
 * Classify one artifact.
 *
 * @param {object} input - artifact.
 * @param {string} input.name - artifact name.
 * @param {Buffer|Uint8Array} input.buffer - artifact bytes.
 * @returns {SniffedArtifact} sniff result.
 */
export function sniffArtifact({ name, buffer }) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const markers = [];
  const warnings = [];
  let weight = 0;
  for (const marker of ASCEND_NAME_MARKERS) {
    if (marker.pattern.test(name)) {
      markers.push(`文件名：${marker.label}`);
      weight += marker.weight;
    }
  }
  const format = archiveFormatOf(name, bytes);
  const head = decodeHead(bytes);
  let kind = 'unknown';
  let supported = true;

  if (format !== 'plain') {
    kind = `archive:${format}`;
  } else if (/\.db$/i.test(name)) {
    kind = 'msprof-db';
    supported = false;
    warnings.push('msprof 的 SQLite 数据库（*.db）需要在 MindStudio Insight 或 msprof-analyze 中导出为 CSV 后再上传。');
  } else if (isCsvLike(name, head)) {
    // A table kind is claimed only for CSV-shaped files: `kernel_details.proto`
    // must not be routed to the CSV reader just because its stem matches a
    // known table name.
    const tableKind = tableKindOf(name);
    kind = tableKind === 'generic' ? 'csv-unknown' : tableKind;
  } else if (/\.(?:json|txt|log)$/i.test(name) || /^\s*[[{]/.test(head)) {
    kind = looksLikeTrace(head) ? 'trace-json' : /\.json$/i.test(name) ? 'json-generic' : 'text';
  } else if (/\.(?:proto|pb|bin)$/i.test(name)) {
    kind = 'proto';
  }

  for (const marker of ASCEND_CONTENT_MARKERS) {
    if (marker.pattern.test(head)) {
      markers.push(`内容：${marker.label}`);
      weight += marker.weight;
    }
  }
  for (const marker of VLLM_MARKERS) {
    if (marker.pattern.test(head)) {
      markers.push(`vLLM：${marker.label}`);
      weight += marker.weight;
    }
  }

  if (!SUPPORTED_EXTENSIONS.test(name) && format === 'plain' && kind === 'unknown') {
    supported = false;
    warnings.push(`不支持的文件类型：${name}（支持 .json/.csv/.proto/.bin 以及 .gz/.zip/.tar 压缩包）`);
  }
  // A JSON artifact that is not a trace is only worth flagging when it is not one
  // of the sidecar files a profiling export always ships.
  if (kind === 'json-generic' && !isKnownSidecar(name)) {
    warnings.push(`${name}：是 JSON 但不是 trace 事件数组，将尝试作为元数据/通信矩阵解析。`);
  }

  return {
    name,
    kind,
    bytes: bytes.length,
    confidence: Math.min(1, weight / 12),
    weight,
    markers,
    warnings,
    supported,
  };
}

/**
 * Validate a whole artifact set.
 *
 * @param {SniffedArtifact[]} artifacts - sniffed artifacts.
 * @param {object} [options] - validation options.
 * @param {number} [options.minWeight] - evidence weight required to accept the set.
 * @returns {{
 *   ok: boolean, profileType: 'vllm-ascend'|'ascend'|'unknown',
 *   weight: number, confidence: number, errors: string[], warnings: string[],
 *   markers: string[], artifacts: SniffedArtifact[]
 * }} verdict.
 */
export function validateProfileSet(artifacts, { minWeight = 4 } = {}) {
  const errors = [];
  const warnings = [];
  const markers = [];
  let weight = 0;
  let vllmWeight = 0;
  let readable = 0;

  for (const artifact of artifacts) {
    weight += artifact.weight;
    for (const marker of artifact.markers) {
      if (marker.startsWith('vLLM：')) vllmWeight += 1;
    }
    if (artifact.supported) readable += 1;
    markers.push(...artifact.markers.map((marker) => `${artifact.name} → ${marker}`));
    warnings.push(...artifact.warnings);
  }

  if (artifacts.length === 0) {
    errors.push('没有收到任何文件：请上传 vLLM-Ascend profiling 产物（trace_view.json、op_statistic.csv、kernel_details.csv、operator_details.csv 或 *_ascend_pt 目录打包）。');
    return { ok: false, profileType: 'unknown', weight: 0, confidence: 0, errors, warnings, markers, artifacts };
  }
  if (readable === 0) {
    errors.push('上传的文件都不受支持：请提供 trace_view.json / *.csv / *.proto / *.bin，或将其打成 zip/tar.gz 后上传。');
  }

  const hasTrace = artifacts.some((artifact) => artifact.kind === 'trace-json');
  const hasTable = artifacts.some((artifact) => ['op_statistic', 'op_summary', 'kernel_details', 'operator_details', 'api_statistic', 'step_trace_time', 'communication_statistic'].includes(artifact.kind));
  const hasProto = artifacts.some((artifact) => artifact.kind === 'proto');
  if (!hasTrace && !hasTable && !hasProto) {
    errors.push('未识别到任何 profiling 数据表：至少需要一个 trace_view.json、一个 .proto/.bin 二进制记录，或 op_statistic/kernel_details/operator_details/op_summary 之一。');
  }

  const profileType = vllmWeight > 0 ? 'vllm-ascend' : weight >= minWeight ? 'ascend' : 'unknown';
  if (profileType === 'ascend') {
    warnings.push('识别为 Ascend NPU profiling 产物，但未发现 vLLM 专属标记（execute_model / prefill / kv-cache 等）：将按通用昇腾推理负载分析，prefill/decode 阶段划分可能退化为按步长推断。');
  }
  if (profileType === 'unknown') {
    errors.push(
      `文件证据不足以判定为 vLLM-Ascend profiling 产物（累计证据权重 ${String(weight)} < ${String(minWeight)}）。`
      + '期望的产物包括：torch_npu 的 trace_view.json、kernel_details.csv、operator_details.csv、op_statistic.csv，'
      + '或 msprof 的 op_summary.csv / step_trace_time.csv，或上述文件所在的 *_ascend_pt 目录。',
    );
  }
  if (!hasTrace && hasTable) {
    warnings.push('没有 trace 时间线文件，模块一的泳道图将仅依据 CSV 中的起止时间列绘制（若 CSV 无时间列则该模块为空）。');
  }
  if (hasProto && !hasTrace && !hasTable) {
    warnings.push('仅提供二进制/proto 产物：事件由通用 wire 解码 + 启发式识别得到，置信度低；建议同时提供 trace_view.json 或 kernel_details.csv 交叉校验（也可用 protoFieldMap 指定字段号获得精确映射）。');
  }
  if (hasTrace && !hasTable) {
    warnings.push('没有算子统计 CSV，算子耗时统计将完全由 trace 事件聚合得到（结论可信度略降，因为缺少 CANN 侧的流水利用率数据）。');
  }

  return {
    ok: errors.length === 0,
    profileType,
    weight,
    confidence: Math.min(1, weight / 20),
    errors,
    warnings,
    markers,
    artifacts,
  };
}

/**
 * Build the artifact list of a directory ingest by sniffing each member.
 * @param {{ name: string, buffer: Buffer|Uint8Array }[]} files - member files.
 * @returns {SniffedArtifact[]} sniff results.
 */
export function sniffAll(files) {
  return files.map((file) => sniffArtifact(file));
}

/**
 * Whether an artifact is a delimited text table.
 *
 * The extension decides for `.csv`; extension-less or `.txt`/`.log` exports
 * (msprof writes some tables without an extension) are accepted when their first
 * line is delimiter-heavy, which a binary record or a JSON document never is.
 *
 * @param {string} name - artifact name.
 * @param {string} head - decoded head text.
 * @returns {boolean} whether the file should be read as CSV.
 */
function isCsvLike(name, head) {
  if (/\.csv$/i.test(name)) return true;
  if (!/\.(?:txt|log)$/i.test(name) && /\.[a-z0-9]+$/i.test(name)) return false;
  const firstLine = head.split(/\r?\n/).find((line) => line.trim() !== '') ?? '';
  if (/^\s*[[{]/.test(firstLine)) return false;
  const delimiters = (firstLine.match(/[,\t;]/g) ?? []).length;
  return delimiters >= 2;
}

/** Whether the head text looks like a chrome trace document. */
function looksLikeTrace(head) {
  if (/"traceEvents"\s*:/.test(head)) return true;
  // A bare array whose first object already carries event keys.
  return /^\s*\[\s*\{/.test(head) && /"(?:ph|ts|dur|name)"\s*:/.test(head);
}

/**
 * Sidecar files a profiling export ships next to the trace. They are metadata,
 * not traces, and their being non-trace JSON is expected rather than notable.
 */
const SIDECAR_PATTERN = /(?:profiler_info|profiler_metadata|communication|comm_matrix|memory|mstx|msprof_tx|analysis|config|version|summary)/i;

/** @param {string} name - artifact name. @returns {boolean} whether it is a known sidecar. */
function isKnownSidecar(name) {
  return SIDECAR_PATTERN.test(name);
}

/** Decode the head of a buffer as UTF-8 text for marker matching. */
function decodeHead(bytes) {
  const slice = bytes.subarray(0, Math.min(bytes.length, HEAD_BYTES));
  return new TextDecoder('utf-8', { fatal: false }).decode(slice);
}
