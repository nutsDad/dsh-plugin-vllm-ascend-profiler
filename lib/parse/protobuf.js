/**
 * Best-effort protobuf decoding for Ascend profiling binaries.
 *
 * `torch_npu` mostly exports JSON and CSV, but Ascend tooling also emits
 * proto-encoded records (`*.proto`, `*.pb`, per-device `*.bin` blobs from
 * `msprof`). Those files carry no self-describing schema, so a full decode is
 * impossible without the matching `.proto`. What this module does instead is
 * honest and bounded:
 *
 * 1. decode the protobuf **wire format** generically (varint, 64-bit, length
 *    delimited, 32-bit), recursing into nested messages;
 * 2. pick out the fields a trace record must have — a name-like string plus two
 *    large integers that can be read as a start and either an end or a
 *    duration;
 * 3. emit candidate events tagged `confidence: "low"` so the UI and the report
 *    label them as heuristic instead of pretending they were parsed exactly.
 *
 * A deployment that knows its schema can pin the field numbers through the
 * plugin's `protoFieldMap` config, which turns the heuristic into an exact
 * mapping.
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/protobuf
 */

import { normalizeOperatorName } from '../model/classify.js';

const WIRE_VARINT = 0;
const WIRE_FIXED64 = 1;
const WIRE_LENGTH = 2;
const WIRE_FIXED32 = 5;

/** Operator-ish strings: anything that looks like a real kernel/API name. */
const NAME_PATTERN = /[A-Za-z_][A-Za-z0-9_.:]{2,}/;
/** Absolute timestamps are Unix-microsecond instants; durations never are. */
const TIMESTAMP_FLOOR = 1e6;
/** Upper bound for a plausible value: 4.2e15 µs is roughly the year 2103. */
const TIMESTAMP_CEILING = 4.2e15;
/** Durations above this are not plausible for a single operator (60s). */
const MAX_DURATION = 6e7;

/**
 * Decode a protobuf buffer into a tree of fields.
 *
 * @param {Buffer|Uint8Array} input - message bytes.
 * @param {object} [options] - decode options.
 * @param {number} [options.maxDepth] - nesting ceiling.
 * @param {number} [options.maxFields] - field ceiling across the whole message.
 * @param {number} [options.stringLimit] - longest string promoted to text.
 * @returns {{ fields: object[], warnings: string[], consumed: number }} decoded fields.
 */
export function decodeProtoMessage(input, { maxDepth = 12, maxFields = 20000, stringLimit = 4096 } = {}) {
  const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
  const warnings = [];
  let fields = 0;
  let consumed = 0;

  /**
   * Decode one message body from an explicit buffer.
   *
   * The buffer is a parameter rather than a closure over the outermost message:
   * a nested message must be decoded in its own coordinates, and decoding the
   * outer buffer at a nested offset silently produces nothing.
   *
   * @param {Uint8Array} buffer - the buffer holding this message.
   * @param {number} start - message start offset.
   * @param {number} end - message end offset.
   * @param {number} depth - current nesting depth.
   * @param {boolean} [collectWarnings] - whether diagnostics should be reported
   *   (speculative nested decodes of string fields run silently).
   * @returns {{ fields: object[], at: number }} decoded fields and the offset reached.
   */
  function read(buffer, start, end, depth, collectWarnings = true) {
    const out = [];
    const warn = (message) => {
      if (collectWarnings) warnings.push(message);
    };
    let at = start;
    while (at < end) {
      const tag = readVarint(buffer, at);
      if (tag === undefined) {
        warn(`偏移 ${String(at)} 处的字段标签无法解码，停止解析该消息。`);
        break;
      }
      at = tag.next;
      const field = tag.value >>> 3;
      const wire = tag.value & 0x7;
      if (field === 0) {
        warn('出现字段号为 0 的非法标签，停止解析该消息。');
        break;
      }
      if (wire === WIRE_VARINT) {
        const value = readVarint(buffer, at);
        if (value === undefined) break;
        at = value.next;
        out.push({ field, wire, value: value.value });
      } else if (wire === WIRE_FIXED64) {
        if (at + 8 > end) break;
        out.push({ field, wire, value: readDouble(buffer, at) });
        at += 8;
      } else if (wire === WIRE_FIXED32) {
        if (at + 4 > end) break;
        out.push({ field, wire, value: readFloat(buffer, at) });
        at += 4;
      } else if (wire === WIRE_LENGTH) {
        const size = readVarint(buffer, at);
        if (size === undefined) break;
        at = size.next;
        if (at + size.value > end) {
          warn('长度前缀超出消息边界，停止解析该消息。');
          break;
        }
        const slice = buffer.subarray(at, at + size.value);
        const text = decodeText(slice, stringLimit);
        // Nested speculation only runs when the slice is NOT valid printable
        // text. A string like "AllGather" can parse as a syntactically perfect
        // message by accident (0x41 is `field 8, wire 1`), which would inject a
        // garbage 64-bit value into the heuristic — the text reading wins.
        const nested = text === undefined && depth < maxDepth ? tryNested(slice, depth) : undefined;
        out.push({ field, wire, value: text ?? undefined, bytes: slice, nested });
        at += size.value;
      } else {
        warn(`不支持的 wire type ${String(wire)}，停止解析该消息。`);
        break;
      }
      fields += 1;
      if (fields > maxFields) {
        warn(`字段数超过上限 ${String(maxFields)}，停止解析。`);
        return { fields: out, at };
      }
    }
    consumed = Math.max(consumed, at);
    return { fields: out, at };
  }

  /**
   * Speculatively decode a length-delimited field as a nested message.
   *
   * A field is accepted as a message only when the nested decode consumes its
   * slice **exactly**; anything else was a string or an opaque blob. Speculation
   * is silent, so a string containing protobuf-looking bytes cannot pollute the
   * diagnostics or the extracted numbers.
   */
  function tryNested(slice, depth) {
    if (slice.length === 0) return undefined;
    const inner = read(slice, 0, slice.length, depth + 1, false);
    if (inner.fields.length === 0) return undefined;
    return inner.at === slice.length ? inner.fields : undefined;
  }

  const top = read(bytes, 0, bytes.length, 0);
  return { fields: top.fields, warnings, consumed };
}

/**
 * Split a length-delimited protobuf stream (the layout Ascend uses for
 * per-event record files) into individual messages.
 *
 * A split is only credible when it consumes the buffer exactly; a raw
 * concatenation of unframed messages produces a plausible-looking but wrong
 * segmentation, so the caller needs to know which case it got.
 *
 * @param {Uint8Array} bytes - stream bytes.
 * @param {number} [limit] - maximum number of messages.
 * @returns {{ messages: Uint8Array[], warnings: string[], consumedAll: boolean }} split result.
 */
export function splitDelimitedStream(bytes, limit = 200000) {
  const messages = [];
  const warnings = [];
  let at = 0;
  while (at < bytes.length && messages.length < limit) {
    const size = readVarint(bytes, at);
    if (size === undefined) break;
    const start = size.next;
    if (start + size.value > bytes.length || size.value === 0) break;
    messages.push(bytes.subarray(start, start + size.value));
    at = start + size.value;
  }
  if (messages.length >= limit) warnings.push(`消息数量超过上限 ${String(limit)}，仅解析前 ${String(limit)} 条。`);
  return { messages, warnings, consumedAll: at === bytes.length };
}

/**
 * Extract candidate trace events from a proto file.
 *
 * @param {object} input - extraction input.
 * @param {string} input.name - artifact name.
 * @param {Buffer|Uint8Array} input.buffer - artifact bytes.
 * @param {object} [input.fieldMap] - exact field numbers: `{ name, start, end, duration, device, opType }`.
 * @param {number} [input.limit] - maximum events to produce.
 * @returns {{ events: object[], warnings: string[], confidence: 'low'|'mapped', messages: number }} extraction result.
 */
export function extractProtoEvents({ name, buffer, fieldMap, limit = 200000 }) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const warnings = [];
  const events = [];
  const { messages, warnings: splitWarnings, consumedAll } = splitDelimitedStream(bytes);
  warnings.push(...splitWarnings);
  const framed = messages.length > 0 && consumedAll;
  const candidates = framed ? messages : [bytes];
  const exact = isCompleteFieldMap(fieldMap);
  if (!framed) {
    warnings.push(
      '未检测到长度前缀消息流，已把整个文件当作单个消息解析：未分帧的连续记录无法在无 schema 的情况下可靠切分，'
      + '因此只会提取首个可识别记录。请提供 protoFieldMap 或分帧后的记录文件以获得完整事件。',
    );
  }

  for (const message of candidates) {
    if (events.length >= limit) break;
    const { fields, warnings: decodeWarnings } = decodeProtoMessage(message);
    if (decodeWarnings.length > 0 && events.length === 0) warnings.push(...decodeWarnings.slice(0, 3));
    const event = exact ? eventFromFieldMap(fields, fieldMap) : heuristicEvent(fields);
    if (event !== undefined) events.push(event);
  }

  if (events.length === 0) {
    warnings.push(
      'proto 文件采用启发式解析但未识别出可用事件记录：Ascend proto 缺少自描述 schema，'
      + '请在插件配置 protoFieldMap 中给出字段号映射，或改用 trace_view.json / CSV 产物。',
    );
  } else if (!exact) {
    warnings.push('proto 事件为启发式识别结果（置信度低），建议同时提供 trace_view.json 或 kernel_details.csv 交叉校验。');
  }
  return { events, warnings, confidence: exact ? 'mapped' : 'low', messages: candidates.length, file: name };
}

function isCompleteFieldMap(map) {
  return typeof map === 'object' && map !== null
    && Number.isInteger(map.name) && Number.isInteger(map.start)
    && (Number.isInteger(map.end) || Number.isInteger(map.duration));
}

function eventFromFieldMap(fields, map) {
  const flat = flattenFields(fields);
  const name = asText(flat.get(map.name));
  const start = asNumber(flat.get(map.start));
  const end = asNumber(flat.get(map.end));
  const duration = asNumber(flat.get(map.duration));
  if (name === undefined || start === undefined) return undefined;
  const dur = duration ?? (end === undefined ? undefined : end - start);
  if (dur === undefined) return undefined;
  return buildEvent(name, start, dur, asText(flat.get(map.opType)), asText(flat.get(map.device)), 'mapped');
}

/**
 * Heuristic record recognition.
 *
 * A message qualifies when it carries a name-like string and a timestamp-like
 * integer (≥ 10^6, i.e. a Unix-µs instant) followed, in wire order, by either a
 * duration or an end timestamp. Field order is preserved by the recursive walk,
 * which is why the search is sequential rather than a set membership test.
 */
function heuristicEvent(fields) {
  const strings = [];
  const numbers = [];
  collect(fields, strings, numbers, 0);
  const name = strings.find((value) => NAME_PATTERN.test(value));
  if (name === undefined) return undefined;
  const startIndex = numbers.findIndex((value) => value >= TIMESTAMP_FLOOR);
  if (startIndex === -1) return undefined;
  const start = numbers[startIndex];
  for (let at = startIndex + 1; at < numbers.length; at += 1) {
    const value = numbers[at];
    if (value <= 0) continue;
    if (value >= TIMESTAMP_FLOOR) {
      // A second absolute timestamp: accept it as the end when the difference is
      // a plausible duration, otherwise keep looking.
      if (value > start && value - start <= MAX_DURATION) return buildEvent(name, start, value - start, undefined, undefined, 'low');
      continue;
    }
    if (value <= MAX_DURATION) return buildEvent(name, start, value, undefined, undefined, 'low');
  }
  return undefined;
}

function collect(fields, strings, numbers, depth) {
  if (depth > 8) return;
  for (const field of fields) {
    if (typeof field.value === 'string' && field.value.length > 0) strings.push(field.value);
    else if (typeof field.value === 'number' && Number.isFinite(field.value)
      && field.value > 0 && field.value <= TIMESTAMP_CEILING) {
      // Magnitude filtering keeps a misread 64-bit field (which decodes as an
      // astronomically large double) from being treated as a timestamp.
      numbers.push(field.value);
    }
    if (Array.isArray(field.nested)) collect(field.nested, strings, numbers, depth + 1);
  }
}

function flattenFields(fields, out = new Map(), depth = 0) {
  if (depth > 8) return out;
  for (const field of fields) {
    if (!out.has(field.field)) {
      if (typeof field.value === 'string' || typeof field.value === 'number') out.set(field.field, field.value);
    }
    if (Array.isArray(field.nested)) flattenFields(field.nested, out, depth + 1);
  }
  return out;
}

function buildEvent(name, startUs, durUs, opType, deviceId, confidence) {
  // A proto record whose name matches the host framework vocabulary is a host
  // row; everything else is treated as a device row, which is what the
  // heuristic recognizer mostly picks up (kernel names with timestamps).
  const hostish = /aten::|python|engine|scheduler|sampler|execute_model/i.test(name);
  const device = hostish ? 'host' : 'device';
  return {
    name,
    normalizedName: normalizeOperatorName(name),
    tsUs: startUs,
    durUs: Math.max(0, durUs),
    device,
    lane: `${device}:proto${deviceId === undefined ? '' : `-${deviceId}`}`,
    pid: 'proto',
    tid: deviceId ?? '0',
    pidLabel: 'Proto (heuristic)',
    tidLabel: undefined,
    stream: deviceId,
    rank: deviceId,
    opType,
    taskType: undefined,
    coreType: undefined,
    cat: 'proto',
    message: undefined,
    flops: undefined,
    waitUs: undefined,
    tokenCount: undefined,
    shapesIn: undefined,
    shapesOut: undefined,
    callStack: undefined,
    args: {},
    confidence,
  };
}

function readVarint(bytes, start) {
  let value = 0;
  let shift = 0;
  let at = start;
  while (at < bytes.length) {
    const byte = bytes[at];
    value += (byte & 0x7f) * 2 ** shift;
    at += 1;
    if ((byte & 0x80) === 0) return { value, next: at };
    shift += 7;
    if (shift > 63) return undefined;
  }
  return undefined;
}

function readDouble(bytes, at) {
  const view = new DataView(bytes.buffer, bytes.byteOffset + at, 8);
  return view.getFloat64(0, true);
}

function readFloat(bytes, at) {
  const view = new DataView(bytes.buffer, bytes.byteOffset + at, 4);
  return view.getFloat32(0, true);
}

function decodeText(slice, limit) {
  if (slice.length === 0 || slice.length > limit) return undefined;
  const text = new TextDecoder('utf-8', { fatal: false }).decode(slice);
  if (text.includes('\uFFFD')) return undefined;
  // Printable-ASCII strings only; binary blobs stay opaque.
  for (let at = 0; at < text.length; at += 1) {
    const code = text.charCodeAt(at);
    if (code < 0x20 && code !== 0x09 && code !== 0x0a) return undefined;
  }
  return text.trim() === '' ? undefined : text;
}

function asText(value) {
  return typeof value === 'string' ? value : undefined;
}

function asNumber(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}
