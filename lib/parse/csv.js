/**
 * CSV decoding and a streaming RFC4180-style record reader.
 *
 * Ascend profiling exports are written by CANN tooling and by
 * `msprof-analyze`, so the reader must survive what those writers actually
 * emit: a UTF-8 BOM, CRLF line endings, quoted fields that contain the
 * delimiter, doubled quotes inside a quoted field, trailing empty columns, and
 * files large enough (hundreds of MB of `kernel_details.csv`) that reading the
 * whole file as one string is not an option.
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/csv
 */

/** Byte-order mark that Windows-authored exports prepend. */
const BOM = '\uFEFF';

/**
 * Decode a profiling text artifact to a string.
 *
 * UTF-8 is the CANN default. Some Windows-authored summaries are GBK; Node
 * ships full ICU, so `gbk` is available as a fallback when the UTF-8 decode
 * produces a suspicious number of replacement characters.
 *
 * @param {Buffer|Uint8Array} buffer - raw artifact bytes.
 * @returns {{ text: string, encoding: string, warning: string|undefined }} decoded text.
 */
export function decodeText(buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let text = new TextDecoder('utf-8', { fatal: false }).decode(bytes);
  let encoding = 'utf-8';
  let warning;
  const replacements = countChar(text, '\uFFFD');
  if (replacements > 0 && replacements / Math.max(1, text.length) > 0.002) {
    try {
      const gbk = new TextDecoder('gbk', { fatal: false }).decode(bytes);
      if (countChar(gbk, '\uFFFD') < replacements) {
        text = gbk;
        encoding = 'gbk';
        warning = '文件不是合法 UTF-8，已按 GBK 解码。';
      }
    } catch {
      warning = `文件包含 ${String(replacements)} 个非法 UTF-8 字节，已按替换字符处理。`;
    }
  }
  if (text.startsWith(BOM)) text = text.slice(BOM.length);
  return { text, encoding, warning };
}

function countChar(text, char) {
  let count = 0;
  let at = text.indexOf(char);
  while (at !== -1) {
    count += 1;
    at = text.indexOf(char, at + 1);
  }
  return count;
}

/**
 * Guess a delimiter from a header line by counting candidates outside quotes.
 * @param {string} line - the header line.
 * @returns {string} the most likely delimiter.
 */
export function detectDelimiter(line) {
  const candidates = [',', '\t', ';'];
  let best = ',';
  let bestCount = -1;
  for (const candidate of candidates) {
    let count = 0;
    let quoted = false;
    for (const char of line) {
      if (char === '"') quoted = !quoted;
      else if (char === candidate && !quoted) count += 1;
    }
    if (count > bestCount) {
      best = candidate;
      bestCount = count;
    }
  }
  return best;
}

/**
 * Split one CSV record's text into fields.
 * @param {string} line - record text without the line terminator.
 * @param {string} delimiter - field delimiter.
 * @returns {string[]} field values, unquoted and unescaped.
 */
export function splitRecord(line, delimiter) {
  const fields = [];
  let field = '';
  let quoted = false;
  let index = 0;
  while (index < line.length) {
    const char = line[index];
    if (quoted) {
      if (char === '"') {
        if (line[index + 1] === '"') {
          field += '"';
          index += 2;
          continue;
        }
        quoted = false;
        index += 1;
        continue;
      }
      field += char;
      index += 1;
      continue;
    }
    if (char === '"' && field.trim() === '') {
      quoted = true;
      field = '';
      index += 1;
      continue;
    }
    if (char === delimiter) {
      fields.push(field.trim());
      field = '';
      index += 1;
      continue;
    }
    field += char;
    index += 1;
  }
  fields.push(field.trim());
  return fields;
}

/**
 * Normalize a header cell for alias lookup: lowercase, collapse every run of
 * non-alphanumeric characters to a single `_`, trim underscores.
 * @param {string} value - raw header cell.
 * @returns {string} normalized key.
 */
export function normalizeHeaderKey(value) {
  return String(value ?? '')
    .replace(/^\uFEFF/, '')
    .replace(/[\uFF08(]([^)\uFF09]*)[)\uFF09]/g, ' $1 ')
    .toLowerCase()
    .replace(/[^a-z0-9\u4e00-\u9fff]+/g, '_')
    .replace(/^_+|_+$/g, '');
}

/**
 * Build a header index: canonical field name → column position, using an alias
 * table so that column drift between CANN releases does not break parsing.
 *
 * @param {string[]} header - header cells in file order.
 * @param {Record<string, string[]>} aliases - canonical name → accepted spellings.
 * @returns {{ index: Record<string, number>, header: string[], unmatched: string[] }} lookup result.
 */
export function buildHeaderIndex(header, aliases) {
  const normalized = header.map((cell) => normalizeHeaderKey(cell));
  const index = {};
  for (const [canonical, spellings] of Object.entries(aliases)) {
    for (const spelling of spellings) {
      const at = normalized.indexOf(normalizeHeaderKey(spelling));
      if (at !== -1) {
        index[canonical] = at;
        break;
      }
    }
  }
  const matched = new Set(Object.values(index));
  const unmatched = header.filter((_, position) => !matched.has(position));
  return { index, header, unmatched };
}

/**
 * Read one value out of a record by canonical field name.
 *
 * `N/A` is a legitimate value in CANN numeric columns (HCCL rows carry
 * `Task ID = N/A`), so the placeholders are normalized to `undefined` here
 * rather than leaking into lanes, statistics, or the report.
 *
 * @param {string[]} record - split record.
 * @param {Record<string, number>} index - header lookup.
 * @param {string} field - canonical field name.
 * @returns {string|undefined} raw cell text.
 */
export function cell(record, index, field) {
  const at = index[field];
  if (at === undefined) return undefined;
  const value = record[at];
  if (value === undefined) return undefined;
  const trimmed = value.trim();
  if (trimmed === '' || /^(?:n\/?a|null|nan|-|--|unknown)$/i.test(trimmed)) return undefined;
  return trimmed;
}

/**
 * Parse a numeric cell tolerantly: strips thousands separators, units, and a
 * trailing percent sign; accepts `1.5e3`; returns `undefined` for blanks.
 * @param {string|number|undefined} value - raw cell.
 * @returns {number|undefined} parsed number.
 */
export function num(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === 'number') return Number.isFinite(value) ? value : undefined;
  const cleaned = String(value).replace(/[\s,_]/g, '').replace(/%$/, '');
  if (cleaned === '' || cleaned === '-' || cleaned === 'N/A') return undefined;
  const parsed = Number(cleaned);
  return Number.isFinite(parsed) ? parsed : undefined;
}

/**
 * Parse a duration cell into microseconds.
 *
 * Ascend summaries label the unit inside the column name (`Duration(us)`,
 * `Task Duration(us)`, `Total Time(ms)`), so the caller passes the unit it
 * detected from the header rather than assuming one.
 *
 * @param {string|undefined} value - raw cell.
 * @param {'us'|'ms'|'ns'|'s'} unit - unit named by the header.
 * @returns {number|undefined} microseconds.
 */
export function durationToUs(value, unit) {
  const parsed = num(value);
  if (parsed === undefined) return undefined;
  switch (unit) {
    case 'ms': return parsed * 1000;
    case 'ns': return parsed / 1000;
    case 's': return parsed * 1e6;
    default: return parsed;
  }
}

/** Duration units recognized in column headers, longest token first. */
const UNIT_PATTERNS = [
  { unit: 'ms', pattern: /\((?:ms|msec|millisecond)s?\)|_ms$/i },
  { unit: 'us', pattern: /\((?:us|µs|μs|usec|microsecond)s?\)|_us$/i },
  { unit: 'ns', pattern: /\((?:ns|nsec|nanosecond)s?\)|_ns$/i },
  { unit: 's', pattern: /\((?:s|sec|second)s?\)|_s$/i },
];

/**
 * Read the duration unit a header cell declares.
 * @param {string|undefined} header - raw header cell.
 * @returns {'us'|'ms'|'ns'|'s'|undefined} the declared unit.
 */
export function unitOfHeader(header) {
  const text = String(header ?? '');
  for (const { unit, pattern } of UNIT_PATTERNS) {
    if (pattern.test(text)) return unit;
  }
  return undefined;
}

/**
 * Incremental CSV reader.
 *
 * Feed decoded text chunks with {@link CsvReader#push}; every complete record
 * is delivered to the `onRecord` callback as a field array. Records split
 * across chunks are held until complete, so a chunk boundary inside a quoted
 * field is safe.
 */
export class CsvReader {
  /**
   * @param {object} options - reader options.
   * @param {(record: string[], rowIndex: number) => void} options.onRecord - record sink.
   * @param {number} [options.maxRows] - stop after this many records.
   * @param {number} [options.maxFieldBytes] - guard against a runaway quote.
   */
  constructor({ onRecord, maxRows = Infinity, maxFieldBytes = 8 * 1024 * 1024 }) {
    this.onRecord = onRecord;
    this.maxRows = maxRows;
    this.maxFieldBytes = maxFieldBytes;
    this.delimiter = undefined;
    this.pending = '';
    this.rowIndex = 0;
    this.rows = [];
    this.truncated = false;
    this.header = undefined;
  }

  /** @returns {string[]} the first record read, treated as the header. */
  get headerRow() {
    return this.header;
  }

  /**
   * Feed one decoded chunk.
   * @param {string} chunk - decoded text.
   */
  push(chunk) {
    if (this.truncated) return;
    this.pending += chunk;
    if (this.pending.length > this.maxFieldBytes) {
      // A single unterminated record this large cannot be a profiling table
      // row; drop the buffer instead of growing without bound.
      this.pending = '';
      this.truncated = true;
      return;
    }
    let start = 0;
    for (let at = 0; at < this.pending.length; at += 1) {
      const char = this.pending[at];
      if (char !== '\n' && char !== '\r') continue;
      const line = this.pending.slice(start, at);
      // Swallow the second half of a CRLF pair.
      let next = at + 1;
      if (char === '\r' && this.pending[next] === '\n') next += 1;
      start = next;
      at = next - 1;
      this.#emit(line);
      if (this.truncated) break;
    }
    this.pending = this.pending.slice(start);
  }

  /** Flush a trailing record that had no line terminator. */
  end() {
    if (!this.truncated && this.pending !== '') this.#emit(this.pending);
    this.pending = '';
  }

  #emit(line) {
    const trimmed = line.trim();
    // msprof writers prefix banner/comment lines with '#'.
    if (trimmed === '' || trimmed.startsWith('#')) return;
    if (this.delimiter === undefined) {
      this.delimiter = detectDelimiter(line);
    }
    const fields = splitRecord(line, this.delimiter);
    if (fields.length === 1 && fields[0] === '') return;
    if (this.header === undefined) {
      this.header = fields;
      this.rows.push(fields);
      this.onRecord(fields, 0);
      return;
    }
    if (this.rowIndex + 1 >= this.maxRows) {
      this.truncated = true;
      return;
    }
    this.rowIndex += 1;
    this.rows.push(fields);
    this.onRecord(fields, this.rowIndex);
  }
}

/**
 * Convenience wrapper: parse a decoded CSV text into header + records.
 * @param {string} text - decoded CSV text.
 * @param {number} [maxRows] - record budget.
 * @returns {{ header: string[], rows: string[][], truncated: boolean }} parsed table.
 */
export function parseCsvText(text, maxRows = Infinity) {
  const rows = [];
  const reader = new CsvReader({ onRecord: (record) => rows.push(record), maxRows });
  reader.push(text);
  reader.end();
  const header = rows.shift() ?? [];
  return { header, rows, truncated: reader.truncated };
}
