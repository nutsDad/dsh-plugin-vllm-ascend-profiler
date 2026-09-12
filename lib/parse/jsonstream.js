/**
 * A streaming JSON element scanner.
 *
 * `trace_view.json` from torch_npu / Ascend profiling is frequently hundreds of
 * megabytes, so it is never loaded as one string. This scanner consumes decoded
 * text chunk by chunk, tracks object/array nesting, and hands back the raw text
 * of every element of every array (for traces: the elements of `traceEvents`, or
 * of a bare top-level array) the moment that element is complete.
 *
 * The caller parses each returned element with `JSON.parse`, which keeps peak
 * memory proportional to one event instead of one file.
 *
 * Design: one capture is active at a time. A capture starts when an array
 * element begins (its opening brace, or the first character of a scalar
 * element) and is remembered as the stack depth of the owning array. Everything
 * nested inside is appended to the capture verbatim; when the stack unwinds
 * back to the owning array's depth, the element is complete and is emitted.
 * Object keys are tracked separately so each element can report which array key
 * it came from (`traceEvents` vs `metadata` vs `otherData`).
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/jsonstream
 */

/** Containers that open a new nesting level. */
const OPENERS = new Set(['{', '[']);
/** Containers that close a nesting level. */
const CLOSERS = new Set(['}', ']']);
/** Whitespace that terminates a scalar token. */
const WHITESPACE = new Set([' ', '\n', '\r', '\t']);

export class JsonStreamScanner {
  /**
   * @param {object} options - scanner options.
   * @param {(element: {path: string[], key: string|undefined, kind: string, text: string}) => void} options.onElement - complete-element sink.
   * @param {number} [options.maxElements] - stop the scan after this many elements.
   * @param {number} [options.maxElementBytes] - emit an oversized element as `kind: 'oversized'` without its text.
   * @param {number} [options.maxDepth] - nesting guard.
   * @param {(reason: string) => void} [options.onTruncate] - notified when a budget stops the scan.
   */
  constructor({ onElement, maxElements = Infinity, maxElementBytes = 8 * 1024 * 1024, maxDepth = 128, onTruncate }) {
    this.onElement = onElement;
    this.maxElements = maxElements;
    this.maxElementBytes = maxElementBytes;
    this.maxDepth = maxDepth;
    this.onTruncate = onTruncate;

    /** @type {{ type: 'object'|'array', key: string|undefined, expect: 'key'|'value' }[]} */
    this.stack = [];
    this.inString = false;
    this.escape = false;
    /** Where the characters of the string being read belong. */
    this.stringRole = undefined;
    this.stringParts = undefined;
    /** @type {{ depth: number, key: string|undefined, path: string[], parts: string[], bytes: number, oversized: boolean }|undefined} */
    this.capture = undefined;
    /** Scalar array element in progress (string, number, boolean, null). */
    this.scalar = undefined;
    this.primitiveActive = false;
    this.elementCount = 0;
    this.truncated = false;
    this.truncateReason = undefined;
    this.error = undefined;
    this.sawAnyContainer = false;
  }

  /**
   * Feed one decoded chunk.
   * @param {string} chunk - decoded text.
   */
  push(chunk) {
    if (this.truncated) return;
    const length = chunk.length;
    let at = 0;
    while (at < length) {
      if (this.truncated) return;
      const char = chunk[at];
      if (this.inString) {
        if (this.escape) {
          this.escape = false;
          this.#append(char);
          at += 1;
          continue;
        }
        if (char === '\\') {
          this.escape = true;
          this.#append(char);
          at += 1;
          continue;
        }
        if (char === '"') {
          this.inString = false;
          this.#append(char);
          at += 1;
          this.#stringClosed();
          continue;
        }
        const stop = nextStringStop(chunk, at);
        this.#append(chunk.slice(at, stop));
        at = stop;
        continue;
      }
      if (char === '"') {
        this.inString = true;
        this.#beginString();
        this.#append(char);
        at += 1;
        continue;
      }
      if (OPENERS.has(char)) {
        this.#open(char);
        at += 1;
        if (this.truncated) return;
        continue;
      }
      if (CLOSERS.has(char)) {
        this.#close(char);
        at += 1;
        if (this.truncated) return;
        continue;
      }
      if (char === ':') {
        this.#append(char);
        this.#colon();
        at += 1;
        continue;
      }
      if (char === ',') {
        // Close a scalar element before the separator joins the captured text.
        if (this.primitiveActive) this.#primitiveEnd();
        this.#append(char);
        this.#comma();
        at += 1;
        continue;
      }
      if (WHITESPACE.has(char)) {
        if (this.primitiveActive) this.#primitiveEnd();
        // Whitespace inside a captured element must survive: dropping the space
        // in `[1, 2]` would silently turn it into `[12]`.
        this.#append(char);
        at += 1;
        continue;
      }
      this.#primitiveChar(char);
      at += 1;
    }
  }

  /** Flush a trailing scalar element at end of input. */
  end() {
    if (this.primitiveActive) this.#primitiveEnd();
  }

  // ── internals ────────────────────────────────────────────────────────────

  #markTruncated(reason) {
    if (this.truncated) return;
    this.truncated = true;
    this.truncateReason = reason;
    this.onTruncate?.(reason);
  }

  #frame() {
    return this.stack[this.stack.length - 1];
  }

  /** Object keys on the path from the root, including the current frame's key. */
  #path(extraKey) {
    const keys = [];
    for (const frame of this.stack) {
      if (frame.type === 'object' && frame.key !== undefined) keys.push(frame.key);
    }
    if (extraKey !== undefined) keys.push(extraKey);
    return keys;
  }

  /**
   * Route one character (or a bulk slice) to whichever buffer owns it: the
   * active capture, the JSON key being read, or a scalar array element.
   * @param {string} text - character or slice to route.
   */
  #append(text) {
    const capture = this.capture;
    if (capture !== undefined) {
      if (capture.oversized) return;
      capture.bytes += text.length;
      if (capture.bytes > this.maxElementBytes) {
        capture.oversized = true;
        capture.parts.length = 0;
        return;
      }
      capture.parts.push(text);
      return;
    }
    if (this.stringRole !== undefined && this.stringParts !== undefined) {
      this.stringParts.push(text);
      return;
    }
    if (this.primitiveActive && this.scalar !== undefined) this.scalar.parts.push(text);
  }

  #beginString() {
    if (this.capture !== undefined) {
      this.stringRole = 'capture';
      return;
    }
    const frame = this.#frame();
    if (frame === undefined) {
      this.stringRole = 'discard';
      this.stringParts = [];
      return;
    }
    if (frame.type === 'object' && frame.expect === 'key') {
      this.stringRole = 'key';
      this.stringParts = [];
      return;
    }
    if (frame.type === 'array') {
      this.stringRole = 'scalar';
      this.stringParts = [];
      this.scalar = {
        depth: this.stack.length,
        key: frame.key,
        path: this.#path(),
        parts: this.stringParts,
        bytes: 0,
        oversized: false,
        kind: 'string',
      };
      return;
    }
    this.stringRole = 'discard';
    this.stringParts = [];
  }

  #stringClosed() {
    const role = this.stringRole;
    const parts = this.stringParts;
    this.stringRole = undefined;
    this.stringParts = undefined;
    if (role === 'key') {
      const frame = this.#frame();
      if (frame !== undefined && frame.type === 'object') {
        frame.key = unquote(parts.join(''));
        frame.expect = 'value';
      }
      return;
    }
    if (role === 'scalar') {
      const scalar = this.scalar;
      this.scalar = undefined;
      if (scalar !== undefined) this.#emit(scalar.path, scalar.key, 'string', parts.join(''), scalar.depth);
    }
    // 'capture' and 'discard' need no further action; #append already routed them.
  }

  #open(char) {
    if (this.stack.length >= this.maxDepth) {
      this.#markTruncated(`JSON 嵌套深度超过 ${String(this.maxDepth)}`);
      return;
    }
    this.sawAnyContainer = true;
    if (this.primitiveActive) this.#primitiveEnd();
    const parent = this.#frame();
    if (parent !== undefined && parent.type === 'array' && this.capture === undefined) {
      this.capture = {
        depth: this.stack.length,
        key: parent.key,
        path: this.#path(),
        parts: [],
        bytes: 0,
        oversized: false,
      };
    }
    this.#append(char);
    // A container introduced by an object key carries that key so that the path
    // of everything inside it stays accurate; an array element has no key.
    const introducedByKey = parent !== undefined && parent.type === 'object' ? parent.key : undefined;
    this.stack.push({
      type: char === '{' ? 'object' : 'array',
      key: introducedByKey,
      expect: char === '{' ? 'key' : 'value',
    });
    if (parent !== undefined && parent.type === 'object') parent.key = undefined;
  }

  #close(char) {
    const frame = this.#frame();
    if (frame === undefined) {
      this.error = `JSON 结构错误：遇到多余的 ${char}`;
      this.#markTruncated(this.error);
      return;
    }
    if (this.primitiveActive) this.#primitiveEnd();
    this.#append(char);
    this.stack.pop();
    const parent = this.#frame();
    if (parent !== undefined) parent.expect = 'value';
    const capture = this.capture;
    if (capture !== undefined && this.stack.length === capture.depth) {
      this.capture = undefined;
      if (this.stringRole !== undefined) {
        this.stringRole = undefined;
        this.stringParts = undefined;
      }
      if (capture.oversized) this.#emit(capture.path, capture.key, 'oversized', '', capture.depth);
      else this.#emit(capture.path, capture.key, 'container', capture.parts.join(''), capture.depth);
    }
  }

  #colon() {
    if (this.primitiveActive) this.#primitiveEnd();
    const frame = this.#frame();
    if (frame !== undefined && frame.type === 'object') frame.expect = 'value';
  }

  #comma() {
    if (this.primitiveActive) this.#primitiveEnd();
    const frame = this.#frame();
    if (frame !== undefined && frame.type === 'object') {
      frame.expect = 'key';
      frame.key = undefined;
    }
  }

  #primitiveChar(char) {
    const frame = this.#frame();
    if (frame === undefined) return;
    if (frame.type === 'object' && frame.expect === 'key') {
      this.#markTruncated(`JSON 结构错误：对象键位置出现裸字符 "${char}"`);
      return;
    }
    this.primitiveActive = true;
    if (this.capture !== undefined) this.#append(char);
    else if (frame.type === 'array') {
      if (this.scalar === undefined) {
        this.scalar = {
          depth: this.stack.length,
          key: frame.key,
          path: this.#path(),
          parts: [],
          bytes: 0,
          oversized: false,
          kind: 'primitive',
        };
      }
      this.scalar.parts.push(char);
    }
  }

  #primitiveEnd() {
    this.primitiveActive = false;
    const frame = this.#frame();
    const scalar = this.scalar;
    this.scalar = undefined;
    if (frame !== undefined && frame.type === 'object' && frame.expect === 'value') frame.expect = 'key';
    if (scalar === undefined) return;
    this.#emit(scalar.path, scalar.key, 'primitive', scalar.parts.join(''), scalar.depth);
  }

  #emit(path, key, kind, text, depth) {
    this.elementCount += 1;
    if (this.elementCount > this.maxElements) {
      this.#markTruncated(`元素数量超过上限 ${String(this.maxElements)}`);
      return;
    }
    this.onElement({ path, key, kind, text });
    void depth;
  }
}

/**
 * Find the next character that could end the current string run.
 * @param {string} chunk - text being scanned.
 * @param {number} from - scan start.
 * @returns {number} index of the next `"` or `\`, or the chunk length.
 */
function nextStringStop(chunk, from) {
  for (let at = from; at < chunk.length; at += 1) {
    const char = chunk[at];
    if (char === '\\' || char === '"') return at;
  }
  return chunk.length;
}

/**
 * Turn a raw JSON string literal (with quotes) into its value.
 * @param {string} raw - raw literal.
 * @returns {string} decoded string, or the raw text when it is not a literal.
 */
function unquote(raw) {
  const text = raw.trim();
  if (text.startsWith('"') && text.endsWith('"') && text.length >= 2) {
    try {
      const parsed = JSON.parse(text);
      return typeof parsed === 'string' ? parsed : text.slice(1, -1);
    } catch {
      return text.slice(1, -1);
    }
  }
  return text;
}

/**
 * Stream a JSON text and deliver every element of the arrays whose owning key
 * matches `arrayKeys` (when given), or every array element (when omitted).
 *
 * @param {string} text - decoded JSON text.
 * @param {object} options - scan options.
 * @param {string[]} [options.arrayKeys] - owning array keys to accept.
 * @param {(value: unknown, meta: { key: string|undefined, path: string[] }) => void} options.onValue - parsed-element sink.
 * @param {number} [options.maxElements] - element budget.
 * @param {number} [options.maxElementBytes] - per-element byte budget.
 * @returns {{ elements: number, delivered: number, truncated: boolean, reason: string|undefined, error: string|undefined }} scan statistics.
 */
export function scanJsonText(text, { arrayKeys, onValue, maxElements = Infinity, maxElementBytes = 8 * 1024 * 1024 }) {
  const stats = {
    elements: 0,
    delivered: 0,
    truncated: false,
    reason: undefined,
    error: undefined,
    oversized: 0,
  };
  const scanner = new JsonStreamScanner({
    maxElements,
    maxElementBytes,
    onElement: (element) => {
      stats.elements += 1;
      if (element.kind === 'oversized') {
        stats.oversized += 1;
        return;
      }
      const owningKey = element.key ?? element.path[element.path.length - 1];
      if (arrayKeys !== undefined && owningKey !== undefined && !arrayKeys.includes(owningKey)) return;
      let parsed;
      try {
        parsed = JSON.parse(element.text);
      } catch {
        return;
      }
      stats.delivered += 1;
      onValue(parsed, { key: owningKey, path: element.path });
    },
    onTruncate: (reason) => {
      stats.truncated = true;
      stats.reason = reason;
    },
  });
  scanner.push(text);
  scanner.end();
  stats.error = scanner.error;
  stats.truncated = stats.truncated || scanner.truncated;
  if (stats.reason === undefined) stats.reason = scanner.truncateReason;
  stats.bytes = text.length;
  return stats;
}
