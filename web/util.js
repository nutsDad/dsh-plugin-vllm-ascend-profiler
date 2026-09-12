/**
 * Small shared helpers for the analyzer page: formatting, DOM building, and
 * interval math used by the swimlane interactions.
 *
 * Loaded as a classic script (no bundler in the delivery path); everything is
 * exposed under the global `VAP` namespace.
 */
(function attachUtil(global) {
  'use strict';

  const CATEGORY_COLORS = {
    compute: '#4f7cff',
    comm: '#f2994a',
    copy: '#27ae60',
    schedule: '#9b59b6',
    other: '#7f8c9b',
  };

  const CATEGORY_LABELS = {
    compute: '计算算子',
    comm: '通信算子',
    copy: '数据拷贝算子',
    schedule: '调度算子',
    other: '其他/同步',
  };

  const PHASE_LABELS = {
    prefill: 'Prefill',
    decode: 'Decode',
    both: '通用',
    overall: '全量窗口',
    unknown: '未划分',
  };

  const BOTTLENECK_COLORS = {
    host: '#9b59b6',
    compute: '#4f7cff',
    comm: '#f2994a',
    copy: '#27ae60',
    balanced: '#7f8c9b',
  };

  /** Format a microsecond value with a unit adapted to its magnitude. */
  function formatUs(value) {
    if (value === undefined || value === null || !Number.isFinite(value)) return 'N/A';
    if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(3)}s`;
    if (Math.abs(value) >= 1000) return `${(value / 1000).toFixed(value >= 1e5 ? 1 : 2)}ms`;
    return `${value.toFixed(value >= 100 ? 1 : 2)}µs`;
  }

  /** Format a millisecond value. */
  function formatMs(value) {
    if (!Number.isFinite(value)) return 'N/A';
    return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${value.toFixed(value >= 10 ? 1 : 3)}ms`;
  }

  /** Format a percentage. */
  function formatPct(value, digits) {
    if (!Number.isFinite(value)) return 'N/A';
    return `${value.toFixed(digits === undefined ? 1 : digits)}%`;
  }

  /** Format a ratio (0–1) as a percentage. */
  function formatRatio(value) {
    if (!Number.isFinite(value)) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  }

  /** Format a byte count. */
  function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) return 'N/A';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
      value /= 1024;
      unit += 1;
    }
    return `${value.toFixed(unit === 0 ? 0 : 1)}${units[unit]}`;
  }

  /** Format a count with thousands separators. */
  function formatCount(value) {
    if (!Number.isFinite(value)) return 'N/A';
    return value.toLocaleString('zh-CN');
  }

  /** Compact JSON for hover cards: arrays of shapes become `[1,1024]×[1024,4096]`. */
  function formatShapes(value) {
    if (value === undefined || value === null || value === '') return undefined;
    if (typeof value === 'string') return value.length > 160 ? `${value.slice(0, 157)}…` : value;
    try {
      const text = JSON.stringify(value);
      return text.length > 160 ? `${text.slice(0, 157)}…` : text;
    } catch {
      return String(value);
    }
  }

  /** Truncate a stack trace to its first frames. */
  function formatStack(value) {
    if (value === undefined || value === null || value === '') return undefined;
    const text = typeof value === 'string' ? value : JSON.stringify(value);
    const frames = text.split(/[;\n]/).filter((part) => part.trim() !== '');
    const head = frames.slice(0, 3).join(' › ');
    return frames.length > 3 ? `${head} › …（共 ${frames.length} 帧）` : head;
  }

  /** Escape text for HTML insertion. */
  function escapeHtml(text) {
    return String(text)
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  }

  /**
   * Build an element tree.
   *
   * Class and id tokens accept any non-space character, including CJK: a tag like
   * `div.advice.pri-高` must stay a real `div` with those classes, because
   * `createElement` on an unparsed tag string yields an HTMLUnknownElement that
   * no stylesheet rule matches.
   *
   * @param {string} tag - tag name, `div.k1.k2` or `span#id` forms allowed.
   * @param {object|string} [attrs] - attributes, or text content.
   * @param {Array|string} [children] - children.
   */
  function h(tag, attrs, children) {
    const match = /^([a-zA-Z][a-zA-Z0-9-]*)((?:[.#][^\s.#]+)*)$/.exec(tag);
    const name = match === null ? tag : match[1];
    const element = document.createElement(name);
    if (match !== null && match[2] !== '') {
      for (const token of match[2].split(/(?=[.#])/).filter(Boolean)) {
        if (token.startsWith('.')) element.classList.add(token.slice(1));
        else element.id = token.slice(1);
      }
    }
    let content = children;
    if (typeof attrs === 'string' || typeof attrs === 'number') content = attrs;
    else if (attrs !== undefined && attrs !== null) {
      for (const [key, value] of Object.entries(attrs)) {
        if (value === undefined || value === null || value === false) continue;
        if (key === 'text') element.textContent = String(value);
        else if (key === 'html') element.innerHTML = String(value);
        else if (key === 'dataset') Object.assign(element.dataset, value);
        else if (key.startsWith('on') && typeof value === 'function') element.addEventListener(key.slice(2), value);
        else if (value === true) element.setAttribute(key, '');
        else element.setAttribute(key, String(value));
      }
    }
    if (Array.isArray(content)) {
      for (const child of content) {
        if (child === undefined || child === null || child === false) continue;
        element.append(child instanceof Node ? child : document.createTextNode(String(child)));
      }
    } else if (content !== undefined && content !== null) {
      element.append(content instanceof Node ? content : document.createTextNode(String(content)));
    }
    return element;
  }

  /** Merge overlapping intervals; returns sorted, disjoint intervals. */
  function unionIntervals(intervals) {
    if (intervals.length === 0) return [];
    const sorted = [...intervals].sort((left, right) => left.start - right.start);
    const merged = [{ start: sorted[0].start, end: sorted[0].end }];
    for (const interval of sorted.slice(1)) {
      const last = merged[merged.length - 1];
      if (interval.start <= last.end) last.end = Math.max(last.end, interval.end);
      else merged.push({ start: interval.start, end: interval.end });
    }
    return merged;
  }

  /** Length covered by both interval sets (both sides merged first). */
  function overlapLength(a, b) {
    if (a.length === 0 || b.length === 0) return 0;
    const left = unionIntervals(a);
    const right = unionIntervals(b);
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

  /** Total duration of a set of events, by operator name. */
  function sumBy(rows, pick) {
    return rows.reduce((sum, row) => sum + (pick(row) ?? 0), 0);
  }

  /** A stable colour per operator name, for the swimlane's non-category rows. */
  function hashColor(text) {
    let hash = 0;
    for (let at = 0; at < text.length; at += 1) hash = (hash * 31 + text.charCodeAt(at)) >>> 0;
    const hue = hash % 360;
    return `hsl(${String(hue)} 62% 55%)`;
  }

  /** Debounce a function. */
  function debounce(fn, wait) {
    let timer;
    return function debounced(...args) {
      clearTimeout(timer);
      timer = setTimeout(() => fn.apply(this, args), wait);
    };
  }

  // ── motion helpers ───────────────────────────────────────────────────────
  // Every animation goes through these so one switch can turn the whole page
  // static (the topbar 动效 toggle adds `no-motion` to <body>, and the OS-level
  // `prefers-reduced-motion` is honoured by the stylesheet as well).

  /** Whether animation is currently allowed. */
  function motionEnabled() {
    if (typeof document === 'undefined') return false;
    if (document.body?.classList.contains('no-motion') === true) return false;
    return !(typeof matchMedia === 'function' && matchMedia('(prefers-reduced-motion: reduce)').matches);
  }

  /**
   * Animate a number from `from` to `to`, formatting each frame.
   * @param {object} options - `{ from, to, duration, onFrame, onDone }`.
   * @returns {() => void} cancel function.
   */
  function animateNumber({ from = 0, to, duration = 650, onFrame, onDone }) {
    if (typeof to !== 'number' || !Number.isFinite(to)) {
      onFrame?.(to);
      onDone?.();
      return () => {};
    }
    if (!motionEnabled() || duration <= 0) {
      onFrame?.(to);
      onDone?.();
      return () => {};
    }
    let frame = 0;
    const started = performance.now();
    const tick = (now) => {
      const progress = Math.min(1, (now - started) / duration);
      // easeOutCubic: fast start, settles gently — reads as "counting up".
      const eased = 1 - (1 - progress) ** 3;
      onFrame?.(from + (to - from) * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
      else onDone?.();
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }

  /**
   * Tween a set of numeric properties with an easing function.
   * Used by the swimlane for smooth zoom (instead of jumping between scales).
   *
   * @param {object} input - `{ from, to, duration, onFrame, onDone }`.
   * @returns {() => void} cancel function.
   */
  function tween({ from, to, duration = 220, onFrame, onDone }) {
    const keys = Object.keys(to);
    if (!motionEnabled() || duration <= 0) {
      onFrame?.(to);
      onDone?.();
      return () => {};
    }
    let frame = 0;
    const started = performance.now();
    const tick = (now) => {
      const progress = Math.min(1, (now - started) / duration);
      const eased = 1 - (1 - progress) ** 3;
      const current = {};
      for (const key of keys) current[key] = from[key] + (to[key] - from[key]) * eased;
      onFrame?.(current);
      if (progress < 1) frame = requestAnimationFrame(tick);
      else onDone?.();
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }

  /** Re-trigger a CSS entrance animation on a section that just became visible. */
  function playEnter(node, className = 'enter') {
    if (node === null || node === undefined) return;
    node.classList.remove('enter', 'enter-2', 'enter-3');
    if (!motionEnabled()) return;
    // Reading offsetWidth restarts the animation without a forced timeout.
    void node.offsetWidth;
    node.classList.add(className);
  }

  global.VAP = global.VAP ?? {};
  Object.assign(global.VAP, {
    CATEGORY_COLORS,
    CATEGORY_LABELS,
    PHASE_LABELS,
    BOTTLENECK_COLORS,
    formatUs,
    formatMs,
    formatPct,
    formatRatio,
    formatBytes,
    formatCount,
    formatShapes,
    formatStack,
    escapeHtml,
    h,
    unionIntervals,
    overlapLength,
    sumBy,
    hashColor,
    debounce,
    motionEnabled,
    animateNumber,
    tween,
    playEnter,
  });
})(window);
