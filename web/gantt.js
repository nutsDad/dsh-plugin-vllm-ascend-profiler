/**
 * Module 1: the Host / Device operator swimlane (Gantt) timeline.
 *
 * Canvas rather than DOM: a real profile holds thousands of bars per lane and
 * hundreds of lanes, and DOM nodes cannot be created for them at 60 fps.
 *
 * Motion (all suppressible through `VAP.motionEnabled()`):
 *   * bars grow in from the left when a dataset is loaded (a mask sweeps right);
 *   * zoom and pan are tweened instead of jumping, so the eye can follow;
 *   * selecting an operator fades the other rows out instead of hiding them;
 *   * the time cursor can be played back, sweeping the window and lighting up
 *     whatever is executing at that instant — this is the one animation that also
 *     answers a question ("what is the device doing while the host is stuck?").
 *
 * Interactions: wheel scrolls rows, ctrl/⌘+wheel zooms around the cursor,
 * shift+wheel (or drag) pans, hover shows the card, click filters to that
 * operator, escape clears, double-click fits the window.
 */
(function attachGantt(global) {
  'use strict';

  const {
    CATEGORY_COLORS, CATEGORY_LABELS, formatUs, formatCount, formatShapes, formatStack,
    motionEnabled, tween,
  } = global.VAP;

  const ROW_HEIGHT = 16;
  const ROW_GAP = 3;
  const GROUP_HEADER = 20;
  const LEFT_GUTTER = 214;
  const TOP_AXIS = 22;
  const FONT = '11px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
  const FONT_SMALL = '10px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
  const FONT_LABEL = '600 11px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';

  class GanttView {
    /**
     * @param {object} options - view wiring.
     * @param {HTMLCanvasElement} options.canvas - target canvas.
     * @param {HTMLElement} options.tooltip - hover card element.
     * @param {HTMLElement} options.cursorLabel - footer coordinate label.
     * @param {HTMLElement} options.hintLabel - footer hint label.
     * @param {(selection: object|undefined) => void} [options.onSelect] - selection callback.
     */
    constructor({ canvas, tooltip, cursorLabel, hintLabel, onSelect }) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.cursorLabel = cursorLabel;
      this.hintLabel = hintLabel;
      this.onSelect = onSelect;
      this.ctx = canvas.getContext('2d');

      /** @type {object|undefined} */
      this.data = undefined;
      this.viewStart = 0;
      this.viewEnd = 1;
      this.scrollY = 0;
      this.hiddenCategories = new Set();
      this.hiddenGroups = new Set();
      this.sortMode = 'total';
      this.rowLimit = 40;
      this.filterName = undefined;
      this.filterCategory = undefined;
      this.hover = undefined;
      this.dragging = undefined;
      this.rows = [];
      this.totalHeight = 0;
      /** Entry animation progress (0 → 1) and the play cursor position. */
      this.reveal = 1;
      this.revealFrame = 0;
      this.playhead = undefined;
      this.playing = false;
      this.playFrame = 0;
      this.cancelTween = undefined;

      this.#bindEvents();
      this.resize();
    }

    // ── data ──────────────────────────────────────────────────────────────

    /** Load a dataset projection and animate it in. */
    setData(data) {
      this.data = data;
      this.viewStart = data.meta.window.start;
      this.viewEnd = data.meta.window.end;
      if (this.viewEnd <= this.viewStart) this.viewEnd = this.viewStart + 1;
      this.scrollY = 0;
      this.filterName = undefined;
      this.filterCategory = undefined;
      this.hiddenCategories.clear();
      this.hiddenGroups.clear();
      this.playhead = undefined;
      this.#layout();
      this.#startReveal();
      this.#emitSelection();
    }

    /** Apply row ordering / limit controls. */
    setOptions({ sortMode, rowLimit }) {
      if (sortMode !== undefined) this.sortMode = sortMode;
      if (rowLimit !== undefined) this.rowLimit = rowLimit;
      this.#layout();
      this.draw();
    }

    /** Toggle a category, driven by the interactive legend. */
    toggleCategory(category) {
      if (category === undefined) return;
      if (this.hiddenCategories.has(category)) this.hiddenCategories.delete(category);
      else this.hiddenCategories.add(category);
      this.#layout();
      this.draw();
      this.#emitSelection();
    }

    /** Set the visible categories outright (used by chart → swimlane linking). */
    setCategories(categories) {
      this.hiddenCategories = new Set(
        Object.keys(CATEGORY_LABELS).filter((category) => !categories.includes(category)),
      );
      this.#layout();
      this.draw();
      this.#emitSelection();
    }

    /** Toggle a device group. */
    toggleGroup(group) {
      if (this.hiddenGroups.has(group)) this.hiddenGroups.delete(group);
      else this.hiddenGroups.add(group);
      this.#layout();
      this.draw();
    }

    /** Show only one operator; `undefined` clears it. */
    filterBy(name) {
      this.filterName = name === undefined || name === '' ? undefined : name;
      if (this.filterName !== undefined) {
        // A focused operator is only useful if its row is actually visible.
        this.hiddenCategories.clear();
      }
      this.#layout();
      this.scrollY = Math.min(this.scrollY, this.maxScrollY);
      this.#scrollToFiltered();
      this.draw();
      this.#emitSelection();
    }

    /** Fit the whole window, with an eased transition when motion is on. */
    resetView() {
      if (this.data === undefined) return;
      this.#animateView(this.data.meta.window.start, this.data.meta.window.end);
    }

    /** Zoom by a factor around the centre of the viewport. */
    zoom(factor) {
      const centre = (this.viewStart + this.viewEnd) / 2;
      this.#zoomAround(centre, factor);
    }

    /** Zoom/pan so that one operator's events fill the viewport. */
    focusOperator(name) {
      if (this.data === undefined) return false;
      const row = this.rows.find((entry) => entry.kind === 'row' && entry.row.name === name)?.row
        ?? this.data.timeline.lanes.flatMap((group) => group.rows).find((candidate) => candidate.name === name);
      if (row === undefined) return false;
      this.filterBy(name);
      const events = row.events;
      if (events.length === 0) return true;
      const start = Math.min(...events.map((event) => event.start));
      const end = Math.max(...events.map((event) => event.start + event.dur));
      const pad = Math.max((end - start) * 0.08, (this.viewEnd - this.viewStart) * 0.02);
      this.#animateView(start - pad, end + pad);
      return true;
    }

    /** Resize the canvas to its CSS box at device pixel ratio. */
    resize() {
      const ratio = global.devicePixelRatio || 1;
      const width = this.canvas.clientWidth || 900;
      const height = Number(this.canvas.getAttribute('height')) || 520;
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      this.canvas.style.height = `${String(height)}px`;
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.cssWidth = width;
      this.cssHeight = height;
      this.draw();
    }

    /** @returns {number} maximum vertical scroll offset. */
    get maxScrollY() {
      return Math.max(0, this.totalHeight - (this.cssHeight - TOP_AXIS));
    }

    // ── playback ──────────────────────────────────────────────────────────

    /**
     * Play the timeline: a vertical cursor sweeps the window at `speed`× real
     * time, and events under it are highlighted. Useful for reading causality
     * (for example: is the device idle *because* the host is inside a synchronise?).
     *
     * @param {boolean} playing - start or stop.
     * @param {number} [speed] - multiplier of real time.
     */
    setPlaying(playing, speed = 4) {
      this.playing = playing;
      this.playSpeed = speed;
      cancelAnimationFrame(this.playFrame);
      if (!playing || this.data === undefined) {
        this.draw();
        return;
      }
      const window_ = this.data.meta.window;
      if (this.playhead === undefined || this.playhead >= window_.end) this.playhead = window_.start;
      const step = (now) => {
        if (!this.playing) return;
        const previous = this.lastFrameAt ?? now;
        this.lastFrameAt = now;
        // Wall-clock based so the sweep speed does not depend on frame rate:
        // `speed` × real time, clamped per frame so a stalled tab does not jump
        // a whole window at once.
        const advance = Math.min(400, now - previous) * (this.playSpeed ?? speed) * 1000;
        this.playhead = (this.playhead ?? window_.start) + advance;
        if (this.playhead >= window_.end) this.playhead = window_.start;
        this.draw();
        this.playFrame = requestAnimationFrame(step);
      };
      this.lastFrameAt = undefined;
      this.playFrame = requestAnimationFrame(step);
    }

    // ── layout ────────────────────────────────────────────────────────────

    #layout() {
      const rows = [];
      if (this.data === undefined) {
        this.rows = rows;
        this.totalHeight = 0;
        return;
      }
      for (const group of this.data.timeline.lanes) {
        if (this.hiddenGroups.has(group.id)) continue;
        rows.push({ kind: 'header', group: group.id, label: group.label, totalUs: group.totalUs, eventCount: group.eventCount });
        const eligible = group.rows.filter((row) => this.#rowVisible(row));
        const sorted = this.#sortRows(eligible).slice(0, this.rowLimit);
        for (const row of sorted) rows.push({ kind: 'row', group: group.id, row });
        if (eligible.length > sorted.length) {
          rows.push({ kind: 'more', group: group.id, label: `其余 ${String(eligible.length - sorted.length)} 个算子未显示（可上调行数）` });
        }
      }
      this.rows = rows;
      let y = TOP_AXIS;
      for (const entry of rows) {
        entry.y = y;
        entry.height = entry.kind === 'row' ? ROW_HEIGHT : entry.kind === 'header' ? GROUP_HEADER : 16;
        y += entry.height + (entry.kind === 'row' ? ROW_GAP : 0);
      }
      this.totalHeight = y + 6;
    }

    #rowVisible(row) {
      if (this.hiddenCategories.has(row.category) && row.overflow !== true) return false;
      if (this.filterName !== undefined && row.name !== this.filterName) return false;
      return true;
    }

    #sortRows(rows) {
      const sorted = [...rows];
      switch (this.sortMode) {
        case 'count': sorted.sort((left, right) => right.count - left.count); break;
        case 'name': sorted.sort((left, right) => left.name.localeCompare(right.name)); break;
        case 'max': sorted.sort((left, right) => right.maxUs - left.maxUs); break;
        default: sorted.sort((left, right) => right.totalUs - left.totalUs);
      }
      return sorted;
    }

    /** Scroll the viewport so the filtered row is on screen. */
    #scrollToFiltered() {
      if (this.filterName === undefined) return;
      const entry = this.rows.find((candidate) => candidate.kind === 'row' && candidate.row.name === this.filterName);
      if (entry === undefined) return;
      const top = entry.y - TOP_AXIS;
      if (top < this.scrollY || top > this.scrollY + this.cssHeight - TOP_AXIS - ROW_HEIGHT) {
        this.scrollY = Math.max(0, Math.min(this.maxScrollY, top - 40));
      }
    }

    // ── animation ─────────────────────────────────────────────────────────

    /** Sweep the timeline from left to right once, as a "loaded" cue. */
    #startReveal() {
      cancelAnimationFrame(this.revealFrame);
      if (!motionEnabled()) {
        this.reveal = 1;
        this.draw();
        return;
      }
      const started = performance.now();
      const duration = 620;
      const step = (now) => {
        const progress = Math.min(1, (now - started) / duration);
        this.reveal = 1 - (1 - progress) ** 2;
        this.draw();
        if (progress < 1) this.revealFrame = requestAnimationFrame(step);
        else this.reveal = 1;
      };
      this.reveal = 0;
      this.revealFrame = requestAnimationFrame(step);
    }

    /** Tween the visible window (zoom / fit / focus). */
    #animateView(start, end) {
      this.cancelTween?.();
      const full = this.data?.meta.window ?? { start: 0, end: 1 };
      const clamp = (from, to) => {
        const span = to - from;
        const fullSpan = Math.max(1, full.end - full.start);
        if (span >= fullSpan) return { start: full.start, end: full.end };
        let nextStart = from;
        if (nextStart < full.start) nextStart = full.start;
        if (nextStart + span > full.end) nextStart = full.end - span;
        return { start: nextStart, end: nextStart + span };
      };
      const target = clamp(start, end);
      this.cancelTween = tween({
        from: { start: this.viewStart, end: this.viewEnd },
        to: target,
        duration: 260,
        onFrame: ({ start: nextStart, end: nextEnd }) => {
          this.viewStart = nextStart;
          this.viewEnd = nextEnd;
          this.draw();
        },
      });
    }

    // ── drawing ───────────────────────────────────────────────────────────

    /** Draw the whole view. */
    draw() {
      const ctx = this.ctx;
      if (this.cssWidth === undefined) return;
      const styles = getComputedStyle(document.body);
      const ink = styles.getPropertyValue('--ink').trim() || '#111';
      const ink3 = styles.getPropertyValue('--ink-3').trim() || '#888';
      const line = styles.getPropertyValue('--line').trim() || '#ddd';
      const panel = styles.getPropertyValue('--panel').trim() || '#fff';
      const panel2 = styles.getPropertyValue('--panel-2').trim() || '#fafbfc';
      const accent = styles.getPropertyValue('--accent').trim() || '#1d4ed8';

      ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);

      if (this.data === undefined) {
        ctx.fillStyle = ink3;
        ctx.font = FONT;
        ctx.fillText('尚未加载数据集', LEFT_GUTTER, 40);
        return;
      }

      const plotLeft = LEFT_GUTTER;
      const plotWidth = Math.max(20, this.cssWidth - LEFT_GUTTER - 8);
      const span = this.viewEnd - this.viewStart || 1;
      const timeToX = (us) => plotLeft + ((us - this.viewStart) / span) * plotWidth;

      this.#drawAxis(ctx, { timeToX, plotLeft, plotWidth, ink3, line, panel });

      ctx.save();
      ctx.beginPath();
      ctx.rect(0, TOP_AXIS, this.cssWidth, this.cssHeight - TOP_AXIS);
      ctx.clip();
      ctx.translate(0, -this.scrollY);

      // Reveal mask: bars left of `revealX` are painted.
      const revealX = plotLeft + this.reveal * (plotWidth + 8);

      for (const entry of this.rows) {
        const top = entry.y;
        if (top - this.scrollY + entry.height < TOP_AXIS) continue;
        if (top - this.scrollY > this.cssHeight) break;
        if (entry.kind === 'header') {
          ctx.fillStyle = panel2;
          ctx.fillRect(0, top, this.cssWidth, entry.height);
          ctx.fillStyle = ink;
          ctx.font = FONT_LABEL;
          ctx.fillText(entry.label, 8, top + 14);
          ctx.fillStyle = ink3;
          ctx.font = FONT_SMALL;
          const detail = `${formatUs(entry.totalUs)} · ${formatCount(entry.eventCount)} 条`;
          ctx.fillText(detail, LEFT_GUTTER - ctx.measureText(detail).width - 10, top + 14);
          ctx.strokeStyle = line;
          ctx.beginPath();
          ctx.moveTo(0, top + entry.height + 0.5);
          ctx.lineTo(this.cssWidth, top + entry.height + 0.5);
          ctx.stroke();
          continue;
        }
        if (entry.kind === 'more') {
          ctx.fillStyle = ink3;
          ctx.font = FONT_SMALL;
          ctx.fillText(entry.label, 12, top + 11);
          continue;
        }
        this.#drawRow(ctx, entry, { timeToX, plotLeft, revealX, accent });
      }

      // Time cursor: playback head or hover guide.
      const cursorUs = this.playhead ?? (this.hover?.event === undefined ? undefined : this.hover.event.start);
      if (cursorUs !== undefined) {
        const x = timeToX(cursorUs);
        const playing = this.playhead !== undefined;
        ctx.strokeStyle = playing ? accent : line;
        ctx.lineWidth = playing ? 1.5 : 1;
        ctx.setLineDash(playing ? [] : [3, 3]);
        ctx.globalAlpha = playing ? 0.9 : 1;
        ctx.beginPath();
        ctx.moveTo(x, TOP_AXIS - this.scrollY);
        ctx.lineTo(x, this.cssHeight - this.scrollY);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.globalAlpha = 1;
        ctx.lineWidth = 1;
        if (playing) {
          ctx.fillStyle = accent;
          ctx.font = FONT_SMALL;
          const label = formatAxis(cursorUs - this.data.meta.window.start);
          const width = ctx.measureText(label).width + 8;
          const labelX = Math.min(this.cssWidth - width - 2, x + 4);
          ctx.fillRect(labelX, TOP_AXIS - this.scrollY + 2, width, 14);
          ctx.fillStyle = panel;
          ctx.fillText(label, labelX + 4, TOP_AXIS - this.scrollY + 12);
        }
      }
      ctx.restore();

      // Left gutter last, so rows scroll underneath it.
      ctx.fillStyle = panel;
      ctx.fillRect(0, TOP_AXIS, LEFT_GUTTER, this.cssHeight - TOP_AXIS);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, TOP_AXIS, LEFT_GUTTER, this.cssHeight - TOP_AXIS);
      ctx.clip();
      ctx.translate(0, -this.scrollY);
      const dimmed = this.filterName !== undefined;
      for (const entry of this.rows) {
        const top = entry.y;
        if (top - this.scrollY + entry.height < TOP_AXIS) continue;
        if (top - this.scrollY > this.cssHeight) break;
        if (entry.kind !== 'row') continue;
        const row = entry.row;
        const selected = this.filterName === row.name;
        ctx.globalAlpha = dimmed && !selected ? 0.45 : 1;
        ctx.fillStyle = selected ? accent : ink;
        ctx.font = selected ? FONT_LABEL : FONT;
        ctx.fillText(truncate(row.label, 28), 10, top + 12);
        ctx.fillStyle = ink3;
        ctx.font = FONT_SMALL;
        const meta = `${formatUs(row.totalUs)} ×${formatCount(row.count)}`;
        ctx.fillText(meta, LEFT_GUTTER - ctx.measureText(meta).width - 10, top + 12);
        ctx.globalAlpha = 1;
      }
      ctx.restore();
      ctx.strokeStyle = line;
      ctx.beginPath();
      ctx.moveTo(LEFT_GUTTER + 0.5, TOP_AXIS);
      ctx.lineTo(LEFT_GUTTER + 0.5, this.cssHeight);
      ctx.stroke();

      this.#drawScrollHint(ctx, ink3, line);
    }

    #drawAxis(ctx, { timeToX, plotLeft, plotWidth, ink3, line, panel }) {
      ctx.fillStyle = panel;
      ctx.fillRect(0, 0, this.cssWidth, TOP_AXIS);
      const ticks = niceTicks(this.viewStart, this.viewEnd, Math.max(3, Math.floor(plotWidth / 120)));
      ctx.font = FONT_SMALL;
      ctx.strokeStyle = line;
      for (const tick of ticks) {
        const x = Math.round(timeToX(tick)) + 0.5;
        if (x < plotLeft) continue;
        ctx.beginPath();
        ctx.moveTo(x, TOP_AXIS - 5);
        ctx.lineTo(x, this.cssHeight);
        ctx.globalAlpha = 0.3;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink3;
        ctx.fillText(formatAxis(tick - this.data.meta.window.start), x + 3, 11);
      }
      ctx.fillStyle = ink3;
      ctx.fillText(`窗口 ${formatUs(this.viewEnd - this.viewStart)}`, 8, 11);
    }

    #drawRow(ctx, entry, { timeToX, plotLeft, revealX, accent }) {
      const row = entry.row;
      const top = entry.y;
      const selected = this.filterName === row.name;
      if (selected) {
        ctx.fillStyle = 'rgba(29,78,216,0.08)';
        ctx.fillRect(0, top - 1, this.cssWidth, ROW_HEIGHT + 2);
      }
      const color = row.overflow === true ? '#aab' : (CATEGORY_COLORS[row.category] ?? '#888');
      const dim = this.filterName !== undefined && !selected;
      const span = this.viewEnd - this.viewStart || 1;
      const plotWidth = this.cssWidth - LEFT_GUTTER - 8;
      for (const event of row.events) {
        const x = timeToX(event.start);
        if (x > revealX) break; // entry animation: not yet revealed
        const width = Math.max(0.6, (event.dur / span) * plotWidth);
        if (x + width < plotLeft) continue;
        const hovered = this.hover !== undefined && this.hover.rowKey === row.key && this.hover.event === event;
        const atPlayhead = this.playhead !== undefined && this.playhead >= event.start && this.playhead <= event.start + event.dur;
        ctx.globalAlpha = dim ? 0.22 : (hovered || atPlayhead ? 1 : 0.85);
        ctx.fillStyle = atPlayhead && !hovered ? accent : color;
        ctx.fillRect(x, top, Math.max(0.7, width), ROW_HEIGHT);
        ctx.globalAlpha = 1;
        if (hovered) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x - 0.5, top - 0.5, Math.max(1.2, width) + 1, ROW_HEIGHT + 1);
          ctx.lineWidth = 1;
        }
      }
    }

    /** A slim scroll indicator on the right edge when rows overflow. */
    #drawScrollHint(ctx, ink3, line) {
      const viewport = this.cssHeight - TOP_AXIS;
      if (this.totalHeight <= viewport) return;
      const trackHeight = viewport - 8;
      const thumbHeight = Math.max(24, (viewport / this.totalHeight) * trackHeight);
      const thumbTop = TOP_AXIS + 4 + (this.scrollY / this.maxScrollY || 0) * (trackHeight - thumbHeight);
      ctx.fillStyle = line;
      ctx.globalAlpha = 0.7;
      ctx.fillRect(this.cssWidth - 4, thumbTop, 3, thumbHeight);
      ctx.globalAlpha = 1;
      void ink3;
    }

    // ── interaction ───────────────────────────────────────────────────────

    #bindEvents() {
      const canvas = this.canvas;

      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const { x } = this.#pointer(event);
        if (event.ctrlKey || event.metaKey) {
          this.#zoomAround(this.#xToTime(x), event.deltaY > 0 ? 1.16 : 1 / 1.16);
          return;
        }
        if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
          const delta = (event.deltaX !== 0 ? event.deltaX : event.deltaY) / Math.max(1, this.cssWidth - LEFT_GUTTER);
          const shift = delta * (this.viewEnd - this.viewStart);
          this.viewStart += shift;
          this.viewEnd += shift;
          this.#clampView();
          this.draw();
          return;
        }
        this.scrollY = Math.min(this.maxScrollY, Math.max(0, this.scrollY + event.deltaY));
        this.draw();
      }, { passive: false });

      canvas.addEventListener('pointerdown', (event) => {
        const point = this.#pointer(event);
        if (point.x < LEFT_GUTTER) return;
        this.dragging = { x: point.x, viewStart: this.viewStart, viewEnd: this.viewEnd, moved: false };
        canvas.setPointerCapture(event.pointerId);
      });

      canvas.addEventListener('pointermove', (event) => {
        const point = this.#pointer(event);
        if (this.dragging !== undefined) {
          const dx = point.x - this.dragging.x;
          if (Math.abs(dx) > 2) this.dragging.moved = true;
          const span = this.dragging.viewEnd - this.dragging.viewStart;
          const shift = (dx / Math.max(1, this.cssWidth - LEFT_GUTTER)) * span;
          this.viewStart = this.dragging.viewStart - shift;
          this.viewEnd = this.dragging.viewEnd - shift;
          this.#clampView();
          this.draw();
          return;
        }
        this.#updateHover(point);
      });

      canvas.addEventListener('pointerup', (event) => {
        const point = this.#pointer(event);
        const dragging = this.dragging;
        this.dragging = undefined;
        if (dragging === undefined || dragging.moved) return;
        const hit = this.#hitTest(point);
        if (hit === undefined) {
          this.filterBy(undefined);
          return;
        }
        this.filterBy(this.filterName === hit.row.name ? undefined : hit.row.name);
      });

      canvas.addEventListener('pointerleave', () => {
        this.tooltip.hidden = true;
        this.hover = undefined;
        this.cursorLabel.textContent = '';
        this.draw();
      });

      canvas.addEventListener('dblclick', () => this.resetView());
    }

    /** Escape clears the operator filter (bound by the controller). */
    clearFilter() {
      this.filterBy(undefined);
    }

    #pointer(event) {
      const rect = this.canvas.getBoundingClientRect();
      return { x: event.clientX - rect.left, y: event.clientY - rect.top };
    }

    #xToTime(x) {
      const plotWidth = Math.max(20, this.cssWidth - LEFT_GUTTER - 8);
      return this.viewStart + ((x - LEFT_GUTTER) / plotWidth) * (this.viewEnd - this.viewStart);
    }

    #zoomAround(time, factor) {
      const span = this.viewEnd - this.viewStart;
      const nextSpan = Math.min(Math.max(span * factor, 1), 1e9);
      const ratio = span === 0 ? 0.5 : (time - this.viewStart) / span;
      const start = time - nextSpan * ratio;
      this.#animateView(start, start + nextSpan);
    }

    #clampView() {
      const full = this.data.meta.window;
      const span = this.viewEnd - this.viewStart;
      const fullSpan = Math.max(1, full.end - full.start);
      if (span >= fullSpan) {
        this.viewStart = full.start;
        this.viewEnd = full.end;
        return;
      }
      if (this.viewStart < full.start) {
        this.viewStart = full.start;
        this.viewEnd = full.start + span;
      }
      if (this.viewEnd > full.end) {
        this.viewEnd = full.end;
        this.viewStart = full.end - span;
      }
    }

    /** Find the row and bar under a point, honouring the scroll offset. */
    #hitTest(point) {
      if (this.data === undefined) return undefined;
      const contentY = point.y + this.scrollY;
      for (const entry of this.rows) {
        if (entry.kind !== 'row') continue;
        if (contentY < entry.y || contentY > entry.y + ROW_HEIGHT) continue;
        const time = this.#xToTime(point.x);
        const event = entry.row.events.find((item) => time >= item.start && time <= item.start + item.dur);
        return { row: entry.row, event };
      }
      return undefined;
    }

    #updateHover(point) {
      if (this.data === undefined) return;
      const time = this.#xToTime(point.x);
      const relative = time - this.data.meta.window.start;
      this.cursorLabel.textContent = point.x >= LEFT_GUTTER ? `t = ${formatAxis(relative)}` : '';

      // Row-level highlight even when no bar is under the cursor.
      const contentY = point.y + this.scrollY;
      const entry = this.rows.find((candidate) => candidate.kind === 'row'
        && contentY >= candidate.y && contentY <= candidate.y + ROW_HEIGHT);
      const hit = this.#hitTest(point);
      this.hover = hit === undefined ? undefined : { rowKey: hit.row.key, event: hit.event, row: hit.row };

      if (hit === undefined) {
        this.tooltip.hidden = true;
        this.draw();
        return;
      }
      // While playing, rows light up as the cursor crosses them and the tooltip
      // would fight the animation for attention — the footer carries the time.
      if (this.playhead === undefined) this.#renderTooltip(hit.row, hit.event, point, entry !== undefined);
      else this.tooltip.hidden = true;
      this.draw();
    }

    #renderTooltip(row, event, point, rowAligned) {
      const sample = row.sample ?? {};
      const lines = [
        ['算子', row.name],
        ['类别', `${CATEGORY_LABELS[row.category] ?? row.category}${sample.subtype === undefined ? '' : ` · ${sample.subtype}`}`],
        ['侧', `${row.group === 'host' ? 'Host（CPU）' : 'Device（昇腾 NPU）'}${sample.rank === undefined ? '' : ` · rank ${String(sample.rank)}`}`],
      ];
      if (event !== undefined) {
        lines.push(
          ['开始', formatAxis(event.start - (this.data.meta.window.start ?? 0))],
          ['本次耗时', formatUs(event.dur)],
        );
      }
      lines.push(
        ['调用次数', formatCount(row.count)],
        ['累计', `${formatUs(row.totalUs)}（均值 ${formatUs(row.avgUs)}，p95 ${formatUs(row.p95Us)}）`],
      );
      const opType = sample.opType ?? sample.taskType;
      if (opType !== undefined) lines.push(['OP/Task Type', String(opType)]);
      const shapesIn = formatShapes(sample.shapesIn);
      if (shapesIn !== undefined) lines.push(['输入 shape', shapesIn]);
      const stack = formatStack(sample.callStack);
      if (stack !== undefined) lines.push(['调用栈', stack]);
      if (row.eventsTruncated === true) {
        lines.push(['视图抽样', `显示 ${formatCount(row.eventsShipped)}/${formatCount(row.eventsTotal)} 条（点击可筛选）`]);
      }
      lines.push(['操作', rowAligned && event !== undefined ? '点击筛选该算子' : '点击行内算子条可筛选']);

      const body = lines.map(([key, value]) => [global.VAP.h('dt', {}, key), global.VAP.h('dd', {}, String(value))]);
      this.tooltip.replaceChildren(
        global.VAP.h('div.tt-title', {}, row.name),
        global.VAP.h('dl', {}, body.flat()),
      );
      this.tooltip.hidden = false;
      const box = this.tooltip.getBoundingClientRect();
      const left = Math.min(point.x + 16, Math.max(4, this.cssWidth - box.width - 8));
      const top = Math.min(point.y + 16, Math.max(4, this.cssHeight - box.height - 8));
      this.tooltip.style.left = `${String(left)}px`;
      this.tooltip.style.top = `${String(top)}px`;
    }

    #emitSelection() {
      if (this.hintLabel !== undefined) {
        const parts = [
          this.filterName === undefined ? '点击算子条筛选' : `已筛选：${this.filterName}`,
          'Ctrl/⌘+滚轮缩放 · 拖拽平移 · 双击重置 · Esc 清除',
        ];
        this.hintLabel.textContent = parts.join(' · ');
      }
      if (typeof this.onSelect === 'function') {
        this.onSelect({ operator: this.filterName, categories: Object.keys(CATEGORY_LABELS).filter((category) => !this.hiddenCategories.has(category)) });
      }
    }
  }

  /** Choose "nice" tick values covering a range. */
  function niceTicks(start, end, count) {
    const span = Math.max(1, end - start);
    const rawStep = span / Math.max(1, count);
    const magnitude = 10 ** Math.floor(Math.log10(rawStep));
    const candidates = [1, 2, 2.5, 5, 10].map((factor) => factor * magnitude);
    const step = candidates.find((value) => value >= rawStep) ?? candidates[candidates.length - 1];
    const ticks = [];
    for (let value = Math.ceil(start / step) * step; value <= end; value += step) ticks.push(value);
    return ticks;
  }

  /** Axis label in a magnitude-appropriate unit. */
  function formatAxis(us) {
    if (!Number.isFinite(us)) return 'N/A';
    if (Math.abs(us) >= 1e6) return `${(us / 1e6).toFixed(2)}s`;
    if (Math.abs(us) >= 1000) return `${(us / 1000).toFixed(2)}ms`;
    return `${us.toFixed(1)}µs`;
  }

  function truncate(text, limit) {
    const value = String(text);
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  global.VAP = global.VAP ?? {};
  global.VAP.GanttView = GanttView;
  global.VAP.ganttAxisLabel = formatAxis;
})(window);
