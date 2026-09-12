/**
 * Module 1: the Host / Device operator swimlane (Gantt) timeline.
 *
 * Canvas rather than DOM: a real profile holds thousands of bars per lane and
 * hundreds of lanes, and DOM nodes cannot be created for them at 60 fps.
 *
 * Interactions:
 *   * wheel / ctrl+wheel zoom around the cursor; drag to pan; double-click reset
 *   * hover shows the operator card (name, start, duration, call count, shapes,
 *     category, device, rank, stream, call stack)
 *   * click a bar selects that operator and filters the ranking table to it
 *   * per-category and per-group toggles hide lanes/bars
 *   * the row order follows the dataset's total-duration ranking, so the most
 *     expensive operator of each group is always at the top of its block
 *
 * The renderer is device-pixel-ratio aware and only draws what the viewport
 * shows, which is what keeps panning smooth on a sampled 150k-event timeline.
 */
(function attachGantt(global) {
  'use strict';

  const {
    CATEGORY_COLORS, CATEGORY_LABELS, formatUs, formatCount, formatShapes, formatStack,
  } = global.VAP;

  const ROW_HEIGHT = 16;
  const ROW_GAP = 3;
  const GROUP_HEADER = 22;
  const LEFT_GUTTER = 232;
  const TOP_AXIS = 24;
  const FONT = '11px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
  const FONT_SMALL = '10px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';

  class GanttView {
    /**
     * @param {object} options - view wiring.
     * @param {HTMLCanvasElement} options.canvas - target canvas.
     * @param {HTMLElement} options.tooltip - hover card element.
     * @param {HTMLElement} options.cursorLabel - footer coordinate label.
     * @param {HTMLElement} options.selectionLabel - footer selection label.
     * @param {(selection: object|undefined) => void} [options.onSelect] - selection callback.
     */
    constructor({ canvas, tooltip, cursorLabel, selectionLabel, onSelect }) {
      this.canvas = canvas;
      this.tooltip = tooltip;
      this.cursorLabel = cursorLabel;
      this.selectionLabel = selectionLabel;
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
      this.rowLimit = 60;
      this.filterName = undefined;
      this.hover = undefined;
      this.dragging = undefined;
      this.rows = [];
      this.totalHeight = 0;

      this.#bindEvents();
      this.resize();
    }

    /** Load a dataset projection. */
    setData(data) {
      this.data = data;
      this.viewStart = data.meta.window.start;
      this.viewEnd = data.meta.window.end;
      if (this.viewEnd <= this.viewStart) this.viewEnd = this.viewStart + 1;
      this.scrollY = 0;
      this.filterName = undefined;
      this.#layout();
      this.draw();
      this.#emitSelection();
    }

    /** Apply the row ordering / limit controls. */
    setOptions({ sortMode, rowLimit }) {
      if (sortMode !== undefined) this.sortMode = sortMode;
      if (rowLimit !== undefined) this.rowLimit = rowLimit;
      this.#layout();
      this.draw();
    }

    /** Toggle a category's visibility. */
    toggleCategory(category) {
      if (this.hiddenCategories.has(category)) this.hiddenCategories.delete(category);
      else this.hiddenCategories.add(category);
      this.#layout();
      this.draw();
    }

    /** Toggle a device group's visibility. */
    toggleGroup(group) {
      if (this.hiddenGroups.has(group)) this.hiddenGroups.delete(group);
      else this.hiddenGroups.add(group);
      this.#layout();
      this.draw();
    }

    /** Show only one operator (click-to-filter); `undefined` clears it. */
    filterBy(name) {
      this.filterName = name === undefined || name === '' ? undefined : name;
      // The row list depends on the filter, so the layout must be rebuilt before
      // the next paint (otherwise the previous rows stay on screen).
      this.#layout();
      this.scrollY = Math.min(this.scrollY, this.maxScrollY);
      this.draw();
      this.#emitSelection();
    }

    /** Fit the whole window. */
    resetView() {
      if (this.data === undefined) return;
      this.viewStart = this.data.meta.window.start;
      this.viewEnd = this.data.meta.window.end;
      this.draw();
    }

    /**
     * Render the whole capture window into a PNG data URL for the PDF report.
     *
     * The viewport is reset, drawn, captured and restored, so the export always
     * shows the complete timeline rather than whatever slice happened to be on
     * screen — and the user's zoom is unchanged afterwards.
     *
     * @returns {string|undefined} `data:image/png;base64,…`, or undefined when no data is loaded.
     */
    toDataUrl() {
      if (this.data === undefined || typeof this.canvas.toDataURL !== 'function') return undefined;
      const savedStart = this.viewStart;
      const savedEnd = this.viewEnd;
      const savedScroll = this.scrollY;
      try {
        this.viewStart = this.data.meta.window.start;
        this.viewEnd = this.data.meta.window.end;
        this.scrollY = 0;
        this.draw();
        return this.canvas.toDataURL('image/png');
      } catch {
        return undefined;
      } finally {
        this.viewStart = savedStart;
        this.viewEnd = savedEnd;
        this.scrollY = savedScroll;
        this.draw();
      }
    }

    /** Zoom by a factor around the centre of the viewport. */
    zoom(factor) {
      const centre = (this.viewStart + this.viewEnd) / 2;
      this.#zoomAround(centre, factor);
      this.draw();
    }

    /** Resize the canvas to its CSS box at device pixel ratio. */
    resize() {
      const ratio = global.devicePixelRatio || 1;
      const width = this.canvas.clientWidth || 900;
      const height = Number(this.canvas.getAttribute('height')) || 520;
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
      // The canvas element's layout height must be set explicitly: assigning
      // `canvas.height` sets the backing store, whose default CSS size would
      // otherwise double the row area on a HiDPI display.
      this.canvas.style.height = `${String(height)}px`;
      this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      this.cssWidth = width;
      this.cssHeight = height;
      this.draw();
    }

    /** Maximum vertical scroll offset for the current row layout. */
    get maxScrollY() {
      return Math.max(0, this.totalHeight - (this.cssHeight - TOP_AXIS));
    }

    // ── layout ────────────────────────────────────────────────────────────

    /** Build the visible row list (group headers + operator rows). */
    #layout() {
      const rows = [];
      if (this.data === undefined) {
        this.rows = rows;
        this.totalHeight = 0;
        return;
      }
      for (const group of this.data.timeline.lanes) {
        if (this.hiddenGroups.has(group.id)) continue;
        rows.push({ kind: 'header', group: group.id, label: group.label, description: group.description, totalUs: group.totalUs, eventCount: group.eventCount });
        const eligible = group.rows.filter((row) => this.#rowVisible(row));
        const sorted = this.#sortRows(eligible).slice(0, this.rowLimit);
        for (const row of sorted) rows.push({ kind: 'row', group: group.id, row });
        if (eligible.length > sorted.length) {
          rows.push({
            kind: 'more',
            group: group.id,
            label: `其余 ${String(eligible.length - sorted.length)} 个算子未显示（可在工具栏把行数上调，或查看算子耗时排行表）`,
          });
        }
      }
      this.rows = rows;
      let y = TOP_AXIS;
      for (const entry of rows) {
        entry.y = y;
        entry.height = entry.kind === 'row' ? ROW_HEIGHT : entry.kind === 'header' ? GROUP_HEADER : 18;
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

    // ── drawing ───────────────────────────────────────────────────────────

    /** Draw the whole view. */
    draw() {
      const ctx = this.ctx;
      if (this.cssWidth === undefined) return;
      ctx.clearRect(0, 0, this.cssWidth, this.cssHeight);
      const styles = getComputedStyle(document.body);
      const ink = styles.getPropertyValue('--ink').trim() || '#111';
      const ink3 = styles.getPropertyValue('--ink-3').trim() || '#888';
      const line = styles.getPropertyValue('--line').trim() || '#ddd';
      const panel = styles.getPropertyValue('--panel').trim() || '#fff';
      const panel2 = styles.getPropertyValue('--panel-2').trim() || '#fafbfc';
      const accent = styles.getPropertyValue('--accent').trim() || '#1d4ed8';

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

      // ── time axis ──
      this.#drawAxis(ctx, { timeToX, plotLeft, plotWidth, span, ink3, line });

      // ── rows ──
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, TOP_AXIS, this.cssWidth, this.cssHeight - TOP_AXIS);
      ctx.clip();
      ctx.translate(0, -this.scrollY);

      for (const entry of this.rows) {
        const top = entry.y;
        if (top - this.scrollY + entry.height < TOP_AXIS) continue;
        if (top - this.scrollY > this.cssHeight) break;
        if (entry.kind === 'header') {
          ctx.fillStyle = panel2;
          ctx.fillRect(0, top, this.cssWidth, entry.height);
          ctx.fillStyle = ink;
          ctx.font = '600 11px "Segoe UI", "Microsoft YaHei", system-ui, sans-serif';
          ctx.fillText(entry.label, 8, top + 15);
          ctx.fillStyle = ink3;
          ctx.font = FONT_SMALL;
          const detail = `${formatUs(entry.totalUs)} · ${formatCount(entry.eventCount)} 个算子条`;
          ctx.fillText(detail, LEFT_GUTTER - ctx.measureText(detail).width - 10, top + 15);
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
          ctx.fillText(entry.label, 12, top + 12);
          continue;
        }
        this.#drawRow(ctx, entry, { timeToX, plotLeft, plotWidth, ink, ink3, line, accent });
      }

      // ── selection / hover guides ──
      if (this.hover !== undefined) {
        ctx.strokeStyle = accent;
        ctx.setLineDash([3, 3]);
        const x = timeToX(this.hover.event.start);
        ctx.beginPath();
        ctx.moveTo(x, TOP_AXIS - this.scrollY);
        ctx.lineTo(x, this.cssHeight - this.scrollY);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.restore();

      // ── left gutter (drawn last so rows scroll under it) ──
      ctx.fillStyle = panel;
      ctx.fillRect(0, TOP_AXIS, LEFT_GUTTER, this.cssHeight - TOP_AXIS);
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, TOP_AXIS, LEFT_GUTTER, this.cssHeight - TOP_AXIS);
      ctx.clip();
      ctx.translate(0, -this.scrollY);
      for (const entry of this.rows) {
        const top = entry.y;
        if (top - this.scrollY + entry.height < TOP_AXIS) continue;
        if (top - this.scrollY > this.cssHeight) break;
        if (entry.kind !== 'row') continue;
        const row = entry.row;
        ctx.fillStyle = this.filterName === row.name ? accent : ink;
        ctx.font = FONT;
        const label = truncate(row.label, 30);
        ctx.fillText(label, 10, top + 12);
        ctx.fillStyle = ink3;
        ctx.font = FONT_SMALL;
        const meta = `${formatUs(row.totalUs)} ×${formatCount(row.count)}`;
        ctx.fillText(meta, LEFT_GUTTER - ctx.measureText(meta).width - 10, top + 12);
      }
      ctx.restore();
      ctx.strokeStyle = line;
      ctx.beginPath();
      ctx.moveTo(LEFT_GUTTER + 0.5, TOP_AXIS);
      ctx.lineTo(LEFT_GUTTER + 0.5, this.cssHeight);
      ctx.stroke();
    }

    #drawAxis(ctx, { timeToX, plotLeft, plotWidth, span, ink3, line }) {
      ctx.fillStyle = 'var(--panel)';
      const styles = getComputedStyle(document.body);
      ctx.fillStyle = styles.getPropertyValue('--panel').trim() || '#fff';
      ctx.fillRect(0, 0, this.cssWidth, TOP_AXIS);
      const ticks = niceTicks(this.viewStart, this.viewEnd, Math.max(3, Math.floor(plotWidth / 110)));
      ctx.font = FONT_SMALL;
      ctx.strokeStyle = line;
      for (const tick of ticks) {
        const x = Math.round(timeToX(tick)) + 0.5;
        if (x < plotLeft) continue;
        ctx.beginPath();
        ctx.moveTo(x, TOP_AXIS - 6);
        ctx.lineTo(x, this.cssHeight);
        ctx.globalAlpha = 0.35;
        ctx.stroke();
        ctx.globalAlpha = 1;
        ctx.fillStyle = ink3;
        const label = formatAxis(tick - this.data.meta.window.start);
        ctx.fillText(label, x + 3, 12);
      }
      ctx.fillStyle = ink3;
      ctx.fillText(`窗口 ${formatUs(span)}`, 8, 12);
    }

    #drawRow(ctx, entry, { timeToX, plotLeft, ink3, line, accent }) {
      const row = entry.row;
      const top = entry.y;
      const selected = this.filterName === row.name;
      if (selected) {
        ctx.fillStyle = 'rgba(29,78,216,0.07)';
        ctx.fillRect(0, top - 1, this.cssWidth, ROW_HEIGHT + 2);
      }
      ctx.strokeStyle = line;
      ctx.globalAlpha = 0.25;
      ctx.beginPath();
      ctx.moveTo(0, top + ROW_HEIGHT + 1.5);
      ctx.lineTo(this.cssWidth, top + ROW_HEIGHT + 1.5);
      ctx.stroke();
      ctx.globalAlpha = 1;

      const color = row.overflow === true ? '#aab' : (CATEGORY_COLORS[row.category] ?? '#888');
      const dimmed = this.hiddenCategories.has(row.category) && row.overflow !== true;
      if (dimmed) return;
      for (const event of row.events) {
        const x = timeToX(event.start);
        const width = Math.max(0.6, (event.dur / (this.viewEnd - this.viewStart || 1)) * (this.cssWidth - LEFT_GUTTER - 8));
        if (x + width < plotLeft || x > this.cssWidth) continue;
        const isHovered = this.hover !== undefined && this.hover.rowKey === row.key && this.hover.event === event;
        ctx.fillStyle = color;
        ctx.globalAlpha = isHovered ? 1 : 0.82;
        ctx.fillRect(x, top, Math.max(0.7, width), ROW_HEIGHT);
        ctx.globalAlpha = 1;
        if (isHovered) {
          ctx.strokeStyle = accent;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x - 0.5, top - 0.5, Math.max(1.2, width) + 1, ROW_HEIGHT + 1);
          ctx.lineWidth = 1;
        }
      }
      void ink3;
    }

    // ── interaction ───────────────────────────────────────────────────────

    #bindEvents() {
      const canvas = this.canvas;

      canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        const { x } = this.#pointer(event);
        if (event.ctrlKey || event.metaKey) {
          // Ctrl/Cmd + wheel zooms around the cursor.
          this.#zoomAround(this.#xToTime(x), event.deltaY > 0 ? 1.18 : 1 / 1.18);
          this.draw();
          return;
        }
        if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
          // Shift + wheel (or a horizontal trackpad) pans in time.
          const delta = (event.deltaX !== 0 ? event.deltaX : event.deltaY) / Math.max(1, this.cssWidth - LEFT_GUTTER);
          const shift = delta * (this.viewEnd - this.viewStart);
          this.viewStart += shift;
          this.viewEnd += shift;
          this.#clampView();
          this.draw();
          return;
        }
        // Plain wheel scrolls the rows vertically.
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
      this.viewStart = time - nextSpan * ratio;
      this.viewEnd = this.viewStart + nextSpan;
      this.#clampView();
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
        if (event !== undefined) return { row: entry.row, event };
        return { row: entry.row, event: undefined };
      }
      return undefined;
    }

    #updateHover(point) {
      if (this.data === undefined) return;
      const time = this.#xToTime(point.x);
      this.cursorLabel.textContent = point.x >= LEFT_GUTTER
        ? `时间 ${formatAxis(time - this.data.meta.window.start)}（绝对值 ${formatUs(time)}）`
        : '';
      const hit = this.#hitTest(point);
      this.hover = hit === undefined ? undefined : { rowKey: hit.row.key, event: hit.event, row: hit.row };
      if (hit === undefined || hit.event === undefined) {
        this.tooltip.hidden = true;
        this.draw();
        return;
      }
      this.#renderTooltip(hit.row, hit.event, point);
      this.draw();
    }

    #renderTooltip(row, event, point) {
      const sample = row.sample ?? {};
      const lines = [
        ['算子', row.name],
        ['类别', `${CATEGORY_LABELS[row.category] ?? row.category}${row.subtypeLabel === undefined ? '' : ` · ${row.subtypeLabel}`}`],
        ['设备', `${row.group === 'host' ? 'Host（CPU）' : 'Device（昇腾 NPU）'}${sample.rank === undefined ? '' : ` · rank ${String(sample.rank)}`}${sample.stream === undefined ? '' : ` · stream ${String(sample.stream)}`}`],
        ['开始（相对）', formatAxis(event.start - (this.data.meta.window.start ?? 0))],
        ['开始（绝对 µs）', String((event.start + (this.data.meta.window.start ?? 0)).toFixed(1))],
        ['本次耗时', formatUs(event.dur)],
        ['调用次数', formatCount(row.count)],
        ['累计耗时', `${formatUs(row.totalUs)}（均值 ${formatUs(row.avgUs)}，p95 ${formatUs(row.p95Us)}，最长 ${formatUs(row.maxUs)}）`],
      ];
      const opType = sample.opType ?? sample.taskType;
      if (opType !== undefined) lines.push(['OP/Task Type', String(opType)]);
      const shapesIn = formatShapes(sample.shapesIn);
      if (shapesIn !== undefined) lines.push(['输入 shape', shapesIn]);
      const shapesOut = formatShapes(sample.shapesOut);
      if (shapesOut !== undefined) lines.push(['输出 shape', shapesOut]);
      if (row.waitUs > 0) lines.push(['等待时间合计', formatUs(row.waitUs)]);
      const stack = formatStack(sample.callStack);
      if (stack !== undefined) lines.push(['调用栈', stack]);
      if (row.eventsTruncated === true) {
        lines.push(['视图抽样', `本行显示 ${formatCount(row.eventsShipped)}/${formatCount(row.eventsTotal)} 个算子条（点击可筛选该算子）`]);
      }

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
      if (this.selectionLabel !== undefined) {
        this.selectionLabel.textContent = this.filterName === undefined
          ? '未按算子筛选（点击算子条可筛选，再次点击取消）'
          : `仅显示算子：${this.filterName}`;
      }
      if (typeof this.onSelect === 'function') this.onSelect(this.filterName);
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
