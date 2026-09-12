/**
 * Module 2 charts: the category cost-share donut and the Top-N ranked bar chart,
 * drawn with native SVG (no charting dependency, works offline).
 *
 * Both charts are pure functions of a projection plus the current filter, and
 * both are interactive in the same direction as the swimlane: clicking a slice
 * or a bar reports the operator/category upward so the page can link the views.
 *
 * Motion:
 *   * the donut ring draws itself: slices are stroked arcs whose dash offset
 *     animates from "invisible" to "drawn", staggered by rank;
 *   * bars grow from zero through a single shared tween;
 *   * hovering isolates one item, and a selection dims the rest.
 */
(function attachCharts(global) {
  'use strict';

  const { h, formatUs, formatCount, CATEGORY_COLORS, CATEGORY_LABELS, motionEnabled, tween, playEnter } = global.VAP;

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /** Create an SVG element; the root gets the namespace for standalone export. */
  function svg(tag, attrs = {}, children = []) {
    const element = document.createElementNS(SVG_NS, tag);
    if (tag === 'svg') element.setAttribute('xmlns', SVG_NS);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null) continue;
      element.setAttribute(key, String(value));
    }
    for (const child of children) {
      if (child === undefined || child === null) continue;
      element.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return element;
  }

  /** Point on a circle. */
  function polar(cx, cy, radius, angle) {
    return { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
  }

  /** Ring-segment path used by the donut: stroked, never filled. */
  function ringArc(cx, cy, radius, from, to) {
    const start = polar(cx, cy, radius, from);
    const end = polar(cx, cy, radius, to);
    const largeArc = to - from > Math.PI ? 1 : 0;
    return `M ${start.x.toFixed(2)} ${start.y.toFixed(2)} A ${String(radius)} ${String(radius)} 0 ${String(largeArc)} 1 ${end.x.toFixed(2)} ${end.y.toFixed(2)}`;
  }

  /**
   * Category share donut.
   *
   * @param {object} input - `{ items, scopeLabel, totalUs, onSelect }`.
   * @returns {{element: SVGElement, legend: HTMLElement}} chart.
   */
  function renderPie({ items, scopeLabel, onSelect }) {
    const width = 300;
    const height = 220;
    const cx = 110;
    const cy = 110;
    const outer = 86;
    const thickness = 34;
    const radius = outer - thickness / 2;
    const root = svg('svg', {
      viewBox: `0 0 ${String(width)} ${String(height)}`,
      role: 'img',
      'aria-label': '算子大类耗时占比',
      style: `--slice-w:${String(thickness)}px; --slice-w-hover:${String(thickness + 7)}px`,
    });
    const legend = h('ul.pie-legend');
    const positive = items.filter((item) => item.totalUs > 0);
    const sum = positive.reduce((total, item) => total + item.totalUs, 0);

    if (sum <= 0) {
      root.append(svg('circle', { cx, cy, r: radius, fill: 'none', stroke: 'var(--line)', 'stroke-width': thickness }));
      root.append(svg('text', { x: cx, y: cy + 4, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '12' }, ['该口径下没有耗时数据']));
      return { element: root, legend };
    }

    const gap = positive.length > 1 ? 0.014 : 0;
    let angle = -Math.PI / 2;
    const slices = [];
    for (const [index, item] of positive.entries()) {
      const sweep = (item.totalUs / sum) * Math.PI * 2;
      const from = angle + gap / 2;
      const to = angle + sweep - gap / 2;
      angle += sweep;
      if (to <= from) continue;
      const color = item.color ?? CATEGORY_COLORS[item.id] ?? '#888';
      const length = radius * (to - from);
      const path = svg('path', {
        class: 'slice',
        d: ringArc(cx, cy, radius, from, to),
        fill: 'none',
        stroke: color,
        'stroke-width': thickness,
        'stroke-dasharray': length.toFixed(2),
        'stroke-dashoffset': motionEnabled() ? length.toFixed(2) : 0,
        'stroke-linecap': 'butt',
        'data-category': item.id,
        style: `transition-delay:${String(Math.min(index * 55, 320))}ms`,
      });
      path.append(svg('title', {}, [`${item.label}：${formatUs(item.totalUs)}（${item.sharePct.toFixed(2)}%，${formatCount(item.count)} 次）`]));
      if (typeof onSelect === 'function') {
        path.addEventListener('click', () => onSelect({ category: item.id }));
      }
      root.append(path);
      slices.push(path);

      const legendButton = h('button', { type: 'button', 'data-category': item.id, title: `只看${item.label}` }, [
        h('i', { style: `background:${color}` }),
        h('span', {}, item.label),
      ]);
      if (typeof onSelect === 'function') {
        legendButton.addEventListener('click', () => onSelect({ category: item.id }));
      }
      legend.append(h('li', {}, [legendButton, h('span.val', {}, `${item.sharePct.toFixed(2)}% · ${formatUs(item.totalUs)}`)]));
    }

    root.append(svg('text', { x: cx, y: cy - 3, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '11' }, ['合计']));
    root.append(svg('text', { x: cx, y: cy + 15, 'text-anchor': 'middle', fill: 'var(--ink)', 'font-size': '15', 'font-weight': '600' }, [formatUs(sum)]));
    root.append(svg('text', { x: cx, y: cy + 34, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '10.5' }, [scopeLabel ?? '']));

    // Draw-in: one rAF batch flipping every dash offset to zero.
    if (motionEnabled()) {
      requestAnimationFrame(() => {
        for (const path of slices) path.setAttribute('stroke-dashoffset', '0');
      });
    }

    return { element: root, legend };
  }

  /**
   * Top-N ranked horizontal bars.
   *
   * @param {object} input - `{ rows, dimension, maxRows, onSelect, selected }`.
   * @returns {{element: SVGElement, note: string}} chart.
   */
  function renderBars({ rows, dimension, maxRows = 50, onSelect, selected }) {
    const visible = rows.slice(0, maxRows);
    const rowHeight = 20;
    const gap = 6;
    const labelWidth = 250;
    const valueWidth = 120;
    const barValueWidth = 84;
    const width = 940;
    const height = Math.max(rowHeight, visible.length * (rowHeight + gap) + 30);
    const plotWidth = width - labelWidth - valueWidth - barValueWidth;
    const max = visible.reduce((top, row) => Math.max(top, row.value), 0);
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, width: '100%', height: String(height), role: 'img', 'aria-label': 'TopN 算子耗时排行' });

    root.append(svg('text', { x: labelWidth, y: 12, fill: 'var(--ink-3)', 'font-size': '10.5' }, [dimension === 'average' ? '单次执行耗时' : '累计总耗时']));
    root.append(svg('text', { x: width - valueWidth + 4, y: 12, fill: 'var(--ink-3)', 'font-size': '10.5' }, ['占算子总耗时']));

    const bars = [];
    visible.forEach((row, index) => {
      const y = 22 + index * (rowHeight + gap);
      const color = CATEGORY_COLORS[row.category] ?? '#888';
      const group = svg('g', { class: `bar-row${selected === row.name ? ' selected' : ''}`, 'data-operator': row.name, tabindex: '0', role: 'button' });
      group.append(svg('title', {}, [`${row.name}\n${row.group === 'host' ? 'Host' : 'Device'} · ${CATEGORY_LABELS[row.category] ?? row.category}\n调用 ${formatCount(row.count)} 次 · 累计 ${formatUs(row.totalUs)} · 均值 ${formatUs(row.avgUs)}`]));
      group.append(svg('text', { x: labelWidth - 10, y: y + 14, 'text-anchor': 'end', fill: 'var(--ink)', 'font-size': '12' }, [`${String(row.rank)}. ${truncate(row.name, 32)}`]));
      const bar = svg('rect', { class: 'bar', x: labelWidth, y, width: 0, height: rowHeight, rx: 4, fill: color, 'fill-opacity': '0.88' });
      group.append(bar);
      group.append(svg('text', { class: 'bar-value', x: labelWidth + 8, y: y + 14, fill: 'var(--ink-2)', 'font-size': '11', opacity: motionEnabled() ? 0 : 1 }, [dimension === 'average' ? formatUs(row.value) : formatUs(row.value)]));
      group.append(svg('text', { x: width - valueWidth + 4, y: y + 14, fill: 'var(--ink-3)', 'font-size': '11' }, [`${row.shareOfOpsPct.toFixed(2)}%`]));
      if (typeof onSelect === 'function') {
        group.addEventListener('click', () => onSelect({ operator: row.name }));
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSelect({ operator: row.name });
          }
        });
      }
      root.append(group);
      bars.push({ bar, group, target: max <= 0 ? 0 : Math.max(2, (row.value / max) * plotWidth), labelWidth });
    });

    // Grow-in: one tween drives every bar, staggered by rank.
    if (motionEnabled()) {
      const delayWindow = 0.45;
      tween({
        from: { p: 0 },
        to: { p: 1 },
        duration: 620 + Math.min(visible.length, 20) * 12,
        onFrame: ({ p }) => {
          for (const [index, entry] of bars.entries()) {
            const delayed = Math.min(1, Math.max(0, (p - (index / Math.max(1, bars.length)) * delayWindow) / (1 - delayWindow)));
            entry.bar.setAttribute('width', (entry.target * delayed).toFixed(2));
          }
        },
        onDone: () => {
          for (const entry of bars) {
            entry.bar.setAttribute('width', entry.target.toFixed(2));
            const value = entry.group.querySelector('.bar-value');
            if (value !== null) value.setAttribute('opacity', '1');
          }
        },
      });
    } else {
      for (const entry of bars) entry.bar.setAttribute('width', entry.target.toFixed(2));
    }

    const truncated = rows.length - visible.length;
    return {
      element: root,
      note: truncated > 0 ? `显示前 ${String(visible.length)} / ${String(rows.length)} 项` : `${String(visible.length)} 项`,
    };
  }

  /**
   * Apply the current selection to an already-rendered chart without rebuilding
   * it — rebuilding would replay the draw-in animation on every interaction.
   *
   * @param {SVGElement|null} root - chart root.
   * @param {{ category?: string, operator?: string }} filter - active filter.
   */
  function applyHighlight(root, filter) {
    if (root === null || root === undefined) return;
    const category = filter?.category;
    const operator = filter?.operator;
    for (const slice of root.querySelectorAll?.('path.slice') ?? []) {
      const match = category === undefined || slice.getAttribute('data-category') === category;
      slice.classList.toggle('dimmed', !match);
    }
    for (const group of root.querySelectorAll?.('g.bar-row') ?? []) {
      const name = group.getAttribute('data-operator');
      const match = operator === undefined || name === operator;
      group.classList.toggle('dimmed', !match);
      group.classList.toggle('selected', operator !== undefined && name === operator);
    }
  }

  function truncate(text, limit) {
    const value = String(text);
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  /**
   * Build the ranking rows used by the bar chart and the data table.
   *
   * @param {object} input - `{ dataset, dimension, scope, topN, category }`.
   * @returns {{rows: object[], totalUs: number, scopeLabel: string}} rows.
   */
  function buildRanking({ dataset, dimension, scope, topN, category }) {
    const operators = dataset.operators.filter((row) => row.overflow !== true);
    const scoped = operators
      .filter((row) => (scope === 'all' ? true : row.group === scope))
      .filter((row) => (category === undefined ? true : row.category === category));
    const rows = scoped.map((row) => ({
      name: row.name,
      group: row.group,
      category: row.category,
      subtype: row.subtype,
      subtypeLabel: row.subtypeLabel,
      count: row.count,
      totalUs: row.totalUs,
      avgUs: row.avgUs,
      maxUs: row.maxUs,
      p95Us: row.p95Us,
      waitUs: row.waitUs,
      value: dimension === 'average' ? row.avgUs : row.totalUs,
    }));
    const sum = rows.reduce((total, row) => total + row.totalUs, 0);
    const sorted = [...rows].sort((left, right) => right.value - left.value).slice(0, topN);
    sorted.forEach((row, index) => {
      row.rank = index + 1;
      row.shareOfOpsPct = sum > 0 ? (row.totalUs / sum) * 100 : 0;
    });
    return {
      rows: sorted,
      totalUs: sum,
      scopeLabel: scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧算子' : '设备侧算子',
    };
  }

  /**
   * Category items for the donut, recomputed for the chosen scope so the donut and
   * the ranking always describe the same population.
   *
   * @param {object} input - `{ dataset, scope }`.
   * @returns {object[]} category rows.
   */
  function buildCategories({ dataset, scope }) {
    const base = dataset.categories.items.map((item) => ({
      id: item.id,
      label: CATEGORY_LABELS[item.id] ?? item.label,
      color: CATEGORY_COLORS[item.id] ?? item.color,
      totalUs: scope === 'all' ? item.totalUs : scope === 'host' ? item.hostUs : item.deviceUs,
      count: item.count,
    }));
    const sum = base.reduce((total, item) => total + item.totalUs, 0);
    return base
      .map((item) => ({ ...item, sharePct: sum > 0 ? (item.totalUs / sum) * 100 : 0 }))
      .filter((item) => item.totalUs > 0)
      .sort((left, right) => right.totalUs - left.totalUs);
  }

  /** Render the ranking data table. */
  function renderRankingTable(rows, { onSelect, selected } = {}) {
    const head = ['#', '算子', '类别', '设备', '调用次数', '累计', '单次均值', 'p95', '占算子总耗时'];
    const table = h('table');
    const headRow = h('tr');
    for (const [index, label] of head.entries()) headRow.append(h('th', { class: index >= 4 ? 'num' : '' }, label));
    table.append(h('thead', {}, [headRow]));
    const tbody = h('tbody');
    for (const row of rows) {
      const tr = h('tr', { style: selected === row.name ? 'background:var(--accent-soft)' : undefined, title: row.name }, [
        h('td.num', {}, String(row.rank)),
        h('td.mono', {}, row.name),
        h('td', {}, CATEGORY_LABELS[row.category] ?? row.category),
        h('td', {}, row.group === 'host' ? 'Host' : 'Device'),
        h('td.num', {}, formatCount(row.count)),
        h('td.num', {}, formatUs(row.totalUs)),
        h('td.num', {}, formatUs(row.avgUs)),
        h('td.num', {}, formatUs(row.p95Us)),
        h('td.num', {}, `${row.shareOfOpsPct.toFixed(2)}%`),
      ]);
      if (typeof onSelect === 'function') {
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', () => onSelect({ operator: row.name }));
      }
      tbody.append(tr);
    }
    table.append(tbody);
    return table;
  }

  /**
   * Serialize an SVG element into a PNG data URL through an offscreen image, so
   * the printable report embeds the same charts the page shows.
   *
   * @param {SVGElement} element - chart root.
   * @param {number} [scale] - raster scale.
   * @returns {Promise<string|undefined>} `data:image/png;base64,…`.
   */
  function svgToPngDataUrl(element, scale = 2) {
    return new Promise((resolve) => {
      try {
        const box = typeof element.getBoundingClientRect === 'function' ? element.getBoundingClientRect() : { width: 0, height: 0 };
        const width = Math.max(320, Math.ceil(box.width || 640));
        const height = Math.max(180, Math.ceil(box.height || 320));
        const clone = element.cloneNode(true);
        clone.setAttribute('width', String(width));
        clone.setAttribute('height', String(height));
        const markup = new XMLSerializer().serializeToString(clone);
        const source = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup)}`;
        const image = new Image();
        image.onload = () => {
          try {
            const canvas = document.createElement('canvas');
            canvas.width = width * scale;
            canvas.height = height * scale;
            const context = canvas.getContext('2d');
            context.fillStyle = '#ffffff';
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.setTransform(scale, 0, 0, scale, 0, 0);
            context.drawImage(image, 0, 0, width, height);
            resolve(canvas.toDataURL('image/png'));
          } catch {
            resolve(undefined);
          }
        };
        image.onerror = () => resolve(undefined);
        image.src = source;
      } catch {
        resolve(undefined);
      }
    });
  }

  void playEnter;

  global.VAP = global.VAP ?? {};
  Object.assign(global.VAP, {
    charts: { renderPie, renderBars, buildRanking, buildCategories, renderRankingTable, svgToPngDataUrl, applyHighlight },
  });
})(window);
