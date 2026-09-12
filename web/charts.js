/**
 * Module 2 charts: the category cost-share pie and the Top-N ranked bar chart,
 * drawn with native SVG (no charting dependency, works offline).
 *
 * Both charts are pure functions of the dataset: they return an SVG element and
 * never read global state, so the same projection can be screenshotted into the
 * printable report later without re-running any analysis.
 */
(function attachCharts(global) {
  'use strict';

  const { h, formatUs, formatCount, CATEGORY_COLORS, CATEGORY_LABELS, escapeHtml } = global.VAP;

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /** Create an SVG element with attributes. */
  function svg(tag, attrs = {}, children = []) {
    const element = document.createElementNS(SVG_NS, tag);
    // `xmlns` is not required for inline SVG in HTML but IS required for a
    // standalone document, which is what the PDF export serializes.
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

  /**
   * Category share donut.
   *
   * A donut rather than a filled pie: the hole carries the total, which is the
   * number a reader wants next to the shares.
   *
   * @param {object} input - `{ items, totalUs, scopeLabel, note }`.
   * @returns {{element: SVGElement, legend: HTMLElement}} chart.
   */
  function renderPie({ items, totalUs, scopeLabel }) {
    const width = 320;
    const height = 240;
    const cx = 120;
    const cy = 120;
    const outer = 88;
    const inner = 52;
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, class: 'pie' });

    const positive = items.filter((item) => item.totalUs > 0);
    const sum = positive.reduce((total, item) => total + item.totalUs, 0);
    const legend = h('ul.pie-legend');

    if (sum <= 0) {
      root.append(svg('circle', { cx, cy, r: (outer + inner) / 2, fill: 'none', stroke: 'var(--line)', 'stroke-width': outer - inner }));
      root.append(svg('text', { x: cx, y: cy + 4, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '12' }, ['该口径下没有耗时数据']));
      return { element: root, legend };
    }

    let angle = -Math.PI / 2;
    for (const item of positive) {
      const sweep = (item.totalUs / sum) * Math.PI * 2;
      const end = angle + sweep;
      const largeArc = sweep > Math.PI ? 1 : 0;
      const path = [
        `M ${String(cx + outer * Math.cos(angle))} ${String(cy + outer * Math.sin(angle))}`,
        `A ${String(outer)} ${String(outer)} 0 ${String(largeArc)} 1 ${String(cx + outer * Math.cos(end))} ${String(cy + outer * Math.sin(end))}`,
        `L ${String(cx + inner * Math.cos(end))} ${String(cy + inner * Math.sin(end))}`,
        `A ${String(inner)} ${String(inner)} 0 ${String(largeArc)} 0 ${String(cx + inner * Math.cos(angle))} ${String(cy + inner * Math.sin(angle))}`,
        'Z',
      ].join(' ');
      const color = item.color ?? CATEGORY_COLORS[item.id] ?? '#888';
      const slice = svg('path', { d: path, fill: color, 'fill-opacity': '0.92', stroke: 'var(--panel)', 'stroke-width': '1.2' });
      slice.append(svg('title', {}, [`${item.label}：${formatUs(item.totalUs)}（${item.sharePct.toFixed(2)}%，${formatCount(item.count)} 次）`]));
      root.append(slice);
      angle = end;

      legend.append(h('li', {}, [
        h('i', { style: `background:${color}` }),
        h('span', {}, item.label),
        h('span.val', {}, `${item.sharePct.toFixed(2)}% · ${formatUs(item.totalUs)}`),
      ]));
    }

    root.append(svg('text', { x: cx, y: cy - 4, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '11' }, ['合计']));
    root.append(svg('text', { x: cx, y: cy + 14, 'text-anchor': 'middle', fill: 'var(--ink)', 'font-size': '14', 'font-weight': '600' }, [formatUs(sum)]));
    root.append(svg('text', { x: 236, y: 24, fill: 'var(--ink-3)', 'font-size': '11' }, [scopeLabel ?? '']));
    return { element: root, legend };
  }

  /**
   * Top-N ranked horizontal bars.
   *
   * Bars are drawn in rank order with the value printed at the end, and each bar
   * is coloured by category so the ranking and the donut tell one story.
   *
   * @param {object} input - `{ rows, dimension, metricOf, maxRows }`.
   * @returns {{element: SVGElement, note: string, rowHeight: number}} chart.
   */
  function renderBars({ rows, dimension, maxRows = 50 }) {
    const visible = rows.slice(0, maxRows);
    const rowHeight = 22;
    const gap = 6;
    const labelWidth = 268;
    const valueWidth = 132;
    // Space reserved for the value printed after each bar. Without it the longest
    // bar's label runs into the percentage column and the two texts overlap.
    const barValueWidth = 88;
    const width = 960;
    const height = Math.max(rowHeight, visible.length * (rowHeight + gap) + 34);
    const plotWidth = width - labelWidth - valueWidth - barValueWidth;
    const max = visible.reduce((top, row) => Math.max(top, row.value), 0);
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, width: '100%', height: String(height), class: 'bars' });

    root.append(svg('text', { x: labelWidth, y: 14, fill: 'var(--ink-3)', 'font-size': '11' }, [dimension === 'average' ? '单次执行耗时（µs）' : '累计总耗时']));
    root.append(svg('text', { x: width - valueWidth + 4, y: 14, fill: 'var(--ink-3)', 'font-size': '11' }, ['占算子总耗时']));

    visible.forEach((row, index) => {
      const y = 24 + index * (rowHeight + gap);
      const barWidth = max <= 0 ? 0 : Math.max(1.5, (row.value / max) * plotWidth);
      const color = CATEGORY_COLORS[row.category] ?? '#888';
      const group = svg('g', { class: 'bar-row' });

      group.append(svg('text', {
        x: labelWidth - 10, y: y + 15, 'text-anchor': 'end', fill: 'var(--ink)', 'font-size': '12',
      }, [`${String(row.rank)}. ${truncate(row.name, 34)}`]));
      group.append(svg('title', {}, [`${row.name}\n${row.group === 'host' ? 'Host' : 'Device'} · ${CATEGORY_LABELS[row.category] ?? row.category}\n调用 ${formatCount(row.count)} 次 · 累计 ${formatUs(row.totalUs)} · 均值 ${formatUs(row.avgUs)}`]));
      group.append(svg('rect', { x: labelWidth, y, width: barWidth, height: rowHeight, rx: 4, fill: color, 'fill-opacity': '0.88' }));
      group.append(svg('text', {
        x: labelWidth + barWidth + 8, y: y + 15, fill: 'var(--ink-2)', 'font-size': '11',
      }, [dimension === 'average' ? formatUs(row.value) : formatUs(row.value)]));
      group.append(svg('text', {
        x: width - valueWidth + 4, y: y + 15, fill: 'var(--ink-3)', 'font-size': '11',
      }, [`${row.shareOfOpsPct.toFixed(2)}%`]));
      root.append(group);
    });

    const truncated = rows.length - visible.length;
    return {
      element: root,
      note: truncated > 0 ? `显示前 ${String(visible.length)} 项（共 ${String(rows.length)} 项）` : `${String(visible.length)} 项`,
      rowHeight: rowHeight + gap,
    };
  }

  function truncate(text, limit) {
    const value = String(text);
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  /**
   * Build the ranking rows used by the bar chart and the data table.
   *
   * @param {object} input - `{ dataset, dimension, scope, topN }`.
   * @returns {{rows: object[], totalUs: number, scopeLabel: string}} rows.
   */
  function buildRanking({ dataset, dimension, scope, topN }) {
    const operators = dataset.operators.filter((row) => row.overflow !== true);
    const scoped = operators.filter((row) => (scope === 'all' ? true : row.group === scope));
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
   * Category items for the pie, recomputed for the chosen scope so the donut and
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
  function renderRankingTable(rows) {
    const head = ['#', '算子', '类别', '亚类', '设备', '调用次数', '累计', '单次均值', 'p95', '占算子总耗时'];
    const table = h('table');
    const thead = h('thead');
    const headRow = h('tr');
    for (const [index, label] of head.entries()) {
      headRow.append(h('th', { class: index >= 5 ? 'num' : '' }, label));
    }
    thead.append(headRow);
    const tbody = h('tbody');
    for (const row of rows) {
      tbody.append(h('tr', {}, [
        h('td.num', {}, String(row.rank)),
        h('td.mono', { title: row.name }, row.name),
        h('td', {}, CATEGORY_LABELS[row.category] ?? row.category),
        h('td', {}, row.subtypeLabel ?? row.subtype ?? ''),
        h('td', {}, row.group === 'host' ? 'Host' : 'Device'),
        h('td.num', {}, formatCount(row.count)),
        h('td.num', {}, formatUs(row.totalUs)),
        h('td.num', {}, formatUs(row.avgUs)),
        h('td.num', {}, formatUs(row.p95Us)),
        h('td.num', {}, `${row.shareOfOpsPct.toFixed(2)}%`),
      ]));
    }
    table.append(thead, tbody);
    void escapeHtml;
    return table;
  }

  global.VAP = global.VAP ?? {};
  Object.assign(global.VAP, {
    charts: { renderPie, renderBars, buildRanking, buildCategories, renderRankingTable, svgToPngDataUrl },
  });
})(window);
