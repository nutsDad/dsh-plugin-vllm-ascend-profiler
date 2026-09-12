/**
 * Module 2 projections: the numbers behind the composition strip, the treemap and
 * the ranking table.
 *
 * The page draws those three from `buildCategories()` / `buildRanking()` — the
 * donut and the ranked bar chart that used to live here were replaced by diagrams
 * (`diagram.js`), which carry the same hierarchy, ranking and share in one shape.
 * What remains is the projection layer plus the PNG export used by the report.
 */
(function attachCharts(global) {
  'use strict';

  const { h, formatUs, formatCount, CATEGORY_COLORS, CATEGORY_LABELS } = global.VAP;

  /**
   * Apply the current selection to an already-rendered diagram without rebuilding
   * it — rebuilding would replay the reveal animation on every interaction.
   *
   * @param {Element|null} root - diagram root (treemap or composition strip).
   * @param {{ category?: string, operator?: string }} filter - active filter.
   */
  function applyHighlight(root, filter) {
    if (root === null || root === undefined) return;
    const category = filter?.category;
    const operator = filter?.operator;
    for (const tile of root.querySelectorAll?.('g.tm-tile') ?? []) {
      const name = tile.getAttribute('data-operator');
      const inCategory = category === undefined || tile.getAttribute('data-category') === category;
      const isOperator = operator === undefined || name === operator;
      tile.classList.toggle('dimmed', !(inCategory && isOperator));
      tile.classList.toggle('selected', operator !== undefined && name === operator);
    }
    for (const segment of root.querySelectorAll?.('g.share-seg') ?? []) {
      const id = segment.getAttribute('data-category');
      segment.classList.toggle('dimmed', category !== undefined && id !== category);
    }
  }

  /**
   * Build the ranking rows used by the treemap and the data table.
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
   * Category items for the composition strip, recomputed for the chosen scope so
   * the strip and the treemap always describe the same population.
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

  /** Render the ranking data table — the exact numbers behind the treemap. */
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
   * the printable report embeds the same diagrams the page shows.
   *
   * @param {SVGElement} element - diagram root.
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

  global.VAP = global.VAP ?? {};
  Object.assign(global.VAP, {
    charts: { buildRanking, buildCategories, renderRankingTable, svgToPngDataUrl, applyHighlight },
  });
})(window);
