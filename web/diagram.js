/**
 * Diagram primitives for the analyzer page.
 *
 * These are the visuals that replace prose: a hierarchy whose area *is* the cost,
 * a composition strip, threshold comparisons, a reasoning chain, and gain ranges.
 * All are pure functions returning SVG or DOM nodes, all are keyboard-reachable
 * where they are interactive, and all degrade to a static drawing when the page
 * turns motion off.
 *
 * @module dsp-plugin-vllm-ascend-profiler/web/diagram
 */
(function attachDiagram(global) {
  'use strict';

  const { h, formatUs, formatPct, formatCount, CATEGORY_COLORS, motionEnabled } = global.VAP;

  const SVG_NS = 'http://www.w3.org/2000/svg';

  /** Create an SVG element (`svg` root gets the namespace for standalone export). */
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

  function truncate(text, limit) {
    const value = String(text ?? '');
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  // ── squarified treemap ───────────────────────────────────────────────────

  /**
   * Lay items out as a squarified treemap (area ∝ value).
   *
   * @param {{name: string, value: number}[]} items - items with a positive value.
   * @param {{x: number, y: number, w: number, h: number}} box - target rectangle.
   * @returns {object[]} items with `x`, `y`, `w`, `h` added, in input order.
   */
  function squarify(items, box) {
    const positive = items.filter((item) => item.value > 0);
    const total = positive.reduce((sum, item) => sum + item.value, 0);
    if (total <= 0 || box.w <= 0 || box.h <= 0) return [];
    const scale = (box.w * box.h) / total;
    const queue = positive
      .map((item) => ({ ...item, area: item.value * scale }))
      .sort((left, right) => right.area - left.area);
    const rect = { ...box };
    const out = [];
    let row = [];

    const worst = (candidate, side) => {
      const sum = candidate.reduce((total2, node) => total2 + node.area, 0);
      const max = Math.max(...candidate.map((node) => node.area));
      const min = Math.min(...candidate.map((node) => node.area));
      if (sum <= 0 || min <= 0) return Infinity;
      return Math.max((side * side * max) / (sum * sum), (sum * sum) / (side * side * min));
    };

    const flush = () => {
      if (row.length === 0) return;
      const sum = row.reduce((total2, node) => total2 + node.area, 0);
      if (rect.w >= rect.h) {
        // Fill a column along the shorter (vertical) side.
        const columnWidth = rect.h <= 0 ? rect.w : Math.min(rect.w, sum / rect.h);
        let y = rect.y;
        for (const node of row) {
          const height = columnWidth <= 0 ? 0 : Math.min(rect.y + rect.h - y, node.area / columnWidth);
          out.push({ ...node, x: rect.x, y, w: columnWidth, h: height });
          y += height;
        }
        rect.x += columnWidth;
        rect.w = Math.max(0, rect.w - columnWidth);
      } else {
        // Fill a row along the shorter (horizontal) side.
        const rowHeight = rect.w <= 0 ? rect.h : Math.min(rect.h, sum / rect.w);
        let x = rect.x;
        for (const node of row) {
          const width = rowHeight <= 0 ? 0 : Math.min(rect.x + rect.w - x, node.area / rowHeight);
          out.push({ ...node, x, y: rect.y, w: width, h: rowHeight });
          x += width;
        }
        rect.y += rowHeight;
        rect.h = Math.max(0, rect.h - rowHeight);
      }
      row = [];
    };

    for (const node of queue) {
      const side = Math.min(rect.w, rect.h);
      if (row.length === 0 || worst([...row, node], side) <= worst(row, side)) {
        row.push(node);
        continue;
      }
      flush();
      row.push(node);
    }
    flush();
    return out;
  }

  /**
   * Operator-level treemap: area is cumulative time, colour is the category.
   *
   * Replaces a donut plus a ranking chart with one diagram: it shows the
   * hierarchy (total → category → operator), the ranking, and the composition
   * at the same time. Tiles carry their name and share when they are big enough
   * to read; everything else lives in the hover card.
   *
   * @param {object} input - `{ rows, dimension, onSelect, selected, totalUs }`.
   * @returns {{element: SVGElement, shown: number}} diagram.
   */
  function treemap({ rows, dimension, onSelect, selected, totalUs }) {
    const width = 960;
    const height = 340;
    const items = rows.map((row) => ({
      name: row.name,
      value: dimension === 'average' ? row.avgUs * row.count : row.totalUs,
      category: row.category,
      count: row.count,
      totalUs: row.totalUs,
      avgUs: row.avgUs,
      p95Us: row.p95Us,
      sharePct: row.sharePct ?? 0,
    }));
    const sum = items.reduce((total, item) => total + item.value, 0);
    const root = svg('svg', {
      viewBox: `0 0 ${String(width)} ${String(height)}`,
      width: '100%',
      height: String(height),
      role: 'img',
      'aria-label': '算子耗时分布（面积代表耗时）',
    });
    if (sum <= 0) {
      root.append(svg('text', { x: width / 2, y: height / 2, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '12' }, ['该口径下没有耗时数据']));
      return { element: root, shown: 0 };
    }

    const tiles = squarify(items, { x: 0, y: 0, w: width, h: height });
    let revealIndex = 0;
    for (const tile of tiles) {
      const color = CATEGORY_COLORS[tile.category] ?? '#888';
      const group = svg('g', {
        class: `tm-tile${selected === tile.name ? ' selected' : ''}`,
        'data-operator': tile.name,
        'data-category': tile.category,
        tabindex: '0',
        role: onSelect === undefined ? 'img' : 'button',
        style: motionEnabled() ? `opacity:0;transition:opacity .32s ease ${String(Math.min(revealIndex * 22, 420))}ms` : undefined,
      });
      revealIndex += 1;
      group.append(svg('title', {}, [
        `${tile.name}\n${tile.category}\n占比 ${formatPct((tile.value / sum) * 100, 2)} · 累计 ${formatUs(tile.totalUs)}\n调用 ${formatCount(tile.count)} 次 · 均值 ${formatUs(tile.avgUs)} · p95 ${formatUs(tile.p95Us)}`,
      ]));
      group.append(svg('rect', {
        x: tile.x + 0.5,
        y: tile.y + 0.5,
        width: Math.max(0, tile.w - 1),
        height: Math.max(0, tile.h - 1),
        rx: 3,
        fill: color,
        'fill-opacity': '0.86',
      }));
      // Labels only when the tile can hold them: the diagram must never turn
      // into a pile of clipped text.
      const share = (tile.value / sum) * 100;
      if (tile.w > 58 && tile.h > 26) {
        group.append(svg('text', {
          x: tile.x + 6, y: tile.y + 15, fill: '#fff', 'font-size': '11.5', 'font-weight': '600',
        }, [truncate(tile.name, Math.max(6, Math.floor(tile.w / 7)))]));
        group.append(svg('text', {
          x: tile.x + 6, y: tile.y + 29, fill: 'rgba(255,255,255,.85)', 'font-size': '10.5',
        }, [`${share.toFixed(1)}% · ${formatUs(tile.totalUs)}`]));
      } else if (tile.w > 26 && tile.h > 14) {
        group.append(svg('text', {
          x: tile.x + 5, y: tile.y + 13, fill: '#fff', 'font-size': '10', 'font-weight': '600',
        }, [`${share.toFixed(0)}%`]));
      }
      if (typeof onSelect === 'function') {
        group.addEventListener('click', () => onSelect({ operator: tile.name }));
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSelect({ operator: tile.name });
          }
        });
      }
      root.append(group);
    }
    if (motionEnabled()) {
      requestAnimationFrame(() => {
        for (const tile of root.querySelectorAll('g.tm-tile')) tile.setAttribute('style', 'opacity:1');
      });
    }
    void totalUs;
    return { element: root, shown: tiles.length };
  }

  /**
   * 100% composition strip: one row, category segments, labels inside.
   *
   * @param {object} input - `{ items, onSelect, selected }`.
   * @returns {{element: SVGElement, legend: HTMLElement}} strip and its legend.
   */
  function shareBar({ items, onSelect, selected }) {
    const width = 960;
    const height = 46;
    const positive = items.filter((item) => item.totalUs > 0);
    const sum = positive.reduce((total, item) => total + item.totalUs, 0);
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, width: '100%', height: String(height), role: 'img', 'aria-label': '算子大类构成' });
    const legend = h('div.composition-legend');
    if (sum <= 0) {
      root.append(svg('rect', { x: 0, y: 0, width, height: 34, rx: 6, fill: 'var(--panel-3)' }));
      return { element: root, legend };
    }

    let x = 0;
    for (const item of positive) {
      const segment = (item.totalUs / sum) * width;
      const color = CATEGORY_COLORS[item.id] ?? '#888';
      const group = svg('g', { class: 'share-seg', 'data-category': item.id, tabindex: '0', role: 'button' });
      group.append(svg('title', {}, [`${item.label}：${formatPct(item.sharePct, 2)} · ${formatUs(item.totalUs)} · ${formatCount(item.count)} 次`]));
      group.append(svg('rect', { x: x + 1, y: 0, width: Math.max(0, segment - 2), height: 34, rx: 5, fill: color, 'fill-opacity': selected === item.id ? '1' : '0.86' }));
      if (segment > 66) {
        group.append(svg('text', { x: x + 9, y: 21, fill: '#fff', 'font-size': '11.5', 'font-weight': '600' }, [`${item.label} ${item.sharePct.toFixed(1)}%`]));
      } else if (segment > 26) {
        group.append(svg('text', { x: x + 6, y: 21, fill: '#fff', 'font-size': '10.5', 'font-weight': '600' }, [`${item.sharePct.toFixed(0)}%`]));
      }
      if (typeof onSelect === 'function') {
        group.addEventListener('click', () => onSelect({ category: item.id }));
        group.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onSelect({ category: item.id });
          }
        });
      }
      root.append(group);
      x += segment;
      // The legend repeats the strip in text form and is clickable, so the
      // smallest categories stay reachable without pixel-hunting the strip.
      const legendItem = h('button.composition-item', {
        type: 'button',
        dataset: { category: item.id },
        title: `只看${item.label}`,
      }, [
        h('i', { style: `background:${color}` }),
        h('b', {}, item.label),
        h('span.mono', {}, formatPct(item.sharePct, 1)),
      ]);
      if (typeof onSelect === 'function') {
        legendItem.addEventListener('click', () => onSelect({ category: item.id }));
      }
      legend.append(legendItem);
    }
    return { element: root, legend };
  }

  /**
   * Threshold comparison: one row per metric, measured bar against its gate.
   *
   * This is the evidence section as a diagram — the pass/fail verdict is the
   * colour, the gap to the gate is the visible length, and the numbers stay in
   * the row without any prose.
   *
   * @param {object} input - `{ rows, onSelect }`.
   * @returns {{element: SVGElement, legend: HTMLElement}} diagram.
   */
  function thresholdBars({ rows, onSelect }) {
    const usable = rows.filter((row) => Number.isFinite(row.value));
    const rowHeight = 26;
    const labelWidth = 200;
    const valueWidth = 118;
    const width = 940;
    const height = Math.max(rowHeight, usable.length * rowHeight + 16);
    const plotWidth = width - labelWidth - valueWidth;
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, width: '100%', height: String(height), role: 'img', 'aria-label': '指标与门限对比' });
    const legend = h('div.threshold-legend', {}, [
      h('span', {}, [h('i.pass'), '达门限']),
      h('span', {}, [h('i.fail'), '未达门限']),
      h('span', {}, [h('i.gate'), '门限']),
    ]);
    if (usable.length === 0) {
      root.append(svg('text', { x: width / 2, y: 24, 'text-anchor': 'middle', fill: 'var(--ink-3)', 'font-size': '12' }, ['本次没有可比较的门限指标']));
      return { element: root, legend };
    }

    usable.forEach((row, index) => {
      const y = 8 + index * rowHeight;
      const gate = Number.isFinite(row.threshold) ? row.threshold : undefined;
      const scale = Math.max(row.value, gate ?? 0) * 1.25 || 1;
      const measured = (row.value / scale) * plotWidth;
      const gateX = gate === undefined ? undefined : (gate / scale) * plotWidth;
      const pass = row.passed !== false;
      const group = svg('g', { class: 'tb-row' });
      group.append(svg('text', { x: labelWidth - 12, y: y + 13, 'text-anchor': 'end', fill: 'var(--ink)', 'font-size': '11.5' }, [truncate(row.metric, 22)]));
      group.append(svg('rect', { x: labelWidth, y: y + 3, width: plotWidth, height: 12, rx: 6, fill: 'var(--panel-3)' }));
      group.append(svg('rect', {
        x: labelWidth, y: y + 3, width: Math.max(2, measured), height: 12, rx: 6,
        fill: pass ? 'var(--ok)' : 'var(--err)', 'fill-opacity': '0.75',
      }));
      if (gateX !== undefined) {
        group.append(svg('line', {
          x1: labelWidth + gateX, y1: y, x2: labelWidth + gateX, y2: y + 18,
          stroke: 'var(--ink-2)', 'stroke-width': '1.5', 'stroke-dasharray': '3 2',
        }));
      }
      group.append(svg('text', {
        x: labelWidth + Math.max(measured, gateX ?? 0) + 8, y: y + 13, fill: 'var(--ink-2)', 'font-size': '11',
      }, [row.unit === '%' ? formatPct(row.value, 1) : `${String(Number(row.value.toFixed(2)))}${row.unit ?? ''}`]));
      if (gate !== undefined) {
        group.append(svg('text', {
          x: width - 6, y: y + 13, 'text-anchor': 'end', fill: 'var(--ink-3)', 'font-size': '10.5',
        }, [`门限 ${String(gate)}${row.unit ?? ''}`]));
      }
      group.append(svg('title', {}, [`${row.metric}\n实测 ${String(row.value)}${row.unit ?? ''}${gate === undefined ? '' : ` · 门限 ${String(gate)}${row.unit ?? ''}`}\n${row.source ?? ''}`]));
      if (typeof onSelect === 'function' && row.category !== undefined) {
        group.style.cursor = 'pointer';
        group.addEventListener('click', () => onSelect({ category: row.category }));
      }
      root.append(group);
    });
    return { element: root, legend };
  }

  /**
   * Horizontal range bar for one expected gain (0 → max).
   *
   * @param {object} input - `{ rangePct, estimatePct, confidence, max, width }`.
   * @returns {SVGElement} bar.
   */
  function gainBar({ rangePct, estimatePct, confidence, max, width = 190 }) {
    const height = 16;
    const scale = Math.max(1, max);
    const root = svg('svg', { viewBox: `0 0 ${String(width)} ${String(height)}`, width: String(width), height: String(height), 'aria-hidden': 'true' });
    const from = Array.isArray(rangePct) ? (rangePct[0] / scale) * width : 0;
    const to = Array.isArray(rangePct) ? (rangePct[1] / scale) * width : (estimatePct / scale) * width;
    const mark = (estimatePct / scale) * width;
    const color = confidence === 'low' ? 'var(--warn)' : 'var(--accent)';
    root.append(svg('rect', { x: 0, y: 5, width, height: 6, rx: 3, fill: 'var(--panel-3)' }));
    root.append(svg('rect', {
      x: Math.max(0, from), y: 5, width: Math.max(2, to - from), height: 6, rx: 3,
      fill: color, 'fill-opacity': confidence === 'low' ? '0.45' : '0.85',
    }));
    root.append(svg('line', { x1: mark, y1: 2, x2: mark, y2: 14, stroke: color, 'stroke-width': '2' }));
    return root;
  }

  /**
   * Reasoning chain: ①→⑤ nodes that act as tabs over one detail panel.
   *
   * @param {object} input - `{ nodes, active, onSelect }`.
   * @returns {HTMLElement} the flow.
   */
  function chainFlow({ nodes, active, onSelect }) {
    const flow = h('div.chain-flow', { role: 'tablist', 'aria-label': '分析推理链' });
    nodes.forEach((node, index) => {
      const button = h('button', {
        type: 'button',
        class: `flow-node${node.id === active ? ' active' : ''}${node.muted === true ? ' muted' : ''}`,
        role: 'tab',
        'aria-selected': String(node.id === active),
        dataset: { node: node.id },
        title: node.hint,
      }, [
        h('span.flow-index', {}, node.index),
        h('span.flow-body', {}, [
          h('span.flow-title', {}, node.title),
          h('span.flow-value', {}, node.value),
        ]),
      ]);
      button.addEventListener('click', () => onSelect?.(node.id));
      flow.append(button);
      if (index < nodes.length - 1) flow.append(h('span.flow-arrow', { 'aria-hidden': 'true' }, '→'));
    });
    return flow;
  }

  /**
   * Cause chains: trigger chips → one-line mechanism → effect chip.
   *
   * The prose stays available, but the default read is three labelled boxes per
   * hypothesis instead of two paragraphs.
   *
   * @param {object} input - `{ items }`.
   * @returns {HTMLElement} list of chains.
   */
  function causeChains({ items }) {
    const list = h('div.cause-list');
    for (const item of items) {
      const triggers = item.triggers.slice(0, 3);
      list.append(h('div.cause-row', {}, [
        h('div.cause-cell.triggers', {}, triggers.map((trigger) => h('span.chip.mono', {
          title: trigger.note,
        }, `${trigger.metric} ${trigger.value === undefined ? '' : String(trigger.value)}${trigger.unit ?? ''}`))),
        h('span.cause-arrow', { 'aria-hidden': 'true' }, '→'),
        h('div.cause-cell.mechanism', {}, [
          h('div.cause-title', {}, item.title),
          h('div.cause-line', {}, oneLine(item.mechanism)),
        ]),
        h('span.cause-arrow', { 'aria-hidden': 'true' }, '→'),
        h('div.cause-cell.effect', {}, [
          h('span.chip.phase', {}, item.bottleneck),
          h('span.chip.phase', {}, item.phase),
        ]),
        h('details.cause-more', {}, [
          h('summary', {}, '依据'),
          h('p', {}, item.mechanism),
          h('p.hint', {}, item.vllmBehaviour),
          h('ul', {}, item.checks.map((check) => h('li', {}, check))),
        ]),
      ]));
    }
    return list;
  }

  /** First sentence of a paragraph, capped — the "one line" for a diagram. */
  function oneLine(text) {
    const value = String(text ?? '').trim();
    // A sentence ends at CJK punctuation, or at a Latin period that is followed by
    // a space (or the end) — never at the decimal point inside a number, which
    // would truncate "Host 独占 30.22%" into "Host 独占 30.".
    const stop = /[。；;！!？?]|\.(?=\s|$)/.exec(value);
    const first = stop === null ? value : value.slice(0, stop.index + 1);
    return truncate(first, 78);
  }

  global.VAP = global.VAP ?? {};
  global.VAP.diagram = { squarify, treemap, shareBar, thresholdBars, gainBar, chainFlow, causeChains, oneLine, svg };
})(window);
