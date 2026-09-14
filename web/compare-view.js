/**
 * Step 6 rendering: the before/after comparison.
 *
 * The comparison itself is computed host-side (`lib/analysis/compare.js`) so the
 * page never re-derives a delta; what lives here is purely presentation:
 *
 * * `renderCompareSummary` — the step-6 panel: headline sentence, the four
 *   metrics an optimisation is expected to move, comparability warnings, and one
 *   row per recommendation saying whether its target metric actually improved;
 * * `renderDeltaStrip` — the compact chips shown above the two swimlanes;
 * * `renderDeltaTable` — category and operator deltas for step 4.
 *
 * @module dsh-plugin-vllm-ascend-profiler/web/compare-view
 */
(function attachCompareView(global) {
  'use strict';

  const { h, formatUs, formatPct, CATEGORY_COLORS } = global.VAP;

  /** Headline metrics shown as cards (order matters — this is the reading order). */
  const HEADLINE_CARDS = [
    { key: 'avgStepUs', label: '每步墙钟', unit: 'µs', format: (value) => formatUs(value), digits: 1 },
    { key: 'deviceBusyPct', label: 'NPU 忙碌率', unit: '%', format: (value) => formatPct(value, 1), digits: 1 },
    { key: 'hostExclusivePerStepUs', label: 'Host 独占/步', unit: 'µs', format: (value) => formatUs(value), digits: 0 },
    { key: 'dispatchPerStep', label: '派发算子数/步', unit: '个', format: (value) => String(Math.round(value)), digits: 0 },
  ];

  /** Chips shown above the swimlanes (step 3). */
  const STRIP_KEYS = ['avgStepUs', 'deviceBusyPct', 'hostExclusivePerStepUs', 'dispatchPerStep', 'hostSyncPerStep', 'commExposedPct'];

  const VERDICT_LABEL = {
    achieved: '已达成',
    partial: '部分达成',
    missed: '未达成',
    unknown: '无法判定',
  };

  /** One metric row out of the payload. */
  function metricOf(comparison, key) {
    return (comparison.metrics ?? []).find((row) => row.key === key);
  }

  /**
   * CSS class for a metric delta: green when the metric moved the good way, red
   * when it moved the bad way, grey for context metrics that have no direction.
   */
  function deltaClass(metric) {
    if (metric === undefined || metric.improved === undefined) return 'delta-neutral';
    return metric.improved ? 'delta-up' : 'delta-down';
  }

  /** `−45.2%` / `+3.1%` / `—`. */
  function deltaText(metric) {
    if (metric === undefined || !Number.isFinite(metric.deltaPct)) return '—';
    const sign = metric.deltaPct > 0 ? '+' : metric.deltaPct < 0 ? '−' : '';
    return `${sign}${Math.abs(metric.deltaPct).toFixed(1)}%`;
  }

  /** Human phrasing that respects the metric's improvement direction. */
  function deltaPhrase(metric) {
    if (metric === undefined || metric.improved === undefined) return '变化';
    if (metric.improved) return '改善';
    return '变差';
  }

  /**
   * The step-6 panel.
   *
   * @param {object} comparison - payload from `/api/datasets/<id>/compare`.
   * @returns {HTMLElement} panel.
   */
  function renderCompareSummary(comparison) {
    const root = h('div.compare-summary-body');
    const headline = comparison.headline ?? {};
    const sides = comparison.sides ?? {};

    root.append(h('div.compare-headline', {}, [
      h('span.k', {}, '结论'),
      h('span.v', {}, headline.summary ?? '—'),
      h('span.k', {}, `${sides.before?.label ?? '优化前'} → ${sides.after?.label ?? '优化后'}`),
    ]));

    const cards = HEADLINE_CARDS.map((card) => {
      const metric = metricOf(comparison, card.key);
      if (metric === undefined || metric.before === undefined || metric.after === undefined) return undefined;
      return h('div.compare-card', {}, [
        h('div.k', {}, card.label),
        h('div.v', {}, [
          h('span.from', {}, card.format(metric.before)),
          h('span', { 'aria-hidden': 'true' }, '→'),
          h('span.to', {}, card.format(metric.after)),
          h('span', { class: `d ${deltaClass(metric)}` }, `${deltaText(metric)} ${deltaPhrase(metric)}`),
        ]),
      ]);
    }).filter((node) => node !== undefined);
    root.append(h('div.compare-cards', {}, cards));

    // How comparable are the two captures? Warnings first, then context notes.
    const notes = comparison.comparability?.notes ?? [];
    if (notes.length > 0) {
      root.append(h('details.boundary', { open: comparison.comparability?.comparable === false }, [
        h('summary', {}, `可比性：${comparison.comparability?.level === 'high' ? '高' : comparison.comparability?.level === 'medium' ? '中' : '低'}（${String(notes.length)} 条说明）`),
        h('ul', {}, notes.map((note) => h('li', {}, note.text))),
      ]));
    }

    const bottleneck = comparison.bottleneck ?? {};
    if (bottleneck.changed === true) {
      root.append(h('p.compare-note', {}, `瓶颈类型变化：${bottleneck.before?.label} ${String(bottleneck.before?.score)} → ${bottleneck.after?.label} ${String(bottleneck.after?.score)}（得分 ${bottleneck.scoreDelta >= 0 ? '+' : ''}${String(bottleneck.scoreDelta)}）`));
    } else if (bottleneck.after !== undefined) {
      root.append(h('p.compare-note', {}, `瓶颈类型未变：${bottleneck.after.label} ${String(bottleneck.before?.score)} → ${String(bottleneck.after.score)} 分（${bottleneck.scoreDelta >= 0 ? '+' : ''}${String(bottleneck.scoreDelta)}）`));
    }

    // Recommendation verification: the reason this step exists.
    const items = comparison.recommendations ?? [];
    if (items.length > 0) {
      const rows = items.map((item) => {
        const verdict = String(item.verdict ?? 'unknown');
        return h('div.verify-row', { dataset: { adviceId: item.id, verdict } }, [
          h('div.verify-head', {}, [
            h('span', { class: `verify-verdict ${verdict}` }, VERDICT_LABEL[verdict] ?? verdict),
            h('span.title', {}, item.title),
            h('span.chip.phase', {}, (item.phase ?? []).map((phase) => global.VAP.PHASE_LABELS?.[phase] ?? phase).join('/')),
            item.expectedPct === undefined ? undefined : h('span.hint', {}, `预期 ${formatPct(item.expectedPct)}`),
            item.observedPct === undefined ? undefined : h('span.hint', {}, `实测 ${item.observedPct > 0 ? '改善' : '变化'} ${formatPct(Math.abs(item.observedPct))}`),
          ]),
          h('div.verify-targets', {}, (item.targets ?? []).map((target) => h('span.verify-target', {}, [
            `${target.label} `,
            h('b', { class: target.improvementPct > 0 ? 'delta-up' : target.improvementPct < 0 ? 'delta-down' : 'delta-flat' },
              `${target.improvementPct > 0 ? '−' : target.improvementPct < 0 ? '+' : ''}${Math.abs(Number(target.improvementPct ?? 0)).toFixed(1)}%`),
          ]))),
          h('div.note', {}, item.note ?? ''),
        ]);
      });
      root.append(h('h3', {}, `建议达成校验（${String(items.filter((item) => item.verdict === 'achieved').length)} 达成 / ${String(items.filter((item) => item.verdict === 'partial').length)} 部分 / ${String(items.filter((item) => item.verdict === 'missed').length)} 未达成 / ${String(items.filter((item) => item.verdict === 'unknown').length)} 无法判定）`));
      root.append(h('div.verify-list', {}, rows));
    }

    const warnings = comparison.warnings ?? [];
    if (warnings.length > 0) {
      root.append(h('details.boundary', {}, [
        h('summary', {}, `结论边界（${String(warnings.length)} 条）`),
        h('ul', {}, warnings.map((warning) => h('li', {}, warning))),
      ]));
    }
    return root;
  }

  /**
   * Compact chips over the swimlanes.
   *
   * @param {object} comparison - payload.
   * @returns {HTMLElement} strip.
   */
  function renderDeltaStrip(comparison) {
    const strip = h('div.delta-strip');
    for (const key of STRIP_KEYS) {
      const metric = metricOf(comparison, key);
      if (metric === undefined) continue;
      strip.append(h('span.delta-chip', {}, [
        h('span.k', {}, metric.label),
        h('span.v', { class: deltaClass(metric) }, deltaText(metric)),
        h('span.k', {}, `${String(metric.before)}${metric.unit} → ${String(metric.after)}${metric.unit}`),
      ]));
    }
    return strip;
  }

  /**
   * Category and operator deltas for step 4.
   *
   * @param {object} comparison - payload.
   * @returns {HTMLElement} table block.
   */
  function renderDeltaTable(comparison) {
    const root = h('div');
    const categories = comparison.categories ?? [];
    if (categories.length > 0) {
      const maxAbs = Math.max(...categories.map((row) => Math.abs(row.deltaUs)), 1);
      const table = h('table');
      // `<caption>` must be the first child of the table: as a sibling it is laid
      // out as a cramped inline box and the header column collapses.
      table.append(h('caption', {}, '大类耗时：优化前 → 优化后（绿色 = 节省）'));
      table.append(h('thead', {}, [h('tr', {}, [
        h('th', {}, '大类'), h('th.num', {}, '优化前'), h('th.num', {}, '优化后'), h('th.num', {}, '变化'),
        h('th.num', {}, '占比 前 → 后'), h('th', {}, '幅度'),
      ])]));
      table.append(h('tbody', {}, categories.map((row) => h('tr', {}, [
        h('td', {}, [h('i', { style: `display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:6px;background:${CATEGORY_COLORS[row.id] ?? '#888'}` }), row.label]),
        h('td.num.before-col', {}, formatUs(row.beforeUs)),
        h('td.num.after-col', {}, formatUs(row.afterUs)),
        h('td.num', { class: row.deltaUs < 0 ? 'delta-up' : row.deltaUs > 0 ? 'delta-down' : 'delta-flat' }, `${row.deltaUs > 0 ? '+' : row.deltaUs < 0 ? '−' : ''}${formatUs(Math.abs(row.deltaUs))}（${row.deltaPct > 0 ? '+' : row.deltaPct < 0 ? '−' : ''}${Math.abs(row.deltaPct).toFixed(1)}%）`),
        h('td.num', {}, `${row.beforeSharePct.toFixed(1)}% → ${row.afterSharePct.toFixed(1)}%`),
        h('td', {}, [h('span.bar-mini', { class: row.deltaUs > 0 ? 'regress' : '', style: `width:${String(Math.max(3, (Math.abs(row.deltaUs) / maxAbs) * 90))}px` })]),
      ]))));
      root.append(h('div.delta-table', {}, [table]));
    }

    const improved = comparison.operators?.improved ?? [];
    const regressed = comparison.operators?.regressed ?? [];
    if (improved.length > 0 || regressed.length > 0) {
      const maxAbs = Math.max(...[...improved, ...regressed].map((row) => Math.abs(row.deltaUs)), 1);
      const table = h('table');
      table.append(h('caption', {}, '算子变化 Top（按节省 / 增加的时间排序）'));
      table.append(h('thead', {}, [h('tr', {}, [
        h('th', {}, '算子'), h('th', {}, '类别'), h('th.num', {}, '优化前'), h('th.num', {}, '优化后'), h('th.num', {}, '变化'), h('th', {}, '幅度'),
      ])]));
      const row = (entry) => h('tr', {}, [
        h('td.mono', {}, entry.name),
        h('td', {}, global.VAP.CATEGORY_LABELS?.[entry.category] ?? entry.category),
        h('td.num.before-col', {}, formatUs(entry.beforeUs)),
        h('td.num.after-col', {}, formatUs(entry.afterUs)),
        h('td.num', { class: entry.deltaUs < 0 ? 'delta-up' : 'delta-down' }, `${entry.deltaUs > 0 ? '+' : '−'}${formatUs(Math.abs(entry.deltaUs))}（${entry.deltaPct > 0 ? '+' : '−'}${Math.abs(entry.deltaPct).toFixed(1)}%）`),
        h('td', {}, [h('span.bar-mini', { class: entry.deltaUs > 0 ? 'regress' : '', style: `width:${String(Math.max(3, (Math.abs(entry.deltaUs) / maxAbs) * 90))}px` })]),
      ]);
      table.append(h('tbody', {}, [...improved, ...regressed].map(row)));
      root.append(h('div.delta-table', {}, [table]));
    }

    if (improved.length === 0 && regressed.length === 0) {
      root.append(h('p.compare-note', {}, '两次采集的算子耗时没有可察觉的差异。'));
    }
    return root;
  }

  global.VAP = global.VAP ?? {};
  global.VAP.compareView = { renderCompareSummary, renderDeltaStrip, renderDeltaTable, deltaClass, deltaText };
})(window);
