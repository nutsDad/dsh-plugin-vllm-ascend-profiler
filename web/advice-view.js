/**
 * Module 5 renderer: the reasoning chain as a diagram instead of a document.
 *
 * Layout: a ①→⑤ flow across the top whose nodes carry the key number of each
 * step, and a single panel below that shows the *selected* step graphically.
 * Nothing is hidden — every step is one click away and the flow itself states
 * the order and the counts — but the default reading is shapes and numbers, not
 * paragraphs. Prose survives inside `依据` disclosures for whoever wants to
 * audit a claim.
 *
 * Panel by step:
 *   ① 定位   score cards per scope + a candidate comparison bar
 *   ② 证据   threshold bars (measured vs gate), i.e. the evidence as a diagram
 *   ③ 根因   cause chains: trigger chips → one-line mechanism → effect chips
 *   ④ 行动   ranked action rows with gain-range bars
 *   ⑤ 收益   per-item gain bars plus conservative/optimistic totals
 */
(function attachAdvice(global) {
  'use strict';

  const { h, formatUs, formatPct, formatCount, PHASE_LABELS, BOTTLENECK_COLORS, diagram } = global.VAP;

  /** Priority label → CSS-safe class token (class names stay ASCII). */
  const PRIORITY_CLASS = { 高: 'pri-high', 中: 'pri-mid', 低: 'pri-low' };

  /** Category filter each advice id should apply to the timeline. */
  const ADVICE_FILTER = {
    'host.enable-graph-mode': { category: 'schedule' },
    'host.async-sampling': { category: 'schedule' },
    'host.reduce-dispatch': { category: 'schedule' },
    'compute.quantize': { category: 'compute' },
    'compute.decode-increase-batch': { category: 'compute' },
    'compute.fuse-small-ops': { category: 'compute' },
    'compute.tune-chunked-prefill': { category: 'compute' },
    'comm.enable-overlap-fusion': { category: 'comm' },
    'comm.batch-amortize': { category: 'comm' },
    'comm.parallel-strategy': { category: 'comm' },
    'copy.async-d2h': { category: 'copy' },
    'copy.kv-locality': { category: 'copy' },
    'copy.reduce-profiling-overhead': { category: 'copy' },
  };

  /** Category label per bottleneck candidate, for the evidence diagram. */
  const CANDIDATE_CATEGORY = { host: 'schedule', compute: 'compute', comm: 'comm', copy: 'copy' };

  /**
   * Render the chain: flow + one panel.
   *
   * @param {object} viewModel - dataset view model (carries `analysis`).
   * @param {object} [options] - `{ priorityFilter, activeStep, onFocus, onStep }`.
   * @returns {{element: HTMLElement, setStep: (id: string) => void}} view.
   */
  function renderAdvice(viewModel, options = {}) {
    const analysis = viewModel.analysis;
    const root = h('div.advice-root');
    const nodes = flowNodes(analysis);
    let active = options.activeStep ?? nodes[0].id;

    const flowHost = h('div');
    const panelHost = h('div.panel-host');
    root.append(flowHost, panelHost);

    const paint = () => {
      flowHost.replaceChildren(diagram.chainFlow({
        nodes,
        active,
        onSelect: (id) => setStep(id),
      }));
      panelHost.replaceChildren(renderPanel(analysis, active, options, nodes));
      panelHost.classList.remove('enter');
      void panelHost.offsetWidth;
      panelHost.classList.add('enter');
    };
    const setStep = (id) => {
      active = id;
      options.onStep?.(id);
      paint();
    };
    paint();

    if (analysis.warnings.length > 0) {
      root.append(h('details.boundary', {}, [
        h('summary', {}, `结论边界（${String(analysis.warnings.length)} 条）`),
        h('ul', {}, analysis.warnings.map((warning) => h('li', {}, warning))),
      ]));
    }
    return { element: root, setStep };
  }

  /** The five flow nodes, each reduced to one number. */
  function flowNodes(analysis) {
    const counts = { 高: 0, 中: 0, 低: 0 };
    for (const item of analysis.steps.actions.items) counts[item.priority] = (counts[item.priority] ?? 0) + 1;
    const gates = analysis.steps.locate.perScope.flatMap((scope) => scope.candidates.flatMap((candidate) => candidate.evidence))
      .filter((row) => row.threshold !== undefined);
    const passed = gates.filter((row) => row.passed !== false).length;
    const bottleneck = analysis.bottleneck;
    return [
      {
        id: 'locate',
        index: '①',
        title: '瓶颈定位',
        value: `${bottleneck.short ?? bottleneck.label} ${bottleneck.score.toFixed(0)}`,
        hint: '四类候选分别打分，全量窗口与 Prefill/Decode 独立判定',
      },
      {
        id: 'evidence',
        index: '②',
        title: '量化证据',
        value: `${String(passed)}/${String(gates.length)} 达门限`,
        hint: '每个指标的实测值与门限对比',
      },
      {
        id: 'cause',
        index: '③',
        title: '根因推断',
        value: analysis.steps.cause.items.length === 0 ? '无' : `${String(analysis.steps.cause.items.length)} 条机理`,
        hint: '触发数据 → 作用机理 → 影响',
        muted: analysis.steps.cause.items.length === 0,
      },
      {
        id: 'actions',
        index: '④',
        title: '优化行动',
        value: `高${String(counts.高)} 中${String(counts.中)} 低${String(counts.低)}`,
        hint: '按优先级排序的可执行项',
      },
      {
        id: 'benefit',
        index: '⑤',
        title: '预期收益',
        value: `保守 ${formatPct(analysis.steps.benefit.combined.conservativePct)}`,
        hint: '由本次数据推算，或标注为经验区间',
      },
    ];
  }

  /** Body of the selected step. */
  function renderPanel(analysis, step, options, nodes) {
    switch (step) {
      case 'evidence': return evidencePanel(analysis, options);
      case 'cause': return causePanel(analysis);
      case 'actions': return actionsPanel(analysis, options);
      case 'benefit': return benefitPanel(analysis);
      default: return locatePanel(analysis, nodes);
    }
  }

  // ── ① locate ────────────────────────────────────────────────────────────

  function locatePanel(analysis, nodes) {
    const scopes = analysis.steps.locate.perScope;
    const candidates = ['host', 'compute', 'comm', 'copy'];
    const labels = Object.fromEntries(scopes[0].candidates.map((candidate) => [candidate.id, candidate.label]));

    // Candidate × scope comparison: one legend, then one row of four bars per
    // scope. A zero score is drawn as an empty slot with its number, so "this
    // mechanism is not implicated" is visible instead of blank space.
    const chart = h('div.compare-chart', {}, [
      h('div.compare-legend', {}, candidates.map((id) => h('span', { title: labels[id] }, [
        h('i', { style: `background:${BOTTLENECK_COLORS[id] ?? 'var(--accent)'}` }),
        labels[id],
      ]))),
    ]);
    for (const scope of scopes) {
      const bars = h('div.compare-bars');
      for (const id of candidates) {
        const score = scope.candidates.find((entry) => entry.id === id)?.score ?? 0;
        bars.append(h('div.compare-bar', { title: `${labels[id]}：${score.toFixed(0)}/100` }, [
          h('i', {
            class: score < 1 ? 'zero' : '',
            style: `height:${String(Math.max(0, Math.min(100, score)))}%;background:${BOTTLENECK_COLORS[id] ?? 'var(--accent)'}`,
          }),
          h('span', { class: score < 1 ? 'zero' : '' }, score.toFixed(0)),
        ]));
      }
      chart.append(h('div.compare-row', {}, [h('div.compare-label', {}, scope.scopeLabel), bars]));
    }

    const blocks = [chart];
    for (const scope of scopes) {
      const primary = scope.primaryCandidate;
      blocks.push(h('div.scope-line', {}, [
        h('span.chip.verdict', { style: `background:${BOTTLENECK_COLORS[primary.id] ?? 'var(--accent)'}` }, `${scope.scopeLabel} · ${primary.short ?? primary.label} ${primary.score.toFixed(0)}`),
        h('span.scope-text', {}, oneLine(primary.summary)),
        h('details.compact', {}, [
          h('summary', {}, '依据'),
          evidenceTable(primary.evidence),
        ]),
      ]));
      if ((primary.missing ?? []).length > 0) {
        blocks.push(h('ul.missing', {}, primary.missing.slice(0, 3).map((item) => h('li', {}, item))));
      }
    }
    void nodes;
    return h('div.panel', {}, blocks);
  }

  function evidenceTable(rows) {
    const table = h('table.compact-table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '指标'), h('th.num', {}, '实测'), h('th', {}, '门限判定'),
    ])]));
    table.append(h('tbody', {}, rows.map((row) => h('tr', {}, [
      h('td', {}, row.metric),
      h('td.num', {}, row.unit === '%' ? formatPct(row.value, 2) : row.value === undefined ? 'N/A' : `${String(row.value)}${row.unit ?? ''}`),
      h('td', { class: row.passed === false ? 'conf-low' : 'conf-high' }, row.comparison),
    ]))));
    return table;
  }

  // ── ② evidence ──────────────────────────────────────────────────────────

  function evidencePanel(analysis, options) {
    // Prefer the located bottleneck's own evidence (it is the reason for the
    // verdict); fall back to the overall indicator list.
    const scope = analysis.steps.locate.perScope[0];
    const primary = scope.candidates.find((candidate) => candidate.id === scope.primaryCandidate.id) ?? scope.candidates[0];
    const rows = primary.evidence
      .filter((row) => Number.isFinite(row.value))
      .map((row) => ({
        metric: row.metric,
        value: row.value,
        unit: row.unit,
        threshold: row.threshold,
        passed: row.passed,
        source: row.source,
        category: CANDIDATE_CATEGORY[primary.id],
      }));
    const bars = diagram.thresholdBars({
      rows,
      onSelect: typeof options.onFocus === 'function' ? ({ category }) => options.onFocus({ category }) : undefined,
    });
    const groups = [h('div.panel-head', {}, [
      h('span.chip.verdict', { style: `background:${BOTTLENECK_COLORS[primary.id] ?? 'var(--accent)'}` }, `${primary.short ?? primary.label} 的判定依据`),
      h('span.hint', {}, '点击任一指标可在第 3 步按时序筛选同类算子'),
    ])];
    groups.push(h('div.diagram-host', {}, [bars.element, bars.legend]));
    return h('div.panel', {}, groups);
  }

  // ── ③ cause ─────────────────────────────────────────────────────────────

  function causePanel(analysis) {
    const items = analysis.steps.cause.items;
    if (items.length === 0) {
      return h('div.panel', {}, [h('p.hint', {}, '未定位到达到门限的瓶颈，因此不输出根因推断（避免无据结论）。')]);
    }
    return h('div.panel', {}, [diagram.causeChains({ items })]);
  }

  // ── ④ actions ───────────────────────────────────────────────────────────

  function actionsPanel(analysis, options) {
    const all = analysis.steps.actions.items;
    const allowed = options.priorityFilter === undefined || options.priorityFilter === 'all'
      ? undefined
      : new Set(String(options.priorityFilter).split(','));
    const items = allowed === undefined ? all : all.filter((item) => allowed.has(item.priority));
    if (items.length === 0) {
      return h('div.panel', {}, [h('p.hint', {}, all.length === 0 ? '没有匹配到可执行的优化项：请先按"补齐采集"建议完善 profiling 产物。' : '当前优先级筛选下没有优化项。')]);
    }
    const max = Math.max(...items.map((item) => item.expectedGain?.rangePct?.[1] ?? item.expectedGain?.estimatePct ?? 0), 10);

    const rows = items.map((item, index) => {
      const link = ADVICE_FILTER[item.id];
      const gain = item.expectedGain;
      const head = h('div.action-row', { dataset: { adviceId: item.id } }, [
        h('span', { class: `pri ${PRIORITY_CLASS[item.priority] ?? 'pri-mid'}` }, item.priority),
        h('span.action-index.mono', {}, String(index + 1)),
        h('div.action-main', {}, [
          h('div.action-title', {}, item.title),
          h('div.action-meta', {}, [
            h('span.chip.phase', {}, (item.phases ?? [item.phase]).map((phase) => PHASE_LABELS[phase] ?? phase).join('/')),
            item.confirmInVersion === true ? h('span.chip.phase.warn', {}, '需版本确认') : undefined,
          ]),
        ]),
        gain === undefined ? h('span.hint', {}, '—') : h('div.action-gain', {}, [
          h('span.gain-value.mono', {}, formatPct(gain.estimatePct)),
          diagram.gainBar({ rangePct: gain.rangePct, estimatePct: gain.estimatePct, confidence: gain.confidence, max }),
          h('span', { class: `gain-conf ${gain.confidence === 'low' ? 'conf-low' : 'conf-high'}` }, confidenceLabel(gain.confidence)),
        ]),
        h('div.action-actions', {}, [
          link !== undefined && typeof options.onFocus === 'function'
            ? h('button.small.ghost', { type: 'button', onclick: () => options.onFocus(link) }, '查看')
            : undefined,
        ]),
      ]);
      const detail = h('details.compact', {}, [
        h('summary', {}, '依据 / 动作 / 验证'),
        h('dl', {}, [
          h('dt', {}, '依据'), h('dd', {}, item.rationale),
          h('dt', {}, '动作'), h('dd', {}, h('ul', {}, item.actions.map((action) => h('li', {}, [`${action.type} · `, action.text])))),
          gain === undefined ? undefined : h('dt', {}, '收益推算'),
          gain === undefined ? undefined : h('dd', {}, `${gain.metric} · ${gain.basis}（前提：${gain.assumption}）`),
          h('dt', {}, '验证'), h('dd', {}, item.verification),
          h('dt', {}, '风险'), h('dd', {}, item.risk),
          item.linkedRootCauses.length === 0 ? undefined : h('dt', {}, '关联根因'),
          item.linkedRootCauses.length === 0 ? undefined : h('dd.mono', {}, item.linkedRootCauses.join('、')),
        ]),
      ]);
      return h('div.action-block', {}, [head, detail]);
    });
    return h('div.panel', {}, [
      h('div.panel-head', {}, [
        h('span.hint', {}, `共 ${String(items.length)} 项 · 横条为收益区间（竖线=估算值，浅色=经验区间）`),
      ]),
      ...rows,
    ]);
  }

  // ── ⑤ benefit ───────────────────────────────────────────────────────────

  function benefitPanel(analysis) {
    const benefit = analysis.steps.benefit;
    const items = benefit.items;
    if (items.length === 0) {
      return h('div.panel', {}, [h('p.hint', {}, benefit.note)]);
    }
    const max = Math.max(...items.map((item) => item.rangePct?.[1] ?? item.estimatePct), 1);
    // The benefit list is short, so its bars get the wide variant: the reading is
    // "how much of the achievable range does each item cover", and that needs room.
    const rows = items.map((item) => h('div.benefit-row', {}, [
      h('div.benefit-title', {}, item.title),
      h('div.benefit-bar', {}, [diagram.gainBar({ rangePct: item.rangePct, estimatePct: item.estimatePct, confidence: item.confidence, max, width: 520 })]),
      h('span.benefit-value.mono', {}, formatPct(item.estimatePct)),
      h('span.chip.phase', {}, item.phase.map((phase) => PHASE_LABELS[phase] ?? phase).join('/')),
    ]));
    return h('div.panel', {}, [
      h('div.benefit-totals', {}, [
        h('div.total', {}, [h('span.k', {}, '保守合计'), h('b', {}, formatPct(benefit.combined.conservativePct))]),
        h('div.total', {}, [h('span.k', {}, '乐观合计'), h('b', {}, formatPct(benefit.combined.optimisticPct))]),
        h('div.total', {}, [h('span.k', {}, '可量化项'), h('b', {}, String(benefit.combined.itemCount))]),
      ]),
      ...rows,
      h('p.hint', {}, benefit.combined.note),
    ]);
  }

  function confidenceLabel(confidence) {
    return { high: '置信度高', medium: '置信度中', low: '经验区间' }[confidence] ?? '未知';
  }

  /** First sentence, capped — the diagram-level summary of a paragraph. */
  function oneLine(text) {
    return diagram.oneLine(text);
  }

  /** The three highest-priority actions, as the overview's answer-first list. */
  function topActions(analysis, limit = 3) {
    return analysis.steps.actions.items.slice(0, limit).map((item) => ({
      id: item.id,
      title: item.title,
      priority: item.priority,
      phase: item.phases ?? [item.phase],
      gainPct: item.expectedGain?.estimatePct,
      confidence: item.expectedGain?.confidence,
      filter: ADVICE_FILTER[item.id],
    }));
  }

  void formatUs;
  void formatCount;

  global.VAP = global.VAP ?? {};
  global.VAP.advice = { renderAdvice, topActions, ADVICE_FILTER };
})(window);
