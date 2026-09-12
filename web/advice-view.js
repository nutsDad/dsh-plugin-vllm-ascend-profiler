/**
 * Module 3 renderer: the five-step reasoning chain, laid out so that the logic is
 * visible rather than implied.
 *
 * Reading order and linkage:
 *   * each step is collapsible and states what it consumes and what it produces;
 *   * step ① shows the four scored candidates side by side (the *choice* is the
 *     content, not just the winner);
 *   * step ② keeps every threshold comparison one click away instead of inline,
 *     so the page stays readable while remaining auditable;
 *   * step ④ carries an action link ("在泳道图查看") that applies the matching
 *     filter to step ③ — advice and evidence point at each other.
 */
(function attachAdvice(global) {
  'use strict';

  const { h, formatUs, formatPct, formatCount, PHASE_LABELS, BOTTLENECK_COLORS } = global.VAP;

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

  /**
   * Render the whole chain.
   *
   * @param {object} viewModel - dataset view model (carries `analysis`).
   * @param {object} [options] - `{ priorityFilter, onFocus, collapsed }`.
   * @returns {HTMLElement} root node.
   */
  function renderAdvice(viewModel, options = {}) {
    const analysis = viewModel.analysis;
    const root = h('div.advice-root');
    const collapsed = options.collapsed ?? new Set();

    root.append(chainStep({
      index: '①',
      title: '瓶颈类型定位',
      consumes: '四类候选分别打分，全量窗口与 Prefill/Decode 阶段独立判定',
      summary: analysis.bottleneck.label,
      collapsed: collapsed.has('locate'),
      body: locateBody(analysis),
    }));
    root.append(chainStep({
      index: '②',
      title: '量化证据',
      consumes: `${String(analysis.steps.evidence.overall.length)} 项指标 + 门限比对明细`,
      summary: `NPU 忙碌 ${formatPct(analysis.indicators.deviceBusyPct)} · Host 独占 ${formatPct(analysis.indicators.hostOnlyPct)}`,
      collapsed: collapsed.has('evidence'),
      body: evidenceBody(analysis),
    }));
    root.append(chainStep({
      index: '③',
      title: '根因推断',
      consumes: '基于上一步的证据触发条件',
      summary: analysis.steps.cause.items.length === 0 ? '无（未达门限）' : `${String(analysis.steps.cause.items.length)} 条假设`,
      collapsed: collapsed.has('cause'),
      body: causeBody(analysis),
    }));
    root.append(chainStep({
      index: '④',
      title: '可落地优化方案',
      consumes: '每条方案都绑定根因与证据，并给出验证方法与风险',
      summary: prioritySummary(analysis),
      collapsed: collapsed.has('actions'),
      body: actionBody(analysis, options),
    }));
    root.append(chainStep({
      index: '⑤',
      title: '预期收益',
      consumes: '由本次数据推算，或明确标注为经验区间',
      summary: `保守 ${formatPct(analysis.steps.benefit.combined.conservativePct)} · 乐观 ${formatPct(analysis.steps.benefit.combined.optimisticPct)}`,
      collapsed: collapsed.has('benefit'),
      body: benefitBody(analysis),
    }));
    if (analysis.warnings.length > 0) {
      root.append(h('div.alert.warn', {}, [
        h('strong', {}, '结论边界'),
        h('ul', {}, analysis.warnings.map((warning) => h('li', {}, warning))),
      ]));
    }
    return root;
  }

  /** Chain frame with a clickable head that collapses its body. */
  function chainStep({ index, title, consumes, summary, collapsed, body }) {
    const head = h('div.chain-head', { role: 'button', tabindex: '0', 'aria-expanded': String(!collapsed) }, [
      h('span.idx', {}, index),
      h('h3', {}, title),
      h('span.sub', {}, summary ?? ''),
      h('span.caret', {}, '▼'),
    ]);
    const wrapper = h(`div.chain-step${collapsed ? '.collapsed' : ''}`, {}, [head]);
    const content = h('div.chain-body', {}, [consumes === undefined ? undefined : h('p.hint', {}, consumes), ...body]);
    if (collapsed) content.hidden = true;
    wrapper.append(content);
    const toggle = () => {
      const next = !content.hidden;
      content.hidden = next;
      wrapper.classList.toggle('collapsed', next);
      head.setAttribute('aria-expanded', String(!next));
    };
    head.addEventListener('click', toggle);
    head.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        toggle();
      }
    });
    return wrapper;
  }

  // ── ① locate ────────────────────────────────────────────────────────────

  function locateBody(analysis) {
    const blocks = [];
    for (const scope of analysis.steps.locate.perScope) {
      const cards = scope.candidates.map((candidate) => {
        const isPrimary = candidate.id === scope.primaryCandidate.id;
        return h(`div.score-card${isPrimary ? '.primary' : ''}`, {}, [
          h('div.k', {}, candidate.label),
          h('div.v', {}, candidate.score.toFixed(0)),
          // Width is applied one frame after insertion so the CSS transition runs
          // (see `animateMeters` in app.js) instead of appearing pre-filled.
          h('div.meter', {}, [h('i', { dataset: { meter: String(Math.min(100, candidate.score)) }, style: `background:${BOTTLENECK_COLORS[candidate.id] ?? 'var(--accent)'}` })]),
          h('div.note', {}, candidate.score >= 40 ? '达到门限' : '未达门限'),
        ]);
      });
      const body = [h('p', {}, scope.primaryCandidate.summary), h('div.score-grid', {}, cards)];
      if (scope.assumedPhase !== undefined) {
        body.push(h('p.hint', {}, `全量窗口按主导阶段 ${PHASE_LABELS[scope.assumedPhase] ?? scope.assumedPhase} 评估`));
      }
      for (const candidate of scope.candidates) {
        if (candidate.score === 0) continue;
        body.push(h('details', {}, [
          h('summary', {}, `${candidate.short} · 判定依据（${String(candidate.evidence.length)} 项）`),
          evidenceTable(candidate.evidence),
        ]));
      }
      blocks.push(h('div.scope-block', {}, [h('h4', {}, scope.scopeLabel), ...body]));
    }
    return blocks;
  }

  function evidenceTable(rows) {
    const table = h('table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '指标'), h('th.num', {}, '实测'), h('th', {}, '门限判定'), h('th', {}, '来源'),
    ])]));
    table.append(h('tbody', {}, rows.map((row) => h('tr', {}, [
      h('td', {}, row.metric),
      h('td.num', {}, row.unit === '%' ? formatPct(row.value, 2) : row.value === undefined ? 'N/A' : `${String(row.value)}${row.unit ?? ''}`),
      h('td', { class: row.passed === false ? 'conf-low' : '' }, row.comparison),
      h('td.mono', {}, row.source),
    ]))));
    return table;
  }

  // ── ② evidence ──────────────────────────────────────────────────────────

  function evidenceBody(analysis) {
    const blocks = [h('h4', {}, '全量窗口'), metricTable(analysis.steps.evidence.overall)];
    for (const [phaseId, rows] of Object.entries(analysis.steps.evidence.phases)) {
      blocks.push(h('h4', {}, `阶段 · ${PHASE_LABELS[phaseId] ?? phaseId}`), metricTable(rows));
    }
    return blocks;
  }

  function metricTable(rows) {
    const table = h('table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '分组'), h('th', {}, '指标'), h('th.num', {}, '数值'), h('th', {}, '说明'),
    ])]));
    table.append(h('tbody', {}, rows.map((row) => h('tr', {}, [
      h('td', {}, row.group),
      h('td', {}, row.metric),
      h('td.num', {}, row.value === undefined ? 'N/A' : formatMetric(row.value, row.unit)),
      h('td.hint', {}, row.note),
    ]))));
    return table;
  }

  function formatMetric(value, unit) {
    if (!Number.isFinite(value)) return 'N/A';
    if (unit === '%') return `${value.toFixed(2)}%`;
    if (unit === 'µs') return formatUs(value);
    if (unit === 'ms') return `${value.toFixed(3)}ms`;
    if (unit === '') return value.toFixed(3);
    return `${String(value)}${unit ?? ''}`;
  }

  // ── ③ cause ─────────────────────────────────────────────────────────────

  function causeBody(analysis) {
    const items = analysis.steps.cause.items;
    if (items.length === 0) {
      return [h('p', {}, '未定位到达到门限的瓶颈，因此不输出根因推断（避免无据结论）。')];
    }
    return items.map((item) => h('div.scope-block', {}, [
      h('h4', {}, item.title),
      h('p.hint', {}, `${item.bottleneck} · ${PHASE_LABELS[item.phase] ?? item.phase} · 作用范围 ${(item.scopes ?? [item.scope]).map((scope) => PHASE_LABELS[scope] ?? scope).join('、')}`),
      h('dl', {}, [
        h('dt', {}, '机理'), h('dd', {}, item.mechanism),
        h('dt', {}, 'vLLM 侧原因'), h('dd', {}, item.vllmBehaviour),
      ]),
      h('details', {}, [
        h('summary', {}, `触发数据（${String(item.triggers.length)} 项）`),
        h('table', {}, [
          h('thead', {}, [h('tr', {}, [h('th', {}, '指标'), h('th.num', {}, '数值'), h('th', {}, '说明')])]),
          h('tbody', {}, item.triggers.map((trigger) => h('tr', {}, [
            h('td', {}, trigger.metric),
            h('td.num', {}, trigger.value === undefined ? 'N/A' : formatMetric(trigger.value, trigger.unit)),
            h('td.hint', {}, trigger.note),
          ]))),
        ]),
      ]),
      h('details', {}, [h('summary', {}, '现场确认方法'), h('ul', {}, item.checks.map((check) => h('li', {}, check)))]),
    ]));
  }

  // ── ④ actions ───────────────────────────────────────────────────────────

  function prioritySummary(analysis) {
    const counts = { 高: 0, 中: 0, 低: 0 };
    for (const item of analysis.steps.actions.items) counts[item.priority] = (counts[item.priority] ?? 0) + 1;
    return `高 ${String(counts.高)} / 中 ${String(counts.中)} / 低 ${String(counts.低)}`;
  }

  function actionBody(analysis, options) {
    const all = analysis.steps.actions.items;
    const allowed = options.priorityFilter === undefined || options.priorityFilter === 'all'
      ? undefined
      : new Set(String(options.priorityFilter).split(','));
    const items = allowed === undefined ? all : all.filter((item) => allowed.has(item.priority));
    if (items.length === 0) {
      return [h('p', {}, all.length === 0 ? '没有匹配到可执行的优化项：请先按"补齐采集"建议完善 profiling 产物。' : '当前优先级筛选下没有优化项。')];
    }
    return items.map((item, index) => {
      const link = ADVICE_FILTER[item.id];
      const body = [
        h('div.advice-head', {}, [
          h('span.pri', {}, item.priority),
          h('span.title', {}, `${String(index + 1)}. ${item.title}`),
          h('span.phase', {}, `${(item.phases ?? [item.phase]).map((phase) => PHASE_LABELS[phase] ?? phase).join('/')} · 得分 ${item.priorityScore.toFixed(1)}${item.confirmInVersion === true ? ' · 含需版本确认的开关' : ''}`),
        ]),
        h('dl', {}, [
          h('dt', {}, '依据'), h('dd', {}, item.rationale),
          h('dt', {}, '动作'), h('dd', {}, h('ul', {}, item.actions.map((action) => h('li', {}, [h('span.tag', {}, action.type), ` ${action.text}`])))),
        ]),
      ];
      if (item.expectedGain !== undefined) {
        body.push(h('div.gain', {}, [
          h('div', {}, [
            h('strong', {}, '预期收益 '),
            h('span.big', {}, formatPct(item.expectedGain.estimatePct)),
            item.expectedGain.rangePct === undefined ? undefined : ` （区间 ${item.expectedGain.rangePct[0].toFixed(1)}% – ${item.expectedGain.rangePct[1].toFixed(1)}%）`,
            h('span', { class: item.expectedGain.confidence === 'low' ? 'conf-low' : 'conf-high' }, ` · ${confidenceLabel(item.expectedGain.confidence)}`),
          ]),
          h('div.hint', {}, `${item.expectedGain.metric} · ${item.expectedGain.basis}`),
          h('div.hint', {}, `前提：${item.expectedGain.assumption}`),
        ]));
      }
      body.push(h('dl', {}, [h('dt', {}, '验证'), h('dd', {}, item.verification), h('dt', {}, '风险'), h('dd', {}, item.risk)]));
      const actions = h('div', { style: 'margin-top:7px;display:flex;gap:8px;flex-wrap:wrap' });
      if (link !== undefined && typeof options.onFocus === 'function') {
        actions.append(h('button.small.ghost', {
          type: 'button',
          onclick: () => options.onFocus(link),
        }, link.operator === undefined ? `在第 3 步查看「${categoryLabel(link.category)}」` : `在第 3 步查看「${link.operator}」`));
      }
      if (item.linkedRootCauses.length > 0) {
        actions.append(h('span.hint', {}, `关联根因：${item.linkedRootCauses.join('、')}`));
      }
      body.push(actions);
      return h(`div.advice.${PRIORITY_CLASS[item.priority] ?? 'pri-mid'}`, { dataset: { priority: item.priority, adviceId: item.id } }, body);
    });
  }

  function categoryLabel(category) {
    return { compute: '计算算子', comm: '通信算子', copy: '数据拷贝', schedule: 'Host 调度', other: '其他' }[category] ?? category;
  }

  // ── ⑤ benefit ───────────────────────────────────────────────────────────

  function benefitBody(analysis) {
    const benefit = analysis.steps.benefit;
    const table = h('table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '优化项'), h('th', {}, '优先级'), h('th', {}, '阶段'), h('th', {}, '目标指标'),
      h('th.num', {}, '估算'), h('th.num', {}, '区间'), h('th', {}, '置信度'),
    ])]));
    table.append(h('tbody', {}, benefit.items.map((item) => h('tr', {}, [
      h('td', {}, item.title),
      h('td', {}, item.priority),
      h('td', {}, item.phase.map((phase) => PHASE_LABELS[phase] ?? phase).join('/')),
      h('td', {}, item.metric),
      h('td.num', {}, formatPct(item.estimatePct)),
      h('td.num', {}, item.rangePct === undefined ? 'N/A' : `${item.rangePct[0].toFixed(1)}%–${item.rangePct[1].toFixed(1)}%`),
      h('td', { class: item.confidence === 'low' ? 'conf-low' : 'conf-high' }, confidenceLabel(item.confidence)),
    ]))));
    const body = [
      table,
      h('p', {}, [
        h('strong', {}, `保守合计 ${formatPct(benefit.combined.conservativePct)}`),
        ` · 乐观合计 ${formatPct(benefit.combined.optimisticPct)}（${String(benefit.combined.itemCount)} 项可量化）`,
      ]),
      h('p.hint', {}, benefit.combined.note),
    ];
    if (benefit.items.length === 0) body.unshift(h('p', {}, benefit.note));
    void formatCount;
    return body;
  }

  function confidenceLabel(confidence) {
    return { high: '置信度高', medium: '置信度中', low: '经验区间' }[confidence] ?? '未知';
  }

  /** The three highest-priority actions, as the overview's "answer first" list. */
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

  global.VAP = global.VAP ?? {};
  global.VAP.advice = { renderAdvice, topActions };
})(window);
