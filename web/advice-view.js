/**
 * Module 3 renderer: the five-step reasoning chain as a readable, auditable
 * page section.
 *
 * The structure mirrors the analysis result exactly — locate, evidence, cause,
 * actions, benefit — and every number is shown with the threshold it was
 * compared against, so a reader can disagree with the conclusion on the
 * evidence rather than having to trust a verdict.
 */
(function attachAdvice(global) {
  'use strict';

  const { h, formatUs, formatPct, formatRatio, formatCount, PHASE_LABELS, BOTTLENECK_COLORS } = global.VAP;

  /** Priority label to a CSS-safe class token (class names stay ASCII). */
  const PRIORITY_CLASS = { 高: 'pri-high', 中: 'pri-mid', 低: 'pri-low' };

  /**
   * Render the whole chain.
   * @param {object} viewModel - dataset view model (carries `analysis`).
   * @param {object} [options] - `{ priorityFilter }`.
   * @returns {HTMLElement} root node.
   */
  function renderAdvice(viewModel, options = {}) {
    const analysis = viewModel.analysis;
    const root = h('div.advice-root');
    root.append(stepLocate(analysis));
    root.append(stepEvidence(analysis));
    root.append(stepCause(analysis));
    root.append(stepActions(analysis, options.priorityFilter));
    root.append(stepBenefit(analysis));
    if (analysis.warnings.length > 0) {
      const box = h('div.alert.warn', {}, [
        h('strong', {}, '结论边界'),
        h('ul', {}, analysis.warnings.map((warning) => h('li', {}, warning))),
      ]);
      root.append(box);
    }
    return root;
  }

  /** Chain frame with a numbered head. */
  function chainStep(index, title, subtitle, body) {
    return h('div.chain-step', {}, [
      h('div.chain-head', {}, [
        h('span.idx', {}, String(index)),
        h('h3', {}, title),
        subtitle === undefined ? undefined : h('span.sub', {}, subtitle),
      ]),
      h('div.chain-body', {}, body),
    ]);
  }

  function stepLocate(analysis) {
    const scopes = analysis.steps.locate.perScope.map((scope) => {
      const cards = scope.candidates.map((candidate) => h(`div.score-card${candidate.id === scope.primaryCandidate.id ? '.primary' : ''}`, {}, [
        h('div.k', {}, candidate.label),
        h('div.v', {}, candidate.score.toFixed(0)),
        h('div.meter', {}, [h('i', { style: `width:${String(Math.min(100, candidate.score))}%;background:${BOTTLENECK_COLORS[candidate.id] ?? 'var(--accent)'}` })]),
        h('div.note', {}, candidate.score >= 40 ? '达到瓶颈门限' : '未达门限（证据不足）'),
      ]));
      const body = [
        h('p', {}, scope.primaryCandidate.summary),
        h('div.score-grid', {}, cards),
      ];
      if (scope.assumedPhase !== undefined) {
        body.push(h('p.hint', {}, `全量窗口没有单一阶段，本判定按主导阶段 ${PHASE_LABELS[scope.assumedPhase] ?? scope.assumedPhase} 评估；阶段维度结论见下方对应区块。`));
      }
      for (const candidate of scope.candidates) {
        if (candidate.score === 0) continue;
        body.push(h('details', {}, [
          h('summary', {}, `${candidate.label} · 判定依据（${candidate.evidence.length} 项证据）`),
          evidenceTable(candidate.evidence),
        ]));
      }
      if ((scope.primaryCandidate.missing ?? []).length > 0) {
        body.push(h('ul.hint', {}, scope.primaryCandidate.missing.map((item) => h('li', {}, `未达门限：${item}`))));
      }
      return h('div.scope-block', {}, [h('h4', {}, scope.scopeLabel), ...body]);
    });
    return chainStep('①', '瓶颈类型定位', analysis.bottleneck.label, [
      h('p', {}, analysis.steps.locate.summary),
      ...scopes,
    ]);
  }

  function evidenceTable(rows) {
    const table = h('table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '指标'), h('th.num', {}, '实测'), h('th', {}, '门限判定'), h('th', {}, '数据来源'),
    ])]));
    table.append(h('tbody', {}, rows.map((row) => h('tr', {}, [
      h('td', {}, row.metric),
      h('td.num', {}, row.unit === '%' ? formatPct(row.value, 2) : row.value === undefined ? 'N/A' : `${String(row.value)}${row.unit ?? ''}`),
      h('td', { class: row.passed === false ? 'conf-low' : '' }, row.comparison),
      h('td.mono', {}, row.source),
    ]))));
    return table;
  }

  function stepEvidence(analysis) {
    const blocks = [];
    blocks.push(h('h4', {}, '全量窗口指标'));
    blocks.push(metricTable(analysis.steps.evidence.overall));
    for (const [phaseId, rows] of Object.entries(analysis.steps.evidence.phases)) {
      blocks.push(h('h4', {}, `阶段指标 · ${PHASE_LABELS[phaseId] ?? phaseId}`));
      blocks.push(metricTable(rows));
    }
    const indicator = analysis.indicators;
    blocks.push(h('h4', {}, '口径说明'));
    blocks.push(h('ul', {}, [
      h('li', {}, `设备忙碌时间按区间并集计算：${formatUs(indicator.deviceBusyUs)}，占墙钟 ${formatPct(indicator.deviceBusyPct)}。`),
      h('li', {}, `Host 忙碌时间按区间并集计算：${formatUs(indicator.hostBusyUs)}，占墙钟 ${formatPct(indicator.hostBusyPct)}。`),
      h('li', {}, `通信并集 ${formatUs(indicator.commUs)}，未掩盖 ${formatUs(indicator.commExposedUs)}（重叠率 ${formatPct(indicator.commOverlapPct)}）。`),
      h('li', {}, `设备空闲 ${formatUs(indicator.idleUs)}（${formatPct(indicator.idlePct)}），空闲段 ${formatCount(indicator.gapCount)} 个，最大 ${formatUs(indicator.maxGapUs)}。`),
      h('li', {}, `Host 独占（设备空等）每步 ${formatUs(indicator.hostExclusivePerStepUs)}，Host 派发算子 ${indicator.dispatchPerStep.toFixed(0)} 个/步。`),
    ]));
    return chainStep('②', '量化证据', '全部数值来自解析结果，可回溯到产物字段', blocks);
  }

  function metricTable(rows) {
    const table = h('table');
    table.append(h('thead', {}, [h('tr', {}, [
      h('th', {}, '分组'), h('th', {}, '指标'), h('th.num', {}, '数值'), h('th', {}, '说明'),
    ])]));
    table.append(h('tbody', {}, rows.map((row) => h('tr', {}, [
      h('td', {}, row.group),
      h('td', {}, row.metric),
      h('td.num', {}, row.value === undefined ? 'N/A' : `${formatMetric(row.value, row.unit)}`),
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

  function stepCause(analysis) {
    const items = analysis.steps.cause.items;
    if (items.length === 0) {
      return chainStep('③', '根因推断', '无', [
        h('p', {}, '未定位到达到门限的瓶颈，因此不输出根因推断，避免无据结论。请先按第 ④ 步的"补齐采集"建议完善 profiling 产物。'),
      ]);
    }
    const blocks = items.map((item) => h('div.scope-block', {}, [
      h('h4', {}, item.title),
      h('p.hint', {}, `适用瓶颈：${item.bottleneck} · 适用阶段：${PHASE_LABELS[item.phase] ?? item.phase} · 作用范围：${(item.scopes ?? [item.scope]).map((scope) => PHASE_LABELS[scope] ?? scope).join('、')}`),
      h('dl', {}, [
        h('dt', {}, '作用机理'), h('dd', {}, item.mechanism),
        h('dt', {}, 'vLLM-Ascend 实现原因'), h('dd', {}, item.vllmBehaviour),
      ]),
      item.triggers.length === 0 ? undefined : h('details', { open: true }, [
        h('summary', {}, `触发该推断的数据（${String(item.triggers.length)} 项）`),
        h('table', {}, [
          h('thead', {}, [h('tr', {}, [h('th', {}, '指标'), h('th.num', {}, '数值'), h('th', {}, '说明')])]),
          h('tbody', {}, item.triggers.map((trigger) => h('tr', {}, [
            h('td', {}, trigger.metric),
            h('td.num', {}, trigger.value === undefined ? 'N/A' : `${formatMetric(trigger.value, trigger.unit)}`),
            h('td.hint', {}, trigger.note),
          ]))),
        ]),
      ]),
      h('h4', {}, '现场确认方法'),
      h('ul', {}, item.checks.map((check) => h('li', {}, check))),
    ]));
    return chainStep('③', '根因推断', `${String(items.length)} 条假设`, blocks);
  }

  function stepActions(analysis, priorityFilter) {
    const all = analysis.steps.actions.items;
    const allowed = priorityFilter === undefined || priorityFilter === 'all'
      ? undefined
      : new Set(String(priorityFilter).split(','));
    const items = allowed === undefined ? all : all.filter((item) => allowed.has(item.priority));
    if (items.length === 0) {
      return chainStep('④', '可落地优化方案', '无匹配项', [
        h('p', {}, all.length === 0 ? '没有匹配到可执行的优化项：请先按"补齐采集"建议完善 profiling 产物。' : '当前优先级筛选下没有优化项。'),
      ]);
    }
    const counts = { 高: 0, 中: 0, 低: 0 };
    for (const item of items) counts[item.priority] = (counts[item.priority] ?? 0) + 1;
    const blocks = items.map((item, index) => h(`div.advice.${PRIORITY_CLASS[item.priority] ?? 'pri-mid'}`, { dataset: { priority: item.priority } }, [
      h('div.advice-head', {}, [
        h('span.pri', {}, item.priority),
        h('span.title', {}, `${String(index + 1)}. ${item.title}`),
        h('span.phase', {}, `适用阶段 ${(item.phases ?? [item.phase]).map((phase) => PHASE_LABELS[phase] ?? phase).join('/')} · 优先级得分 ${item.priorityScore.toFixed(1)}${item.confirmInVersion === true ? ' · 含需按版本确认的开关' : ''}`),
      ]),
      h('dl', {}, [
        h('dt', {}, '依据'), h('dd', {}, item.rationale),
        h('dt', {}, '执行动作'), h('dd', {}, h('ul', {}, item.actions.map((action) => h('li', {}, [h('span.tag', {}, action.type), ` ${action.text}`])))),
      ]),
      item.expectedGain === undefined ? undefined : h('div.gain', {}, [
        h('div', {}, [
          h('strong', {}, '预期收益 '),
          h('span.big', {}, `${item.expectedGain.estimatePct.toFixed(1)}%`),
          ` （区间 ${item.expectedGain.rangePct === undefined ? 'N/A' : `${item.expectedGain.rangePct[0].toFixed(1)}% – ${item.expectedGain.rangePct[1].toFixed(1)}%`}）`,
          h('span', { class: item.expectedGain.confidence === 'low' ? 'conf-low' : 'conf-high' }, ` · 置信度 ${confidenceLabel(item.expectedGain.confidence)}`),
        ]),
        h('div.hint', {}, `目标指标：${item.expectedGain.metric}；推算依据：${item.expectedGain.basis}`),
        h('div.hint', {}, `前提假设：${item.expectedGain.assumption}`),
      ]),
      h('dl', {}, [
        h('dt', {}, '验证方法'), h('dd', {}, item.verification),
        h('dt', {}, '风险'), h('dd', {}, item.risk),
      ]),
      item.linkedRootCauses.length === 0 ? undefined : h('p.hint', {}, `关联根因：${item.linkedRootCauses.join('、')}`),
    ]));
    return chainStep('④', '可落地优化方案', `高 ${String(counts.高)} / 中 ${String(counts.中)} / 低 ${String(counts.低)}`, [
      h('p.hint', {}, '排序依据：优先级权重 + 实测严重度 + 估算收益。所有涉及具体版本开关的建议都标注了"需版本确认"，请在目标环境核对后再启用。'),
      ...blocks,
    ]);
  }

  function stepBenefit(analysis) {
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
      h('td.num', {}, `${item.estimatePct.toFixed(1)}%`),
      h('td.num', {}, item.rangePct === undefined ? 'N/A' : `${item.rangePct[0].toFixed(1)}%–${item.rangePct[1].toFixed(1)}%`),
      h('td', { class: item.confidence === 'low' ? 'conf-low' : 'conf-high' }, confidenceLabel(item.confidence)),
    ]))));
    const body = [
      table,
      h('p', {}, [
        h('strong', {}, `保守合计 ${benefit.combined.conservativePct.toFixed(1)}%`),
        ` · 乐观合计 ${benefit.combined.optimisticPct.toFixed(1)}%（共 ${String(benefit.combined.itemCount)} 项可量化优化）`,
      ]),
      h('p.hint', {}, benefit.combined.note),
      h('p.hint', {}, benefit.note),
    ];
    const byPhase = Object.entries(benefit.combined.byPhase);
    if (byPhase.length > 0) {
      body.push(h('ul', {}, byPhase.map(([phase, value]) => h('li', {}, `${PHASE_LABELS[phase] ?? phase} 维度合计：${value.toFixed(1)}%`))));
    }
    void formatRatio;
    return chainStep('⑤', '预期收益', `${String(benefit.items.length)} 项可量化`, body);
  }

  function confidenceLabel(confidence) {
    return { high: '高（数据推导）', medium: '中（数据 + 假设）', low: '低（经验区间）' }[confidence] ?? '未知';
  }

  global.VAP = global.VAP ?? {};
  global.VAP.advice = { renderAdvice };
})(window);
