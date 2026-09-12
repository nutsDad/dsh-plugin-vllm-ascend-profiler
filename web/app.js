/**
 * Page controller.
 *
 * The page is one analysis instrument with five steps, not five independent
 * widgets, and this module is what makes that true. Two ideas carry the logic:
 *
 * 1. **One filter state.** `state.filters = { operator, category, group, phase }`
 *    is the single source of truth. The swimlane, the donut, the bar chart, the
 *    ranking table and the advice links all read and write it, so a click
 *    anywhere is visible everywhere (the bar chart greys out, the chip row
 *    updates, the advice card that mentions the same area stays highlighted).
 *
 * 2. **A visible pipeline.** The stepper tracks the section in view; each step
 *    states what it consumes and what it produces; the overview leads with the
 *    verdict and the three highest-priority actions, each linking into the step
 *    that justifies it.
 *
 * Motion is centralised: the topbar switch (persisted) and the OS
 * `prefers-reduced-motion` setting both disable every animation through
 * `VAP.motionEnabled()` and the `no-motion` body class.
 */
(function bootstrap(global) {
  'use strict';

  const VAP = global.VAP;
  const { h, formatUs, formatPct, formatCount, formatMs, PHASE_LABELS, BOTTLENECK_COLORS, escapeHtml, animateNumber, playEnter } = VAP;

  const state = {
    /** @type {object|undefined} */ viewModel: undefined,
    gantt: undefined,
    health: undefined,
    datasets: [],
    busy: false,
    playing: false,
    /** Shared cross-module filter. */
    filters: { operator: undefined, category: undefined, group: undefined },
    /** Collapsed advice sections, remembered per dataset. */
    collapsed: new Set(),
  };

  const el = {};
  const STORAGE_KEY = 'vap.preferences';

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    cacheElements();
    restorePreferences();
    bindIntake();
    bindControls();
    bindStepper();
    state.gantt = new VAP.GanttView({
      canvas: el.ganttCanvas,
      tooltip: el.ganttTooltip,
      cursorLabel: el.ganttCursor,
      hintLabel: el.ganttHint,
      onSelect: (selection) => {
        state.filters.operator = selection.operator;
        state.filters.category = selection.categories.length === Object.keys(VAP.CATEGORY_LABELS).length
          ? undefined
          : (selection.categories.length === 1 ? selection.categories[0] : undefined);
        renderFilters();
        VAP.charts.applyHighlight(el.pieHost.firstChild, state.filters);
        VAP.charts.applyHighlight(el.barHost.firstChild, state.filters);
      },
    });
    renderLegend();
    await Promise.all([loadHealth(), loadDocs(), refreshDatasetList()]);
    global.addEventListener('resize', VAP.debounce(() => state.gantt.resize(), 120));
    global.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        clearFilters();
        el.modal.hidden = true;
      }
    });
  }

  function cacheElements() {
    const ids = [
      'motion-toggle', 'btn-docs', 'btn-about', 'btn-formats', 'stepper',
      'intake', 'dropzone', 'file-input', 'path-input', 'btn-path', 'path-hint',
      'progress-wrap', 'progress-detail', 'progress-percent', 'progress-bar', 'progress-steps', 'progress-log', 'progress-log-toggle',
      'error-box', 'warn-box', 'dataset-row', 'dataset-list',
      'overview', 'verdict', 'verdict-badge', 'verdict-summary', 'verdict-meta', 'verdict-score',
      'kpis', 'conclusions', 'conclusions-note', 'phase-select', 'btn-report-md', 'btn-report-pdf',
      'module-gantt', 'gantt-legend', 'gantt-groups', 'gantt-filters', 'gantt-play', 'gantt-speed',
      'gantt-zoom-in', 'gantt-zoom-out', 'gantt-reset', 'gantt-sort', 'gantt-limit',
      'gantt-canvas', 'gantt-tooltip', 'gantt-cursor', 'gantt-hint',
      'module-share', 'share-dimension', 'share-scope', 'share-topn', 'share-hint',
      'pie-host', 'pie-legend', 'bar-host', 'bar-note', 'ranking-details', 'ranking-host',
      'module-advice', 'advice-priority', 'advice-chain',
      'modal', 'modal-title', 'modal-body', 'modal-close', 'health-line',
    ];
    for (const id of ids) el[camel(id)] = document.getElementById(id);
  }

  function camel(id) {
    return id.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
  }

  function restorePreferences() {
    let stored = {};
    try {
      stored = JSON.parse(global.localStorage?.getItem(STORAGE_KEY) ?? '{}');
    } catch {
      stored = {};
    }
    const motion = stored.motion !== false;
    el.motionToggle.checked = motion;
    document.body.classList.toggle('no-motion', !motion);
    if (stored.sortMode !== undefined) el.ganttSort.value = stored.sortMode;
    if (stored.rowLimit !== undefined) el.ganttLimit.value = String(stored.rowLimit);
    if (stored.topN !== undefined) el.shareTopn.value = String(stored.topN);
    if (stored.scope !== undefined) el.shareScope.value = stored.scope;
  }

  function savePreferences(patch) {
    let stored = {};
    try {
      stored = JSON.parse(global.localStorage?.getItem(STORAGE_KEY) ?? '{}');
    } catch {
      stored = {};
    }
    try {
      global.localStorage?.setItem(STORAGE_KEY, JSON.stringify({ ...stored, ...patch }));
    } catch {
      // Private mode: preferences simply do not persist.
    }
  }

  // ── intake ──────────────────────────────────────────────────────────────

  function bindIntake() {
    el.dropzone.addEventListener('click', () => el.fileInput.click());
    el.dropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') el.fileInput.click();
    });
    el.fileInput.addEventListener('change', () => {
      if (el.fileInput.files.length > 0) void uploadFiles([...el.fileInput.files]);
      el.fileInput.value = '';
    });
    for (const type of ['dragenter', 'dragover']) {
      el.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        el.dropzone.classList.add('dragover');
      });
    }
    for (const type of ['dragleave', 'drop']) {
      el.dropzone.addEventListener(type, (event) => {
        event.preventDefault();
        el.dropzone.classList.remove('dragover');
      });
    }
    el.dropzone.addEventListener('drop', (event) => {
      const files = [...(event.dataTransfer?.files ?? [])];
      if (files.length > 0) void uploadFiles(files);
    });
    el.btnPath.addEventListener('click', () => void analyzePath());
    el.pathInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') void analyzePath();
    });
    el.progressLogToggle.addEventListener('click', () => {
      el.progressLog.hidden = !el.progressLog.hidden;
      el.progressLogToggle.textContent = el.progressLog.hidden ? '解析日志' : '收起日志';
    });
  }

  /** Upload a file set as one collection job, showing byte-level progress. */
  async function uploadFiles(files) {
    if (state.busy) return;
    setBusy(true);
    clearAlerts();
    showProgress();
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    try {
      logProgress(`创建收集任务（${String(files.length)} 个文件，共 ${VAP.formatBytes(totalBytes)}）`);
      const collection = await VAP.api.postJson('/jobs', { collect: true, label: files.map((file) => file.name).join(' + ').slice(0, 120) });
      let uploaded = 0;
      for (const [index, file] of files.entries()) {
        const base = uploaded;
        await VAP.api.uploadFile(collection.id, file, undefined, (loaded) => {
          const done = base + loaded;
          setProgress((done / Math.max(1, totalBytes)) * 18, `上传 ${file.name}：${VAP.formatBytes(done)} / ${VAP.formatBytes(totalBytes)}`, 'inspect');
        });
        uploaded += file.size;
        logProgress(`已上传 ${String(index + 1)}/${String(files.length)}：${file.name}`);
      }
      setProgress(20, '上传完成，开始解析', 'inspect');
      const started = await VAP.api.postJson(`/jobs/${collection.id}/start`, {});
      await trackJob(started.id ?? collection.id);
    } catch (error) {
      showError(error.message);
      setBusy(false);
    }
  }

  /** Start a server-side path analysis. */
  async function analyzePath() {
    if (state.busy) return;
    const path = el.pathInput.value.trim();
    if (path === '') {
      showError('请填写 profiling 文件或目录路径');
      return;
    }
    setBusy(true);
    clearAlerts();
    showProgress();
    try {
      logProgress(`提交路径分析：${path}`);
      setProgress(6, '服务端读取并校验文件', 'inspect');
      const created = await VAP.api.postJson('/jobs', { path, label: path.split(/[\\/]/).pop() });
      await trackJob(created.id);
    } catch (error) {
      showError(error.message);
      setBusy(false);
    }
  }

  /** Poll a job, driving the staged progress bar, then load the dataset. */
  async function trackJob(jobId) {
    try {
      const job = await VAP.api.waitForJob(jobId, (tick) => {
        setProgress(20 + (tick.progress / 100) * 76, tick.detail ?? '', tick.phase);
        if (tick.detail !== undefined) logProgress(`${phaseLabel(tick.phase)} · ${tick.detail}`);
      });
      setProgress(100, '解析完成', 'analyze');
      logProgress(`解析完成：${String(job.summary?.events ?? 0)} 个事件，用时 ${String(((job.summary?.elapsedMs ?? 0) / 1000).toFixed(1))}s`);
      if ((job.warnings ?? []).length > 0) showWarnings(job.warnings);
      await refreshDatasetList();
      if (job.datasetId !== undefined) await loadDataset(job.datasetId);
    } catch (error) {
      showError(error.message);
      if (error.job !== undefined && (error.job.warnings ?? []).length > 0) showWarnings(error.job.warnings);
    } finally {
      setBusy(false);
    }
  }

  /** Load a dataset projection and render every step. */
  async function loadDataset(id) {
    try {
      const viewModel = await VAP.api.getDataset(id);
      state.viewModel = viewModel;
      state.collapsed = new Set(id === state.viewModel.datasetId ? state.collapsed : []);
      clearFilters({ silent: true });
      for (const section of [el.overview, el.moduleGantt, el.moduleShare, el.moduleAdvice]) section.hidden = false;
      playEnter(el.overview, 'enter');
      playEnter(el.moduleGantt, 'enter-2');
      playEnter(el.moduleShare, 'enter-2');
      playEnter(el.moduleAdvice, 'enter-3');
      setPhaseSelection(viewModel.analysis.options.phaseOverride ?? 'auto');
      renderOverview();
      state.gantt.setData(viewModel);
      state.gantt.setOptions({ sortMode: el.ganttSort.value, rowLimit: Number(el.ganttLimit.value) });
      renderShare();
      renderAdvice();
      markActiveDataset(id);
      el.overview.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'start' });
    } catch (error) {
      showError(error.message);
    }
  }

  async function refreshDatasetList() {
    try {
      const payload = await VAP.api.listDatasets();
      state.datasets = payload.datasets ?? [];
      el.datasetRow.hidden = state.datasets.length === 0;
      el.datasetList.replaceChildren(...state.datasets.map((entry) => h('button', {
        type: 'button',
        dataset: { id: entry.id },
        title: `${entry.label} · ${formatCount(entry.eventCount)} 事件 · ${entry.bottleneck ?? '未定位'}`,
        onclick: () => void loadDataset(entry.id),
      }, [
        h('span.dot'),
        h('span', {}, truncate(entry.label ?? entry.id, 28)),
        h('span.hint', {}, entry.bottleneck ?? ''),
      ])));
      if (state.datasets.length > 0 && state.viewModel === undefined) await loadDataset(state.datasets[0].id);
    } catch {
      el.datasetRow.hidden = true;
    }
  }

  function markActiveDataset(id) {
    for (const item of el.datasetList.children) item.classList.toggle('active', item.dataset.id === id);
  }

  // ── overview (verdict first, then actions) ──────────────────────────────

  function renderOverview() {
    const viewModel = state.viewModel;
    const analysis = viewModel.analysis;
    const indicator = analysis.indicators;
    const bottleneck = analysis.bottleneck;

    el.verdictBadge.style.background = BOTTLENECK_COLORS[bottleneck.id] ?? '#666';
    el.verdictBadge.textContent = bottleneck.short ?? bottleneck.label;
    el.verdictSummary.textContent = bottleneck.summary;
    el.verdictMeta.textContent = `${bottleneck.scopeLabel ?? PHASE_LABELS[bottleneck.scope] ?? '全量窗口'} · 阶段划分 ${viewModel.phases.sourceLabel}（置信度 ${viewModel.phases.confidence}）`;
    animateNumber({
      to: bottleneck.score,
      onFrame: (value) => { el.verdictScore.textContent = String(Math.round(value)); },
    });

    const kpis = [
      { k: '采集窗口', v: formatMs(viewModel.meta.wallUs / 1000), s: `${formatCount(viewModel.meta.eventCount)} 事件 · ${String(indicator.stepCount)} 步`, meter: undefined },
      { k: 'NPU 忙碌率', v: formatPct(indicator.deviceBusyPct), s: `空闲 ${formatPct(indicator.idlePct)}`, meter: indicator.deviceBusyPct },
      { k: 'Host 独占/步', v: formatUs(indicator.hostExclusivePerStepUs), s: `Host 忙 ${formatPct(indicator.hostBusyPct)} · 派发 ${indicator.dispatchPerStep.toFixed(0)} 个/步`, meter: indicator.hostOnlyPct },
      { k: '通信未掩盖', v: formatPct(indicator.commExposedPct), s: `通信占设备 ${formatPct(indicator.commPctOfDevice)} · D2H ${formatUs(indicator.d2hPerStepUs)}/步`, meter: indicator.commExposedPct },
    ];
    el.kpis.replaceChildren(...kpis.map((item) => h('div.kpi', {}, [
      h('div.k', {}, item.k),
      h('div.v', {}, item.v),
      h('div.s', {}, item.s),
      item.meter === undefined ? undefined : h('div.bar', {}, [h('i', { dataset: { meter: String(Math.min(100, item.meter)) } })]),
    ])));

    const actions = VAP.advice.topActions(analysis, 3);
    el.conclusionsNote.textContent = actions.length === 0 ? '（当前没有可执行项）' : '点击右侧按钮可跳到对应证据';
    el.conclusions.replaceChildren(...actions.map((action) => h('li', {}, [
      h(`span.pri.${priorityClass(action.priority)}`, {}, action.priority),
      h('div.text', {}, [
        h('div', {}, action.title),
        h('div.hint', {}, [
          action.phase.map((phase) => PHASE_LABELS[phase] ?? phase).join('/'),
          action.gainPct === undefined ? '' : ` · 预期收益 ${formatPct(action.gainPct)}`,
          action.confidence === 'low' ? '（经验区间）' : '',
        ].join('')),
      ]),
      h('button.link', {
        type: 'button',
        onclick: () => focusAdvice(action.id),
      }, '查看方案'),
    ])));

    animateMeters(el.overview);
  }

  /** The three highest-priority advice cards, as jump targets from the overview. */
  function focusAdvice(adviceId) {
    el.moduleAdvice.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'start' });
    const card = el.adviceChain.querySelector(`[data-advice-id="${adviceId}"]`);
    if (card === null) return;
    // Open the collapsed step that holds the card, then flash it.
    const step = card.closest('.chain-step');
    const body = step?.querySelector('.chain-body');
    if (body !== null && body !== undefined && body.hidden) {
      body.hidden = false;
      step.classList.remove('collapsed');
      step.querySelector('.chain-head')?.setAttribute('aria-expanded', 'true');
    }
    card.classList.remove('enter');
    void card.offsetWidth;
    card.classList.add('enter');
    card.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'center' });
  }

  // ── module 1 ────────────────────────────────────────────────────────────

  function renderLegend() {
    el.ganttLegend.replaceChildren(...Object.entries(VAP.CATEGORY_LABELS).map(([id, label]) => {
      const chip = h('button.chip', { type: 'button', dataset: { cat: id }, 'aria-pressed': 'true', title: `只看/隐藏${label}` }, [
        h('i', { style: `background:${VAP.CATEGORY_COLORS[id]}` }),
        label,
      ]);
      chip.addEventListener('click', () => {
        const pressed = chip.getAttribute('aria-pressed') === 'true';
        chip.setAttribute('aria-pressed', pressed ? 'false' : 'true');
        state.gantt.toggleCategory(id);
        syncCategoryHighlight();
      });
      return chip;
    }));
  }

  /** After the swimlane changes its category filter, keep the donut in sync. */
  function syncCategoryHighlight() {
    const visible = [...el.ganttLegend.querySelectorAll('[data-cat]')]
      .filter((chip) => chip.getAttribute('aria-pressed') === 'true')
      .map((chip) => chip.dataset.cat);
    state.filters.category = visible.length === 1 ? visible[0] : undefined;
    renderFilters();
    VAP.charts.applyHighlight(el.pieHost.firstChild, { category: state.filters.category });
    VAP.charts.applyHighlight(el.barHost.firstChild, state.filters);
  }

  function renderFilters() {
    const chips = [];
    if (state.filters.operator !== undefined) {
      chips.push(filterTag(`算子：${truncate(state.filters.operator, 26)}`, () => {
        state.gantt.filterBy(undefined);
      }));
    }
    if (state.filters.category !== undefined) {
      chips.push(filterTag(`类别：${VAP.CATEGORY_LABELS[state.filters.category] ?? state.filters.category}`, () => {
        for (const chip of el.ganttLegend.querySelectorAll('[data-cat]')) chip.setAttribute('aria-pressed', 'true');
        state.gantt.setCategories(Object.keys(VAP.CATEGORY_LABELS));
        state.filters.category = undefined;
        renderFilters();
        VAP.charts.applyHighlight(el.pieHost.firstChild, {});
        VAP.charts.applyHighlight(el.barHost.firstChild, state.filters);
      }));
    }
    if (chips.length > 1) {
      chips.push(h('button.small.ghost', { type: 'button', onclick: () => clearFilters() }, '清除全部'));
    }
    el.ganttFilters.replaceChildren(...chips);
  }

  function filterTag(label, onClear) {
    return h('span.tag', {}, [label, h('button', { type: 'button', title: '清除', onclick: onClear }, '×')]);
  }

  function clearFilters({ silent = false } = {}) {
    state.filters = { operator: undefined, category: undefined, group: undefined };
    if (!silent && state.gantt !== undefined) {
      for (const chip of el.ganttLegend.querySelectorAll('[data-cat]')) chip.setAttribute('aria-pressed', 'true');
      for (const chip of el.ganttGroups.querySelectorAll('[data-group]')) chip.setAttribute('aria-pressed', 'true');
      state.gantt.setCategories(Object.keys(VAP.CATEGORY_LABELS));
      state.gantt.filterBy(undefined);
      if (state.viewModel !== undefined) renderShare();
    }
    el.ganttFilters.replaceChildren();
  }

  /** Apply a filter coming from a chart or an advice link. */
  function applyFilter({ operator, category }) {
    if (operator !== undefined) {
      state.gantt.filterBy(operator);
    } else if (category !== undefined) {
      const others = Object.keys(VAP.CATEGORY_LABELS).filter((id) => id !== category);
      // A category focus means: show only this category's lanes.
      state.gantt.setCategories([category, ...others.filter((id) => id === 'other' && category !== 'other')]);
      for (const chip of el.ganttLegend.querySelectorAll('[data-cat]')) {
        chip.setAttribute('aria-pressed', chip.dataset.cat === category ? 'true' : 'false');
      }
      state.filters.category = category;
      renderFilters();
    }
    el.moduleGantt.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'start' });
  }

  // ── module 2 ────────────────────────────────────────────────────────────

  function renderShare() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    const dimension = currentDimension();
    const scope = el.shareScope.value;
    const topN = Number(el.shareTopn.value);

    const categories = VAP.charts.buildCategories({ dataset: viewModel, scope });
    const pie = VAP.charts.renderPie({
      items: categories,
      scopeLabel: scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧' : '设备侧',
      onSelect: ({ category }) => applyFilter({ category }),
    });
    el.pieHost.replaceChildren(pie.element);
    // `children` (not `childNodes`) so the transfer is portable across DOM
    // implementations and never depends on text-node bookkeeping.
    el.pieLegend.replaceChildren(...[...pie.legend.children]);
    VAP.charts.applyHighlight(pie.element, { category: state.filters.category });

    const ranking = VAP.charts.buildRanking({
      dataset: viewModel,
      dimension,
      scope,
      topN,
      category: el.shareScope.value === 'device' ? undefined : undefined,
    });
    const bars = VAP.charts.renderBars({
      rows: ranking.rows,
      dimension,
      maxRows: topN,
      selected: state.filters.operator,
      onSelect: ({ operator }) => applyFilter({ operator }),
    });
    el.barHost.replaceChildren(bars.element);
    el.barNote.textContent = `${dimension === 'average' ? '单次执行耗时' : '累计总耗时'} · ${ranking.scopeLabel} · ${bars.note}`;
    VAP.charts.applyHighlight(bars.element, state.filters);

    el.rankingHost.replaceChildren(VAP.charts.renderRankingTable(ranking.rows, {
      selected: state.filters.operator,
      onSelect: ({ operator }) => applyFilter({ operator }),
    }));
    el.shareHint.textContent = state.filters.operator === undefined
      ? '点击饼图或条形图即可回到第 3 步筛选对应算子。'
      : `已从第 3 步带入筛选：${truncate(state.filters.operator, 30)}（点击条形图可切换）`;
  }

  function currentDimension() {
    const pressed = el.shareDimension.querySelector('button[aria-pressed="true"]');
    return pressed?.dataset.dimension ?? 'total';
  }

  // ── module 3 ────────────────────────────────────────────────────────────

  function renderAdvice() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    el.adviceChain.replaceChildren(VAP.advice.renderAdvice(viewModel, {
      priorityFilter: el.advicePriority.value,
      collapsed: state.collapsed,
      onFocus: (filter) => applyFilter(filter),
    }));
    animateMeters(el.adviceChain);
  }

  /** Grow every meter from 0 to its target so progress reads as progress. */
  function animateMeters(root) {
    const meters = [...root.querySelectorAll('[data-meter]')];
    if (meters.length === 0) return;
    if (!VAP.motionEnabled()) {
      for (const meter of meters) meter.style.width = `${meter.dataset.meter}%`;
      return;
    }
    requestAnimationFrame(() => {
      for (const meter of meters) meter.style.width = `${meter.dataset.meter}%`;
    });
  }

  // ── controls ────────────────────────────────────────────────────────────

  function bindControls() {
    el.motionToggle.addEventListener('change', () => {
      const enabled = el.motionToggle.checked;
      document.body.classList.toggle('no-motion', !enabled);
      savePreferences({ motion: enabled });
      if (enabled && state.viewModel !== undefined) {
        playEnter(el.overview, 'enter');
        state.gantt.setData(state.viewModel);
        renderShare();
      }
    });

    el.ganttSort.addEventListener('change', () => {
      state.gantt.setOptions({ sortMode: el.ganttSort.value });
      savePreferences({ sortMode: el.ganttSort.value });
    });
    el.ganttLimit.addEventListener('change', () => {
      state.gantt.setOptions({ rowLimit: Number(el.ganttLimit.value) });
      savePreferences({ rowLimit: Number(el.ganttLimit.value) });
    });
    el.ganttPlay.addEventListener('click', () => {
      state.playing = !state.playing;
      el.ganttPlay.textContent = state.playing ? '⏸ 暂停' : '▶ 播放';
      el.ganttPlay.classList.toggle('primary', !state.playing);
      state.gantt.setPlaying(state.playing, Number(el.ganttSpeed.value));
    });
    el.ganttSpeed.addEventListener('change', () => {
      if (state.playing) state.gantt.setPlaying(true, Number(el.ganttSpeed.value));
    });
    el.ganttZoomIn.addEventListener('click', () => state.gantt.zoom(1 / 1.5));
    el.ganttZoomOut.addEventListener('click', () => state.gantt.zoom(1.5));
    el.ganttReset.addEventListener('click', () => state.gantt.resetView());
    for (const chip of el.ganttGroups.querySelectorAll('[data-group]')) {
      chip.addEventListener('click', () => {
        const pressed = chip.getAttribute('aria-pressed') === 'true';
        chip.setAttribute('aria-pressed', pressed ? 'false' : 'true');
        state.gantt.toggleGroup(chip.dataset.group);
      });
    }

    for (const button of el.shareDimension.querySelectorAll('button[data-dimension]')) {
      button.addEventListener('click', () => {
        for (const sibling of el.shareDimension.querySelectorAll('button[data-dimension]')) {
          sibling.setAttribute('aria-pressed', String(sibling === button));
        }
        renderShare();
      });
    }
    el.shareScope.addEventListener('change', () => {
      savePreferences({ scope: el.shareScope.value });
      renderShare();
    });
    el.shareTopn.addEventListener('change', () => {
      savePreferences({ topN: Number(el.shareTopn.value) });
      renderShare();
    });
    el.advicePriority.addEventListener('change', renderAdvice);

    for (const button of el.phaseSelect.querySelectorAll('button[data-phase]')) {
      button.addEventListener('click', () => void reanalyze(button.dataset.phase));
    }

    el.btnReportMd.addEventListener('click', () => download('report.md'));
    el.btnReportPdf.addEventListener('click', () => void exportPdf());
    el.btnDocs.addEventListener('click', () => void showDocs());
    el.btnFormats.addEventListener('click', () => void showFormats());
    el.btnAbout.addEventListener('click', () => void showAbout());
    el.modalClose.addEventListener('click', () => { el.modal.hidden = true; });
    el.modal.addEventListener('click', (event) => {
      if (event.target === el.modal) el.modal.hidden = true;
    });
  }

  function setPhaseSelection(phase) {
    for (const button of el.phaseSelect.querySelectorAll('button[data-phase]')) {
      button.setAttribute('aria-pressed', String(button.dataset.phase === phase));
    }
  }

  async function reanalyze(phase) {
    const viewModel = state.viewModel;
    if (viewModel === undefined || state.busy) return;
    setPhaseSelection(phase);
    state.busy = true;
    el.phaseSelect.classList.add('busy');
    try {
      const response = await VAP.api.analyze(viewModel.datasetId, { phaseOverride: phase });
      viewModel.analysis = response.analysis;
      renderOverview();
      renderAdvice();
      playEnter(el.overview, 'enter');
    } catch (error) {
      showError(error.message);
    } finally {
      state.busy = false;
      el.phaseSelect.classList.remove('busy');
    }
  }

  function download(format) {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    global.location.href = VAP.api.reportUrl(viewModel.datasetId, format);
  }

  /**
   * Export the PDF: capture the swimlane and both charts first, hand them to the
   * server so the printable report embeds them, then open the print view.
   * A capture failure must never block the report.
   */
  async function exportPdf() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    el.btnReportPdf.disabled = true;
    el.btnReportPdf.textContent = '准备图表…';
    try {
      const charts = {};
      const gantt = state.gantt?.toDataUrl();
      if (typeof gantt === 'string') charts['模块一 · Host/Device 算子执行泳道图（完整采集窗口）'] = gantt;
      const pie = el.pieHost.firstChild;
      if (pie !== null && pie !== undefined) {
        const dataUrl = await VAP.charts.svgToPngDataUrl(pie);
        if (typeof dataUrl === 'string') charts['模块二 · 算子大类耗时占比'] = dataUrl;
      }
      const bars = el.barHost.firstChild;
      if (bars !== null && bars !== undefined) {
        const dataUrl = await VAP.charts.svgToPngDataUrl(bars);
        if (typeof dataUrl === 'string') charts['模块二 · TopN 算子耗时排行'] = dataUrl;
      }
      await VAP.api.postCharts(viewModel.datasetId, charts);
    } catch (error) {
      logProgress(`图表快照上传失败（不影响报告内容）：${error.message}`);
    } finally {
      el.btnReportPdf.disabled = false;
      el.btnReportPdf.textContent = 'PDF';
    }
    global.open(VAP.api.reportUrl(viewModel.datasetId, 'report.print'), '_blank', 'noopener');
  }

  // ── stepper ─────────────────────────────────────────────────────────────

  function bindStepper() {
    for (const step of el.stepper.querySelectorAll('.step')) {
      step.addEventListener('click', () => {
        const target = document.getElementById(step.dataset.step);
        if (target === null) return;
        target.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'start' });
      });
    }
    const sections = ['intake', 'overview', 'module-gantt', 'module-share', 'module-advice'];
    if (typeof IntersectionObserver !== 'function') return;
    const observer = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const index = sections.indexOf(entry.target.id);
        if (index === -1) continue;
        let at = 0;
        for (const step of el.stepper.querySelectorAll('.step')) {
          step.classList.toggle('active', at === index);
          step.classList.toggle('done', at < index);
          at += 1;
        }
      }
    }, { rootMargin: '-45% 0px -50% 0px' });
    for (const id of sections) {
      const node = document.getElementById(id);
      if (node !== null) observer.observe(node);
    }
  }

  // ── docs / health ───────────────────────────────────────────────────────

  async function showDocs() {
    el.modalTitle.textContent = '说明 · 指标口径与产物字段';
    if (el.modalBody.dataset.kind !== 'docs') {
      el.modalBody.replaceChildren(h('p', {}, '加载中…'));
      await loadDocs(true);
    }
    el.modal.hidden = false;
  }

  async function showFormats() {
    el.modalTitle.textContent = '支持的文件与字段';
    await loadDocs();
    el.modal.hidden = false;
    const anchor = el.modalBody.querySelector('.doc-item');
    anchor?.scrollIntoView({ block: 'start' });
  }

  async function loadDocs(intoModal = false) {
    try {
      const bundle = await VAP.api.docs();
      el.modalBody.replaceChildren(VAP.docsView.renderDocs(bundle));
      el.modalBody.dataset.kind = 'docs';
    } catch (error) {
      if (intoModal) el.modalBody.replaceChildren(h('p', {}, `说明文档加载失败：${error.message}`));
    }
  }

  async function loadHealth() {
    try {
      state.health = await VAP.api.health();
      el.healthLine.textContent = `服务正常 · 数据集 ${String(state.health.store.datasets)}/${String(state.health.store.maxDatasets)}`;
    } catch (error) {
      el.healthLine.textContent = `健康检查失败：${error.message}`;
    }
  }

  async function showAbout() {
    el.modalTitle.textContent = '状态与限制';
    if (state.health === undefined) await loadHealth();
    el.modalBody.dataset.kind = 'about';
    el.modalBody.replaceChildren(
      VAP.docsView.renderHealth(state.health ?? {}),
      h('h4', {}, '当前限制'),
      h('ul', {}, Object.entries(state.health?.limits ?? {}).map(([key, value]) => h('li', {}, `${key}: ${String(value)}`))),
      h('h4', {}, '数据与安全'),
      h('ul', {}, [
        h('li', {}, '解析全部在 DSH 主机进程内完成，不向任何外部服务上传数据。'),
        h('li', {}, '上传内容仅驻留内存，解析完成后释放；数据集按空闲 TTL 自动过期。'),
        h('li', {}, '按路径分析默认限制在会话工作区内。'),
      ]),
    );
    el.modal.hidden = false;
  }

  // ── small helpers ───────────────────────────────────────────────────────

  function setBusy(busy) {
    state.busy = busy;
    el.btnPath.disabled = busy;
    el.dropzone.style.pointerEvents = busy ? 'none' : '';
  }

  function showProgress() {
    el.progressWrap.hidden = false;
    el.progressWrap.classList.add('busy');
    el.progressLog.hidden = true;
    el.progressLog.replaceChildren();
    el.progressLogToggle.textContent = '解析日志';
  }

  function setProgress(percent, detail, phase) {
    el.progressBar.style.width = `${String(Math.max(0, Math.min(100, percent)))}%`;
    el.progressPercent.textContent = `${String(Math.round(percent))}%`;
    if (detail !== undefined && detail !== '') el.progressDetail.textContent = detail;
    if (percent >= 100) el.progressWrap.classList.remove('busy');
    if (phase !== undefined) {
      const order = ['inspect', 'parse', 'assemble', 'analyze'];
      const active = order.indexOf(phase);
      for (const item of el.progressSteps.querySelectorAll('li')) {
        const at = order.indexOf(item.dataset.phase);
        item.classList.toggle('active', at === active);
        item.classList.toggle('done', at < active || percent >= 100);
      }
    }
  }

  function logProgress(message) {
    const item = h('li', {}, `${new Date().toLocaleTimeString('zh-CN')}  ${message}`);
    el.progressLog.append(item);
    el.progressLog.scrollTop = el.progressLog.scrollHeight;
    while (el.progressLog.childNodes.length > 80) el.progressLog.removeChild(el.progressLog.firstChild);
  }

  function clearAlerts() {
    for (const box of [el.errorBox, el.warnBox]) {
      box.hidden = true;
      box.replaceChildren();
    }
  }

  function showError(message) {
    el.errorBox.hidden = false;
    el.errorBox.replaceChildren(h('strong', {}, '失败：'), escapeHtml(message));
  }

  function showWarnings(warnings) {
    el.warnBox.hidden = false;
    el.warnBox.replaceChildren(
      h('strong', {}, `解析告警（${String(warnings.length)} 条）`),
      h('ul', {}, warnings.map((warning) => h('li', {}, warning))),
    );
  }

  function priorityClass(priority) {
    return { 高: 'pri-high', 中: 'pri-mid', 低: 'pri-low' }[priority] ?? 'pri-mid';
  }

  function truncate(text, limit) {
    const value = String(text ?? '');
    return value.length <= limit ? value : `${value.slice(0, limit - 1)}…`;
  }

  function phaseLabel(phase) {
    return {
      queued: '排队', inspect: '校验', validate: '校验', parse: '解析', assemble: '汇总', analyze: '分析', done: '完成', failed: '失败',
    }[phase] ?? phase ?? '';
  }
})(window);
