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
    /** Second swimlane for the optimized capture (created on first comparison). */
    ganttAfter: undefined,
    /** `{ beforeId, afterId, view, comparison }` once an optimized capture is paired. */
    compare: undefined,
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
    bindCompare();
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
        highlightShare();
      },
    });
    renderLegend();
    await Promise.all([loadHealth(), loadDocs(), refreshDatasetList()]);
    global.addEventListener('resize', VAP.debounce(() => {
      state.gantt.resize();
      state.ganttAfter?.resize();
    }, 120));
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
      'gantt-compare-grid', 'gantt-before-pane', 'gantt-before-head', 'gantt-before-note',
      'gantt-after-pane', 'gantt-after-note', 'gantt-canvas-after', 'gantt-tooltip-after',
      'gantt-cursor-after', 'gantt-hint-after', 'gantt-deltas', 'gantt-compare-summary',
      'module-share', 'share-dimension', 'share-scope', 'share-topn', 'share-bar-note',
      'share-bar-host', 'share-bar-legend', 'treemap-host', 'ranking-details', 'ranking-host',
      'share-compare-grid', 'share-before-pane', 'share-before-head', 'share-before-note',
      'share-after-pane', 'share-after-note', 'share-bar-note-after', 'share-bar-host-after',
      'share-bar-legend-after', 'treemap-host-after', 'ranking-host-after', 'share-deltas', 'share-compare-summary',
      'module-advice', 'advice-priority', 'advice-chain',
      'module-compare', 'compare-intake', 'compare-dropzone', 'compare-file-input',
      'compare-path-input', 'compare-path-button', 'compare-pair', 'compare-clear',
      'compare-progress-wrap', 'compare-progress-detail', 'compare-progress-percent', 'compare-progress-bar',
      'compare-error', 'compare-summary',
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

  /**
   * Upload a file set as one collection job, showing byte-level progress.
   *
   * @param {File[]} files - files to upload.
   * @param {'primary'|'compare'} [target] - which dataset slot the result fills.
   */
  async function uploadFiles(files, target = 'primary') {
    if (state.busy) return;
    setBusy(true);
    clearAlerts();
    if (target === 'compare') clearCompareError();
    else showProgress();
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    try {
      const label = datasetLabel(files);
      logProgress(`创建收集任务（${String(files.length)} 个文件，共 ${VAP.formatBytes(totalBytes)}）`);
      const collection = await VAP.api.postJson('/jobs', { collect: true, label });
      let uploaded = 0;
      for (const [index, file] of files.entries()) {
        const base = uploaded;
        await VAP.api.uploadFile(collection.id, file, undefined, (loaded) => {
          const done = base + loaded;
          reportProgress(target, (done / Math.max(1, totalBytes)) * 18, `上传 ${file.name}：${VAP.formatBytes(done)} / ${VAP.formatBytes(totalBytes)}`, 'inspect');
        });
        uploaded += file.size;
        logProgress(`已上传 ${String(index + 1)}/${String(files.length)}：${file.name}`);
      }
      reportProgress(target, 20, '上传完成，开始解析', 'inspect');
      const started = await VAP.api.postJson(`/jobs/${collection.id}/start`, {});
      await trackJob(started.id ?? collection.id, target);
    } catch (error) {
      if (target === 'compare') showCompareError(error.message);
      else showError(error.message);
      setBusy(false);
    }
  }

  /**
   * Label a dataset from the uploaded files.
   *
   * Dragging a folder gives every file a `webkitRelativePath`, so the folder name
   * is available and is by far the clearest label ("host-schedule-bound"); a file
   * picker with several files has no such context, and joining the names produced
   * an unreadable chip, so it becomes "N 个文件" instead.
   */
  function datasetLabel(files) {
    const relative = files.map((file) => file.webkitRelativePath ?? '').find((path) => path.includes('/'));
    if (relative !== undefined) return relative.split('/')[0].slice(0, 60);
    if (files.length === 1) return files[0].name;
    return `${String(files.length)} 个文件`;
  }

  /**
   * Start a server-side path analysis.
   *
   * @param {'primary'|'compare'} [target] - which dataset slot the result fills.
   */
  async function analyzePath(target = 'primary') {
    if (state.busy) {
      return;
    }
    const input = target === 'compare' ? el.comparePathInput : el.pathInput;
    const path = input.value.trim();
    if (path === '') {
      if (target === 'compare') showCompareError('请填写优化后产物的路径');
      else showError('请填写 profiling 文件或目录路径');
      return;
    }
    setBusy(true);
    clearAlerts();
    if (target === 'compare') clearCompareError();
    else showProgress();
    try {
      logProgress(`提交路径分析：${path}`);
      reportProgress(target, 6, '服务端读取并校验文件', 'inspect');
      const created = await VAP.api.postJson('/jobs', { path, label: path.split(/[\\/]/).pop() });
      await trackJob(created.id, target);
    } catch (error) {
      if (target === 'compare') {
        showCompareError(error.message);
      } else {
        showError(error.message);
      }
      setBusy(false);
    }
  }

  /** Route job progress to the primary or the compare panel. */
  function reportProgress(target, percent, detail, phase) {
    if (target !== 'compare') {
      setProgress(percent, detail, phase);
      return;
    }
    el.compareProgressWrap.hidden = false;
    el.compareProgressBar.style.width = `${String(Math.max(0, Math.min(100, percent)))}%`;
    el.compareProgressPercent.textContent = `${String(Math.round(percent))}%`;
    if (detail !== undefined && detail !== '') el.compareProgressDetail.textContent = detail;
  }

  /**
   * Poll a job, driving the staged progress bar, then load the dataset.
   *
   * @param {string} jobId - job id.
   * @param {'primary'|'compare'} [target] - which dataset slot the result fills.
   */
  async function trackJob(jobId, target = 'primary') {
    try {
      const job = await VAP.api.waitForJob(jobId, (tick) => {
        reportProgress(target, 20 + (tick.progress / 100) * 76, tick.detail ?? '', tick.phase);
        if (tick.detail !== undefined) logProgress(`${phaseLabel(tick.phase)} · ${tick.detail}`);
      });
      reportProgress(target, 100, '解析完成', 'analyze');
      logProgress(`解析完成：${String(job.summary?.events ?? 0)} 个事件，用时 ${String(((job.summary?.elapsedMs ?? 0) / 1000).toFixed(1))}s`);
      if ((job.warnings ?? []).length > 0 && target !== 'compare') showWarnings(job.warnings);
      await refreshDatasetList();
      if (job.datasetId !== undefined) {
        if (target === 'compare') await attachCompare(job.datasetId);
        else await loadDataset(job.datasetId);
      }
      if (target === 'compare') el.compareProgressWrap.hidden = true;
    } catch (error) {
      if (target === 'compare') {
        showCompareError(error.message);
        el.compareProgressWrap.hidden = true;
      } else {
        showError(error.message);
        if (error.job !== undefined && (error.job.warnings ?? []).length > 0) showWarnings(error.job.warnings);
      }
    } finally {
      setBusy(false);
    }
  }

  /** Load a dataset projection and render every step. */
  async function loadDataset(id) {
    try {
      const viewModel = await VAP.api.getDataset(id);
      // The comparison is tied to one baseline: switching away from it drops the
      // pairing rather than silently comparing against a different capture.
      if (state.compare !== undefined && state.compare.beforeId !== id) clearCompare({ silent: true });
      state.viewModel = viewModel;
      state.collapsed = new Set(id === state.viewModel.datasetId ? state.collapsed : []);
      clearFilters({ silent: true });
      for (const section of [el.overview, el.moduleGantt, el.moduleShare, el.moduleAdvice, el.moduleCompare]) section.hidden = false;
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

  // ── step 6: optimized capture + before/after comparison ──────────────────

  /**
   * Pair the current dataset with an optimized capture and render the comparison.
   *
   * The comparison payload is computed host-side (`/compare`), so the page only
   * has to place it: step 3 gets a second swimlane, step 4 a second composition
   * strip and treemap, and step 6 the headline, warnings and per-advice verdicts.
   */
  async function attachCompare(afterId) {
    const beforeId = state.viewModel?.datasetId;
    if (beforeId === undefined) {
      showCompareError('请先在第 1 步导入优化前的产物，再导入优化后的采集结果');
      return;
    }
    if (afterId === beforeId) {
      showCompareError('优化后的数据集不能与优化前相同：请导入另一次采集的产物');
      return;
    }
    setBusy(true);
    clearCompareError();
    try {
      const [afterView, payload] = await Promise.all([
        VAP.api.getDataset(afterId),
        VAP.api.compare(beforeId, afterId),
      ]);
      state.compare = { beforeId, afterId, view: afterView, comparison: payload.comparison };
      renderCompare();
    } catch (error) {
      showCompareError(error.message);
    } finally {
      setBusy(false);
    }
  }

  /** Render everything that depends on the comparison. */
  function renderCompare() {
    const compare = state.compare;
    if (compare === undefined) return;
    const { comparison, view } = compare;

    el.compareIntake.hidden = true;
    el.compareSummary.hidden = false;
    el.comparePair.textContent = `优化前 ${comparison.sides.before.label} → 优化后 ${comparison.sides.after.label}`;
    el.compareClear.hidden = false;
    el.compareSummary.replaceChildren(VAP.compareView.renderCompareSummary(comparison));

    // ── step 3: two swimlanes side by side ────────────────────────────────
    el.ganttCompareGrid.classList.add('split');
    el.ganttBeforeHead.hidden = false;
    el.ganttBeforeNote.textContent = `${comparison.sides.before.label} · ${formatCount(comparison.sides.before.eventCount)} 事件 · 每步 ${formatUs(comparison.headline.stepBeforeUs)}`;
    el.ganttAfterPane.hidden = false;
    el.ganttAfterNote.textContent = `${comparison.sides.after.label} · ${formatCount(comparison.sides.after.eventCount)} 事件 · 每步 ${formatUs(comparison.headline.stepAfterUs)}`;
    el.ganttDeltas.hidden = false;
    el.ganttDeltas.replaceChildren(VAP.compareView.renderDeltaStrip(comparison));
    if (state.ganttAfter === undefined) {
      state.ganttAfter = new VAP.GanttView({
        canvas: el.ganttCanvasAfter,
        tooltip: el.ganttTooltipAfter,
        cursorLabel: el.ganttCursorAfter,
        hintLabel: el.ganttHintAfter,
      });
    }
    state.ganttAfter.setData(view);
    state.ganttAfter.setOptions({ sortMode: el.ganttSort.value, rowLimit: Number(el.ganttLimit.value) });
    el.ganttCompareSummary.hidden = false;
    el.ganttCompareSummary.replaceChildren(h('p.compare-note', {}, `优化后：${comparison.headline.summary}`));

    // ── step 4: second strip + treemap, plus the delta tables ─────────────
    el.shareCompareGrid.classList.add('split');
    el.shareBeforeHead.hidden = false;
    el.shareBeforeNote.textContent = `${comparison.sides.before.label} · 每步 ${formatUs(comparison.headline.stepBeforeUs)}`;
    el.shareAfterPane.hidden = false;
    el.shareAfterNote.textContent = `${comparison.sides.after.label} · 每步 ${formatUs(comparison.headline.stepAfterUs)}`;
    renderShareAfter(view);
    el.shareDeltas.hidden = false;
    el.shareDeltas.replaceChildren(VAP.compareView.renderDeltaTable(comparison));
    el.shareCompareSummary.hidden = false;
    el.shareCompareSummary.replaceChildren(h('p.compare-note', {}, `优化前 ${(comparison.headline.stepBeforeUs / 1000).toFixed(2)}ms/步 → 优化后 ${(comparison.headline.stepAfterUs / 1000).toFixed(2)}ms/步（${comparison.headline.stepPct > 0 ? '+' : '−'}${Math.abs(comparison.headline.stepPct).toFixed(1)}%）；左右两侧使用同一统计口径与同一门限。`));

    for (const section of [el.moduleGantt, el.moduleShare, el.moduleCompare]) playEnter(section, 'enter');
    state.ganttAfter.resize();
  }

  /** Step 4's "after" strip and treemap (same controls, the other dataset). */
  function renderShareAfter(viewModel) {
    const dimension = currentDimension();
    const scope = el.shareScope.value;
    const tiles = Number(el.shareTopn.value);
    const categories = VAP.charts.buildCategories({ dataset: viewModel, scope });
    const strip = VAP.diagram.shareBar({ items: categories });
    el.shareBarHostAfter.replaceChildren(strip.element);
    el.shareBarLegendAfter.replaceChildren(...[...strip.legend.children]);
    el.shareBarNoteAfter.textContent = `${scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧' : '设备侧'} · 合计 ${formatUs(categories.reduce((sum, item) => sum + item.totalUs, 0))}`;

    const ranking = VAP.charts.buildRanking({ dataset: viewModel, dimension, scope, topN: tiles });
    el.treemapHostAfter.replaceChildren(VAP.diagram.treemap({ rows: ranking.rows, dimension }).element);
    el.rankingHostAfter.hidden = false;
    el.rankingHostAfter.replaceChildren(VAP.charts.renderRankingTable(ranking.rows, {}));
  }

  /** Drop the pairing and restore the single-capture view. */
  function clearCompare({ silent = false } = {}) {
    if (state.compare === undefined) {
      if (!silent) showCompareError('当前没有前后对比');
      return;
    }
    state.compare = undefined;
    el.compareIntake.hidden = false;
    el.compareSummary.hidden = true;
    el.compareSummary.replaceChildren();
    el.comparePair.textContent = '';
    el.compareClear.hidden = true;
    el.compareProgressWrap.hidden = true;
    el.ganttCompareGrid.classList.remove('split');
    el.ganttBeforeHead.hidden = true;
    el.ganttAfterPane.hidden = true;
    el.ganttDeltas.hidden = true;
    el.ganttDeltas.replaceChildren();
    el.ganttCompareSummary.hidden = true;
    el.ganttCompareSummary.replaceChildren();
    el.shareCompareGrid.classList.remove('split');
    el.shareBeforeHead.hidden = true;
    el.shareAfterPane.hidden = true;
    el.shareDeltas.hidden = true;
    el.shareDeltas.replaceChildren();
    el.shareCompareSummary.hidden = true;
    el.shareCompareSummary.replaceChildren();
    el.shareBarHostAfter.replaceChildren();
    el.shareBarLegendAfter.replaceChildren();
    el.treemapHostAfter.replaceChildren();
    el.rankingHostAfter.replaceChildren();
    el.rankingHostAfter.hidden = true;
    state.ganttAfter?.clear();
    clearCompareError();
  }

  function showCompareError(message) {
    el.compareError.hidden = false;
    el.compareError.replaceChildren(h('strong', {}, '对比失败：'), escapeHtml(message));
  }

  function clearCompareError() {
    el.compareError.hidden = true;
    el.compareError.replaceChildren();
  }

  function bindCompare() {
    const open = () => el.compareFileInput.click();
    el.compareDropzone.addEventListener('click', open);
    el.compareDropzone.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') open();
    });
    el.compareFileInput.addEventListener('change', () => {
      if (el.compareFileInput.files.length > 0) void uploadFiles([...el.compareFileInput.files], 'compare');
      el.compareFileInput.value = '';
    });
    for (const type of ['dragenter', 'dragover']) {
      el.compareDropzone.addEventListener(type, (event) => {
        event.preventDefault();
        el.compareDropzone.classList.add('dragover');
      });
    }
    for (const type of ['dragleave', 'drop']) {
      el.compareDropzone.addEventListener(type, (event) => {
        event.preventDefault();
        el.compareDropzone.classList.remove('dragover');
      });
    }
    el.compareDropzone.addEventListener('drop', (event) => {
      const files = [...(event.dataTransfer?.files ?? [])];
      if (files.length > 0) void uploadFiles(files, 'compare');
    });
    el.comparePathButton.addEventListener('click', () => void analyzePath('compare'));
    el.comparePathInput.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') void analyzePath('compare');
    });
    el.compareClear.addEventListener('click', () => clearCompare());
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

  /** Jump from an overview action into the matching card of step ④. */
  function focusAdvice(adviceId) {
    el.moduleAdvice.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'start' });
    const items = state.viewModel?.analysis?.steps.actions.items ?? [];
    const item = items.find((entry) => entry.id === adviceId);
    // The cards live in step ④: select it, and widen the priority filter when it
    // would hide the very item the user asked to see.
    if (item !== undefined && !priorityAllows(el.advicePriority.value, item.priority)) {
      el.advicePriority.value = 'all';
      renderAdvice();
    }
    state.adviceStep = 'actions';
    state.adviceView?.setStep('actions');
    const card = el.adviceChain.querySelector(`[data-advice-id="${adviceId}"]`);
    if (card === null) return;
    card.classList.remove('enter');
    void card.offsetWidth;
    card.classList.add('enter');
    card.scrollIntoView({ behavior: VAP.motionEnabled() ? 'smooth' : 'auto', block: 'center' });
  }

  /** Whether a priority-filter value keeps an item of this priority visible. */
  function priorityAllows(value, priority) {
    return value === undefined || value === 'all' || String(value).split(',').includes(priority);
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

  /** After the swimlane changes its category filter, keep the diagrams in sync. */
  function syncCategoryHighlight() {
    const visible = [...el.ganttLegend.querySelectorAll('[data-cat]')]
      .filter((chip) => chip.getAttribute('aria-pressed') === 'true')
      .map((chip) => chip.dataset.cat);
    state.filters.category = visible.length === 1 ? visible[0] : undefined;
    renderFilters();
    highlightShare();
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
        highlightShare();
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
      highlightShare();
    }
    el.ganttFilters.replaceChildren();
  }

  /**
   * Re-apply the shared filter to the diagrams already on screen.
   *
   * The treemap is *not* rebuilt here: rebuilding would replay its staggered
   * reveal on every click. Only the classes that express the selection change.
   */
  function highlightShare() {
    if (state.viewModel === undefined) return;
    VAP.charts.applyHighlight(el.shareBarHost.firstChild, { category: state.filters.category });
    VAP.charts.applyHighlight(el.treemapHost.firstChild, state.filters);
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
    const tiles = Number(el.shareTopn.value);

    // One composition strip for the category level, one treemap for the operator
    // level: together they carry the whole attribution, so the section needs no
    // explanatory prose at all.
    const categories = VAP.charts.buildCategories({ dataset: viewModel, scope });
    const strip = VAP.diagram.shareBar({
      items: categories,
      selected: state.filters.category,
      onSelect: ({ category }) => applyFilter({ category }),
    });
    el.shareBarHost.replaceChildren(strip.element);
    el.shareBarLegend.replaceChildren(...[...strip.legend.children]);
    VAP.charts.applyHighlight(strip.element, { category: state.filters.category });
    el.shareBarNote.textContent = `${scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧' : '设备侧'} · 合计 ${formatUs(categories.reduce((sum, item) => sum + item.totalUs, 0))}`;

    const ranking = VAP.charts.buildRanking({ dataset: viewModel, dimension, scope, topN: tiles });
    const map = VAP.diagram.treemap({
      rows: ranking.rows,
      dimension,
      selected: state.filters.operator,
      onSelect: ({ operator }) => applyFilter({ operator }),
    });
    el.treemapHost.replaceChildren(map.element);
    VAP.charts.applyHighlight(map.element, state.filters);
    el.rankingHost.replaceChildren(VAP.charts.renderRankingTable(ranking.rows, {
      selected: state.filters.operator,
      onSelect: ({ operator }) => applyFilter({ operator }),
    }));
  }

  function currentDimension() {
    const pressed = el.shareDimension.querySelector('button[aria-pressed="true"]');
    return pressed?.dataset.dimension ?? 'total';
  }

  // ── module 3 ────────────────────────────────────────────────────────────

  function renderAdvice() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    const previousStep = state.adviceView?.activeStep;
    const view = VAP.advice.renderAdvice(viewModel, {
      priorityFilter: el.advicePriority.value,
      activeStep: state.adviceStep ?? previousStep,
      onStep: (id) => { state.adviceStep = id; },
      onFocus: (filter) => applyFilter(filter),
    });
    state.adviceView = view;
    el.adviceChain.replaceChildren(view.element);
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
        renderShareAfterIfComparing();
      });
    }
    el.shareScope.addEventListener('change', () => {
      savePreferences({ scope: el.shareScope.value });
      renderShare();
      renderShareAfterIfComparing();
    });
    el.shareTopn.addEventListener('change', () => {
      savePreferences({ topN: Number(el.shareTopn.value) });
      renderShare();
      renderShareAfterIfComparing();
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

  /** Step 4's after-pane follows the same dimension/scope/topN controls. */
  function renderShareAfterIfComparing() {
    if (state.compare === undefined) return;
    renderShareAfter(state.compare.view);
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
      // The baseline moved, so the verdicts of the comparison must be recomputed
      // against the same optimized capture.
      if (state.compare !== undefined) await reloadComparison();
      playEnter(el.overview, 'enter');
    } catch (error) {
      showError(error.message);
    } finally {
      state.busy = false;
      el.phaseSelect.classList.remove('busy');
    }
  }

  /** Re-fetch the comparison payload for the current pair (after a re-analysis). */
  async function reloadComparison() {
    const compare = state.compare;
    if (compare === undefined) return;
    try {
      const payload = await VAP.api.compare(compare.beforeId, compare.afterId);
      compare.comparison = payload.comparison;
      renderCompare();
    } catch (error) {
      showCompareError(error.message);
    }
  }

  function download(format) {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    global.location.href = VAP.api.reportUrl(viewModel.datasetId, format);
  }

  /**
   * Export the PDF: capture the swimlane and both diagrams first, hand them to
   * the server so the printable report embeds them, then open the print view.
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
      const strip = el.shareBarHost.firstChild;
      if (strip !== null && strip !== undefined) {
        const dataUrl = await VAP.charts.svgToPngDataUrl(strip);
        if (typeof dataUrl === 'string') charts['模块二 · 算子大类耗时构成'] = dataUrl;
      }
      const tiles = el.treemapHost.firstChild;
      if (tiles !== null && tiles !== undefined) {
        const dataUrl = await VAP.charts.svgToPngDataUrl(tiles);
        if (typeof dataUrl === 'string') charts['模块二 · 算子耗时分布（面积=占比）'] = dataUrl;
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
    const sections = ['intake', 'overview', 'module-gantt', 'module-share', 'module-advice', 'module-compare'];
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
