/**
 * Page controller: intake (upload / path), progress, dataset selection, and the
 * three visualization modules.
 *
 * State is intentionally tiny and explicit — one active view model plus the
 * module-level display options — so what the page shows is always a pure
 * function of the server's projection plus the user's toggles.
 */
(function bootstrap(global) {
  'use strict';

  const VAP = global.VAP;
  const { h, formatUs, formatPct, formatCount, formatMs, PHASE_LABELS, BOTTLENECK_COLORS, escapeHtml } = VAP;

  const state = {
    /** @type {object|undefined} */ viewModel: undefined,
    gantt: undefined,
    /** @type {object|undefined} */ health: undefined,
    /** @type {object[]} */ datasets: [],
    busy: false,
    uploadBytes: 0,
  };

  const el = {};

  document.addEventListener('DOMContentLoaded', init);

  async function init() {
    cacheElements();
    bindIntake();
    bindControls();
    state.gantt = new VAP.GanttView({
      canvas: el.ganttCanvas,
      tooltip: el.ganttTooltip,
      cursorLabel: el.ganttCursor,
      selectionLabel: el.ganttSelection,
      onSelect: (name) => {
        el.ganttClear.hidden = name === undefined;
        renderShare();
      },
    });
    renderLegend();
    await Promise.all([loadHealth(), loadDocs(), refreshDatasetList()]);
    global.addEventListener('resize', VAP.debounce(() => {
      state.gantt.resize();
    }, 120));
  }

  function cacheElements() {
    const ids = [
      'dropzone', 'file-input', 'path-input', 'path-hint', 'btn-path', 'phase-prefill-hint', 'phase-decode-hint',
      'progress-wrap', 'progress-detail', 'progress-percent', 'progress-bar', 'progress-log',
      'error-box', 'warn-box', 'ok-box', 'dataset-list', 'dataset-count', 'dataset-list-wrap',
      'overview', 'kpis', 'bottleneck-banner', 'phase-select', 'btn-reanalyze', 'btn-report-md', 'btn-report-pdf',
      'module-gantt', 'gantt-canvas', 'gantt-tooltip', 'gantt-cursor', 'gantt-selection', 'gantt-clear',
      'gantt-legend', 'gantt-sort', 'gantt-limit', 'gantt-host', 'gantt-device', 'gantt-zoom-in', 'gantt-zoom-out', 'gantt-reset', 'gantt-hint',
      'module-share', 'share-dimension', 'share-scope', 'share-topn', 'pie-host', 'pie-legend', 'bar-host', 'bar-note', 'pie-note', 'ranking-host',
      'module-advice', 'advice-chain', 'advice-priority',
      'module-docs', 'docs-body', 'btn-docs', 'btn-about', 'modal', 'modal-title', 'modal-body', 'modal-close', 'health-line',
    ];
    for (const id of ids) el[camel(id)] = document.getElementById(id);
  }

  function camel(id) {
    return id.replace(/-([a-z])/g, (_, char) => char.toUpperCase());
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
  }

  /** Upload a file set as one collection job, showing byte-level progress. */
  async function uploadFiles(files) {
    if (state.busy) return;
    setBusy(true);
    clearAlerts();
    el.progressWrap.hidden = false;
    el.progressLog.replaceChildren();
    const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
    try {
      logProgress(`创建收集任务（${String(files.length)} 个文件，共 ${VAP.formatBytes(totalBytes)}）`);
      const collection = await VAP.api.postJson('/jobs', { collect: true, label: files.map((file) => file.name).join(' + ').slice(0, 120) });
      let uploaded = 0;
      for (const [index, file] of files.entries()) {
        const base = uploaded;
        logProgress(`上传 ${file.name}（${VAP.formatBytes(file.size)}）`);
        await VAP.api.uploadFile(collection.id, file, undefined, (loaded) => {
          const done = base + loaded;
          setProgress((done / Math.max(1, totalBytes)) * 20, `上传 ${file.name}：${VAP.formatBytes(done)} / ${VAP.formatBytes(totalBytes)}`);
        });
        uploaded += file.size;
        logProgress(`已上传 ${index + 1}/${String(files.length)}：${file.name}`);
      }
      setProgress(22, '上传完成，开始解析');
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
    el.progressWrap.hidden = false;
    el.progressLog.replaceChildren();
    try {
      logProgress(`提交路径分析：${path}`);
      setProgress(8, '服务端读取并校验文件');
      const created = await VAP.api.postJson('/jobs', { path, label: path.split(/[\\/]/).pop() });
      await trackJob(created.id);
    } catch (error) {
      showError(error.message);
      setBusy(false);
    }
  }

  /** Poll a job until it settles, then load the dataset. */
  async function trackJob(jobId) {
    try {
      const job = await VAP.api.waitForJob(jobId, (tick) => {
        const percent = 20 + (tick.progress / 100) * 78;
        setProgress(percent, `${phaseLabel(tick.phase)}：${tick.detail ?? ''}`);
        if (tick.detail !== undefined) logProgress(`${phaseLabel(tick.phase)} · ${tick.detail}`);
      });
      setProgress(100, '解析完成');
      logProgress(`解析完成：${String(job.summary?.events ?? 0)} 个事件，用时 ${String(((job.summary?.elapsedMs ?? 0) / 1000).toFixed(1))}s`);
      showOk(`解析完成，主导瓶颈：${job.summary?.bottleneck?.label ?? '未定位'}`);
      if ((job.warnings ?? []).length > 0) showWarnings(job.warnings);
      await refreshDatasetList();
      if (job.datasetId !== undefined) await loadDataset(job.datasetId);
    } catch (error) {
      const job = error.job;
      showError(error.message);
      if (job !== undefined && (job.warnings ?? []).length > 0) showWarnings(job.warnings);
    } finally {
      setBusy(false);
    }
  }

  /** Load a dataset projection and render all three modules. */
  async function loadDataset(id) {
    try {
      const viewModel = await VAP.api.getDataset(id);
      state.viewModel = viewModel;
      el.overview.hidden = false;
      el.moduleGantt.hidden = false;
      el.moduleShare.hidden = false;
      el.moduleAdvice.hidden = false;
      el.phaseSelect.value = viewModel.analysis.options.phaseOverride ?? 'auto';
      renderOverview();
      state.gantt.setData(viewModel);
      state.gantt.setOptions({ sortMode: el.ganttSort.value, rowLimit: Number(el.ganttLimit.value) });
      renderGanttHint();
      renderShare();
      renderAdvice();
      markActiveDataset(id);
      el.overview.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (error) {
      showError(error.message);
    }
  }

  async function refreshDatasetList() {
    try {
      const payload = await VAP.api.listDatasets();
      state.datasets = payload.datasets ?? [];
      el.datasetCount.textContent = String(state.datasets.length);
      el.datasetListWrap.hidden = state.datasets.length === 0;
      el.datasetList.replaceChildren(...state.datasets.map((entry) => h('li', { dataset: { id: entry.id } }, [
        h('button.small.primary', { onclick: () => void loadDataset(entry.id) }, '查看'),
        h('span', {}, entry.label ?? entry.id),
        h('span.meta', {}, `${formatCount(entry.eventCount)} 事件 · ${formatMs(entry.windowMs ?? 0)} · ${entry.bottleneck ?? '未定位'}`),
        h('button.small.ghost', {
          onclick: async () => {
            await VAP.api.deleteDataset(entry.id);
            if (state.viewModel?.datasetId === entry.id) resetView();
            await refreshDatasetList();
          },
        }, '删除'),
      ])));
      if (state.datasets.length > 0 && state.viewModel === undefined) {
        await loadDataset(state.datasets[0].id);
      }
    } catch (error) {
      el.datasetListWrap.hidden = true;
      void error;
    }
  }

  function markActiveDataset(id) {
    for (const item of el.datasetList.querySelectorAll('li')) {
      item.classList.toggle('active', item.dataset.id === id);
    }
  }

  function resetView() {
    state.viewModel = undefined;
    el.overview.hidden = true;
    el.moduleGantt.hidden = true;
    el.moduleShare.hidden = true;
    el.moduleAdvice.hidden = true;
  }

  // ── overview ────────────────────────────────────────────────────────────

  function renderOverview() {
    const viewModel = state.viewModel;
    const analysis = viewModel.analysis;
    const indicator = analysis.indicators;
    const kpis = [
      { k: '采集窗口', v: formatMs(viewModel.meta.wallUs / 1000), s: `${formatCount(viewModel.meta.eventCount)} 个事件` },
      { k: 'NPU 忙碌率', v: formatPct(indicator.deviceBusyPct), s: `空闲 ${formatPct(indicator.idlePct)}` },
      { k: 'Host 忙占比', v: formatPct(indicator.hostBusyPct), s: `独占 ${formatPct(indicator.hostOnlyPct)}` },
      { k: '通信占设备', v: formatPct(indicator.commPctOfDevice), s: `未掩盖 ${formatPct(indicator.commExposedPct)}` },
      { k: '设备侧拷贝', v: formatPct(indicator.copyPct), s: `D2H ${formatUs(indicator.d2hPerStepUs)}/步` },
      { k: '推理步', v: String(indicator.stepCount), s: `平均 ${formatUs(indicator.avgStepUs)}` },
      { k: '算子数', v: formatCount(viewModel.meta.counts.operators), s: `rank ${String(viewModel.meta.counts.ranks || 1)}` },
      { k: '保守收益', v: formatPct(analysis.steps.benefit.combined.conservativePct), s: `乐观 ${formatPct(analysis.steps.benefit.combined.optimisticPct)}` },
    ];
    el.kpis.replaceChildren(...kpis.map((item) => h('div.kpi', {}, [
      h('div.k', {}, item.k),
      h('div.v', {}, item.v),
      h('div.s', {}, item.s),
    ])));

    const bottleneck = analysis.bottleneck;
    el.bottleneckBanner.hidden = false;
    el.bottleneckBanner.replaceChildren(
      h('span.badge', { style: `background:${BOTTLENECK_COLORS[bottleneck.id] ?? '#666'}` }, bottleneck.short ?? bottleneck.label),
      h('div.text', {}, [
        h('div', {}, bottleneck.summary),
        h('div.hint', {}, `得分 ${bottleneck.score.toFixed(0)}/100 · 作用范围 ${bottleneck.scopeLabel ?? PHASE_LABELS[bottleneck.scope] ?? '全量窗口'} · 阶段划分来源：${viewModel.phases.sourceLabel}（置信度 ${viewModel.phases.confidence}）`),
      ]),
    );
  }

  // ── module 1 ────────────────────────────────────────────────────────────

  function renderLegend() {
    el.ganttLegend.replaceChildren(...Object.entries(VAP.CATEGORY_LABELS).map(([id, label]) => h('span', {}, [
      h('i', { style: `background:${VAP.CATEGORY_COLORS[id]}` }),
      label,
    ])));
  }

  function renderGanttHint() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    const timeline = viewModel.timeline;
    const parts = [
      `共 ${formatCount(viewModel.meta.eventCount)} 个事件；视图下发 ${formatCount(timeline.shippedEvents)} 个算子条（预算 ${formatCount(timeline.eventBudget)}）`,
      '滚轮缩放 / 拖拽平移 / 双击重置 / 悬停查看详情 / 点击算子条筛选',
    ];
    if (timeline.truncated) parts.push('部分算子行做了视图抽样（行内优先保留耗时最长的算子条），累计耗时统计不受影响');
    if (viewModel.meta.sampling?.applied === true) parts.push(`trace 解析阶段已采样：${viewModel.meta.sampling.strategy}（步长 ${String(viewModel.meta.sampling.stride)}）`);
    el.ganttHint.textContent = parts.join(' · ');
  }

  // ── module 2 ────────────────────────────────────────────────────────────

  function renderShare() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    const dimension = el.shareDimension.value;
    const scope = el.shareScope.value;
    const topN = Number(el.shareTopn.value);

    const categories = VAP.charts.buildCategories({ dataset: viewModel, scope });
    const pie = VAP.charts.renderPie({
      items: categories,
      totalUs: viewModel.categories.totalUs,
      scopeLabel: scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧' : '设备侧',
    });
    el.pieHost.replaceChildren(pie.element);
    el.pieLegend.replaceChildren(...pie.legend.childNodes);
    el.pieNote.textContent = `口径：${scope === 'all' ? '全部算子' : scope === 'host' ? 'Host 侧算子' : '设备侧算子'} · 总计 ${formatUs(categories.reduce((sum, item) => sum + item.totalUs, 0))} · 占比分母为所选口径下的算子总耗时`;

    const ranking = VAP.charts.buildRanking({ dataset: viewModel, dimension, scope, topN });
    const bars = VAP.charts.renderBars({ rows: ranking.rows, dimension, maxRows: topN });
    el.barHost.replaceChildren(bars.element);
    el.barNote.textContent = `维度：${dimension === 'average' ? '单次执行耗时' : '累计总耗时'} · ${ranking.scopeLabel} · ${bars.note}`;
    el.rankingHost.replaceChildren(VAP.charts.renderRankingTable(ranking.rows));
  }

  // ── module 3 ────────────────────────────────────────────────────────────

  function renderAdvice() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    el.adviceChain.replaceChildren(VAP.advice.renderAdvice(viewModel, { priorityFilter: el.advicePriority.value }));
  }

  // ── controls ────────────────────────────────────────────────────────────

  function bindControls() {
    el.ganttSort.addEventListener('change', () => state.gantt.setOptions({ sortMode: el.ganttSort.value }));
    el.ganttLimit.addEventListener('change', () => state.gantt.setOptions({ rowLimit: Number(el.ganttLimit.value) }));
    el.ganttHost.addEventListener('click', () => {
      state.gantt.toggleGroup('host');
      el.ganttHost.setAttribute('aria-pressed', el.ganttHost.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    });
    el.ganttDevice.addEventListener('click', () => {
      state.gantt.toggleGroup('device');
      el.ganttDevice.setAttribute('aria-pressed', el.ganttDevice.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
    });
    for (const button of el.ganttLegend.parentElement.querySelectorAll('[data-cat]')) {
      button.addEventListener('click', () => {
        state.gantt.toggleCategory(button.dataset.cat);
        button.setAttribute('aria-pressed', button.getAttribute('aria-pressed') === 'true' ? 'false' : 'true');
      });
    }
    el.ganttZoomIn.addEventListener('click', () => state.gantt.zoom(1 / 1.4));
    el.ganttZoomOut.addEventListener('click', () => state.gantt.zoom(1.4));
    el.ganttReset.addEventListener('click', () => state.gantt.resetView());
    el.ganttClear.addEventListener('click', () => state.gantt.filterBy(undefined));

    el.shareDimension.addEventListener('change', renderShare);
    el.shareScope.addEventListener('change', renderShare);
    el.shareTopn.addEventListener('change', renderShare);
    el.advicePriority.addEventListener('change', renderAdvice);

    el.btnReanalyze.addEventListener('click', () => void reanalyze());
    el.btnReportMd.addEventListener('click', () => download('report.md'));
    el.btnReportPdf.addEventListener('click', () => void exportPdf());
    el.btnDocs.addEventListener('click', () => {
      el.moduleDocs.hidden = false;
      el.moduleDocs.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    el.btnAbout.addEventListener('click', () => void showAbout());
    el.modalClose.addEventListener('click', () => {
      el.modal.hidden = true;
    });
    el.modal.addEventListener('click', (event) => {
      if (event.target === el.modal) el.modal.hidden = true;
    });
  }

  async function reanalyze() {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    el.btnReanalyze.disabled = true;
    el.btnReanalyze.textContent = '分析中…';
    try {
      const response = await VAP.api.analyze(viewModel.datasetId, { phaseOverride: el.phaseSelect.value });
      viewModel.analysis = response.analysis;
      renderOverview();
      renderAdvice();
      showOk(`已按“${phaseLabelOf(el.phaseSelect.value)}”口径重新分析`);
    } catch (error) {
      showError(error.message);
    } finally {
      el.btnReanalyze.disabled = false;
      el.btnReanalyze.textContent = '重新分析';
    }
  }

  function phaseLabelOf(phase) {
    return phase === 'auto' ? '自动推断' : phase === 'prefill' ? '仅 Prefill' : '仅 Decode';
  }

  function download(format) {
    const viewModel = state.viewModel;
    if (viewModel === undefined) return;
    global.location.href = VAP.api.reportUrl(viewModel.datasetId, format);
  }

  /**
   * Export the PDF: capture the swimlane and both charts first, hand them to the
   * server so the printable report embeds them, then open the print view.
   * A capture failure must never block the report — the charts are additional
   * evidence, not the report itself.
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
      el.btnReportPdf.textContent = '导出 PDF';
    }
    global.open(VAP.api.reportUrl(viewModel.datasetId, 'report.print'), '_blank', 'noopener');
  }

  // ── docs / health ───────────────────────────────────────────────────────

  async function loadDocs() {
    try {
      const bundle = await VAP.api.docs();
      el.docsBody.replaceChildren(VAP.docsView.renderDocs(bundle));
    } catch (error) {
      el.docsBody.replaceChildren(h('p', {}, `说明文档加载失败：${error.message}`));
    }
  }

  async function loadHealth() {
    try {
      state.health = await VAP.api.health();
      el.healthLine.textContent = `服务正常 · 数据集 ${String(state.health.store.datasets)}/${String(state.health.store.maxDatasets)} · 上传上限 ${VAP.formatBytes(Math.min(state.health.limits.maxUploadBytes, state.health.limits.maxInMemoryBytes))}`;
    } catch (error) {
      el.healthLine.textContent = `健康检查失败：${error.message}`;
    }
  }

  async function showAbout() {
    el.modalTitle.textContent = '关于 / 健康检查';
    if (state.health === undefined) await loadHealth();
    const config = state.health?.limits ?? {};
    el.modalBody.replaceChildren(
      VAP.docsView.renderHealth(state.health ?? {}),
      h('h4', {}, '当前限制'),
      h('ul', {}, Object.entries(config).map(([key, value]) => h('li', {}, `${key}: ${String(value)}`))),
      h('h4', {}, '安全与数据'),
      h('ul', {}, [
        h('li', {}, '解析全部在 DSH 主机进程内完成，不上传任何数据到外部服务。'),
        h('li', {}, '上传内容仅驻留内存，解析完成后即释放；数据集按空闲 TTL 自动过期。'),
        h('li', {}, '按路径分析默认限制在会话工作区内，可在插件配置中放开（allowOutsideWorkspacePaths）。'),
      ]),
    );
    el.modal.hidden = false;
  }

  // ── small helpers ───────────────────────────────────────────────────────

  function setBusy(busy) {
    state.busy = busy;
    el.btnPath.disabled = busy;
    el.dropzone.style.pointerEvents = busy ? 'none' : '';
    if (!busy) {
      el.progressWrap.hidden = false;
    }
  }

  function setProgress(percent, detail) {
    el.progressWrap.hidden = false;
    el.progressBar.style.width = `${String(Math.max(0, Math.min(100, percent)))}%`;
    el.progressPercent.textContent = `${String(Math.round(percent))}%`;
    if (detail !== undefined) el.progressDetail.textContent = detail;
  }

  function logProgress(message) {
    const item = h('li', {}, `${new Date().toLocaleTimeString('zh-CN')}  ${message}`);
    el.progressLog.append(item);
    el.progressLog.scrollTop = el.progressLog.scrollHeight;
    while (el.progressLog.childNodes.length > 60) el.progressLog.removeChild(el.progressLog.firstChild);
  }

  function clearAlerts() {
    for (const box of [el.errorBox, el.warnBox, el.okBox]) {
      box.hidden = true;
      box.replaceChildren();
    }
  }

  function showError(message) {
    el.errorBox.hidden = false;
    el.errorBox.replaceChildren(h('strong', {}, '解析失败：'), escapeHtml(message));
  }

  function showOk(message) {
    el.okBox.hidden = false;
    el.okBox.textContent = message;
  }

  function showWarnings(warnings) {
    el.warnBox.hidden = false;
    el.warnBox.replaceChildren(h('strong', {}, `解析告警（${String(warnings.length)} 条）`), h('ul', {}, warnings.map((warning) => h('li', {}, warning))));
  }

  function phaseLabel(phase) {
    return {
      queued: '排队', inspect: '校验', validate: '校验', parse: '解析', assemble: '汇总', analyze: '分析', done: '完成', failed: '失败',
    }[phase] ?? phase ?? '';
  }
})(window);
