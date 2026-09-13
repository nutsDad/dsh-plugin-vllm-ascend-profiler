/**
 * vLLM-Ascend Profiler Analyzer — host half.
 *
 * A dual-face DSH plugin row: this module is the node half, mounted through the
 * bundle patch in `cordis.patch.yml`. It claims a webserver route prefix and
 * serves
 *
 * * the standalone analyzer page and its assets (`web/`),
 * * a JSON API for upload → parse (with polled progress) → dataset retrieval,
 * * report export as Markdown or as a self-contained printable HTML document,
 * * the built-in metric/artifact documentation.
 *
 * Everything it does is bounded and in-memory: uploads are parsed to a model,
 * the bytes are dropped, datasets expire on an idle TTL, and no artifact is ever
 * written to disk. Large profiling trees are analyzed by path instead of by
 * upload, which is why `POST /api/jobs` also accepts a filesystem path inside
 * the session workspace.
 *
 * @module dsh-plugin-vllm-ascend-profiler
 */

import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve, sep } from 'node:path';

import { analyzeDataset } from './analysis/index.js';
import { documentationBundle } from './docs.js';
import { createThrottledLogger, formatBytes, readBody, readJsonBody, sendHtml, sendJson, sendText, serveStaticFile } from './http.js';
import { buildDataset } from './model/dataset.js';
import { parseProfileSet } from './parse/index.js';
import { buildViewModel } from './view.js';
import { renderMarkdownReport } from './report/markdown.js';
import { renderPrintableReport } from './report/print.js';
import { JobStore } from './store.js';

/** Stable Cordis plugin name. */
export const name = 'vllm-ascend-profiler';

/** Services required before the routes can be registered. */
export const inject = ['webServer'];

/** Plugin configuration defaults (overridable from the bundle patch row). */
export const DEFAULT_CONFIG = Object.freeze({
  routePrefix: '/vllm-ascend-profiler',
  maxUploadBytes: 2 * 1024 * 1024 * 1024,
  maxInMemoryBytes: 256 * 1024 * 1024,
  maxTimelineEvents: 400000,
  maxTableRows: 400000,
  viewEventBudget: 150000,
  maxDatasets: 6,
  datasetTtlMs: 60 * 60 * 1000,
  allowPathIngest: true,
  allowOutsideWorkspacePaths: false,
  peakTflops: 0,
  multiRankMode: 'rebase',
  topN: 20,
  thresholds: {},
  /**
   * Exact protobuf field numbers for `.proto`/`.bin` artifacts, e.g.
   * `{ name: 1, start: 2, duration: 3 }` or `{ name: 1, start: 2, end: 3 }`.
   * Without it, proto events are recognized heuristically and marked
   * low-confidence in the report.
   */
  protoFieldMap: undefined,
});

/**
 * Package version, read once from `package.json` so the health endpoint, the
 * injected shell global and the in-page docs can never disagree with the release.
 */
const VERSION = readPackageVersion();

function readPackageVersion() {
  try {
    const manifest = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
    return typeof manifest.version === 'string' ? manifest.version : '0.0.0';
  } catch {
    return '0.0.0';
  }
}

/**
 * Mount the plugin.
 *
 * @param {object} ctx - cordis context carrying `webServer`.
 * @param {object} [userConfig] - row configuration.
 */
export function apply(ctx, userConfig = {}) {
  const config = { ...DEFAULT_CONFIG, ...userConfig };
  const prefix = normalizePrefix(config.routePrefix);
  const assetRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'web');
  const store = new JobStore({ maxDatasets: config.maxDatasets, datasetTtlMs: config.datasetTtlMs });
  const log = createThrottledLogger(ctx.logger ?? console);

  if (!existsSync(join(assetRoot, 'index.html'))) {
    ctx.logger?.warn(`vllm-ascend-profiler: 前端资源缺失（${assetRoot}），页面将无法加载`);
  }

  const route = (path, handler) => {
    const full = path === '' ? prefix : `${prefix}${path}`;
    ctx.effect(() => ctx.webServer.register({ kind: 'prefix', path: full, handler }), `vllm-ascend-profiler: ${full}`);
  };

  /** Single dispatcher keeps route registration to one prefix entry. */
  route('', async (req, res) => {
    store.sweep();
    const url = new URL(req.url ?? '/', 'http://dsh.invalid');
    const path = url.pathname.slice(prefix.length) || '/';
    try {
      if (path === '/' || path === '/index.html') {
        await serveStaticFile(res, assetRoot, 'index.html', { fallback: 'index.html' });
        return;
      }
      if (path.startsWith('/api/')) {
        await handleApi({ req, res, url, path: path.slice('/api'.length), config, store, log, assetRoot });
        return;
      }
      const served = await serveStaticFile(res, assetRoot, path.replace(/^\/+/, ''));
      if (!served) {
        sendHtml(res, 404, notFoundPage(prefix));
        return;
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      log(`vllm-ascend-profiler: 处理 ${path} 失败：${message}`);
      if (!res.headersSent) sendJson(res, 500, { error: message });
      else res.end();
    }
  });

  ctx.logger?.info(`vllm-ascend-profiler: 分析页面位于 ${prefix}/（trace 解析、可视化与报告导出）`);

  // Structured index injection: publish the route prefix to the browser shell so
  // the client half can build the page URL without duplicating configuration.
  ctx.effect(() => ctx.on('webserver/index-inject', (rows) => {
    rows.push({
      kind: 'global',
      name: '__VLLM_ASCEND_PROFILER__',
      value: { routePrefix: prefix, title: 'vLLM-Ascend Profiler Analyzer', version: VERSION },
    });
  }), 'vllm-ascend-profiler: index injection');

  void VERSION;
}

/**
 * API dispatcher.
 * @param {object} input - request context.
 */
async function handleApi({ req, res, url, path, config, store, log, assetRoot }) {
  const method = req.method ?? 'GET';

  if (path === '/health' && method === 'GET') {
    sendJson(res, 200, {
      ok: true,
      plugin: name,
      version: VERSION,
      routePrefix: config.routePrefix,
      store: store.stats(),
      limits: {
        maxUploadBytes: config.maxUploadBytes,
        maxInMemoryBytes: config.maxInMemoryBytes,
        maxTimelineEvents: config.maxTimelineEvents,
        viewEventBudget: config.viewEventBudget,
        allowPathIngest: config.allowPathIngest,
      },
    });
    return;
  }

  if (path === '/docs' && method === 'GET') {
    sendJson(res, 200, documentationBundle({
      version: VERSION,
      routePrefix: config.routePrefix,
      config: publicConfig(config),
    }));
    return;
  }

  if (path === '/datasets' && method === 'GET') {
    sendJson(res, 200, { datasets: store.listDatasets() });
    return;
  }

  if (path === '/jobs' && method === 'POST') {
    await createJob({ req, res, url, config, store, log });
    return;
  }

  const jobMatch = /^\/jobs\/([0-9a-f-]{8,})(\/start)?$/i.exec(path);
  if (jobMatch !== null && jobMatch[2] === '/start' && method === 'POST') {
    const job = store.getJob(jobMatch[1]);
    if (job === undefined) {
      sendJson(res, 404, { error: '任务不存在或已过期' });
      return;
    }
    if (job.state !== 'collecting') {
      sendJson(res, 409, { error: `任务状态为 ${job.state}，无法重复启动解析` });
      return;
    }
    if (job.inputs.length === 0) {
      sendJson(res, 400, { error: '没有待解析的文件' });
      return;
    }
    void runJob({ job, store, config, log });
    sendJson(res, 202, { id: job.id, state: 'queued', files: job.inputs.length });
    return;
  }
  if (jobMatch !== null && jobMatch[2] === undefined && method === 'GET') {
    const job = store.getJob(jobMatch[1]);
    if (job === undefined) {
      sendJson(res, 404, { error: '任务不存在或已过期（任务仅保留 15 分钟）' });
      return;
    }
    sendJson(res, 200, {
      id: job.id,
      state: job.state,
      progress: job.progress,
      phase: job.phase,
      detail: job.detail,
      warnings: job.warnings,
      errors: job.errors,
      datasetId: job.datasetId,
      elapsedMs: Date.now() - job.createdAt,
      summary: job.summary,
    });
    return;
  }

  if (jobMatch !== null && jobMatch[2] === undefined && method === 'DELETE') {
    sendJson(res, 200, { ok: store.getJob(jobMatch[1]) !== undefined });
    return;
  }

  const datasetMatch = /^\/datasets\/([0-9a-f-]{8,})(\/.*)?$/i.exec(path);
  if (datasetMatch !== null) {
    const record = store.getDataset(datasetMatch[1]);
    if (record === undefined) {
      sendJson(res, 404, { error: '数据集不存在或已过期（默认空闲 1 小时后释放）' });
      return;
    }
    const sub = datasetMatch[2] ?? '';
    if (sub === '' && method === 'GET') {
      sendJson(res, 200, viewFor(record));
      return;
    }
    if (sub === '' && method === 'DELETE') {
      store.deleteDataset(datasetMatch[1]);
      sendJson(res, 200, { ok: true });
      return;
    }
    if (sub === '/report.md' && method === 'GET') {
      const markdown = renderMarkdownReport({
        dataset: record.dataset,
        analysis: record.analysis,
        charts: record.charts ?? {},
      });
      sendText(res, 200, markdown, {
        'content-type': 'text/markdown; charset=utf-8',
        'content-disposition': `attachment; filename="${reportFileName(record, 'md')}"`,
      });
      return;
    }
    if (sub === '/report.json' && method === 'GET') {
      sendJson(res, 200, { dataset: viewFor(record), analysis: record.analysis });
      return;
    }
    if (sub === '/report.print' && method === 'GET') {
      const autoPrint = url.searchParams.get('print') !== '0';
      const html = renderPrintableReport({
        dataset: record.dataset,
        analysis: record.analysis,
        charts: record.charts ?? {},
        autoPrint,
      });
      sendHtml(res, 200, html);
      return;
    }
    if (sub === '/charts' && method === 'POST') {
      // Chart snapshots are captured in the browser (canvas + SVG raster) and
      // stored so the printable report can embed them. They are optional and
      // bounded: a failed or oversized capture never blocks the report.
      const limit = 12 * 1024 * 1024;
      const body = await readJsonBody(req, limit);
      if (body.ok !== true) {
        sendJson(res, body.tooLarge === true ? 413 : 400, { error: body.error ?? '请求体无效' });
        return;
      }
      const value = /** @type {Record<string, unknown>} */ (body.value ?? {});
      const charts = value.charts;
      if (typeof charts !== 'object' || charts === null) {
        sendJson(res, 400, { error: '请求体需要 charts 字段' });
        return;
      }
      const accepted = {};
      let dropped = 0;
      for (const [name, dataUrl] of Object.entries(charts)) {
        if (typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/')) {
          dropped += 1;
          continue;
        }
        if (dataUrl.length > 6 * 1024 * 1024) {
          dropped += 1;
          continue;
        }
        accepted[name] = dataUrl;
      }
      record.charts = accepted;
      sendJson(res, 200, { ok: true, accepted: Object.keys(accepted).length, dropped });
      return;
    }
    if (sub === '/analyze' && method === 'POST') {
      const body = await readJsonBody(req);
      if (body.ok !== true) {
        sendJson(res, 400, { error: body.error ?? '请求体无效' });
        return;
      }
      const options = body.value ?? {};
      const reanalyzed = analyzeDataset(record.dataset, {
        phaseOverride: options.phaseOverride ?? 'auto',
        thresholds: { ...config.thresholds, ...(options.thresholds ?? {}) },
        triggerScore: options.triggerScore,
      });
      store.setAnalysis(datasetMatch[1], reanalyzed);
      sendJson(res, 200, { ok: true, analysis: reanalyzed });
      return;
    }
  }

  if (path === '/report.sample' && method === 'GET') {
    sendText(res, 200, '示例报告不可用：请先解析一份 profiling 产物。');
    return;
  }

  log(`vllm-ascend-profiler: 未匹配的 API 请求 ${method} ${path}`);
  sendJson(res, 404, { error: `未知的 API 路径：${method} ${path}` });
  void assetRoot;
}

/**
 * Project a stored record into the browser view model.
 * @param {object} record - stored dataset record.
 * @returns {object} view model.
 */
function viewFor(record) {
  return buildViewModel({
    dataset: record.dataset,
    analysis: record.analysis,
    datasetId: record.id,
    label: record.label,
    options: { eventBudget: record.viewEventBudget },
    charts: record.charts,
  });
}

/**
 * Create a parse job: either from an uploaded artifact set (raw body plus
 * metadata headers) or from a server-side path.
 */
async function createJob({ req, res, url, config, store, log }) {
  const contentType = String(req.headers['content-type'] ?? '');
  let inputs = [];
  let label = url.searchParams.get('name') ?? 'profiling';

  if (contentType.includes('application/json')) {
    const body = await readJsonBody(req, 1024 * 1024);
    if (body.ok !== true) {
      sendJson(res, 400, { error: body.error ?? '请求体无效' });
      return;
    }
    const value = /** @type {Record<string, unknown>} */ (body.value ?? {});
    if (value.collect === true) {
      // Collection mode: the page creates a job, uploads file by file, then
      // starts the parse. This keeps a multi-artifact upload as one dataset
      // without holding the whole set in a single request.
      const job = store.createJob({
        label: typeof value.label === 'string' && value.label !== '' ? value.label : 'profiling',
        inputs: [],
        state: 'collecting',
        detail: '等待上传文件',
      });
      sendJson(res, 202, { id: job.id, state: 'collecting' });
      return;
    }
    if (typeof value.path === 'string') {
      if (config.allowPathIngest !== true) {
        sendJson(res, 403, { error: '按路径分析已关闭：请在插件配置中开启 allowPathIngest' });
        return;
      }
      const guard = guardPath(value.path, config);
      if (guard.ok !== true) {
        sendJson(res, 403, { error: guard.error });
        return;
      }
      inputs = [{ name: guard.path.split(/[\\/]/).pop() ?? 'profile', path: guard.path }];
      label = typeof value.label === 'string' && value.label !== '' ? value.label : guard.path;
    } else if (Array.isArray(value.paths)) {
      for (const entry of value.paths.slice(0, 64)) {
        if (typeof entry !== 'string') continue;
        const guard = guardPath(entry, config);
        if (guard.ok !== true) {
          sendJson(res, 403, { error: guard.error });
          return;
        }
        inputs.push({ name: guard.path.split(/[\\/]/).pop() ?? 'profile', path: guard.path });
      }
      label = typeof value.label === 'string' && value.label !== '' ? value.label : `${String(inputs.length)} 个路径`;
    } else {
      sendJson(res, 400, { error: '请求体需要 path 或 paths 字段' });
      return;
    }
  } else {
    // Raw upload. An upload must be buffered in memory to be parsed, so the
    // ceiling here is `maxInMemoryBytes`; the larger `maxUploadBytes` ceiling
    // applies to path ingest, which streams from disk.
    const limit = Math.min(config.maxUploadBytes, config.maxInMemoryBytes);
    const declared = Number(req.headers['content-length'] ?? 0);
    if (Number.isFinite(declared) && declared > limit) {
      sendJson(res, 413, {
        error: `文件 ${formatBytes(declared)} 超过上传上限 ${formatBytes(limit)}。`
          + '超大 trace 请改为“按路径分析”：把产物放进会话工作区后填写路径，解析会以流式方式读取，不会整份驻留内存。',
      });
      return;
    }
    const group = url.searchParams.get('group');
    const collected = group === null ? undefined : store.getJob(group);
    const existingBytes = collected?.inputs?.reduce((sum, input) => sum + (input.buffer?.length ?? 0), 0) ?? 0;
    if (limit - existingBytes <= 0) {
      sendJson(res, 413, { error: `累计上传内容已达上限 ${formatBytes(limit)}，请先启动解析或改用按路径分析。` });
      return;
    }
    const body = await readBody(req, limit - existingBytes);
    if (body.tooLarge) {
      sendJson(res, 413, { error: `累计上传内容超过上限 ${formatBytes(limit)}` });
      return;
    }
    if (body.bytes === 0) {
      sendJson(res, 400, { error: '上传内容为空' });
      return;
    }
    if (collected !== undefined && collected.state === 'collecting') {
      collected.inputs.push({ name: label, buffer: body.buffer });
      collected.updatedAt = Date.now();
      log(`vllm-ascend-profiler: 收集文件 ${label}（${formatBytes(body.bytes)}），共 ${String(collected.inputs.length)} 个`);
      sendJson(res, 200, { id: collected.id, state: 'collecting', files: collected.inputs.length });
      return;
    }
    if (group !== null) {
      sendJson(res, 409, { error: '目标收集任务不存在或已开始解析，请重新发起上传' });
      return;
    }
    inputs = [{ name: label, buffer: body.buffer }];
  }

  const job = store.createJob({ label, inputs, state: 'queued' });
  log(`vllm-ascend-profiler: 开始解析 ${label}（${String(inputs.length)} 个输入）`);
  void runJob({ job, store, config, log });
  sendJson(res, 202, { id: job.id, state: job.state });
}

/** Parse, build the dataset, analyze, and record progress on the job. */
async function runJob({ job, store, config, log }) {
  const started = Date.now();
  store.updateJob(job.id, { state: 'parsing', phase: 'inspect', progress: 1, detail: '校验文件' });
  try {
    const parse = await parseProfileSet({
      inputs: job.inputs,
      options: {
        maxUploadBytes: config.maxUploadBytes,
        maxInMemoryBytes: config.maxInMemoryBytes,
        maxTimelineEvents: config.maxTimelineEvents,
        maxTableRows: config.maxTableRows,
        protoFieldMap: config.protoFieldMap,
      },
      onProgress: (progress) => {
        store.updateJob(job.id, {
          phase: progress.phase,
          progress: Math.max(1, Math.min(99, Math.round(progress.percent))),
          detail: progress.detail,
        });
      },
    });
    // The uploaded bytes are no longer needed once parsing returns.
    job.inputs = job.inputs.map((input) => ({ name: input.name, path: input.path, size: input.buffer?.length }));

    if (parse.ok !== true) {
      store.updateJob(job.id, {
        state: 'error',
        progress: 100,
        phase: 'validate',
        detail: '文件校验未通过',
        errors: parse.errors ?? ['解析失败'],
        warnings: parse.warnings ?? [],
      });
      log(`vllm-ascend-profiler: 校验失败 —— ${(parse.errors ?? []).join('；')}`);
      return;
    }

    store.updateJob(job.id, { phase: 'assemble', progress: 96, detail: '构建数据集' });
    const dataset = buildDataset(parse, {
      peakTflops: config.peakTflops,
      multiRankMode: config.multiRankMode,
      topN: config.topN,
    });
    store.updateJob(job.id, { phase: 'analyze', progress: 98, detail: '推理瓶颈与优化建议' });
    const analysis = analyzeDataset(dataset, { thresholds: config.thresholds });

    const record = store.putDataset({
      label: job.label,
      dataset,
      analysis,
      viewEventBudget: config.viewEventBudget,
    });
    store.updateJob(job.id, {
      state: 'done',
      progress: 100,
      phase: 'done',
      detail: '解析完成',
      datasetId: record.id,
      warnings: dataset.meta.warnings,
      summary: {
        events: dataset.meta.eventCount,
        windowMs: dataset.meta.wallUs / 1000,
        operators: dataset.meta.counts.operators,
        phases: dataset.phases.phases.map((phase) => ({ id: phase.id, steps: phase.stepCount })),
        bottleneck: {
          id: analysis.bottleneck.id,
          label: analysis.bottleneck.label,
          score: analysis.bottleneck.score,
          scope: analysis.bottleneck.scope,
        },
        elapsedMs: Date.now() - started,
      },
    });
    log(`vllm-ascend-profiler: ${job.label} 解析完成（${String(dataset.meta.eventCount)} 个事件，${String(Math.round((Date.now() - started) / 1000))}s），主导瓶颈：${analysis.bottleneck.label}`);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    store.updateJob(job.id, { state: 'error', progress: 100, phase: 'failed', detail: message, errors: [message] });
    log(`vllm-ascend-profiler: 解析异常 —— ${message}`);
  }
}

/**
 * Reject a path outside the session workspace unless explicitly allowed.
 * @param {string} target - requested path.
 * @param {object} config - plugin config.
 * @returns {{ok: true, path: string}|{ok: false, error: string}} guard result.
 */
function guardPath(target, config) {
  const resolved = resolve(target);
  if (!existsSync(resolved)) return { ok: false, error: `路径不存在：${resolved}` };
  if (config.allowOutsideWorkspacePaths === true) return { ok: true, path: resolved };
  const workspace = resolve(process.cwd());
  if (resolved !== workspace && !resolved.startsWith(workspace + sep)) {
    return {
      ok: false,
      error: `路径超出会话工作区（${workspace}）：请把 profiling 产物放到工作区内，或在插件配置中开启 allowOutsideWorkspacePaths`,
    };
  }
  return { ok: true, path: resolved };
}

/** Normalize the configured prefix into `/seg` form. */
function normalizePrefix(value) {
  let prefix = String(value ?? '/vllm-ascend-profiler').trim();
  if (prefix === '' || prefix === '/') return '/vllm-ascend-profiler';
  if (!prefix.startsWith('/')) prefix = `/${prefix}`;
  return prefix.replace(/\/+$/, '');
}

/** Config fields safe to expose to the browser. */
function publicConfig(config) {
  return {
    routePrefix: config.routePrefix,
    maxUploadBytes: config.maxUploadBytes,
    maxInMemoryBytes: config.maxInMemoryBytes,
    maxTimelineEvents: config.maxTimelineEvents,
    viewEventBudget: config.viewEventBudget,
    allowPathIngest: config.allowPathIngest,
    allowOutsideWorkspacePaths: config.allowOutsideWorkspacePaths,
    peakTflops: config.peakTflops,
    multiRankMode: config.multiRankMode,
    topN: config.topN,
    thresholds: config.thresholds,
  };
}

/** Report file name built from the dataset label. */
function reportFileName(record, extension) {
  const safe = String(record.label ?? 'profile').replace(/[^\w.\-]+/g, '_').slice(0, 60);
  const stamp = new Date(record.createdAt).toISOString().slice(0, 19).replaceAll(':', '');
  return `vllm-ascend-report-${safe}-${stamp}.${extension}`;
}

/** Minimal 404 page for an unknown asset path. */
function notFoundPage(prefix) {
  return `<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8"><title>404</title>
<style>body{font:14px/1.6 system-ui,"Microsoft YaHei",sans-serif;padding:40px;color:#222}code{background:#f2f4f8;padding:2px 6px;border-radius:4px}</style>
</head><body><h1>404</h1><p>未找到该资源。分析页面位于 <a href="${prefix}/"><code>${prefix}/</code></a>。</p></body></html>`;
}

export { JobStore, buildViewModel, renderMarkdownReport, renderPrintableReport };
