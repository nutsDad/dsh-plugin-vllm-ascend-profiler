import test from 'node:test';
import assert from 'node:assert/strict';
import { Readable, Writable } from 'node:stream';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { apply, DEFAULT_CONFIG } from '../lib/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixtureDir = join(here, 'fixtures', 'decode-comm-bound');

/**
 * Minimal ServerResponse recorder.
 *
 * Extends `Writable` so `stream.pipe(res)` — how the static file server sends
 * assets — writes into the same buffer as `res.end(body)`.
 */
class MockResponse extends Writable {
  constructor() {
    super();
    this.status = 0;
    this.headers = {};
    this.chunks = [];
    this.headersSent = false;
  }

  _write(chunk, _encoding, callback) {
    this.chunks.push(Buffer.from(chunk));
    callback();
  }

  writeHead(status, headers = {}) {
    this.status = status;
    this.headers = { ...this.headers, ...headers };
    this.headersSent = true;
  }

  get body() {
    return Buffer.concat(this.chunks).toString('utf8');
  }

  json() {
    return JSON.parse(this.body);
  }
}

/** Build the plugin's route handler over a mock cordis context. */
function mountPlugin(config = {}) {
  const routes = [];
  const context = {
    logger: { info() {}, warn() {}, error() {} },
    effect(fn) {
      const disposer = fn();
      return typeof disposer === 'function' ? disposer : () => {};
    },
    on() {
      return () => {};
    },
    webServer: {
      register(route) {
        routes.push(route);
        return () => {};
      },
    },
  };
  apply(context, { ...DEFAULT_CONFIG, ...config });
  assert.equal(routes.length, 1, 'the plugin registers exactly one prefix route');
  return routes[0];
}

/** Drive one request through the handler. */
async function request(route, { method = 'GET', path = '/', body, headers = {} } = {}) {
  const payload = body === undefined ? [] : [Buffer.isBuffer(body) ? body : Buffer.from(body)];
  const req = Readable.from(payload);
  req.method = method;
  req.url = path;
  req.headers = { host: '127.0.0.1:3080', ...headers };
  if (payload.length > 0) req.headers['content-length'] = String(payload[0].length);
  const res = new MockResponse();
  await route.handler(req, res);
  return res;
}

/** Poll a job to completion. */
async function waitForJob(route, id, timeoutMs = 30000) {
  const started = Date.now();
  let last;
  while (Date.now() - started < timeoutMs) {
    const res = await request(route, { path: `/vllm-ascend-profiler/api/jobs/${id}` });
    last = res.json();
    if (last.state === 'done' || last.state === 'error') return last;
    await new Promise((resolve) => setTimeout(resolve, 25));
  }
  throw new Error(`job did not settle: ${JSON.stringify(last)}`);
}

test('health endpoint reports limits and store state', async () => {
  const route = mountPlugin();
  const res = await request(route, { path: '/vllm-ascend-profiler/api/health' });
  assert.equal(res.status, 200);
  const payload = res.json();
  assert.equal(payload.ok, true);
  assert.equal(payload.routePrefix, '/vllm-ascend-profiler');
  assert.equal(payload.limits.allowPathIngest, true);
});

test('serves the analyzer page and its assets', async () => {
  const route = mountPlugin();
  const page = await request(route, { path: '/vllm-ascend-profiler/' });
  assert.equal(page.status, 200);
  assert.match(page.headers['content-type'], /text\/html/);
  assert.ok(page.body.includes('vLLM-Ascend Profiler Analyzer'));
  assert.ok(page.body.includes('模块一 · Host / Device 算子执行泳道图'));
  assert.ok(page.body.includes('模块二 · 算子耗时占比'));
  assert.ok(page.body.includes('模块三 · 结构化性能优化建议'));

  for (const asset of ['styles.css', 'app.js', 'util.js', 'api.js', 'charts.js', 'gantt.js', 'advice-view.js', 'docs-view.js']) {
    const res = await request(route, { path: `/vllm-ascend-profiler/${asset}` });
    assert.equal(res.status, 200, `${asset} must be served`);
    assert.ok(res.body.length > 200, `${asset} must not be empty`);
  }
});

test('never leaks files outside the asset root', async () => {
  const route = mountPlugin();
  // `..` segments are normalized away by URL parsing; the request must still not
  // reach package.json (it either renders the page or 404s).
  const normalized = await request(route, { path: '/vllm-ascend-profiler/../../package.json' });
  assert.ok(!normalized.body.includes('"dsh-plugin-vllm-ascend-profiler"'), 'must not leak package.json');

  // A percent-encoded traversal is normalized by URL parsing too, so it also
  // cannot reach the file; what matters is that nothing outside the root leaks.
  const encoded = await request(route, { path: '/vllm-ascend-profiler/%2e%2e/package.json' });
  assert.ok([200, 403, 404].includes(encoded.status), `unexpected status ${String(encoded.status)}`);
  assert.ok(!encoded.body.includes('"dsh-plugin-vllm-ascend-profiler"'), 'must not leak package.json');
});

test('rejects a path outside the workspace with an actionable error', async () => {
  const route = mountPlugin();
  // Must be a path that exists but lies outside the session workspace; the
  // platform-specific candidate keeps this test meaningful on CI too.
  const outside = process.platform === 'win32'
    ? 'C:\\Windows\\System32\\drivers\\etc\\hosts'
    : '/etc/hosts';
  const res = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: outside }),
  });
  assert.equal(res.status, 403);
  assert.match(res.json().error, /工作区|路径不存在/);
});

test('ingests a profiling directory by path and serves the dataset', async () => {
  const route = mountPlugin();
  const create = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: fixtureDir, label: 'decode-comm-bound' }),
  });
  assert.equal(create.status, 202);
  const job = await waitForJob(route, create.json().id);
  assert.equal(job.state, 'done', JSON.stringify(job.errors));
  assert.ok(job.summary.events > 500);
  assert.equal(job.summary.bottleneck.id, 'comm');

  const dataset = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}` });
  assert.equal(dataset.status, 200);
  const view = dataset.json();
  assert.ok(view.timeline.lanes.length === 2);
  assert.ok(view.operators.length > 0);
  assert.ok(view.analysis.steps.actions.items.length > 0);
  assert.ok(view.timeline.shippedEvents > 0);
  assert.ok(view.timeline.shippedEvents <= view.timeline.eventBudget);
});

test('uploads a collection file by file, then parses it', async () => {
  const route = mountPlugin();
  const collection = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ collect: true, label: 'uploaded' }),
  });
  assert.equal(collection.status, 202);
  const groupId = collection.json().id;

  const files = readdirSync(fixtureDir).slice(0, 3);
  for (const name of files) {
    const buffer = readFileSync(join(fixtureDir, name));
    const res = await request(route, {
      method: 'POST',
      path: `/vllm-ascend-profiler/api/jobs?group=${groupId}&name=${encodeURIComponent(name)}`,
      headers: { 'content-type': 'application/octet-stream' },
      body: buffer,
    });
    assert.equal(res.status, 200, `${name}: ${res.body.slice(0, 200)}`);
    assert.equal(res.json().state, 'collecting');
  }

  const started = await request(route, { method: 'POST', path: `/vllm-ascend-profiler/api/jobs/${groupId}/start` });
  assert.equal(started.status, 202);
  const job = await waitForJob(route, groupId);
  assert.equal(job.state, 'done', JSON.stringify(job.errors));
  assert.equal(job.datasetId !== undefined, true);
});

test('a rejected upload produces a precise error, not an empty report', async () => {
  const route = mountPlugin();
  const collection = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ collect: true, label: 'bad' }),
  });
  const groupId = collection.json().id;
  await request(route, {
    method: 'POST',
    path: `/vllm-ascend-profiler/api/jobs?group=${groupId}&name=notes.txt`,
    headers: { 'content-type': 'application/octet-stream' },
    body: 'this is not a profiling artifact, just prose about one.\n',
  });
  await request(route, { method: 'POST', path: `/vllm-ascend-profiler/api/jobs/${groupId}/start` });
  const job = await waitForJob(route, groupId);
  assert.equal(job.state, 'error');
  assert.ok(job.errors.length > 0);
  assert.match(job.errors.join(' '), /vLLM-Ascend|未识别|不受支持|没有可解析/);
});

test('exports markdown and a printable HTML report', async () => {
  const route = mountPlugin();
  const create = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: fixtureDir }),
  });
  const job = await waitForJob(route, create.json().id);
  assert.equal(job.state, 'done');

  const markdown = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/report.md` });
  assert.equal(markdown.status, 200);
  assert.match(markdown.headers['content-type'], /text\/markdown/);
  assert.match(markdown.headers['content-disposition'], /attachment; filename="vllm-ascend-report-/);
  assert.ok(markdown.body.includes('## ① 瓶颈类型定位'));

  const printable = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/report.print` });
  assert.equal(printable.status, 200);
  assert.match(printable.headers['content-type'], /text\/html/);
  assert.ok(printable.body.startsWith('<!DOCTYPE html>'));
  assert.ok(printable.body.includes('预期收益'));
});

test('re-analysis honours a phase override', async () => {
  const route = mountPlugin();
  const create = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: fixtureDir }),
  });
  const job = await waitForJob(route, create.json().id);
  const reanalyzed = await request(route, {
    method: 'POST',
    path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/analyze`,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ phaseOverride: 'prefill' }),
  });
  assert.equal(reanalyzed.status, 200);
  const analysis = reanalyzed.json().analysis;
  assert.equal(analysis.options.phaseOverride, 'prefill');
  assert.equal(analysis.steps.locate.perScope.length, 1);

  const dataset = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}` });
  assert.equal(dataset.json().analysis.options.phaseOverride, 'prefill', 'the stored dataset must reflect the new analysis');
});

test('chart snapshots are accepted and embedded in the printable report', async () => {
  const route = mountPlugin();
  const create = await request(route, {
    method: 'POST',
    path: '/vllm-ascend-profiler/api/jobs',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ path: fixtureDir }),
  });
  const job = await waitForJob(route, create.json().id);
  assert.equal(job.state, 'done');

  const png = `data:image/png;base64,${Buffer.from('fake-png-bytes').toString('base64')}`;
  const upload = await request(route, {
    method: 'POST',
    path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/charts`,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ charts: { '泳道图': png, '坏数据': 'not-an-image', '过大': `data:image/png;base64,${'A'.repeat(6 * 1024 * 1024 + 4)}` } }),
  });
  assert.equal(upload.status, 200);
  const report = upload.json();
  assert.equal(report.accepted, 1);
  assert.equal(report.dropped, 2, 'non-image and oversized payloads must be rejected');

  const printable = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/report.print?print=0` });
  assert.equal(printable.status, 200);
  assert.ok(printable.body.includes('图表快照'));
  assert.ok(printable.body.includes(png), 'the accepted snapshot must be embedded');

  const markdown = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/report.md` });
  assert.equal(markdown.status, 200);
  assert.ok(markdown.body.includes('![泳道图](data:image/png;base64,'), 'the markdown export must embed the snapshot too');

  // A malformed request must not break the stored dataset.
  const bad = await request(route, {
    method: 'POST',
    path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}/charts`,
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ nope: true }),
  });
  assert.equal(bad.status, 400);
  const stillThere = await request(route, { path: `/vllm-ascend-profiler/api/datasets/${job.datasetId}` });
  assert.equal(stillThere.status, 200);
});

test('unknown api paths and expired datasets return typed errors', async () => {
  const route = mountPlugin();
  const unknown = await request(route, { path: '/vllm-ascend-profiler/api/nope' });
  assert.equal(unknown.status, 404);
  assert.match(unknown.json().error, /未知的 API 路径/);

  const missing = await request(route, { path: '/vllm-ascend-profiler/api/datasets/00000000-0000-0000-0000-000000000000' });
  assert.equal(missing.status, 404);
  assert.match(missing.json().error, /数据集不存在或已过期/);
});

test('docs endpoint carries metric, artifact, and usage documentation', async () => {
  const route = mountPlugin();
  const res = await request(route, { path: '/vllm-ascend-profiler/api/docs' });
  assert.equal(res.status, 200);
  const bundle = res.json();
  assert.ok(bundle.metricDocs.length >= 3);
  assert.ok(bundle.artifactDocs.some((entry) => entry.name === 'trace_view.json'));
  assert.ok(bundle.artifactDocs.some((entry) => entry.name === 'step_trace_time.csv'));
  assert.ok(bundle.usage.quickStart.length > 0);
  const trace = bundle.artifactDocs.find((entry) => entry.name === 'trace_view.json');
  assert.ok(trace.notes.some((note) => note.includes('裸 JSON 数组')), 'the bare-array fact must be documented');
});

test('status page reports a 404 for an unknown asset', async () => {
  const route = mountPlugin();
  const res = await request(route, { path: '/vllm-ascend-profiler/missing.js' });
  assert.equal(res.status, 404);
  assert.match(res.body, /未找到该资源/);
});
