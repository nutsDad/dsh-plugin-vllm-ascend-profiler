/**
 * Provenance audit: are the three modules really built from the uploaded
 * profiling artifacts?
 *
 * Three independent checks:
 *   A. HTTP upload path — upload the real fixture bytes, parse, and compare the
 *      returned view model against an *independent* read of the same files
 *      (a hand-written CSV reader, file lineage, internal cross-sums).
 *   B. Mutation — change a duration inside the artifact, upload again, and require
 *      the values to move accordingly (no caching, no hard-coded numbers).
 *   C. Rendered page — drive the real page over CDP, read what each module
 *      actually put on screen (composition strip, treemap, chain flow, panel,
 *      swimlane tooltip) and match it against the same view model JSON.
 *
 * Usage:
 *   node .tools/audit-provenance.mjs --url http://127.0.0.1:3099/vllm-ascend-profiler/ \
 *     --fixture vllm-ascend-profiler-analyzer/test/fixtures/host-schedule-bound --port 9222
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

function parseArgs(argv) {
  const out = {};
  for (let at = 0; at < argv.length; at += 1) {
    const token = argv[at];
    if (!token.startsWith('--')) continue;
    const next = argv[at + 1];
    if (next === undefined || next.startsWith('--')) out[token.slice(2)] = true;
    else {
      out[token.slice(2)] = next;
      at += 1;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const base = String(args.url ?? 'http://127.0.0.1:3099/vllm-ascend-profiler/').replace(/\/$/, '');
const fixture = args.fixture ?? join('test', 'fixtures', 'host-schedule-bound');
const port = Number(args.port ?? 9222);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok: ok === true, detail });
  process.stdout.write(`${ok === true ? 'PASS' : 'FAIL'}  ${name}${detail === undefined ? '' : ` — ${detail}`}\n`);
}

// ── independent helpers (deliberately NOT the plugin's own parsers) ──────────

/** Minimal CSV reader (quote-aware) for the CANN summary tables. */
function readCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let quoted = false;
  for (let at = 0; at < text.length; at += 1) {
    const char = text[at];
    if (quoted) {
      if (char === '"' && text[at + 1] === '"') {
        field += '"';
        at += 1;
      } else if (char === '"') quoted = false;
      else field += char;
      continue;
    }
    if (char === '"') quoted = true;
    else if (char === ',') {
      row.push(field);
      field = '';
    } else if (char === '\n') {
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else if (char !== '\r') field += char;
  }
  if (field !== '' || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  const [header, ...body] = rows.filter((entry) => entry.length > 1);
  return {
    header: (header ?? []).map((key) => key.trim()),
    rows: body.map((entry) => Object.fromEntries((header ?? []).map((key, index) => [key.trim(), entry[index] ?? '']))),
  };
}

/** Rewrite one quote-aware CSV field on a line whose first cell matches `key`. */
function mapCsvLine(text, predicate, fieldName, transform) {
  const header = text.slice(0, text.indexOf('\n')).split(',').map((key) => key.trim());
  const index = header.indexOf(fieldName);
  if (index === -1) throw new Error(`no ${fieldName} column`);
  return text.split('\n').map((line) => {
    if (line === '' || !predicate(line)) return line;
    const cells = [];
    let field = '';
    let quoted = false;
    for (let at = 0; at < line.length; at += 1) {
      const char = line[at];
      if (quoted) {
        field += char;
        if (char === '"') quoted = false;
        continue;
      }
      if (char === '"') {
        quoted = true;
        field += char;
        continue;
      }
      if (char === ',') {
        cells.push(field);
        field = '';
        continue;
      }
      field += char;
    }
    cells.push(field);
    if (cells.length <= index) return line;
    cells[index] = transform(cells[index]);
    return cells.join(',');
  }).join('\n');
}

/** Upload a file set through the real collection endpoint; returns the dataset id. */
async function uploadArtifacts(files, label) {
  const created = await (await fetch(`${base}/api/jobs`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ collect: true, label }),
  })).json();
  if (created.id === undefined) throw new Error(`cannot create job: ${JSON.stringify(created)}`);
  for (const file of files) {
    const response = await fetch(`${base}/api/jobs?name=${encodeURIComponent(file.name)}&group=${String(created.id)}`, {
      method: 'POST',
      headers: { 'content-type': 'application/octet-stream' },
      body: file.buffer,
    });
    if (!response.ok) throw new Error(`upload ${file.name} failed: ${String(response.status)}`);
  }
  const started = await fetch(`${base}/api/jobs/${String(created.id)}/start`, { method: 'POST' });
  if (!started.ok) throw new Error(`start failed: ${String(started.status)}`);
  for (let attempt = 0; attempt < 240; attempt += 1) {
    const job = await (await fetch(`${base}/api/jobs/${String(created.id)}`)).json();
    if (job.state === 'done') return job.datasetId;
    if (job.state === 'error') throw new Error(`parse failed: ${String(job.errors)}`);
    await sleep(250);
  }
  throw new Error('parse did not finish in time');
}

const getView = async (id) => (await fetch(`${base}/api/datasets/${String(id)}`)).json();
const relative = (left, right) => Math.abs(left - right) / Math.max(1, Math.abs(right));

// ── A. upload path vs an independent read of the same bytes ─────────────────

const files = readdirSync(fixture).map((name) => ({ name, buffer: readFileSync(join(fixture, name)) }));
process.stdout.write(`\n=== A. 上传真实产物 → 解析 → 视图模型（${String(files.length)} 个文件）===\n`);
const datasetA = await uploadArtifacts(files, 'audit-original');
const viewA = await getView(datasetA);
process.stdout.write(`dataset ${datasetA} · ${viewA.label} · ${String(viewA.meta.eventCount)} 事件 · ${String(viewA.meta.wallUs)}µs · ${viewA.meta.validation.profileType}\n`);

// A1 lineage: the view model names exactly the uploaded files, with per-file evidence.
const shippedFiles = (viewA.meta.files ?? []).map((file) => (typeof file === 'string' ? { name: file } : file));
const shipped = new Set(shippedFiles.map((file) => file.name));
const missing = files.map((file) => file.name).filter((name) => !shipped.has(name));
const withMarkers = shippedFiles.filter((file) => Array.isArray(file.markers) && file.markers.length > 0).length;
check('A1 视图模型声明的文件 = 上传的文件（含识别证据）', missing.length === 0 && withMarkers >= 4,
  `${String(shippedFiles.length)} 个文件，${String(withMarkers)} 个带识别标记${missing.length === 0 ? '' : `，缺 ${missing.join(',')}`}`);

// A2 independent CANN-summary read vs the reported per-operator totals.
const statistic = readCsv(readFileSync(join(fixture, 'op_statistic.csv'), 'utf8'));
const independent = new Map();
for (const row of statistic.rows) {
  const name = row['OP Type'] ?? row.Name;
  const current = independent.get(name) ?? { totalUs: 0, count: 0 };
  independent.set(name, {
    totalUs: current.totalUs + Number(row['Total Time(us)'] ?? 0),
    count: current.count + Number(row.Count ?? 0),
  });
}
let a2Worst = 0;
let a2Compared = 0;
for (const [name, expected] of independent) {
  const row = (viewA.operators ?? []).find((entry) => entry.name === name);
  if (row === undefined) continue;
  a2Compared += 1;
  a2Worst = Math.max(a2Worst, relative(row.totalUs, expected.totalUs));
  if (row.count !== expected.count) a2Worst = Math.max(a2Worst, 1);
}
check('A2 独立解析 op_statistic.csv = 页面算子累计耗时', a2Compared >= 4 && a2Worst < 0.01,
  `对比 ${String(a2Compared)} 个算子，最大偏差 ${(a2Worst * 100).toFixed(4)}%`);

// A3 independent kernel-table row sum vs the reported device totals.
const kernels = readCsv(readFileSync(join(fixture, 'kernel_details.csv'), 'utf8'));
const kernelSum = new Map();
for (const row of kernels.rows) {
  kernelSum.set(row.Name, (kernelSum.get(row.Name) ?? 0) + Number(row['Duration(us)'] ?? 0));
}
let a3Worst = 0;
let a3Compared = 0;
for (const [name, total] of kernelSum) {
  const row = (viewA.operators ?? []).find((entry) => entry.name === name);
  if (row === undefined) continue;
  a3Compared += 1;
  a3Worst = Math.max(a3Worst, relative(row.totalUs, total));
}
check('A3 独立求和 kernel_details.csv = 设备算子累计耗时', a3Compared >= 3 && a3Worst < 0.01,
  `对比 ${String(a3Compared)} 个算子，最大偏差 ${(a3Worst * 100).toFixed(4)}%`);

// A4 internal consistency: every category equals the sum of its operators.
let a4Worst = 0;
for (const item of viewA.categories.items) {
  const sum = (viewA.operators ?? []).filter((row) => row.category === item.id).reduce((total, row) => total + row.totalUs, 0);
  if (item.totalUs === 0 && sum === 0) continue;
  a4Worst = Math.max(a4Worst, relative(item.totalUs, sum));
}
check('A4 每个大类 = 该大类算子累计之和', a4Worst < 0.02, `最大偏差 ${(a4Worst * 100).toFixed(3)}%`);
const shareSum = viewA.categories.items.reduce((sum, item) => sum + item.sharePct, 0);
check('A5 大类占比合计 = 100%', Math.abs(shareSum - 100) < 0.5, `${shareSum.toFixed(3)}%`);

// A6 the advice quotes the same measurements the overview shows as KPIs.
const evidence = viewA.analysis.steps.locate.perScope[0].primaryCandidate.evidence;
const idle = evidence.find((row) => String(row.metric).includes('NPU 空闲'));
const hostOnly = evidence.find((row) => String(row.metric).includes('Host 独占时间占比'));
check('A6 证据指标 = KPI 指标（同一份实测值）',
  idle !== undefined && Number.isFinite(idle.value) && Math.abs(idle.value - viewA.analysis.indicators.idlePct) < 0.01
  && hostOnly !== undefined && Math.abs(hostOnly.value - viewA.analysis.indicators.hostOnlyPct) < 0.01,
  `NPU 空闲 ${String(idle?.value)}% = idlePct ${viewA.analysis.indicators.idlePct.toFixed(2)}% · Host 独占 ${String(hostOnly?.value)}% = hostOnlyPct ${viewA.analysis.indicators.hostOnlyPct.toFixed(2)}%`);

// A7 the swimlane only ships operators that the artifacts contain.
const laneRows = viewA.timeline.lanes.flatMap((group) => group.rows);
const unknown = laneRows.map((row) => row.name).filter((name) => !(viewA.operators ?? []).some((row) => row.name === name));
const laneEvents = laneRows.reduce((sum, row) => sum + row.events.length, 0);
check('A7 泳道行 / 事件都来自上传产物', unknown.length === 0 && laneEvents > 0,
  `${String(laneRows.length)} 行 · ${String(laneEvents)} 个算子条 · 未知算子 ${unknown.slice(0, 3).join(',') || '无'}`);

// ── B. mutation: change the artifact, the numbers must follow ───────────────

process.stdout.write('\n=== B. 改动产物里的一个耗时 → 图上的数字必须跟着变 ===\n');
const targetOperator = 'MatMulV2';
// The reported total for this dataset comes from trace_view.json (totalsSource
// = "trace"), so the mutation has to hit the trace. The CSV copy is mutated too,
// in a second upload, to check that disagreeing sources are surfaced rather than
// silently averaged.
const mutateTrace = (file) => {
  const text = file.buffer.toString('utf8');
  const pattern = new RegExp(`("name":"${targetOperator}","pid":\\d+,"tid":"?\\d+"?,"ts":"[^"]+","dur":)([0-9.]+)`, 'g');
  let hits = 0;
  const doubled = text.replace(pattern, (_all, head, dur) => {
    hits += 1;
    return `${head}${(Number(dur) * 2).toFixed(3)}`;
  });
  return { file: { name: file.name, buffer: Buffer.from(doubled, 'utf8') }, hits };
};
const mutateCsv = (file, column) => {
  const text = file.buffer.toString('utf8');
  const doubled = mapCsvLine(text, (line) => line.startsWith(`0,${targetOperator},`), column, (value) => (Number(value) * 2).toFixed(5));
  return { name: file.name, buffer: Buffer.from(doubled, 'utf8') };
};

let traceHits = 0;
const traceMutated = files.map((file) => {
  if (file.name !== 'trace_view.json') return file;
  const result = mutateTrace(file);
  traceHits = result.hits;
  return result.file;
});
check('B0 产物里找到并改写了目标算子的每次调用', traceHits >= 100, `trace_view.json 中改写 ${String(traceHits)} 条 ${targetOperator} 事件`);

const datasetB = await uploadArtifacts(traceMutated, 'audit-mutated-trace');
const viewB = await getView(datasetB);
const before = (viewA.operators ?? []).find((row) => row.name === targetOperator);
const after = (viewB.operators ?? []).find((row) => row.name === targetOperator);
const ratio = after.totalUs / before.totalUs;
check(`B1 ${targetOperator} 累计耗时随产物翻倍`, Math.abs(ratio - 2) < 0.05,
  `${before.totalUs.toFixed(1)}µs → ${after.totalUs.toFixed(1)}µs（×${ratio.toFixed(3)}）`);

const computeBefore = viewA.categories.items.find((item) => item.id === 'compute');
const computeAfter = viewB.categories.items.find((item) => item.id === 'compute');
check('B2 计算类占比随之上升', computeAfter.sharePct > computeBefore.sharePct + 0.5,
  `${computeBefore.sharePct.toFixed(2)}% → ${computeAfter.sharePct.toFixed(2)}%`);
// The mutation must appear *exactly* in the aggregate: compute time grows by the
// amount that was added to the artifact (device busy is a union, so it can only
// grow by less — that is overlap, not a lost number).
const addedUs = after.totalUs - before.totalUs;
const computeDelta = viewB.analysis.indicators.computeUs - viewA.analysis.indicators.computeUs;
check('B3 计算时间增量 = 产物里增加的耗时', Math.abs(computeDelta - addedUs) / addedUs < 0.01,
  `computeUs +${computeDelta.toFixed(1)}µs vs 产物 +${addedUs.toFixed(1)}µs`);
// The candidate score is gate-based, so it may legitimately stay at 0; what must
// change is the *evidence* the score is computed from — and it must equal the new
// dataset's own aggregates (recomputed, not cached).
const computeCandidate = (view) => view.analysis.steps.locate.perScope[0].candidates.find((entry) => entry.id === 'compute');
const evidenceA = new Map(computeCandidate(viewA).evidence.map((row) => [row.metric, row.value]));
const changed = computeCandidate(viewB).evidence.filter((row) => evidenceA.get(row.metric) !== row.value);
const selfConsistent = changed.every((row) => {
  if (String(row.metric).includes('忙碌')) return Math.abs(row.value - viewB.analysis.indicators.deviceBusyPct) < 0.6;
  if (String(row.metric).includes('计算占比')) return Math.abs(row.value - viewB.analysis.indicators.computePctOfDevice) < 0.6;
  return true;
});
check('B4 受影响的候选证据随产物重算（且等于新数据集的聚合值）',
  viewA.datasetId !== viewB.datasetId && changed.length >= 1 && selfConsistent,
  changed.slice(0, 2).map((row) => `${row.metric} ${String(evidenceA.get(row.metric))} → ${String(row.value)}`).join(' · ')
  || 'no evidence value changed');
// CSV-only mutation: the CANN summary now disagrees with the trace. The plugin
// prefers the trace totals here, so the reported value must NOT move — but the
// disagreement has to show up (cross-check deviation / warning), because that is
// exactly how a stale or partially-written CSV is caught.
const csvMutated = files.map((file) => {
  if (file.name === 'kernel_details.csv') return mutateCsv(file, 'Duration(us)');
  if (file.name === 'op_statistic.csv') return mutateCsv(mutateCsv(file, 'Total Time(us)'), 'Avg Time(us)');
  return file;
});
const datasetC = await uploadArtifacts(csvMutated, 'audit-mutated-csv');
const viewC = await getView(datasetC);
const csvRow = (viewC.operators ?? []).find((row) => row.name === targetOperator);
const deviation = Number(csvRow.crossCheckPct ?? 0);
const csvWarnings = (viewC.meta.warnings ?? []).concat(viewC.analysis.warnings ?? []).filter((text) => /cross|CSV|偏差|一致/i.test(String(text)));
check('B5 只改 CSV → 仍以 trace 为准，并暴露 cross-check 偏差',
  Math.abs(csvRow.totalUs - before.totalUs) / before.totalUs < 0.001 && deviation > 30,
  `累计仍是 ${csvRow.totalUs.toFixed(1)}µs（traceTotal=${String(csvRow.traceTotalUs)}）· crossCheck ${deviation.toFixed(1)}% · 告警 ${String(csvWarnings.length)} 条`);

// ── C. rendered page: what each module actually drew ────────────────────────

process.stdout.write('\n=== C. 页面三块图实际画出来的东西（CDP 驱动真实页面）===\n');
const targets = await (await fetch(`http://127.0.0.1:${String(port)}/json/list`)).json();
const target = targets.find((entry) => entry.type === 'page') ?? targets[0];
const socket = new WebSocket(target.webSocketDebuggerUrl);
let nextId = 1;
const pending = new Map();
const pageErrors = [];
socket.addEventListener('message', (event) => {
  const message = JSON.parse(String(event.data));
  if (message.id === undefined) {
    if (message.method === 'Runtime.exceptionThrown') {
      pageErrors.push(message.params?.exceptionDetails?.exception?.description ?? 'unknown');
    }
    return;
  }
  const entry = pending.get(message.id);
  if (entry === undefined) return;
  pending.delete(message.id);
  if (message.error !== undefined) entry.reject(new Error(JSON.stringify(message.error)));
  else entry.resolve(message.result);
});
await new Promise((resolve, reject) => {
  socket.addEventListener('open', resolve, { once: true });
  socket.addEventListener('error', () => reject(new Error('devtools socket failed')), { once: true });
});
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const id = nextId;
  nextId += 1;
  pending.set(id, { resolve, reject });
  socket.send(JSON.stringify({ id, method, params }));
});
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result?.value;

try {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1600, height: 1200, deviceScaleFactor: 1, mobile: false });
  await send('Page.navigate', { url: `${base}/` });
  await sleep(2800);

  /** Load one dataset in the page and read back what the three modules drew. */
  const inspect = async (datasetId) => {
    const loaded = await evaluate(`(async () => {
      const button = [...document.querySelectorAll('#dataset-list button')].find((node) => node.dataset.id === ${JSON.stringify(datasetId)});
      if (button === undefined) return 'missing';
      button.click();
      for (let attempt = 0; attempt < 100; attempt += 1) {
        const active = document.querySelector('#dataset-list button.active');
        if (active !== null && active.dataset.id === ${JSON.stringify(datasetId)} && document.querySelector('g.tm-tile') !== null) break;
        await new Promise((resolve) => setTimeout(resolve, 120));
      }
      await new Promise((resolve) => setTimeout(resolve, 700));
      return 'ok';
    })()`);
    if (loaded !== 'ok') return undefined;
    const raw = await evaluate(`JSON.stringify({
      active: document.querySelector('#dataset-list button.active')?.dataset.id ?? '',
      kpis: [...document.querySelectorAll('#kpis .kpi')].map((node) => node.textContent.replace(/\\s+/g, ' ').trim()),
      verdict: document.querySelector('#verdict-badge')?.textContent ?? '',
      score: document.querySelector('#verdict-score')?.textContent ?? '',
      strip: [...document.querySelectorAll('#share-bar-legend .composition-item')].map((node) => node.textContent.replace(/\\s+/g, ' ').trim()),
      stripNote: document.querySelector('#share-bar-note')?.textContent ?? '',
      tiles: [...document.querySelectorAll('g.tm-tile')].map((node) => ({
        op: node.getAttribute('data-operator'),
        cat: node.getAttribute('data-category'),
        area: Number(node.querySelector('rect').getAttribute('width')) * Number(node.querySelector('rect').getAttribute('height')),
        title: node.querySelector('title')?.textContent ?? '',
      })),
      ranking: [...document.querySelectorAll('#ranking-host tbody tr')].slice(0, 5).map((row) => [...row.children].map((cell) => cell.textContent)),
      flow: [...document.querySelectorAll('#advice-chain button.flow-node')].map((node) => node.textContent.replace(/\\s+/g, ' ').trim()),
      laneNames: [...document.querySelectorAll('#gantt-groups [data-group]')].map((node) => node.textContent),
    })`);
    return JSON.parse(raw);
  };

  const pageA = await inspect(datasetA);
  const pageB = await inspect(datasetB);
  check('C0 页面成功切到两个（真实上传的）数据集', pageA !== undefined && pageB !== undefined
    && pageA.active === datasetA && pageB.active === datasetB && pageA.tiles.length > 0 && pageB.tiles.length > 0,
    `A=${String(pageA?.active).slice(0, 8)} tiles=${String(pageA?.tiles.length)} · B=${String(pageB?.active).slice(0, 8)} tiles=${String(pageB?.tiles.length)}`);

  if (pageA !== undefined && pageB !== undefined) {
    // C1 composition strip = the view model's category shares (device scope).
    const deviceCategories = viewA.categories.items.filter((item) => item.deviceUs > 0);
    const deviceTotal = deviceCategories.reduce((sum, item) => sum + item.deviceUs, 0);
    const expectedStrip = deviceCategories.map((item) => ({ label: item.label, share: (item.deviceUs / deviceTotal) * 100 }));
    const stripOk = expectedStrip.length === pageA.strip.length
      && expectedStrip.every((expected) => pageA.strip.some((text) => text.includes(expected.label) && text.includes(expected.share.toFixed(1))));
    check('C1 构成条 = 产物算出的设备侧大类占比', stripOk, pageA.strip.join(' | '));

    // C2 treemap: the drawn share matches the operator's cost share in the data,
    // and the tile carries the category the artifacts imply.
    const topN = viewA.operators.filter((row) => row.group !== 'host' && row.overflow !== true)
      .sort((left, right) => right.totalUs - left.totalUs).slice(0, pageA.tiles.length);
    const topNTotal = topN.reduce((sum, row) => sum + row.totalUs, 0);
    const tileAreaTotal = pageA.tiles.reduce((sum, tile) => sum + tile.area, 0);
    const areaOk = pageA.tiles.every((tile) => {
      const row = topN.find((entry) => entry.name === tile.op);
      if (row === undefined) return false;
      const areaShare = tile.area / tileAreaTotal;
      const dataShare = row.totalUs / topNTotal;
      // The tile label prints the share with two decimals ("占比 46.91%").
      const titleShare = Number(/(\d+(?:\.\d+)?)%/.exec(tile.title)?.[1] ?? 'NaN');
      return Math.abs(areaShare - dataShare) < 0.02
        && Math.abs(titleShare - dataShare * 100) < 0.15
        && tile.cat === row.category;
    });
    check('C2 方块面积与标注 % = 产物里的算子占比', areaOk && pageA.tiles.length === topN.length,
      `${String(pageA.tiles.length)} 块：${pageA.tiles.slice(0, 4).map((tile) => `${tile.op} ${(tile.area / tileAreaTotal * 100).toFixed(1)}%`).join(' · ')}`);

    // C3 ranking table = the same operators, in the same order.
    const rankingOk = pageA.ranking.every((row, index) => row[1] === topN[index].name
      && row[4] === String(topN[index].count)
      && row[8] === `${topN[index].totalUs / topNTotal * 100 >= 0 ? (topN[index].totalUs / viewA.operators.filter((entry) => entry.group !== 'host').reduce((sum, entry) => sum + entry.totalUs, 0) * 100).toFixed(2) : ''}%`);
    check('C3 数据表前三行 = 产物里的算子排行（名字/次数/占比）', rankingOk,
      pageA.ranking.slice(0, 3).map((row) => `${row[1]} ${row[4]}次 ${row[8]}`).join(' | '));

    // C4-C9 read the panels of dataset A: reload it, because the last inspect()
    // left dataset B (the mutated upload) on screen.
    await inspect(datasetA);

    // C4 chain flow node ① carries the analysis verdict.
    const bottleneck = viewA.analysis.bottleneck;
    check('C4 流程节点① = 分析结论（类型 + 得分）',
      pageA.flow[0].includes(bottleneck.short ?? bottleneck.label) && pageA.flow[0].includes(bottleneck.score.toFixed(0)),
      `${pageA.flow[0]} vs ${bottleneck.short} ${bottleneck.score.toFixed(1)}`);
    check('C5 结论横幅与流程节点同源', pageA.verdict === (bottleneck.short ?? bottleneck.label) || pageA.verdict === bottleneck.label,
      `${pageA.verdict} · ${pageA.score}`);

    // C6 evidence panel: drawn measured values = analysis evidence values.
    const drawnEvidence = JSON.parse(await evaluate(`(async () => {
      document.querySelector('#advice-chain button.flow-node[data-node="evidence"]').click();
      await new Promise((resolve) => setTimeout(resolve, 450));
      return JSON.stringify([...document.querySelectorAll('#advice-chain g.tb-row')].map((row) => [...row.querySelectorAll('text')].map((node) => node.textContent)));
    })()`));
    const primaryEvidence = viewA.analysis.steps.locate.perScope[0].primaryCandidate.evidence.filter((row) => Number.isFinite(row.value));
    const formatValue = (row) => (row.unit === '%'
      ? `${row.value.toFixed(1)}%`
      : `${String(Number(row.value.toFixed(2)))}${row.unit ?? ''}`);
    const evidenceOk = primaryEvidence.length === drawnEvidence.length && primaryEvidence.every((row, index) => {
      const drawn = drawnEvidence[index] ?? [];
      const label = String(drawn[0] ?? '');
      const value = String(drawn[1] ?? '');
      return String(row.metric).startsWith(label.slice(0, 5)) && value === formatValue(row);
    });
    check('C6 证据面板实测值 = 分析结果实测值', evidenceOk,
      drawnEvidence.slice(0, 3).map((row) => `${row[0]}=${row[1]}`).join(' · '));

    // C7 actions + benefit: drawn numbers = analysis numbers.
    const drawnActions = JSON.parse(await evaluate(`(async () => {
      document.querySelector('#advice-chain button.flow-node[data-node="actions"]').click();
      await new Promise((resolve) => setTimeout(resolve, 450));
      return JSON.stringify({
        flow: document.querySelector('#advice-chain button.flow-node[data-node="actions"]').textContent.replace(/\\s+/g, ' ').trim(),
        rows: [...document.querySelectorAll('#advice-chain .action-row')].map((row) => ({
          id: row.dataset.adviceId,
          title: row.querySelector('.action-title')?.textContent ?? '',
          gain: row.querySelector('.gain-value')?.textContent ?? '',
          pri: row.querySelector('.pri')?.textContent ?? '',
        })),
      });
    })()`));
    const items = viewA.analysis.steps.actions.items;
    const actionsOk = drawnActions.rows.length === items.length && drawnActions.rows.every((row, index) => row.id === items[index].id
      && row.title === items[index].title
      && row.gain === `${items[index].expectedGain?.estimatePct?.toFixed(1) ?? ''}%`
      && row.pri === items[index].priority);
    check('C7 优化行动行 = 分析结果（标题/优先级/收益估算）', actionsOk,
      drawnActions.rows.map((row) => `${row.pri} ${row.gain} ${row.title.slice(0, 14)}`).join(' | '));
    const counts = { 高: 0, 中: 0, 低: 0 };
    for (const item of items) counts[item.priority] += 1;
    check('C8 流程节点④ 的计数 = 行动列表计数', drawnActions.flow.includes(`高${String(counts.高)}`)
      && drawnActions.flow.includes(`中${String(counts.中)}`) && drawnActions.flow.includes(`低${String(counts.低)}`), drawnActions.flow);

    const drawnBenefit = JSON.parse(await evaluate(`(async () => {
      document.querySelector('#advice-chain button.flow-node[data-node="benefit"]').click();
      await new Promise((resolve) => setTimeout(resolve, 450));
      return JSON.stringify({
        flow: document.querySelector('#advice-chain button.flow-node[data-node="benefit"]').textContent.replace(/\\s+/g, ' ').trim(),
        totals: [...document.querySelectorAll('#advice-chain .benefit-totals .total')].map((node) => node.textContent.replace(/\\s+/g, ' ').trim()),
        rows: [...document.querySelectorAll('#advice-chain .benefit-row')].map((node) => node.textContent.replace(/\\s+/g, ' ').trim()),
      });
    })()`));
    const benefit = viewA.analysis.steps.benefit;
    const benefitOk = drawnBenefit.rows.length === benefit.items.length
      && drawnBenefit.rows.every((text, index) => text.includes(benefit.items[index].title)
        && text.includes(`${benefit.items[index].estimatePct.toFixed(1)}%`))
      && drawnBenefit.totals.some((text) => text.includes(`${benefit.combined.conservativePct.toFixed(1)}%`))
      && drawnBenefit.totals.some((text) => text.includes(`${benefit.combined.optimisticPct.toFixed(1)}%`));
    check('C9 收益面板 = 分析结果（逐项 + 保守/乐观合计）', benefitOk,
      `${drawnBenefit.rows.length} 项 · ${drawnBenefit.totals.join(' / ')}`);

    // C10 mutation reaches the page: the same tile draws different numbers.
    const tileA = pageA.tiles.find((tile) => tile.op === targetOperator);
    const tileB = pageB.tiles.find((tile) => tile.op === targetOperator);
    const shareA = Number(/(\d+(?:\.\d+)?)%/.exec(tileA.title)?.[1] ?? '0');
    const shareB = Number(/(\d+(?:\.\d+)?)%/.exec(tileB.title)?.[1] ?? '0');
    check('C10 改动产物后页面方块数值同步变化', tileB.title !== tileA.title && shareB > shareA + 1,
      `${targetOperator} 占比 ${shareA.toFixed(1)}% → ${shareB.toFixed(1)}%`);
    check('C11 改动产物后页面 KPI 同步变化', JSON.stringify(pageA.kpis) !== JSON.stringify(pageB.kpis),
      `${pageA.kpis[1]?.slice(0, 22)} → ${pageB.kpis[1]?.slice(0, 22)}`);

    // C12 module 1 (canvas) provenance: hover the swimlane, read the tooltip, and
    // match its numbers against the dataset rows.
    const tooltips = [];
    for (let y = 60; y <= 500 && tooltips.length < 3; y += 9) {
      for (const x of [420, 640, 860, 1080]) {
        const text = await evaluate(`(() => {
          const canvas = document.getElementById('gantt-canvas');
          const box = canvas.getBoundingClientRect();
          canvas.dispatchEvent(new PointerEvent('pointermove', { clientX: box.left + ${String(x)}, clientY: box.top + ${String(y)}, bubbles: true }));
          const tip = document.getElementById('gantt-tooltip');
          return tip.hidden ? '' : tip.textContent;
        })()`);
        if (typeof text === 'string' && text !== '' && !tooltips.includes(text)) {
          tooltips.push(text);
          break;
        }
      }
    }
    const matched = tooltips.filter((text) => {
      // The tooltip groups thousands ("3,053") and prints µs values; compare the
      // numbers, not the formatting.
      const plain = text.replaceAll(',', '');
      return laneRows.some((row) => plain.includes(row.name)
        && plain.includes(String(row.count))
        && plain.includes(String(row.avgUs.toFixed(2))));
    });    check('C12 泳道图提示框 = 产物里的算子统计（名称/次数/均值）', matched.length >= 2,
      `${String(matched.length)}/${String(tooltips.length)} 条匹配：${tooltips.map((text) => text.split('\n')[0]).slice(0, 3).join(' · ')}`);
    check('C13 泳道图按产物划分 Host / Device 两组 + 类别图例',
      pageA.laneNames.length === 2 && pageA.laneNames.includes('Host') && pageA.laneNames.includes('Device'),
      pageA.laneNames.join(' / '));
    check('C14 页面无脚本异常', pageErrors.length === 0, pageErrors.slice(0, 2).join(' | ') || '无');
  }
} finally {
  socket.close();
}

// Clean up: the audit datasets must not linger in the instance.
for (const id of [datasetA, datasetB, datasetC]) {
  await fetch(`${base}/api/datasets/${String(id)}`, { method: 'DELETE' });
}

const failed = results.filter((entry) => !entry.ok);
process.stdout.write(`\n=== 结论：${String(results.length - failed.length)}/${String(results.length)} 通过 ===\n`);
for (const entry of failed) process.stdout.write(`  FAIL ${entry.name} — ${String(entry.detail)}\n`);
process.exitCode = failed.length === 0 ? 0 : 1;
