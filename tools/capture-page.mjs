/**
 * Dev-only screenshot tool for the analyzer page.
 *
 * The CLI `--screenshot` flag captures as soon as the load event fires, which is
 * before the page has fetched and rendered a dataset. This script drives a
 * headless browser over the DevTools protocol instead: it navigates, waits until
 * the three modules are actually rendered, then captures the full page or a
 * clipped region.
 *
 * Usage (a browser must already be listening on the debugging port):
 *
 * ```sh
 * msedge --headless=new --remote-debugging-port=9222 --user-data-dir=... about:blank
 * node tools/capture-page.mjs --url http://127.0.0.1:3099/vllm-ascend-profiler/ \
 *   --out docs/screenshots --port 9222
 * ```
 *
 * Shots written: `01-intake`, `02-overview`, `03-swimlane`, `04-share`,
 * `05-locate`, `06-evidence`, `07-actions`, `08-benefit`, `09-full`,
 * `10-linked-filter`, `11-dark-share`, and — when an optimized bundle is
 * available — `12-compare-summary`, `13-compare-gantt`, `14-compare-share`,
 * `15-compare-full`.
 */

import { existsSync, mkdirSync, readdirSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

/** Parse `--key value` / `--flag` arguments. */
function parseArgs(argv) {
  const out = {};
  for (let at = 0; at < argv.length; at += 1) {
    const token = argv[at];
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = argv[at + 1];
    if (next === undefined || next.startsWith('--')) out[key] = true;
    else {
      out[key] = next;
      at += 1;
    }
  }
  return out;
}

const args = parseArgs(process.argv.slice(2));
const url = args.url ?? 'http://127.0.0.1:3099/vllm-ascend-profiler/';
const outDir = args.out ?? 'docs/screenshots';
const port = Number(args.port ?? 9222);
const width = Number(args.width ?? 1600);
const height = Number(args.height ?? 1200);

/** Open one DevTools target and return a JSON-RPC client. */
async function connect() {
  const deadline = Date.now() + 30000;
  let targets;
  while (Date.now() < deadline) {
    try {
      targets = await (await fetch(`http://127.0.0.1:${String(port)}/json/list`)).json();
      if (Array.isArray(targets) && targets.length > 0) break;
    } catch {
      // Browser not up yet.
    }
    await sleep(400);
  }
  const page = (targets ?? []).find((target) => target.type === 'page') ?? targets?.[0];
  if (page === undefined) throw new Error('no DevTools page target available');
  const socket = new WebSocket(page.webSocketDebuggerUrl);
  let nextId = 1;
  const pending = new Map();
  const events = [];
  socket.addEventListener('message', (event) => {
    let message;
    try {
      message = JSON.parse(String(event.data));
    } catch {
      return;
    }
    if (message.id === undefined) {
      // Page-side diagnostics: exceptions, console output, log entries.
      if (message.method === 'Runtime.exceptionThrown') {
        const detail = message.params?.exceptionDetails;
        events.push(`[exception] ${detail?.exception?.description ?? detail?.text ?? 'unknown'}`);
      } else if (message.method === 'Runtime.consoleAPICalled') {
        const text = (message.params?.args ?? []).map((argument) => argument.value ?? argument.description ?? '').join(' ');
        events.push(`[console.${String(message.params?.type)}] ${text}`);
      } else if (message.method === 'Log.entryAdded') {
        events.push(`[log.${String(message.params?.entry?.level)}] ${String(message.params?.entry?.text)}`);
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
    socket.addEventListener('error', () => reject(new Error('DevTools socket failed')), { once: true });
  });
  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = nextId;
    nextId += 1;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
  return { send, close: () => socket.close(), events };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Evaluate an expression in the page and return its value. */
async function evaluate(client, expression) {
  const result = await client.send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  return result.result?.value;
}

/** Wait until all three modules have rendered. */
async function waitForRender(client, attempts = 160) {
  const probe = `JSON.stringify({
    gantt: document.getElementById('module-gantt') ? !document.getElementById('module-gantt').hidden : false,
    share: document.getElementById('module-share') ? !document.getElementById('module-share').hidden : false,
    advice: document.getElementById('module-advice') ? !document.getElementById('module-advice').hidden : false,
    compare: document.getElementById('module-compare') ? !document.getElementById('module-compare').hidden : false,
    kpis: document.querySelectorAll('.kpi').length,
    flowNodes: document.querySelectorAll('button.flow-node').length,
    treemapTiles: document.querySelectorAll('g.tm-tile').length,
    shareSegments: document.querySelectorAll('g.share-seg').length,
    ganttWidth: document.getElementById('gantt-canvas') ? document.getElementById('gantt-canvas').width : 0,
    error: document.getElementById('error-box') && !document.getElementById('error-box').hidden
      ? document.getElementById('error-box').textContent.slice(0, 160) : null,
  })`;
  let last = {};
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const raw = await evaluate(client, probe);
    last = raw === undefined ? {} : JSON.parse(raw);
    if (last.gantt === true && last.share === true && last.advice === true && last.compare === true && last.flowNodes >= 5
      && last.treemapTiles > 0 && last.shareSegments > 0 && last.ganttWidth > 0) return last;
    if (last.error !== null && last.error !== undefined) throw new Error(`page reported an error: ${last.error}`);
    await sleep(250);
  }
  throw new Error(`page did not render in time: ${JSON.stringify(last)}`);
}

/**
 * Select the dataset the shots should be taken from.
 *
 * An instance may hold several datasets (including the optimized captures of a
 * previous run), so relying on "the page loaded the first one" silently captures
 * whatever happened to be first. The label is matched by prefix.
 */
async function selectDataset(client, label) {
  return await evaluate(client, `(async () => {
    const button = [...document.querySelectorAll('#dataset-list button')]
      .find((node) => (node.textContent ?? '').startsWith(${JSON.stringify(label)}));
    if (button === undefined) return 'missing';
    button.click();
    for (let attempt = 0; attempt < 120; attempt += 1) {
      if (button.classList.contains('active') && document.querySelector('g.tm-tile') !== null) return 'ok';
      await new Promise((resolve) => setTimeout(resolve, 150));
    }
    return 'timeout';
  })()`);
}

/** Select one step of the reasoning chain and let its panel paint. */
async function selectChainStep(client, step) {  return await evaluate(client, `(() => {
    const node = document.querySelector('#advice-chain button.flow-node[data-node="${step}"]');
    if (node === null) return 'missing';
    node.click();
    return node.querySelector('.flow-value')?.textContent ?? 'ok';
  })()`);
}

/** Capture the full page (beyond the viewport). */
async function captureFull(client, file) {
  const shot = await client.send('Page.captureScreenshot', { format: 'png', captureBeyondViewport: true });
  writeFileSync(file, Buffer.from(shot.data, 'base64'));
  return file;
}

/** Capture one element by selector, clipped to its box. */
async function captureElement(client, selector, file) {
  const boxRaw = await evaluate(client, `(() => {
    const node = document.querySelector(${JSON.stringify(selector)});
    if (node === null) return null;
    const rect = node.getBoundingClientRect();
    return JSON.stringify({ x: rect.left + window.scrollX, y: rect.top + window.scrollY, width: rect.width, height: rect.height });
  })()`);
  if (boxRaw === null || boxRaw === undefined) throw new Error(`selector not found: ${selector}`);
  const box = JSON.parse(boxRaw);
  const shot = await client.send('Page.captureScreenshot', {
    format: 'png',
    captureBeyondViewport: true,
    clip: {
      x: Math.max(0, Math.floor(box.x) - 4),
      y: Math.max(0, Math.floor(box.y) - 4),
      width: Math.ceil(box.width) + 8,
      height: Math.ceil(box.height) + 8,
      scale: 1,
    },
  });
  writeFileSync(file, Buffer.from(shot.data, 'base64'));
  return file;
}

const client = await connect();
try {
  mkdirSync(outDir, { recursive: true });
  await client.send('Page.enable');
  await client.send('Runtime.enable');
  await client.send('Log.enable');
  await client.send('Emulation.setDeviceMetricsOverride', { width, height, deviceScaleFactor: 1, mobile: false });
  await client.send('Page.navigate', { url });
  let state;
  try {
    state = await waitForRender(client, Number(args.budget ?? 160));
  } catch (error) {
    process.stdout.write(`render wait failed: ${error.message}\n`);
    const detail = await evaluate(client, `JSON.stringify({
      ready: document.readyState,
      hasVap: typeof window.VAP,
      vapKeys: typeof window.VAP === 'object' ? Object.keys(window.VAP) : [],
      hasGanttClass: typeof window.VAP?.GanttView,
      datasetCount: document.getElementById('dataset-count')?.textContent,
      overviewHidden: document.getElementById('overview')?.hidden,
      ganttHidden: document.getElementById('module-gantt')?.hidden,
      progress: document.getElementById('progress-detail')?.textContent,
      errorBox: document.getElementById('error-box')?.textContent?.slice(0, 300),
      warnBox: document.getElementById('warn-box')?.textContent?.slice(0, 300),
    })`);
    process.stdout.write(`page state: ${String(detail)}\n`);
    for (const event of client.events.slice(0, 40)) process.stdout.write(`${event}\n`);
    process.exitCode = 2;
    throw error;
  }
  process.stdout.write(`rendered: ${JSON.stringify(state)}\n`);
  const baseline = String(args.baseline ?? 'host-schedule-bound');
  process.stdout.write(`baseline dataset: ${baseline} → ${String(await selectDataset(client, baseline))}\n`);
  // Let the entry animations finish (KPI count-up, bar grow-in, reveal sweep) so
  // the stills show final values instead of a frame mid-animation.
  await sleep(Number(args.settle ?? 1400));
  // The topbar and stepper are sticky, so they would be stamped over every
  // clipped section. Unstick them for the capture (this is a screenshot
  // concern, not a page concern) and scroll to the top for a clean full shot.
  await evaluate(client, `(() => {
    for (const node of document.querySelectorAll('.topbar, .stepper')) {
      node.dataset.originalPosition = node.style.position;
      node.style.position = 'static';
    }
    window.scrollTo(0, 0);
    return true;
  })()`);
  await sleep(200);

  const written = [];
  written.push(await captureElement(client, '#intake', join(outDir, '01-intake.png')));
  written.push(await captureElement(client, '#overview', join(outDir, '02-overview.png')));
  written.push(await captureElement(client, '#module-gantt', join(outDir, '03-swimlane.png')));
  written.push(await captureElement(client, '#module-share', join(outDir, '04-share.png')));
  // Step 5 is a flow plus one panel: each frame shows the flow (the state) above
  // the diagram of the selected step, which is exactly how a reader meets it.
  written.push(await captureElement(client, '#advice-chain', join(outDir, '05-locate.png')));
  process.stdout.write(`step ② 量化证据: ${String(await selectChainStep(client, 'evidence'))}\n`);
  await sleep(500);
  written.push(await captureElement(client, '#advice-chain', join(outDir, '06-evidence.png')));
  process.stdout.write(`step ④ 优化行动: ${String(await selectChainStep(client, 'actions'))}\n`);
  await sleep(500);
  written.push(await captureElement(client, '#advice-chain', join(outDir, '07-actions.png')));
  process.stdout.write(`step ⑤ 预期收益: ${String(await selectChainStep(client, 'benefit'))}\n`);
  await sleep(500);
  written.push(await captureElement(client, '#advice-chain', join(outDir, '08-benefit.png')));
  await selectChainStep(client, 'locate');
  // Re-selecting redraws the panel: let the bar/gain transitions settle, or the
  // full-page shot catches them half-grown.
  await sleep(900);
  written.push(await captureFull(client, join(outDir, '09-full.png')));

  // Drive one interaction to show the cross-module linkage: clicking a tile in
  // step 4 filters step 3 and adds a chip that can be cleared.
  const clicked = await evaluate(client, `(() => {
    const tile = document.querySelector('g.tm-tile');
    if (tile === null) return 'no tile';
    tile.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return tile.getAttribute('data-operator');
  })()`);
  process.stdout.write(`clicked tile: ${String(clicked)}\n`);
  await sleep(900);
  written.push(await captureElement(client, '#module-gantt', join(outDir, '10-linked-filter.png')));
  await evaluate(client, `document.getElementById('gantt-filters')?.querySelector('button')?.click()`);
  await sleep(400);

  try {
    await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });
    await sleep(600);
    written.push(await captureElement(client, '#module-share', join(outDir, '11-dark-share.png')));
    await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });
  } catch {
    // Media emulation is optional.
  }

  // Step 6: import the *optimized* capture through the page's own file input, then
  // capture the comparison — step 6's verdicts plus the two before/after panes.
  // The paths must be absolute: DevTools hands the renderer a path it can only
  // read verbatim, and a relative one yields zero-length files plus an
  // `ERR_ACCESS_DENIED` upload.
  const compareDir = resolve(String(args['compare-dir'] ?? join('test', 'fixtures', 'host-schedule-bound-optimized')));
  if (existsSync(compareDir)) {
    const files = readdirSync(compareDir).map((name) => join(compareDir, name));
    process.stdout.write(`step 6 导入优化后产物：${String(files.length)} 个文件（${compareDir}）\n`);
    await evaluate(client, `document.getElementById('module-compare')?.scrollIntoView({ behavior: 'smooth', block: 'start' })`);
    await sleep(700);
    const doc = await client.send('DOM.getDocument', { depth: -1 });
    const input = await client.send('DOM.querySelector', { nodeId: doc.root.nodeId, selector: '#compare-file-input' });
    await client.send('DOM.setFileInputFiles', { nodeId: input.nodeId, files });
    for (let attempt = 0; attempt < 120; attempt += 1) {
      const ready = await evaluate(client, `document.getElementById('gantt-after-pane') ? !document.getElementById('gantt-after-pane').hidden : false`);
      if (ready === true) break;
      await sleep(300);
    }
    await sleep(1200);
    const verdict = await evaluate(client, `JSON.stringify({
      pair: document.getElementById('compare-pair')?.textContent ?? '',
      achieved: document.querySelectorAll('#compare-summary .verify-verdict.achieved').length,
      missed: document.querySelectorAll('#compare-summary .verify-verdict.missed').length,
      chips: document.querySelectorAll('#gantt-deltas .delta-chip').length,
      deltaTables: document.querySelectorAll('#share-deltas .delta-table').length,
      afterCanvas: document.getElementById('gantt-canvas-after')?.width ?? 0,
    })`);
    process.stdout.write(`compare state: ${String(verdict)}\n`);
    written.push(await captureElement(client, '#module-compare', join(outDir, '12-compare-summary.png')));
    written.push(await captureElement(client, '#module-gantt', join(outDir, '13-compare-gantt.png')));
    written.push(await captureElement(client, '#module-share', join(outDir, '14-compare-share.png')));
    written.push(await captureFull(client, join(outDir, '15-compare-full.png')));
  } else {
    process.stdout.write(`no optimized bundle at ${compareDir}: skipping the comparison shots\n`);
  }

  for (const file of written) process.stdout.write(`wrote ${file}\n`);
} finally {
  client.close();
}
