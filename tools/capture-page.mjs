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
 * `05-locate`, `06-actions`, `07-benefit`, `08-full`, `09-linked-filter`,
 * `10-dark-share`.
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

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
    kpis: document.querySelectorAll('.kpi').length,
    chainSteps: document.querySelectorAll('.chain-step').length,
    adviceItems: document.querySelectorAll('.advice').length,
    ganttWidth: document.getElementById('gantt-canvas') ? document.getElementById('gantt-canvas').width : 0,
    error: document.getElementById('error-box') && !document.getElementById('error-box').hidden
      ? document.getElementById('error-box').textContent.slice(0, 160) : null,
  })`;
  let last = {};
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const raw = await evaluate(client, probe);
    last = raw === undefined ? {} : JSON.parse(raw);
    if (last.gantt === true && last.share === true && last.advice === true && last.chainSteps >= 5 && last.ganttWidth > 0) return last;
    if (last.error !== null && last.error !== undefined) throw new Error(`page reported an error: ${last.error}`);
    await sleep(250);
  }
  throw new Error(`page did not render in time: ${JSON.stringify(last)}`);
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
  // The chain is very tall, so the readable shots are the individual steps
  // (rendered inside the `advice-root` wrapper).
  written.push(await captureElement(client, '#advice-chain .chain-step:nth-of-type(1)', join(outDir, '05-locate.png')));
  written.push(await captureElement(client, '#advice-chain .chain-step:nth-of-type(4)', join(outDir, '06-actions.png')));
  written.push(await captureElement(client, '#advice-chain .chain-step:nth-of-type(5)', join(outDir, '07-benefit.png')));
  written.push(await captureFull(client, join(outDir, '08-full.png')));

  // Drive one interaction to show the cross-module linkage: clicking a bar in
  // step 4 filters step 3 and adds a chip that can be cleared.
  const clicked = await evaluate(client, `(() => {
    const row = document.querySelector('g.bar-row');
    if (row === null) return 'no bar';
    row.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    return row.getAttribute('data-operator');
  })()`);
  process.stdout.write(`clicked bar: ${String(clicked)}\n`);
  await sleep(900);
  written.push(await captureElement(client, '#module-gantt', join(outDir, '09-linked-filter.png')));
  await evaluate(client, `document.getElementById('gantt-filters')?.querySelector('button')?.click()`);
  await sleep(400);

  try {
    await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'dark' }] });
    await sleep(600);
    written.push(await captureElement(client, '#module-share', join(outDir, '10-dark-share.png')));
    await client.send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: 'light' }] });
  } catch {
    // Media emulation is optional.
  }

  for (const file of written) process.stdout.write(`wrote ${file}\n`);
} finally {
  client.close();
}
