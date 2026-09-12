import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';
import { analyzeDataset } from '../lib/analysis/index.js';
import { buildViewModel } from '../lib/view.js';
import { documentationBundle } from '../lib/docs.js';

const here = dirname(fileURLToPath(import.meta.url));
const webRoot = join(here, '..', 'web');
const fixtureDir = join(here, 'fixtures', 'host-schedule-bound');

/**
 * A deliberately small DOM implementation.
 *
 * It is not a browser: it implements exactly the surface the analyzer page uses
 * (element creation, class/dataset/attribute access, append/replaceChildren,
 * text, the 2D canvas context, and `getComputedStyle`). That is enough to run
 * the real render functions and catch integration mistakes — a missing helper,
 * a wrong property, an undefined id — which is the failure mode that would
 * otherwise only show up as a blank page in the browser.
 */
class FakeNode {
  constructor(tag) {
    this.tagName = tag;
    this.children = [];
    this.parentElement = undefined;
    this.listenerCount = 0;
    this.hidden = false;
    this.disabled = false;
    this.value = '';
    this.style = { setProperty() {}, };
    this.dataset = {};
    this._attributes = new Map();
    this._text = '';
    this._classes = new Set();
    this.classList = {
      add: (...names) => names.forEach((name) => this._classes.add(name)),
      remove: (...names) => names.forEach((name) => this._classes.delete(name)),
      contains: (name) => this._classes.has(name),
      toggle: (name, force) => {
        const on = force === undefined ? !this._classes.has(name) : force;
        if (on) this._classes.add(name);
        else this._classes.delete(name);
        return on;
      },
    };
  }

  set id(value) {
    this._id = value;
    if (value !== undefined) registry.set(value, this);
  }

  get id() {
    return this._id ?? '';
  }

  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }

  get textContent() {
    return this._text + this.children.map((child) => child.textContent).join('');
  }

  set className(value) {
    this._classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  get className() {
    return [...this._classes].join(' ');
  }

  setAttribute(name, value) {
    this._attributes.set(name, String(value));
    if (name === 'id') this.id = String(value);
  }

  getAttribute(name) {
    return this._attributes.get(name) ?? null;
  }

  append(...nodes) {
    for (const node of nodes) {
      if (node === undefined || node === null) continue;
      this.children.push(node);
      if (node instanceof FakeNode) node.parentElement = this;
    }
  }

  replaceChildren(...nodes) {
    this.children = [];
    this._text = '';
    this.append(...nodes);
  }

  addEventListener() {
    this.listenerCount += 1;
  }

  removeEventListener() {}

  setPointerCapture() {}

  getBoundingClientRect() {
    return { left: 0, top: 0, width: 900, height: 520, right: 900, bottom: 520 };
  }

  scrollIntoView() {}

  querySelectorAll(selector) {
    const attribute = /^\[([\w-]+)\]$/.exec(selector);
    const matches = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (!(child instanceof FakeNode)) continue;
        if (attribute !== null && child.getAttribute(attribute[1]) !== null) matches.push(child);
        walk(child);
      }
    };
    walk(this);
    return matches;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }
}

/** Id registry backing `document.getElementById`. */
const registry = new Map();

/** 2D context recorder: every call is accepted and counted. */
function fakeContext() {
  const calls = { fillRect: 0, strokeRect: 0, fillText: 0, stroke: 0, beginPath: 0, clearRect: 0 };
  return {
    calls,
    setTransform() {},
    clearRect() { calls.clearRect += 1; },
    fillRect() { calls.fillRect += 1; },
    strokeRect() { calls.strokeRect += 1; },
    fillText() { calls.fillText += 1; },
    stroke() { calls.stroke += 1; },
    beginPath() { calls.beginPath += 1; },
    moveTo() {},
    lineTo() {},
    rect() {},
    clip() {},
    save() {},
    restore() {},
    translate() {},
    setLineDash() {},
    measureText(text) { return { width: String(text).length * 6 }; },
    set font(value) {},
    get font() { return ''; },
    set fillStyle(value) {},
    get fillStyle() { return ''; },
    set strokeStyle(value) {},
    get strokeStyle() { return ''; },
    set globalAlpha(value) {},
    get globalAlpha() { return 1; },
    set lineWidth(value) {},
    get lineWidth() { return 1; },
  };
}

/** Load the page scripts into a browser-like sandbox, in document order. */
function loadPage({ withHtmlIds = false, fetchImpl } = {}) {
  const documentStub = {
    body: new FakeNode('body'),
    head: new FakeNode('head'),
    createElement: (tag) => {
      if (tag === 'canvas') {
        const canvas = new FakeNode('canvas');
        canvas.width = 0;
        canvas.height = 0;
        canvas.clientWidth = 900;
        canvas.getContext = () => canvas._context ?? (canvas._context = fakeContext());
        return canvas;
      }
      return new FakeNode(tag);
    },
    createElementNS: (_namespace, tag) => new FakeNode(tag),
    createTextNode: (text) => ({ textContent: String(text), nodeType: 3 }),
    getElementById: (id) => registry.get(id) ?? null,
    querySelector: () => null,
    querySelectorAll: () => [],
    _listeners: [],
    addEventListener(type, handler) {
      this._listeners.push({ type, handler });
    },
  };
  if (withHtmlIds) {
    // Materialize every element the page declares, so the controller's element
    // cache resolves exactly as it does in a browser.
    const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
    for (const match of html.matchAll(/id="([^"]+)"/g)) {
      const node = match[1].includes('canvas') ? documentStub.createElement('canvas') : new FakeNode('div');
      node.id = match[1];
    }
    // The legend's parent must expose the category chips bindControls queries.
    const legend = registry.get('gantt-legend');
    const chipsParent = new FakeNode('div');
    for (const category of ['compute', 'comm', 'copy', 'schedule', 'other']) {
      const chip = new FakeNode('button');
      chip.setAttribute('data-cat', category);
      chipsParent.append(chip);
    }
    chipsParent.append(legend);
    // A tooltip card needs a measurable box for positioning.
    registry.get('gantt-tooltip').getBoundingClientRect = () => ({ left: 0, top: 0, width: 300, height: 200, right: 300, bottom: 200 });
  }
  const sandbox = {
    document: documentStub,
    console,
    setTimeout,
    clearTimeout,
    devicePixelRatio: 1,
    location: { origin: 'http://127.0.0.1:3080', pathname: '/vllm-ascend-profiler/', href: '' },
    open() {},
    addEventListener() {},
    getComputedStyle: () => ({ getPropertyValue: () => '' }),
    Node: FakeNode,
    XMLHttpRequest: class {
      open() {}
      setRequestHeader() {}
      addEventListener() {}
      send() {}
    },
    fetch: fetchImpl ?? (async () => ({ ok: true, json: async () => ({}) })),
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  vm.createContext(sandbox);
  for (const file of ['util.js', 'api.js', 'charts.js', 'gantt.js', 'docs-view.js', 'advice-view.js', 'app.js']) {
    const source = readFileSync(join(webRoot, file), 'utf8');
    vm.runInContext(source, sandbox, { filename: file });
  }
  return sandbox;
}

/** Build a real view model from the host-schedule fixture. */
async function loadViewModel() {
  const inputs = readdirSync(fixtureDir).map((name) => ({ name, buffer: readFileSync(join(fixtureDir, name)) }));
  const parse = await parseProfileSet({ inputs });
  assert.equal(parse.ok, true);
  const dataset = buildDataset(parse);
  const analysis = analyzeDataset(dataset);
  return buildViewModel({ dataset, analysis, datasetId: 'test-dataset', label: 'host-schedule-bound' });
}

test('every page script loads in document order without throwing', () => {
  const sandbox = loadPage();
  assert.equal(typeof sandbox.VAP.h, 'function');
  assert.equal(typeof sandbox.VAP.charts.renderPie, 'function');
  assert.equal(typeof sandbox.VAP.GanttView, 'function');
  assert.equal(typeof sandbox.VAP.advice.renderAdvice, 'function');
  assert.equal(typeof sandbox.VAP.docsView.renderDocs, 'function');
});

test('module 2 renders the donut, the ranked bars, and the table', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();

  for (const scope of ['all', 'device', 'host']) {
    const categories = sandbox.VAP.charts.buildCategories({ dataset: viewModel, scope });
    assert.ok(categories.length > 0, `scope ${scope} must produce categories`);
    const pie = sandbox.VAP.charts.renderPie({ items: categories, totalUs: viewModel.categories.totalUs, scopeLabel: scope });
    assert.equal(pie.element.tagName, 'svg');
    assert.ok(pie.element.children.length >= categories.length, 'one slice per category');
    assert.ok(pie.legend.children.length === categories.length);
  }

  for (const dimension of ['total', 'average']) {
    const ranking = sandbox.VAP.charts.buildRanking({ dataset: viewModel, dimension, scope: 'all', topN: 20 });
    assert.ok(ranking.rows.length > 0);
    assert.equal(ranking.rows[0].rank, 1);
    assert.ok(ranking.rows[0].shareOfOpsPct > 0);
    const bars = sandbox.VAP.charts.renderBars({ rows: ranking.rows, dimension, maxRows: 20 });
    assert.equal(bars.element.tagName, 'svg');
    assert.ok(bars.element.children.length > ranking.rows.length, 'axis labels plus one group per row');
    const table = sandbox.VAP.charts.renderRankingTable(ranking.rows);
    assert.equal(table.tagName, 'table');
    assert.equal(table.children[1].children.length, ranking.rows.length);
  }
});

test('module 1 renders lanes for both device groups and reports sampling', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const canvas = sandbox.document.createElement('canvas');
  const tooltip = new FakeNode('div');
  const cursor = new FakeNode('span');
  const selection = new FakeNode('span');
  const view = new sandbox.VAP.GanttView({ canvas, tooltip, cursorLabel: cursor, selectionLabel: selection });
  view.setData(viewModel);

  assert.equal(view.rows.filter((entry) => entry.kind === 'header').length, viewModel.timeline.lanes.length);
  assert.equal(view.rows.filter((entry) => entry.kind === 'row').length > 0, true);
  assert.ok(canvas._context.calls.fillRect > 0, 'bars must be painted');
  assert.ok(canvas._context.calls.fillText > 0, 'labels must be painted');

  // The layout must be bounded by the row limit and total row count.
  view.setOptions({ rowLimit: 5 });
  assert.equal(view.rows.filter((entry) => entry.kind === 'row').length <= viewModel.timeline.lanes.length * 5, true);

  // Zoom, pan and reset must stay inside the window.
  view.zoom(1 / 4);
  assert.ok(view.viewEnd - view.viewStart < viewModel.meta.window.end - viewModel.meta.window.start);
  view.resetView();
  assert.equal(view.viewStart, viewModel.meta.window.start);
  assert.equal(view.viewEnd, viewModel.meta.window.end);

  // Filtering by an operator must keep only its row and report the selection.
  const firstName = viewModel.timeline.lanes[0].rows[0].name;
  view.filterBy(firstName);
  assert.match(selection.textContent, /仅显示算子/);
  const rowsAfterFilter = view.rows.filter((entry) => entry.kind === 'row');
  assert.ok(rowsAfterFilter.every((entry) => entry.row.name === firstName));
  view.filterBy(undefined);
  assert.match(selection.textContent, /未按算子筛选/);

  // Hiding a group must remove its header row.
  view.toggleGroup('device');
  assert.equal(view.rows.some((entry) => entry.kind === 'header' && entry.group === 'device'), false);
});

test('module 3 renders the whole five-step chain with evidence and gains', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const root = sandbox.VAP.advice.renderAdvice(viewModel, { priorityFilter: 'all' });
  assert.equal(root.children.length >= 5, true, 'five chain steps');
  const text = root.textContent;
  assert.match(text, /瓶颈类型定位/);
  assert.match(text, /量化证据/);
  assert.match(text, /根因推断/);
  assert.match(text, /可落地优化方案/);
  assert.match(text, /预期收益/);
  assert.match(text, /Host 独占/);
  assert.match(text, /门限/);
  assert.match(text, /置信度/);
  assert.ok(!text.includes('undefined'), 'no undefined must leak into the rendered advice');
  assert.ok(!text.includes('NaN'));

  // The recommendation cards must be real elements with the `advice` class: if
  // the tag builder fails to parse a class token (for example a CJK priority in
  // the class name) the card becomes an HTMLUnknownElement that no stylesheet
  // rule matches, and the section silently loses its layout.
  const cards = [];
  const walk = (node) => {
    for (const child of node.children ?? []) {
      if (child.classList?.contains('advice')) cards.push(child);
      walk(child);
    }
  };
  walk(root);
  assert.ok(cards.length > 0, 'at least one recommendation card must render');
  for (const card of cards) {
    assert.equal(card.tagName, 'div', 'a recommendation card must be a real div');
    assert.match(card.className, /advice pri-(high|mid|low)/, `unexpected card classes: ${card.className}`);
    assert.ok(['高', '中', '低'].includes(card.dataset.priority), 'the Chinese priority must be carried as data, not as a class');
  }

  const highOnly = sandbox.VAP.advice.renderAdvice(viewModel, { priorityFilter: '高' });
  assert.ok(highOnly.textContent.length <= root.textContent.length);
});

test('the documentation panel renders metrics and artifact fields', async () => {
  const sandbox = loadPage();
  const root = sandbox.VAP.docsView.renderDocs(documentationBundle({ version: '1.0.0' }));
  const text = root.textContent;
  assert.match(text, /指标含义与计算口径/);
  assert.match(text, /trace_view.json/);
  assert.match(text, /NPU 忙碌率/);
  assert.match(text, /裸 JSON 数组/);
  assert.ok(!text.includes('undefined'));
});

test('the advice view renders for every scenario without undefined leakage', async () => {
  const sandbox = loadPage();
  for (const scenario of ['decode-comm-bound', 'prefill-compute-bound']) {
    const directory = join(here, 'fixtures', scenario);
    const inputs = readdirSync(directory).map((name) => ({ name, buffer: readFileSync(join(directory, name)) }));
    const parse = await parseProfileSet({ inputs });
    const dataset = buildDataset(parse);
    const viewModel = buildViewModel({
      dataset,
      analysis: analyzeDataset(dataset),
      datasetId: scenario,
      label: scenario,
    });
    const root = sandbox.VAP.advice.renderAdvice(viewModel, {});
    assert.ok(root.textContent.length > 500, `${scenario}: advice must be substantial`);
    assert.ok(!root.textContent.includes('undefined'), `${scenario}: no undefined`);
    assert.ok(!root.textContent.includes('NaN'), `${scenario}: no NaN`);
  }
});

test('the page HTML references only assets the plugin serves', () => {
  const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
  const served = new Set(readdirSync(webRoot));
  const referenced = [...html.matchAll(/(?:src|href)="([^"]+)"/g)].map((match) => match[1]);
  assert.ok(referenced.length > 0);
  for (const asset of referenced) {
    if (asset.startsWith('http') || asset.startsWith('#')) continue;
    assert.ok(served.has(asset), `index.html references ${asset}, which the plugin does not serve`);
  }
  for (const id of ['dropzone', 'gantt-canvas', 'pie-host', 'bar-host', 'ranking-host', 'advice-chain', 'docs-body']) {
    assert.ok(html.includes(`id="${id}"`), `index.html must define #${id}`);
  }
});

test('the stylesheet forces the hidden attribute to win over layout rules', () => {
  // `.modal` declares `display: grid`, which overrides the UA rule for `[hidden]`
  // and would leave the dialog permanently on screen.
  const css = readFileSync(join(webRoot, 'styles.css'), 'utf8');
  assert.match(css, /\[hidden\]\s*\{\s*display:\s*none\s*!important;?\s*\}/, 'a [hidden] rule with !important is required');
  const hiddenElements = ['modal', 'progress-wrap', 'module-gantt', 'module-share', 'module-advice', 'gantt-tooltip'];
  const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
  for (const id of hiddenElements) {
    assert.match(html, new RegExp(`id="${id}"[^>]*hidden`), `#${id} must start hidden`);
  }
});

test('every element id the controller caches exists in index.html', () => {
  // A cached id that the document does not define yields `undefined` and blows up
  // later inside an event binding — a runtime failure no unit test of the
  // renderers would catch. This check is static and catches drift immediately.
  const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
  const declared = new Set([...html.matchAll(/id="([^"]+)"/g)].map((match) => match[1]));
  const source = readFileSync(join(webRoot, 'app.js'), 'utf8');
  const listMatch = /const ids = \[([\s\S]*?)\];/.exec(source);
  assert.ok(listMatch !== null, 'the controller must declare its cached element list');
  const ids = [...listMatch[1].matchAll(/'([^']+)'/g)].map((match) => match[1]);
  assert.ok(ids.length > 40, `expected a substantial id list, got ${String(ids.length)}`);
  const missing = ids.filter((id) => !declared.has(id));
  assert.deepEqual(missing, [], `controller caches ids missing from index.html: ${missing.join(', ')}`);
});

test('the controller initializes against the real page markup', async () => {
  // Drives `init()` with every declared element present, which is what exercises
  // cacheElements/bindIntake/bindControls — the code paths that only fail in a
  // real document.
  const bundle = documentationBundle({ version: '1.0.0' });
  const sandbox = loadPage({
    withHtmlIds: true,
    fetchImpl: async (url) => {
      const path = String(url);
      if (path.includes('/api/docs')) return { ok: true, json: async () => bundle };
      if (path.includes('/api/health')) {
        return { ok: true, json: async () => ({ ok: true, plugin: 'vllm-ascend-profiler', version: '1.0.0', store: { datasets: 0, jobs: 0, maxDatasets: 6 }, limits: { maxUploadBytes: 1000, maxInMemoryBytes: 500, allowPathIngest: true } }) };
      }
      if (path.includes('/api/datasets')) return { ok: true, json: async () => ({ datasets: [] }) };
      return { ok: true, json: async () => ({}) };
    },
  });
  const listeners = sandbox.document._listeners.filter((entry) => entry.type === 'DOMContentLoaded');
  assert.equal(listeners.length, 1, 'the controller registers one DOMContentLoaded handler');
  await listeners[0].handler();
  // The Gantt view must have been constructed (its canvas was sized) and the
  // health line filled in by the successful API stubs.
  const canvas = registry.get('gantt-canvas');
  assert.ok(canvas.width > 0, 'the swimlane canvas must be sized during init');
  assert.match(registry.get('health-line').textContent, /服务正常/);
  assert.equal(registry.get('dataset-list-wrap').hidden, true, 'an empty dataset list hides its panel');
});

test('init survives an unreachable API without breaking the page', async () => {
  const sandbox = loadPage({
    withHtmlIds: true,
    fetchImpl: async () => {
      throw new Error('network down');
    },
  });
  const handler = sandbox.document._listeners.find((entry) => entry.type === 'DOMContentLoaded');
  // `loadHealth` reports the failure in the footer; the rest of init must still run.
  await assert.doesNotReject(async () => handler.handler());
  assert.match(registry.get('health-line').textContent, /健康检查失败/);
  assert.ok(registry.get('gantt-canvas').width > 0);
});
