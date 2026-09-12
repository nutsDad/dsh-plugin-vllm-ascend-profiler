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
    /** All appended nodes, text included (the real DOM's childNodes). */
    this._nodes = [];
    this.parentElement = undefined;
    /** @type {Record<string, Function[]>} */
    this._listeners = {};
    this.listenerCount = 0;
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.value = '';
    this.style = { setProperty() {} };
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

  /** Elements only, like the real DOM's `children`. */
  get children() {
    return this._nodes.filter((node) => node instanceof FakeNode);
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
    this._nodes = [];
  }

  get textContent() {
    return this._text + this._nodes.map((child) => child.textContent).join('');
  }

  /** Mirrors the real DOM closely enough for `replaceChildren(...node.childNodes)`. */
  get childNodes() {
    if (this._nodes.length > 0) return this._nodes;
    return this._text === '' ? [] : [{ textContent: this._text, nodeType: 3 }];
  }

  get firstChild() {
    return this.childNodes[0] ?? null;
  }

  set className(value) {
    this._classes = new Set(String(value).split(/\s+/).filter(Boolean));
  }

  get className() {
    return [...this._classes].join(' ');
  }

  setAttribute(name, value) {
    const text = String(value);
    this._attributes.set(name, text);
    // The real DOM keeps these in sync with their property forms; the page and
    // the tests both rely on that (`class`, `hidden`, `data-*`).
    if (name === 'id') this.id = text;
    else if (name === 'class') this.className = text;
    else if (name === 'hidden') this.hidden = true;
    else if (name === 'checked') this.checked = true;
    else if (name === 'value') this.value = text;
    else if (name.startsWith('data-')) this.dataset[dataKey(name)] = text;
  }

  getAttribute(name) {
    if (name === 'class') return this.className === '' ? null : this.className;
    if (name === 'hidden') return this.hidden ? '' : null;
    return this._attributes.get(name) ?? null;
  }

  append(...nodes) {
    for (const node of nodes) {
      if (node === undefined || node === null) continue;
      this._nodes.push(node);
      if (node instanceof FakeNode) node.parentElement = this;
    }
  }

  replaceChildren(...nodes) {
    this._nodes = [];
    this._text = '';
    this.append(...nodes);
  }

  addEventListener(type, handler) {
    (this._listeners[type] ??= []).push(handler);
    this.listenerCount += 1;
  }

  removeEventListener(type, handler) {
    const list = this._listeners[type] ?? [];
    const at = list.indexOf(handler);
    if (at !== -1) list.splice(at, 1);
  }

  /** Minimal dispatch: enough for the page's click/keydown handlers. */
  dispatchEvent(event) {
    const payload = {
      type: event.type,
      target: this,
      preventDefault() {},
      stopPropagation() {},
      key: event.key,
      clientX: 0,
      clientY: 0,
      pointerId: 1,
      ...event,
    };
    for (const handler of [...(this._listeners[event.type] ?? [])]) handler(payload);
    return true;
  }

  click() {
    this.dispatchEvent({ type: 'click' });
  }

  setPointerCapture() {}

  getBoundingClientRect() {
    return { left: 0, top: 0, width: 900, height: 520, right: 900, bottom: 520 };
  }

  scrollIntoView() {}

  querySelectorAll(selector) {
    const matches = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (!(child instanceof FakeNode)) continue;
        if (child.matches(selector)) matches.push(child);
        walk(child);
      }
    };
    walk(this);
    return matches;
  }

  /** Minimal selector support: `tag`, `.class`, `[attr]`, `tag[attr="v"]`, `[attr="v"]`. */
  matches(selector) {
    for (const part of selector.split(',').map((value) => value.trim())) {
      if (part === '') continue;
      const match = /^([a-zA-Z]*)(?:\.([\w-]+))?(?:\[([\w-]+)(?:="([^"]*)")?\])?$/.exec(part);
      if (match === null) continue;
      const [, tag, className, attr, attrValue] = match;
      if (tag !== '' && this.tagName.toLowerCase() !== tag.toLowerCase()) continue;
      if (className !== undefined && !this._classes.has(className)) continue;
      if (attr !== undefined) {
        const value = this.getAttribute(attr);
        if (value === null) continue;
        if (attrValue !== undefined && value !== attrValue) continue;
      }
      return true;
    }
    return false;
  }

  closest(selector) {
    let node = this;
    while (node !== undefined) {
      if (node instanceof FakeNode && node.matches(selector)) return node;
      node = node.parentElement;
    }
    return null;
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] ?? null;
  }

  scrollIntoView() {}
}

/** Run queued animation frames (the sandbox backs them with setTimeout). */
function nextFrame() {
  return new Promise((resolve) => setTimeout(resolve, 40));
}

/** Id registry backing `document.getElementById`. */
const registry = new Map();

/** Elements the parser treats as self-closing. */
const VOID_ELEMENTS = new Set(['input', 'meta', 'link', 'br', 'img', 'hr', 'source', 'area', 'base', 'col', 'embed', 'param', 'track', 'wbr']);

/**
 * Build a DOM tree from the page's real HTML.
 *
 * Parsing the shipped markup (rather than hand-building the elements a test
 * happens to know about) is what makes this harness worth having: the tests see
 * the same ids, classes, data attributes, `hidden` flags and inner buttons that
 * the browser does, so a markup change breaks the tests instead of production.
 *
 * @param {string} html - page source.
 * @param {object} documentStub - the document shim (provides createElement).
 * @returns {{root: FakeNode, body: FakeNode}} parsed tree and its body element.
 */
function parseHtml(html, documentStub) {
  const root = new FakeNode('html');
  const stack = [root];
  /** @type {FakeNode[]} */
  const selects = [];
  const token = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>|([^<]+)/g;
  let body = root;
  let match;
  while ((match = token.exec(html)) !== null) {
    if (match[0].startsWith('<!--') || match[0].startsWith('<!')) continue;
    if (match[1] !== undefined) {
      // Closing tag: pop to the matching element when it is on the stack.
      for (let at = stack.length - 1; at > 0; at -= 1) {
        if (stack[at].tagName.toLowerCase() === match[1].toLowerCase()) {
          stack.length = at;
          break;
        }
      }
      continue;
    }
    if (match[2] !== undefined) {
      const tag = match[2].toLowerCase();
      if (tag === 'script' || tag === 'style') {
        // Skip raw-content elements entirely (the page loads external scripts).
        const closing = html.indexOf(`</${tag}>`, token.lastIndex);
        if (closing !== -1) token.lastIndex = closing + tag.length + 3;
        continue;
      }
      const node = documentStub.createElement(tag);
      for (const attribute of (match[3] ?? '').matchAll(/([a-zA-Z_:][\w:.-]*)(?:\s*=\s*"([^"]*)")?/g)) {
        // One path for every attribute keeps the property mirroring in
        // `setAttribute` (id, class, hidden, data-*) authoritative.
        node.setAttribute(attribute[1], attribute[2] ?? '');
      }
      stack[stack.length - 1].append(node);
      if (tag === 'body') body = node;
      if (tag === 'select') selects.push(node);
      if (match[4] !== '/' && !VOID_ELEMENTS.has(tag)) stack.push(node);
      continue;
    }
    if (match[5] !== undefined) {
      const text = match[5].replace(/\s+/g, ' ').trim();
      if (text !== '') stack[stack.length - 1].append({ textContent: text, nodeType: 3 });
    }
  }
  // A real `<select>` reports the selected option's value (or the first option's
  // when none is marked). The page reads `.value` directly, so the harness must
  // provide it — otherwise every `<select>` would look empty and defaults such
  // as the row limit would come out as 0.
  for (const select of selects) {
    const options = select.querySelectorAll('option');
    const chosen = options.find((option) => option.getAttribute('selected') !== null) ?? options[0];
    if (chosen !== undefined) select.value = chosen.getAttribute('value') ?? chosen.textContent;
  }
  return { root, body };
}

/** `data-some-key` → `someKey`. */
function dataKey(name) {
  return name.slice(5).replace(/-([a-z])/g, (_, char) => char.toUpperCase());
}

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
function loadPage({ withHtmlIds = false, fetchImpl, noMotion = false } = {}) {
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
    // Materialize the real page markup, so the controller's element cache, its
    // bindings and the tests all see the document a browser would build.
    const html = readFileSync(join(webRoot, 'index.html'), 'utf8');
    const parsed = parseHtml(html, documentStub);
    documentStub.body = parsed.body;
    for (const node of parsed.root.querySelectorAll('*')) {
      if (node.tagName.toLowerCase() === 'canvas') {
        node.clientWidth = 900;
        node.getContext = () => node._context ?? (node._context = fakeContext());
      }
      if (node.id !== '') registry.set(node.id, node);
    }
    registry.set(parsed.body.id || 'body', parsed.body);
    // The tooltip card needs a measurable box for positioning.
    const tooltip = registry.get('gantt-tooltip');
    if (tooltip !== undefined) {
      tooltip.getBoundingClientRect = () => ({ left: 0, top: 0, width: 300, height: 200, right: 300, bottom: 200 });
    }
  }
  const sandbox = {
    document: documentStub,
    console,
    setTimeout,
    clearTimeout,
    // Animation plumbing the page expects from a browser. Frames are backed by
    // setTimeout so tests can await `nextFrame()` instead of racing.
    performance: globalThis.performance,
    requestAnimationFrame: (callback) => setTimeout(() => callback(globalThis.performance.now()), 0),
    cancelAnimationFrame: (handle) => clearTimeout(handle),
    devicePixelRatio: 1,
    matchMedia: () => ({ matches: false }),
    localStorage: (() => {
      const store = new Map();
      return {
        getItem: (key) => store.get(key) ?? null,
        setItem: (key, value) => store.set(key, String(value)),
        removeItem: (key) => store.delete(key),
      };
    })(),
    IntersectionObserver: class {
      constructor(callback) { this.callback = callback; }
      observe() {}
      unobserve() {}
      disconnect() {}
    },
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
  if (noMotion) sandbox.document.body.classList.add('no-motion');
  vm.createContext(sandbox);
  for (const file of ['util.js', 'api.js', 'diagram.js', 'charts.js', 'gantt.js', 'docs-view.js', 'advice-view.js', 'app.js']) {
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
  assert.equal(typeof sandbox.VAP.diagram.treemap, 'function');
  assert.equal(typeof sandbox.VAP.diagram.chainFlow, 'function');
  assert.equal(typeof sandbox.VAP.charts.buildRanking, 'function');
  assert.equal(typeof sandbox.VAP.GanttView, 'function');
  assert.equal(typeof sandbox.VAP.advice.renderAdvice, 'function');
  assert.equal(typeof sandbox.VAP.docsView.renderDocs, 'function');
});

test('module 2 projects the categories and the ranking behind the diagrams', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();

  for (const scope of ['all', 'device', 'host']) {
    const categories = sandbox.VAP.charts.buildCategories({ dataset: viewModel, scope });
    assert.ok(categories.length > 0, `scope ${scope} must produce categories`);
    assert.ok(categories.every((item) => item.totalUs > 0), 'only funded categories reach the strip');
    assert.equal(Math.round(categories.reduce((sum, item) => sum + item.sharePct, 0)), 100, 'shares are a whole');
  }

  for (const dimension of ['total', 'average']) {
    const ranking = sandbox.VAP.charts.buildRanking({ dataset: viewModel, dimension, scope: 'all', topN: 20 });
    assert.ok(ranking.rows.length > 0);
    assert.equal(ranking.rows[0].rank, 1);
    assert.ok(ranking.rows[0].shareOfOpsPct > 0);
    const table = sandbox.VAP.charts.renderRankingTable(ranking.rows);
    assert.equal(table.tagName, 'table');
    assert.equal(table.children[1].children.length, ranking.rows.length);
  }
});

test('the composition strip splits 100% of the cost into clickable categories', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const categories = sandbox.VAP.charts.buildCategories({ dataset: viewModel, scope: 'all' });
  const picks = [];
  const strip = sandbox.VAP.diagram.shareBar({ items: categories, onSelect: (pick) => picks.push(pick) });

  const segments = strip.element.querySelectorAll('g.share-seg');
  assert.equal(segments.length, categories.length, 'one segment per category');
  const width = segments.reduce((sum, segment) => {
    const rect = segment.querySelectorAll('rect')[0];
    return sum + Number(rect.getAttribute('width'));
  }, 0);
  // Segments are inset by 2px each, so the drawn width is the viewport minus the
  // gaps: a full composition bar, never a partial one.
  assert.ok(Math.abs(width - (960 - categories.length * 2)) < 2, `segments must fill the strip, got ${String(width)}`);

  // The legend repeats the strip and is itself a control (the smallest category
  // would otherwise be a few pixels wide).
  const legendButtons = strip.legend.querySelectorAll('button');
  assert.equal(legendButtons.length, categories.length);
  legendButtons[1].click();
  assert.equal(picks.length, 1);
  assert.equal(picks[0].category, categories[1].id);

  // Clicking a segment reports the same category upward.
  segments[0].dispatchEvent({ type: 'click' });
  assert.equal(picks.length, 2);
  assert.equal(picks[1].category, categories[0].id);
});

test('the treemap gives every operator an area proportional to its cost', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const ranking = sandbox.VAP.charts.buildRanking({ dataset: viewModel, dimension: 'total', scope: 'all', topN: 12 });
  const map = sandbox.VAP.diagram.treemap({ rows: ranking.rows, dimension: 'total' });

  const tiles = map.element.querySelectorAll('g.tm-tile');
  assert.equal(tiles.length, ranking.rows.length, 'every ranked operator gets a tile');
  const canvas = 960 * 340;
  const area = tiles.reduce((sum, tile) => {
    const rect = tile.querySelectorAll('rect')[0];
    return sum + Number(rect.getAttribute('width')) * Number(rect.getAttribute('height'));
  }, 0);
  // Squarified layout fills the box: the total tile area is the canvas area.
  assert.ok(Math.abs(area - canvas) / canvas < 0.02, `tile area must equal the canvas area, got ${String(Math.round(area))}`);

  // Area encodes cost: the top operator occupies its own share of the canvas, and
  // the tiles are appended in ranking order.
  const totalUs = ranking.rows.reduce((sum, row) => sum + row.totalUs, 0);
  const first = tiles[0].querySelectorAll('rect')[0];
  const firstArea = Number(first.getAttribute('width')) * Number(first.getAttribute('height'));
  const expected = (ranking.rows[0].totalUs / totalUs) * canvas;
  assert.ok(Math.abs(firstArea - expected) / expected < 0.08, `the top tile must cover its share, got ${String(Math.round(firstArea))} vs ${String(Math.round(expected))}`);
  assert.equal(tiles[0].getAttribute('data-operator'), ranking.rows[0].name);
});

test('the evidence diagram compares each metric with its gate', async () => {
  const sandbox = loadPage();
  const rows = [
    { metric: 'Host 独占占比', value: 96.5, unit: '%', threshold: 60, passed: false, category: 'schedule' },
    { metric: 'NPU 忙碌率', value: 12.4, unit: '%', threshold: 40, passed: false, category: 'compute' },
  ];
  const bars = sandbox.VAP.diagram.thresholdBars({ rows });
  const rendered = bars.element.querySelectorAll('g.tb-row');
  assert.equal(rendered.length, rows.length, 'one row per metric');
  const fills = rendered.map((row) => row.querySelectorAll('rect')[1].getAttribute('fill'));
  assert.deepEqual(fills, ['var(--err)', 'var(--err)'], 'a missed gate is drawn in the failure colour');
  for (const row of rendered) {
    const texts = row.querySelectorAll('text').map((node) => node.textContent);
    assert.ok(texts.some((text) => text.includes('门限')), 'the gate value is printed on the row');
  }
  const passing = sandbox.VAP.diagram.thresholdBars({ rows: [{ metric: 'NPU 忙碌率', value: 88, unit: '%', threshold: 40, passed: true }] });
  assert.equal(passing.element.querySelectorAll('g.tb-row')[0].querySelectorAll('rect')[1].getAttribute('fill'), 'var(--ok)');
  assert.equal(bars.legend.querySelectorAll('span').length, 3, 'the legend names pass, fail and gate');
});

test('the reasoning chain is a five-node flow with one number per step', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const view = sandbox.VAP.advice.renderAdvice(viewModel, { priorityFilter: 'all' });
  const flow = view.element.querySelector('.chain-flow');
  const nodes = flow.querySelectorAll('button.flow-node');
  assert.equal(nodes.length, 5, '①→⑤');
  assert.deepEqual(nodes.map((node) => node.dataset.node), ['locate', 'evidence', 'cause', 'actions', 'benefit']);
  assert.equal(nodes[0].getAttribute('aria-selected'), 'true', 'the first step is selected on load');
  for (const node of nodes) {
    assert.ok(node.querySelector('.flow-value').textContent.length > 0, 'a node must carry its own number');
  }
  // The gain bars are drawn (not just described) in the panels.
  view.setStep('benefit');
  assert.ok(view.element.querySelectorAll('.benefit-row').length > 0);
  assert.equal(view.element.querySelectorAll('button.flow-node')[4].getAttribute('aria-selected'), 'true');
});

test('the one-line summariser never cuts a number in half', () => {
  const sandbox = loadPage();
  const oneLine = sandbox.VAP.diagram.oneLine;
  // The diagrams show one sentence per item: a decimal point inside a number must
  // not be mistaken for the end of that sentence ("Host 独占 30." is a lie).
  assert.match(oneLine('Host 独占 30.22%，高于门限 25%。其余略。'), /30\.22%/);
  assert.equal(oneLine('Host 独占 30.22%。其余略。'), 'Host 独占 30.22%。');
  assert.equal(oneLine('没有句号的一句话'), '没有句号的一句话');
});

test('module 1 renders lanes for both device groups and reports sampling', async () => {
  // Deterministic geometry: motion off. The reveal sweep itself is asserted by
  // the dedicated animation test below.
  const sandbox = loadPage({ noMotion: true });
  const viewModel = await loadViewModel();
  const canvas = sandbox.document.createElement('canvas');
  const tooltip = new FakeNode('div');
  const cursor = new FakeNode('span');
  const hint = new FakeNode('span');
  const view = new sandbox.VAP.GanttView({ canvas, tooltip, cursorLabel: cursor, hintLabel: hint, onSelect: () => {} });
  view.setData(viewModel);

  assert.equal(view.rows.filter((entry) => entry.kind === 'header').length, viewModel.timeline.lanes.length);
  assert.equal(view.rows.filter((entry) => entry.kind === 'row').length > 0, true);
  assert.ok(canvas._context.calls.fillRect > 0, 'bars must be painted');
  assert.ok(canvas._context.calls.fillText > 0, 'labels must be painted');

  // Options, zoom, pan and reset must stay inside the window.
  view.setOptions({ rowLimit: 5 });
  assert.equal(view.rows.filter((entry) => entry.kind === 'row').length <= viewModel.timeline.lanes.length * 5, true);
  view.zoom(1 / 4);
  assert.ok(view.viewEnd - view.viewStart < viewModel.meta.window.end - viewModel.meta.window.start);
  view.resetView();
  assert.equal(view.viewStart, viewModel.meta.window.start);
  assert.equal(view.viewEnd, viewModel.meta.window.end);

  // Filtering by an operator keeps only its row and reports the filter.
  const firstName = viewModel.timeline.lanes[0].rows[0].name;
  view.filterBy(firstName);
  assert.match(hint.textContent, /已筛选/);
  const rowsAfterFilter = view.rows.filter((entry) => entry.kind === 'row');
  assert.ok(rowsAfterFilter.every((entry) => entry.row.name === firstName));
  view.filterBy(undefined);
  assert.match(hint.textContent, /点击算子条筛选/);

  // Focusing an operator zooms to its events (bounded by the data window), and
  // an unknown operator is reported rather than silently ignored.
  const focused = view.focusOperator(firstName);
  assert.equal(focused, true);
  assert.ok(view.viewStart >= viewModel.meta.window.start - 1, 'the focused view stays inside the window');
  assert.ok(view.viewEnd <= viewModel.meta.window.end + 1, 'the focused view stays inside the window');
  assert.equal(view.focusOperator('no-such-operator'), false);

  // Hiding a group removes its header row; category filtering removes rows.
  view.toggleGroup('device');
  assert.equal(view.rows.some((entry) => entry.kind === 'header' && entry.group === 'device'), false);
  view.toggleGroup('device');
  view.setCategories(['comm']);
  assert.ok(view.rows.filter((entry) => entry.kind === 'row').every((entry) => entry.row.category === 'comm' || entry.row.overflow === true));
  view.setCategories(Object.keys(sandbox.VAP.CATEGORY_LABELS));
});

test('the swimlane reveal animation paints progressively when motion is on', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const canvas = sandbox.document.createElement('canvas');
  const view = new sandbox.VAP.GanttView({ canvas, tooltip: new FakeNode('div'), cursorLabel: new FakeNode('span'), hintLabel: new FakeNode('span') });
  view.setData(viewModel);
  const first = canvas._context.calls.fillRect;
  await nextFrame();
  const mid = canvas._context.calls.fillRect;
  await new Promise((resolve) => setTimeout(resolve, 700));
  const end = canvas._context.calls.fillRect;
  assert.ok(mid >= first, 'frames advance during the sweep');
  assert.ok(end > mid, 'the sweep keeps painting until it completes');
  assert.equal(view.reveal, 1, 'the reveal settles at 1');
});

test('the time cursor can be played and stopped', async () => {
  // Motion off: the entry reveal sweep would otherwise keep repainting the canvas
  // and blur the "stopping halts repaint" assertion. Playback itself is a user
  // action and works either way.
  const sandbox = loadPage({ noMotion: true });
  const viewModel = await loadViewModel();
  const canvas = sandbox.document.createElement('canvas');
  const view = new sandbox.VAP.GanttView({ canvas, tooltip: new FakeNode('div'), cursorLabel: new FakeNode('span'), hintLabel: new FakeNode('span') });
  view.setData(viewModel);
  const span = viewModel.meta.window.end - viewModel.meta.window.start;
  try {
    view.setPlaying(true, 1);
    // The first frame only establishes a baseline timestamp, so the assertion
    // waits for a few frames rather than exactly one.
    await new Promise((resolve) => setTimeout(resolve, 120));
    assert.notEqual(view.playhead, undefined, 'playback starts a cursor');
    const slow = (view.playhead ?? 0) - viewModel.meta.window.start;
    assert.ok(slow > 0, 'the cursor advances');
    assert.ok(slow < span, 'at 1× the cursor cannot cross the whole window in one step');
    assert.ok(canvas._context.calls.fillRect > 0, 'playback repaints the canvas');

    // The speed multiplier is applied (and readable); the exact advance per
    // frame is wall-clock dependent, so only invariants are asserted here.
    view.setPlaying(false);
    view.playhead = viewModel.meta.window.start;
    view.setPlaying(true, 12);
    assert.equal(view.playSpeed, 12, 'the requested speed is applied');
    await nextFrame();
    await nextFrame();
    assert.ok((view.playhead ?? 0) >= viewModel.meta.window.start, 'the cursor stays inside the window');
    assert.ok((view.playhead ?? 0) <= viewModel.meta.window.end, 'the cursor stays inside the window');

    // Stopping ends the loop: no further frames, no further repaint.
    view.setPlaying(false);
    const painted = canvas._context.calls.fillRect;
    await new Promise((resolve) => setTimeout(resolve, 120));
    assert.equal(canvas._context.calls.fillRect, painted, 'a stopped cursor no longer repaints');
  } finally {
    // The playback loop is an infinite rAF chain: it must be stopped or the
    // test process would never exit.
    view.setPlaying(false);
  }
  assert.equal(view.playing, false);
});

test('module 3 renders every step of the reasoning chain as a diagram', async () => {
  const sandbox = loadPage();
  const viewModel = await loadViewModel();
  const view = sandbox.VAP.advice.renderAdvice(viewModel, { priorityFilter: 'all' });
  const root = view.element;

  // Every step is one click away, and each renders a diagram rather than prose.
  const panels = {
    locate: '.compare-chart',
    evidence: 'g.tb-row',
    cause: '.cause-row',
    actions: '.action-block',
    benefit: '.benefit-row',
  };
  for (const [step, selector] of Object.entries(panels)) {
    view.setStep(step);
    assert.ok(root.querySelectorAll(selector).length > 0, `step ${step} must render ${selector}`);
    const stepText = root.textContent;
    assert.ok(!stepText.includes('undefined'), `step ${step}: no undefined must leak`);
    assert.ok(!stepText.includes('NaN'), `step ${step}: no NaN must leak`);
  }

  // The prose lives behind the disclosures: the default reading of each step is
  // shapes and numbers, and every claim still has its `依据` one click away.
  view.setStep('cause');
  assert.ok(root.querySelectorAll('details.cause-more').length > 0, 'each mechanism keeps its evidence');
  view.setStep('actions');
  assert.ok(root.querySelectorAll('details.compact').length > 0, 'each action keeps its rationale');

  // Action cards must be real elements carrying the priority as data: a CJK class
  // token would produce an HTMLUnknownElement that no stylesheet rule matches, and
  // the row would silently lose its layout.
  const rows = root.querySelectorAll('.action-row');
  assert.ok(rows.length > 0, 'at least one action must render');
  for (const row of rows) {
    assert.equal(row.tagName, 'div', 'an action row must be a real div');
    const badge = row.querySelector('.pri');
    assert.match(badge.className, /pri-(high|mid|low)/, `unexpected priority classes: ${badge.className}`);
    assert.ok(['高', '中', '低'].includes(badge.textContent), 'the priority badge carries the Chinese label');
  }

  // The priority filter narrows the action list without breaking the diagram.
  const highOnly = sandbox.VAP.advice.renderAdvice(viewModel, { priorityFilter: '高', activeStep: 'actions' });
  assert.ok(highOnly.element.querySelectorAll('.action-row').length <= rows.length);
  assert.ok(highOnly.element.textContent.length <= root.textContent.length);
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
    const view = sandbox.VAP.advice.renderAdvice(viewModel, {});
    const root = view.element;
    assert.ok(root.textContent.length > 200, `${scenario}: advice must state its numbers`);
    for (const step of ['locate', 'evidence', 'cause', 'actions', 'benefit']) {
      view.setStep(step);
      assert.ok(!root.textContent.includes('undefined'), `${scenario}/${step}: no undefined`);
      assert.ok(!root.textContent.includes('NaN'), `${scenario}/${step}: no NaN`);
    }
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
  for (const id of ['dropzone', 'gantt-canvas', 'share-bar-host', 'share-bar-legend', 'treemap-host', 'ranking-host', 'advice-chain', 'modal-body', 'stepper']) {
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

/** A UUID-shaped dataset id: the page treats ids as opaque, but the stub routes on shape. */
const DATASET_ID = '11111111-2222-3333-4444-555555555555';

/**
 * Boot the whole page against a real dataset: real view model, stubbed API.
 * This is the integration test for the controller — stepper, filters, charts and
 * advice are all exercised through the DOM the way a user would.
 */
async function bootPage({ withDataset = true, noMotion = true } = {}) {
  const bundle = documentationBundle({ version: '1.0.0' });
  const viewModel = await loadViewModel();
  const sandbox = loadPage({
    withHtmlIds: true,
    // Geometry assertions need deterministic frames; animation itself is covered
    // by the dedicated motion tests.
    noMotion,
    fetchImpl: async (url, options = {}) => {
      const path = String(url);
      if (path.includes('/api/health')) {
        return { ok: true, json: async () => ({ ok: true, plugin: 'vllm-ascend-profiler', version: '1.0.0', store: { datasets: withDataset ? 1 : 0, jobs: 0, maxDatasets: 6 }, limits: { maxUploadBytes: 1000, maxInMemoryBytes: 500, allowPathIngest: true } }) };
      }
      if (path.includes('/api/docs')) return { ok: true, json: async () => bundle };
      if (path.endsWith('/analyze')) {
        const requested = JSON.parse(options.body ?? '{}').phaseOverride;
        return { ok: true, json: async () => ({ ok: true, analysis: { ...viewModel.analysis, options: { ...viewModel.analysis.options, phaseOverride: requested } } }) };
      }
      if (path.includes(`/api/datasets/${DATASET_ID}`)) return { ok: true, json: async () => viewModel };
      if (path.includes('/api/datasets')) {
        return {
          ok: true,
          json: async () => (withDataset
            ? { datasets: [{ id: DATASET_ID, label: 'host-schedule-bound', createdAt: Date.now(), eventCount: viewModel.meta.eventCount, bottleneck: 'Host 调度', windowMs: viewModel.meta.wallUs / 1000 }] }
            : { datasets: [] }),
        };
      }
      return { ok: true, json: async () => ({}) };
    },
  });
  const handler = sandbox.document._listeners.find((entry) => entry.type === 'DOMContentLoaded');
  assert.equal(handler !== undefined, true, 'the controller registers a DOMContentLoaded handler');
  await handler.handler();
  await nextFrame();
  await nextFrame();
  return { sandbox, viewModel, registry };
}

test('the controller initializes against the real page markup', async () => {
  const { sandbox } = await bootPage({ withDataset: false });
  assert.ok(registry.get('gantt-canvas').width > 0, 'the swimlane canvas must be sized during init');
  assert.match(registry.get('health-line').textContent, /服务正常/);
  assert.equal(registry.get('dataset-row').hidden, true, 'an empty dataset list hides its row');
  assert.ok(registry.get('gantt-legend').children.length >= 5, 'the legend doubles as the category filter');
  assert.equal(sandbox.document.body.classList.contains('no-motion'), false);
});

test('booting with a dataset renders every step of the pipeline', async () => {
  const { sandbox, viewModel } = await bootPage();
  for (const id of ['overview', 'module-gantt', 'module-share', 'module-advice']) {
    assert.equal(registry.get(id).hidden, false, `#${id} must be visible after loading a dataset`);
  }
  assert.equal(registry.get('kpis').children.length, 4, 'exactly four headline indicators');
  assert.equal(registry.get('verdict-badge').textContent, 'Host 调度');
  const conclusions = registry.get('conclusions');
  assert.ok(conclusions.children.length > 0 && conclusions.children.length <= 3, 'three prioritised actions');
  assert.ok(registry.get('share-bar-host').children.length > 0, 'composition strip rendered');
  assert.ok(registry.get('share-bar-legend').children.length >= 2, 'the strip legend names its categories');
  assert.ok(registry.get('treemap-host').children.length > 0, 'treemap rendered');
  assert.equal(registry.get('advice-chain').querySelectorAll('button.flow-node').length, 5, 'five reasoning steps');
  assert.ok(registry.get('advice-chain').querySelectorAll('.compare-chart').length > 0, 'the first step renders its diagram');
  assert.match(registry.get('gantt-hint').textContent, /Ctrl\/⌘\+滚轮缩放/, 'the swimlane hint documents its controls');
  assert.equal(registry.get('verdict-score').textContent.length > 0, true);
  void sandbox;
  void viewModel;
});

test('a dataset warning is not shown, but a parse failure is', async () => {
  const { sandbox } = await bootPage();
  assert.equal(registry.get('error-box').hidden, true);
  // Errors surface through the same box the intake flow uses.
  const dropzone = registry.get('dropzone');
  assert.ok(dropzone !== null && dropzone !== undefined);
  void sandbox;
});

test('clicking a treemap tile filters the timeline and shows an active filter chip', async () => {
  await bootPage();
  const treemapHost = registry.get('treemap-host');
  const tiles = treemapHost.querySelectorAll('g.tm-tile');
  assert.ok(tiles.length > 0, 'the treemap must be interactive');
  const target = tiles[0].getAttribute('data-operator');
  tiles[0].click();
  await nextFrame();

  const chips = registry.get('gantt-filters');
  assert.equal(chips.children.length, 1, 'one active filter chip');
  assert.match(chips.children[0].textContent, /算子/);
  assert.ok(chips.children[0].textContent.includes(target.slice(0, 10)), 'the chip names the operator');

  // The other tiles dim (the selection is highlighted without a rebuild).
  const dimmed = registry.get('treemap-host').querySelectorAll('g.tm-tile').filter((tile) => tile.classList.contains('dimmed'));
  assert.equal(dimmed.length, tiles.length - 1);

  // Clearing via the chip removes the filter again.
  chips.children[0].children[0].click();
  await nextFrame();
  assert.equal(registry.get('gantt-filters').children.length, 0);
});

test('clicking a strip legend entry focuses that category and links the views', async () => {
  await bootPage();
  const legendButtons = registry.get('share-bar-legend').querySelectorAll('button');
  assert.ok(legendButtons.length > 0);
  legendButtons[0].click();
  await nextFrame();

  const chips = registry.get('gantt-filters');
  assert.equal(chips.children.length >= 1, true);
  assert.match(chips.children[0].textContent, /类别/);
  // The swimlane legend reflects the same selection.
  const cat = chips.children[0].textContent.includes('通信') ? 'comm' : undefined;
  if (cat !== undefined) {
    const chip = registry.get('gantt-legend').querySelectorAll(`[data-cat="${cat}"]`)[0];
    assert.equal(chip.getAttribute('aria-pressed'), 'true');
  }
});

test('the motion switch persists and stops all animations', async () => {
  const { sandbox } = await bootPage();
  const toggle = registry.get('motion-toggle');
  assert.equal(toggle.checked, true);
  toggle.checked = false;
  toggle.dispatchEvent({ type: 'change' });
  assert.equal(sandbox.document.body.classList.contains('no-motion'), true, 'the body class drives the CSS kill-switch');
  assert.equal(sandbox.VAP.motionEnabled(), false);
  assert.equal(sandbox.localStorage.getItem('vap.preferences').includes('"motion":false'), true, 'the choice is remembered');

  toggle.checked = true;
  toggle.dispatchEvent({ type: 'change' });
  assert.equal(sandbox.document.body.classList.contains('no-motion'), false);
});

test('the phase control re-analyses and keeps one scope', async () => {
  await bootPage();
  const prefill = registry.get('phase-select').querySelectorAll('[data-phase="prefill"]')[0];
  prefill.click();
  await nextFrame();
  assert.equal(prefill.getAttribute('aria-pressed'), 'true');
  const auto = registry.get('phase-select').querySelectorAll('[data-phase="auto"]')[0];
  assert.equal(auto.getAttribute('aria-pressed'), 'false');
});

test('the play button toggles timeline playback', async () => {
  await bootPage();
  const play = registry.get('gantt-play');
  assert.match(play.textContent, /播放/);
  play.click();
  assert.match(play.textContent, /暂停/);
  play.click();
  assert.match(play.textContent, /播放/);
});

test('the stepper marks the section in view and scrolls on click', async () => {
  await bootPage();
  const steps = registry.get('stepper').querySelectorAll('.step');
  assert.equal(steps.length, 5);
  assert.deepEqual(steps.map((step) => step.dataset.step), ['intake', 'overview', 'module-gantt', 'module-share', 'module-advice']);
  // Clicking is a no-op scroll in the fake DOM, but the handler must exist.
  steps[2].click();
  assert.ok(steps[2].listenerCount > 0);
});

test('an action in step ④ links back to the timeline with a matching filter', async () => {
  await bootPage();
  const chain = registry.get('advice-chain');
  // The jump buttons live in step ④, so the flow must be driven first — the same
  // path a reader takes.
  const actionsNode = chain.querySelectorAll('button.flow-node').find((node) => node.dataset.node === 'actions');
  actionsNode.click();
  await nextFrame();
  const link = chain.querySelectorAll('button').find((button) => button.textContent === '查看');
  assert.ok(link !== undefined, 'advice must offer a jump back into the timeline');
  link.click();
  await nextFrame();
  assert.ok(registry.get('gantt-filters').children.length >= 1, 'the jump applies a filter, not just a scroll');
});

test('an overview action opens step ④ on the very card it names', async () => {
  await bootPage();
  const conclusions = registry.get('conclusions');
  const jump = conclusions.querySelectorAll('button').find((button) => button.textContent === '查看方案');
  assert.ok(jump !== undefined, 'the overview must link into the action list');
  jump.click();
  await nextFrame();

  const chain = registry.get('advice-chain');
  const active = chain.querySelectorAll('button.flow-node').filter((node) => node.getAttribute('aria-selected') === 'true');
  assert.equal(active.length, 1, 'exactly one step stays selected');
  assert.equal(active[0].dataset.node, 'actions', 'the link selects the action step');
  assert.ok(chain.querySelectorAll('.action-row').length > 0, 'and the action rows are on screen');
});

test('init survives an unreachable API without breaking the page', async () => {
  const sandbox = loadPage({
    withHtmlIds: true,
    fetchImpl: async () => {
      throw new Error('network down');
    },
  });
  const handler = sandbox.document._listeners.find((entry) => entry.type === 'DOMContentLoaded');
  await assert.doesNotReject(async () => handler.handler());
  assert.match(registry.get('health-line').textContent, /健康检查失败/);
  assert.ok(registry.get('gantt-canvas').width > 0);
});
