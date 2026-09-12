import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const bundlePath = join(here, '..', 'lib', 'client.js');

/**
 * Load the hand-authored browser bundle in a sandbox that mimics the DSH shell:
 * `window.__ModuleLoader__.load({id, factory})`, a `require` that only resolves
 * the platform seeds, and a minimal `ctx` carrying the slots service.
 *
 * This is the only way to catch a client-bundle mistake outside a browser, and
 * client bundles are loaded as classic scripts, so a syntax error there would
 * otherwise surface as a blank GUI.
 */
function loadBundle({ slots } = {}) {
  const registrations = [];
  const warnings = [];
  const styles = [];
  const loader = {
    load(registration) {
      registrations.push(registration);
    },
  };
  const documentStub = {
    head: { append(node) { styles.push(node); } },
    createElement(tag) {
      return { tagName: tag, dataset: {}, textContent: '', setAttribute() {}, append() {} };
    },
    querySelector() {
      return null;
    },
  };
  // In a browser `globalThis === window`, so the sandbox must be one object that
  // plays both roles — otherwise `globalThis.location` silently differs from
  // `window.location` and the test would diverge from the real shell.
  const sandbox = {
    document: documentStub,
    console: { warn: (...args) => warnings.push(args.join(' ')), error: (...args) => warnings.push(args.join(' ')) },
    location: { origin: 'http://127.0.0.1:3080', pathname: '/vllm-ascend-profiler/' },
    open(url, target) {
      sandbox.opened = { url, target };
    },
    __ModuleLoader__: loader,
    __VLLM_ASCEND_PROFILER__: { routePrefix: '/vllm-ascend-profiler' },
  };
  const seeds = {
    react: {
      createElement(...args) {
        return { type: 'element', args };
      },
      useCallback(fn) {
        return fn;
      },
    },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  const source = readFileSync(bundlePath, 'utf8');
  vm.runInContext(source, sandbox, { filename: 'client.js' });

  assert.equal(registrations.length, 1, 'the bundle must register exactly one module');
  const registration = registrations[0];
  const factoryExports = registration.factory((specifier) => {
    if (seeds[specifier] === undefined) throw new Error(`unexpected require("${specifier}")`);
    return seeds[specifier];
  });

  const registered = [];
  const context = {
    slots: slots ?? {
      inject(name, callback) {
        registered.push({ name, value: callback() });
      },
      register(options, component) {
        return { options, component };
      },
    },
    get() {
      return undefined;
    },
  };
  return { registration, factoryExports, context, registered, warnings, styles, windowStub: sandbox };
}

test('client bundle registers under the package id with the expected face', () => {
  const { registration, factoryExports } = loadBundle();
  assert.equal(registration.id, 'dsh-plugin-vllm-ascend-profiler');
  assert.equal(typeof factoryExports.apply, 'function');
  assert.ok(Array.isArray(factoryExports.inject));
  assert.equal(factoryExports.inject.length, 0);
});

test('client bundle registers a sidebar footer entry', () => {
  const { factoryExports, context, registered, styles } = loadBundle();
  factoryExports.apply(context);
  assert.equal(registered.length, 1);
  assert.equal(registered[0].name, 'sidebar.footer.action');
  const { options, component } = registered[0].value;
  assert.equal(options.name, 'sidebar.footer.action');
  assert.equal(options.id, 'vllm-ascend-profiler');
  assert.equal(typeof component, 'function');
  assert.ok(styles.length >= 1, 'the entry must inject its stylesheet');
});

test('the entry renders and opens the analyzer page', () => {
  const { factoryExports, context, registered, windowStub } = loadBundle();
  factoryExports.apply(context);
  const component = registered[0].value.component;
  const wide = component({ wide: true });
  assert.equal(wide.type, 'element');
  assert.equal(wide.args[0], 'div');
  assert.equal(wide.args[1].dataset.wide, 'true');

  // The wrapper's child is the button; its handler opens the analyzer page.
  const button = wide.args[2];
  assert.equal(button.args[0], 'button');
  assert.equal(typeof button.args[1].onClick, 'function');
  button.args[1].onClick();
  assert.equal(windowStub.opened.url, 'http://127.0.0.1:3080/vllm-ascend-profiler/');
  assert.equal(windowStub.opened.target, '_blank');

  // The rail (compact) rendering drops the text label but keeps the button.
  const compact = component({ wide: false });
  assert.equal(compact.args[1].dataset.wide, 'false');
  assert.equal(compact.args[2].args[0], 'button');
  const labels = compact.args[2].args.slice(2).filter(Boolean);
  assert.equal(labels.length, 1, 'compact mode renders only the icon');
});

test('the entry URL follows the injected route prefix', () => {
  const { factoryExports, context, registered } = loadBundle();
  factoryExports.apply(context);
  const url = factoryExports.analyzerUrl();
  assert.equal(url, 'http://127.0.0.1:3080/vllm-ascend-profiler/');
  void registered;
});

test('a missing slot service degrades to a warning instead of throwing', () => {
  const { factoryExports, warnings } = loadBundle();
  const context = {
    slots: undefined,
    get() {
      return undefined;
    },
  };
  assert.doesNotThrow(() => factoryExports.apply(context));
  assert.ok(warnings.some((warning) => warning.includes('侧边栏插槽服务不可用')));
});

test('a throwing slot service is caught', () => {
  const { factoryExports } = loadBundle();
  const context = {
    slots: {
      inject() {
        throw new Error('slot API changed');
      },
      register() {
        throw new Error('slot API changed');
      },
    },
    get() {
      return undefined;
    },
  };
  assert.doesNotThrow(() => factoryExports.apply(context));
});
