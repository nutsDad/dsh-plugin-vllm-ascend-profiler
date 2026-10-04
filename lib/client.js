/**
 * vLLM-Ascend Profiler Analyzer — browser half.
 *
 * Hand-authored DSH client module: `@deepseek-ai/dsh-client-modules` discovers
 * this package through its `dsh.client` declaration, serves this bundle from
 * `/plugins/<id>/client.js`, and the shell's module loader calls the factory
 * with a synchronous `require` that resolves the platform seeds (`react`, the
 * cordis runtime, the slot and primitive packages).
 *
 * What it contributes is deliberately small and defensive: one entry in the
 * session sidebar's footer action slot that opens the standalone analyzer page
 * (which the node half serves at the configured route prefix). All analysis UI
 * lives in that page, so a slot API change can never break the analyzer itself.
 *
 * The route prefix is read from the `__VLLM_ASCEND_PROFILER__` global the node
 * half injects into the shell's index, which keeps the two halves decoupled:
 * changing `routePrefix` in the patch row needs no change here.
 */
window.__ModuleLoader__.load({
  id: 'dsh-plugin-vllm-ascend-profiler',
  factory: (require) => {
    const module = { exports: {} };
    const exports = module.exports;
    Object.defineProperty(exports, Symbol.toStringTag, { value: 'Module' });

    const react = require('react');

    /** Fallback prefix; the injected global wins when present. */
    const DEFAULT_PREFIX = '/vllm-ascend-profiler';
    /** Cache-busting tag for this bundle's stylesheet. */
    const STYLE_TAG = 'dsh-plugin-vllm-ascend-profiler/sidebar-entry.css';

    const CSS = `
.vap-entry { display: flex; align-items: center; width: 100%; }
.vap-btn {
  display: inline-flex; align-items: center; gap: 8px; width: 100%; height: 38px; margin: 0 -2px;
  padding: 0 10px 0 8px; border: none; border-radius: 12px; background: transparent; cursor: pointer;
  color: var(--dsw-alias-label-primary, #1a1d21); font: inherit; font-size: 13px; overflow: hidden;
}
.vap-btn:hover { background: var(--dsw-alias-interactive-bg-hover, rgba(127, 127, 127, .12)); }
.vap-btn:focus-visible { outline: 2px solid var(--dsw-alias-brand-primary, #1d4ed8); outline-offset: 1px; }
.vap-entry[data-wide="false"] { justify-content: center; }
.vap-entry[data-wide="false"] .vap-btn { width: 38px; height: 38px; justify-content: center; padding: 0; border-radius: 50%; }
.vap-icon {
  flex: none; width: 22px; height: 22px; border-radius: 7px; display: grid; place-items: center;
  background: linear-gradient(135deg, #1d4ed8, #7c3aed); color: #fff; font-size: 11px; font-weight: 700;
}
.vap-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.vap-dot { margin-left: auto; font-size: 10px; color: var(--dsw-alias-label-tertiary, #7b828c); }
`;

    /** Inject the stylesheet once. */
    function ensureStyles() {
      if (typeof document === 'undefined') return;
      if (document.querySelector(`style[data-plugin-css="${STYLE_TAG}"]`) !== null) return;
      const tag = document.createElement('style');
      tag.dataset.plugin = 'dsh-plugin-vllm-ascend-profiler';
      tag.dataset.pluginCss = STYLE_TAG;
      tag.textContent = CSS;
      document.head.append(tag);
    }

    /** Resolve the analyzer page URL from the injected global, with a fallback. */
    function analyzerUrl() {
      const injected = globalThis.__VLLM_ASCEND_PROFILER__;
      const configured = injected !== null && typeof injected === 'object' && typeof injected.routePrefix === 'string'
        ? injected.routePrefix
        : DEFAULT_PREFIX;
      const prefix = configured.endsWith('/') ? configured.slice(0, -1) : configured;
      return `${globalThis.location.origin}${prefix}/`;
    }

    /**
     * Open the analyzer inside the shell's right-hand browser tab.
     *
     * The DSH shell registers a `browser` tab there and routes external links to
     * it, so this is the one way to show the analyzer while keeping the chat
     * visible. Both services are read defensively: a shell without them must not
     * break the entry.
     *
     * @param {object} ctx - client root context.
     * @param {string} url - analyzer URL.
     * @returns {boolean} whether the tab was opened.
     */
    function openInShellTab(ctx, url) {
      try {
        if (ctx === undefined || typeof ctx.get !== 'function') return false;
        const tabs = ctx.get('sidebarRightTabs');
        if (tabs === undefined || typeof tabs.get !== 'function' || tabs.get('browser') === undefined) return false;
        const right = ctx.get('sidebarRight');
        if (right === undefined || typeof right.openTab !== 'function') return false;
        right.openTab('browser', { params: { url } });
        return true;
      } catch {
        return false;
      }
    }

    /**
     * Open the analyzer page, in the order the desktop shell allows.
     *
     * 1. the shell's right-hand browser tab (keeps the chat visible);
     * 2. `window.open` — works in a browser, but the Electron desktop shell denies
     *    every popup (`setWindowOpenHandler` returns `deny` and only hands `https:`
     *    URLs to the OS browser), so it returns null for this http URL and the
     *    button used to do nothing at all;
     * 3. same-window navigation — allowed because the analyzer is served from the
     *    shell's own origin.
     *
     * @param {object} ctx - client root context (optional).
     * @returns {'tab'|'window'|'self'} which strategy was used.
     */
    function openAnalyzer(ctx) {
      const url = analyzerUrl();
      if (openInShellTab(ctx, url)) return 'tab';
      let opened;
      try {
        opened = globalThis.open(url, '_blank', 'noopener');
      } catch {
        opened = null;
      }
      if (opened !== null && opened !== undefined) return 'window';
      // Same-window navigation loses the chat, so hand the analyzer a `back` link
      // to return to; the page only honours a same-origin URL.
      let back = url;
      try {
        back = `${url}?back=${encodeURIComponent(globalThis.location.href)}`;
      } catch {
        back = url;
      }
      globalThis.location.assign(back);
      return 'self';
    }

    /**
     * The sidebar footer entry.
     *
     * Purely presentational: the opener arrives as `props.onOpen` so the entry can
     * be rendered outside the cordis context (tests, previews) and still work.
     */
    function ProfilerEntry(props) {
      const wide = props.wide !== false;
      const onOpen = typeof props.onOpen === 'function' ? props.onOpen : () => openAnalyzer(undefined);
      const open = react.useCallback(() => { onOpen(); }, [onOpen]);
      return react.createElement(
        'div',
        { className: 'vap-entry', dataset: { wide: String(wide) } },
        react.createElement(
          'button',
          {
            type: 'button',
            className: 'vap-btn',
            title: `打开 vLLM-Ascend Profiling 分析器（上传 profiling 产物，查看泳道图与优化建议）· ${analyzerUrl()}`,
            'aria-label': '打开 vLLM-Ascend Profiling 分析器',
            onClick: open,
          },
          react.createElement('span', { className: 'vap-icon', 'aria-hidden': true }, '昇'),
          wide ? react.createElement('span', { className: 'vap-label' }, 'Profiler 分析') : null,
          wide ? react.createElement('span', { className: 'vap-dot' }, 'vLLM-Ascend') : null,
        ),
      );
    }

    /**
     * Client plugin body. Registration is guarded because the slot service is
     * owned by the shell: if a future shell drops `sidebar.footer.action`, this
     * plugin must degrade to a no-op rather than fail the browser boot.
     *
     * @param {object} ctx - client root context, carrying the injected `slots`.
     */
    function apply(ctx) {
      ensureStyles();
      try {
        const slots = ctx.slots ?? (typeof ctx.get === 'function' ? ctx.get('slots') : undefined);
        if (slots === undefined || typeof slots.inject !== 'function' || typeof slots.register !== 'function') {
          console.warn('[vllm-ascend-profiler] 侧边栏插槽服务不可用，分析器入口未注册；可直接访问 ' + analyzerUrl());
          return;
        }
        slots.inject('sidebar.footer.action', () => slots.register({
          name: 'sidebar.footer.action',
          id: 'vllm-ascend-profiler',
        }, (props) => ProfilerEntry({ ...props, onOpen: () => openAnalyzer(ctx) })));
      } catch (error) {
        console.warn('[vllm-ascend-profiler] 注册侧边栏入口失败：', error);
      }
    }

    exports.apply = apply;
    /**
     * The `slots` service must be *declared*, not merely read: the cordis client
     * runner resolves injected services before `apply` runs, so a module that
     * declares nothing sees `ctx.slots === undefined` and the entry below would
     * never register (silently — the analyzer page itself keeps working, which is
     * exactly why this went unnoticed for a release).
     */
    exports.inject = ['slots'];
    exports.ProfilerEntry = ProfilerEntry;
    exports.analyzerUrl = analyzerUrl;
    exports.openAnalyzer = openAnalyzer;
    return module.exports;
  },
});
