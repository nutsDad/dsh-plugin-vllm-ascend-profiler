/**
 * Documentation view: renders the metric definitions, the artifact/field
 * reference, and the usage guidance served by the plugin's `/api/docs`.
 *
 * Keeping this rendered from server-side data (rather than hard-coded in the
 * page) means the documentation and the analysis always ship together: a metric
 * that exists in the report always has its definition available next to it.
 */
(function attachDocsView(global) {
  'use strict';

  const { h } = global.VAP;

  /**
   * Render the documentation bundle.
   * @param {object} bundle - payload from `/api/docs`.
   * @returns {HTMLElement} root node.
   */
  function renderDocs(bundle) {
    const root = h('div');
    root.append(h('p.hint', {}, `插件 ${bundle.plugin.title} v${bundle.plugin.version} · 路由前缀 ${bundle.plugin.routePrefix}`));

    root.append(h('h3', {}, '一、快速开始'));
    root.append(h('ol', {}, bundle.usage.quickStart.map((item) => h('li', {}, item))));

    root.append(h('h3', {}, '二、阶段口径（Prefill / Decode）'));
    root.append(h('ul', {}, bundle.usage.phaseAdvice.map((item) => h('li', {}, item))));

    root.append(h('h3', {}, '三、指标含义与计算口径'));
    for (const group of bundle.metricDocs) {
      root.append(h('h4', {}, group.group));
      for (const metric of group.metrics) {
        root.append(h('div.doc-item', {}, [
          h('div.name', {}, metric.name),
          h('dl', {}, [
            h('dt', {}, '定义'), h('dd', {}, metric.definition),
            h('dt', {}, '计算'), h('dd', {}, metric.formula),
            h('dt', {}, '注意'), h('dd', {}, metric.caveats),
          ]),
        ]));
      }
    }

    root.append(h('h3', {}, '四、vLLM-Ascend / 昇腾 profiling 产物与字段'));
    for (const artifact of bundle.artifactDocs) {
      root.append(h('div.doc-item', {}, [
        h('div.name', {}, artifact.name),
        h('dl', {}, [
          h('dt', {}, '产出方'), h('dd', {}, artifact.producer),
          h('dt', {}, '内容'), h('dd', {}, artifact.content),
        ]),
        h('ul', {}, artifact.notes.map((note) => h('li', {}, note))),
      ]));
    }

    root.append(h('h3', {}, '五、性能提示'));
    root.append(h('ul', {}, bundle.usage.performanceTips.map((item) => h('li', {}, item))));
    return root;
  }

  /** Render the health-check payload into a modal body. */
  function renderHealth(health) {
    return h('div', {}, [
      h('p', {}, `插件状态：${health.ok === true ? '正常' : '异常'}`),
      h('pre', {}, JSON.stringify(health, null, 2)),
    ]);
  }

  global.VAP = global.VAP ?? {};
  global.VAP.docsView = { renderDocs, renderHealth };
})(window);
