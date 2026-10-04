/**
 * Step ⑤ ④ 优化行动 · operator-optimization handoff panel.
 *
 * One advice item, one task: this renders what the host prepared when 执行 was
 * pressed — the operators the advice acts on, the Ascend skill chain that runs it,
 * the acceptance bar, the files written into the workspace, and the instruction to
 * paste into a session. Keeping it in its own module keeps the advice panel about
 * *deciding* and this panel about *handing off*.
 *
 * @module dsh-web/handoff-view
 */
(function attachHandoffView(global) {
  'use strict';

  const VAP = global.VAP ?? (global.VAP = {});
  const { h } = VAP;

  const VERDICT_LABEL = {
    'ready': '可执行',
    'needs-skills': '需先安装 skills',
    'prepared': '已交接',
  };

  /** Skill role → Chinese label. */
  const ROLE_LABEL = { primary: '主', support: '支撑' };

  /**
   * Render the handoff panel for one advice item.
   *
   * @param {object} task - `task` payload from `POST /advice/<id>/handoff`.
   * @param {object} [options] - `{ onCopy, copied }`.
   * @returns {HTMLElement} panel node.
   */
  function renderHandoff(task, options = {}) {
    if (task === undefined || task === null || task.ok === false) {
      return h('p.error-inline', {}, `算子优化任务不可用：${task?.error ?? '缺少任务数据'}`);
    }
    const root = h('div.handoff-panel');
    const status = VERDICT_LABEL[task.status] ?? task.status;
    root.append(h('div.handoff-head', {}, [
      h('span.handoff-title', {}, task.project?.ok === true ? '算子工程已生成' : '算子优化任务已生成'),
      h('span.chip.phase', {}, status),
      h('span.hint', {}, `目标 ${task.operators.length === 0 ? '—' : String(task.operators.length) + ' 个算子'} · skills ${String(task.skills.chain.length)} 个`),
    ]));

    // 0) generated operator project (执行 produces this in the same round trip)
    const project = task.project;
    if (project !== undefined && project.ok === true) {
      const fileCount = (project.files ?? []).length;
      root.append(h('div.handoff-section.project', {}, [
        h('div.handoff-section-title', {}, `已生成算子工程 · ${project.title}`),
        h('p.handoff-optimisation', {}, project.summary?.optimization ?? ''),
        h('p.hint', {}, `导出按钮在建议行「执行」右侧（导出后拷到昇腾机器上 bash build.sh 编译；本工程 ${String(fileCount)} 个文件）`),
        h('dl.handoff-paths', {}, [
          h('dt', {}, '工程目录'), h('dd.mono', {}, project.dir ?? '—'),
          h('dt', {}, '生成类别'), h('dd', {}, `${project.kind}（${project.reason ?? ''}）`),
        ]),
        h('table.handoff-table', {}, [
          h('thead', {}, [h('tr', {}, [h('th', {}, '生成文件'), h('th', {}, '大小')])]),
          h('tbody', {}, (project.files ?? []).map((file) => h('tr', {}, [
            h('td.mono', {}, file.path),
            h('td.mono', {}, `${String(Math.round(file.bytes / 102.4) / 10)} KB`),
          ]))),
        ]),
        h('p.hint', {}, '编译、精度与性能需要 CANN + 昇腾 NPU：按工程目录里的 README 与 register-patch.md 继续。'),
      ]));
    } else if (project !== undefined && project.ok === false) {
      root.append(h('p.error-inline', {}, `算子工程生成失败：${project.error}`));
    }

    // 1) operators
    if (task.operators.length > 0) {
      root.append(h('div.handoff-section', {}, [
        h('div.handoff-section-title', {}, '目标算子'),
        h('table.handoff-table', {}, [
          h('thead', {}, [h('tr', {}, [
            h('th', {}, '算子'), h('th', {}, '类别'), h('th', {}, '调用'), h('th', {}, '累计'), h('th', {}, '单次'), h('th', {}, '占设备耗时'),
          ])]),
          h('tbody', {}, task.operators.map((operator) => h('tr', {}, [
            h('td.mono', {}, operator.name),
            h('td', {}, operator.category ?? '—'),
            h('td.mono', {}, String(operator.count ?? '—')),
            h('td.mono', {}, VAP.formatUs(operator.totalUs)),
            h('td.mono', {}, VAP.formatUs(operator.avgUs)),
            h('td.mono', {}, operator.sharePct === undefined ? '—' : `${String(operator.sharePct)}%`),
          ]))),
        ]),
      ]));
    } else {
      root.append(h('p.hint', {}, '本优化项面向采集/口径改造，不针对单个算子。'));
    }

    // 2) skill chain
    root.append(h('div.handoff-section', {}, [
      h('div.handoff-section-title', {}, '执行的 skills（按顺序）'),
      h('ol.handoff-skills', {}, task.skills.chain.map((skill) => h('li', {}, [
        h('span.mono.skill-name', {}, skill.name),
        h('span.chip.phase', {}, ROLE_LABEL[skill.role] ?? skill.role),
        skill.bundle === undefined ? undefined : h('span.hint', {}, skill.bundle),
        h('span.handoff-skill-purpose', {}, skill.purpose),
        h('span', { class: `skill-state ${skill.installed ? 'ok' : 'missing'}` }, skill.installed ? '已安装' : '未安装'),
      ]))),
    ]));

    // 3) acceptance
    if (task.acceptance.length > 0) {
      root.append(h('div.handoff-section', {}, [
        h('div.handoff-section-title', {}, '验收标准（与第 6 步对比同口径）'),
        h('table.handoff-table', {}, [
          h('thead', {}, [h('tr', {}, [
            h('th', {}, '指标'), h('th', {}, '当前值'), h('th', {}, '预期收益'), h('th', {}, '判定阈值'), h('th', {}, '方向'),
          ])]),
          h('tbody', {}, task.acceptance.map((row) => h('tr', {}, [
            h('td', {}, row.label),
            h('td.mono', {}, row.current === undefined ? '—' : `${String(row.current)} ${row.unit}`),
            h('td.mono', {}, row.expectedPct === undefined ? '—' : `${String(row.expectedPct)}%`),
            h('td.mono', {}, `≥ ${String(row.thresholdPct)}%`),
            h('td', {}, row.direction === 'higher' ? '越高越好' : row.direction === 'lower' ? '越低越好' : '仅背景'),
          ]))),
        ]),
        h('p.hint', {}, task.acceptanceHowTo),
      ]));
    }

    // 4) artifacts + instruction
    const paths = task.artifacts ?? {};
    root.append(h('div.handoff-section', {}, [
      h('div.handoff-section-title', {}, '交接产物'),
      h('dl.handoff-paths', {}, [
        h('dt', {}, '任务说明'), h('dd.mono', {}, paths.markdownPath ?? '—'),
        h('dt', {}, '结构化数据'), h('dd.mono', {}, paths.jsonPath ?? '—'),
      ]),
      h('div.handoff-prompt-actions', {}, [
        h('button.small.primary', {
          type: 'button',
          onclick: () => {
            if (typeof options.onCopy === 'function') options.onCopy(task.prompt, task.adviceId);
          },
        }, options.copied === true ? '已复制指令' : '复制指令'),
        h('span.hint', {}, '把指令粘贴给会话中的 agent（或按任务说明里的步骤手动执行）'),
      ]),
      h('textarea.handoff-prompt', { readonly: 'readonly', rows: '10', spellcheck: 'false' }, task.prompt),
    ]));

    // 5) install hint when skills are missing
    if (task.skills.missing.length > 0) {
      root.append(h('div.handoff-section.warn', {}, [
        h('div.handoff-section-title', {}, `缺少 ${String(task.skills.missing.length)} 个 skills`),
        h('p.mono', {}, task.installHint?.npx ?? ''),
        h('p.hint', {}, `来源仓库：${task.installHint?.repo ?? ''}`),
      ]));
    }
    return root;
  }

  Object.assign(VAP, { handoffView: { renderHandoff } });
})(window);
