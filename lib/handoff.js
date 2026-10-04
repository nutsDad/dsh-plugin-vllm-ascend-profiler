/**
 * Operator-optimization handoff: turn one advice item into a task an Ascend
 * operator-development agent can execute.
 *
 * Step 5 answers "what to change"; this module covers the gap between that answer
 * and someone opening a kernel repository — the part where an operator developer
 * has to ask "which operator, on what evidence, against what bar, with which
 * skill?". It builds one task package per advice item:
 *
 * * **target operators** picked from the capture by the advice's own focus
 *   (compute / communication / copy / dispatch), with counts and time shares, so
 *   the work starts from data instead of a guess;
 * * **the skill chain** — which of the Ascend bundles ({@link SKILL_BUNDLES})
 *   apply, in what order, and for what purpose;
 * * **acceptance criteria** in the metric vocabulary of the comparison
 *   ({@link module:dsh-plugin-vllm-ascend-profiler/analysis/compare}), including
 *   the same threshold the later before/after comparison will apply;
 * * **a ready-to-run instruction** (Markdown) plus the package written into the
 *   workspace, so an agent — or a person — can start from the file.
 *
 * Skills are *not* vendored here: the three bundles live in the public
 * `ascend-ai-coding/awesome-ascend-skills` repository and are discovered by DSH
 * from its own skill roots. This module probes those roots so the page can say
 * "installed" or "install this first" instead of failing silently at run time.
 *
 * @module dsh-plugin-vllm-ascend-profiler/handoff
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { acceptanceThresholdPct, adviceTargets, describeMetrics } from './analysis/compare.js';
import { round } from './model/stats.js';

/** Where task packages are written, relative to the session workspace. */
export const TASK_DIR = '.dsh-vap-tasks';

/** Upstream repository that owns the skills (no skill content is vendored here). */
export const SKILLS_REPO = 'ascend-ai-coding/awesome-ascend-skills';

/**
 * The three official bundles this integration drives.
 *
 * Transcribed from the repository's marketplace manifest (`ascend-base` 7 skills,
 * `ascend-profiling` 8 entries over 5 directories, `ascend-ops` 4 skills) so the
 * page can list, probe and explain them without a network call at request time.
 */
export const SKILL_BUNDLES = Object.freeze({
  'ascend-base': Object.freeze({
    purpose: '设备与容器环境检查、torch_npu 基础能力（确认优化前环境是健康的）',
    skills: Object.freeze(['npu-smi', 'ascend-docker', 'torch_npu', 'remote-server-guide', 'npu-docker-launcher', 'ascend-dmi', 'ascend-avi-vnpu']),
  }),
  'ascend-profiling': Object.freeze({
    purpose: 'Profiling 解析与瓶颈判定、重新采集、MFU/收益复核',
    skills: Object.freeze(['profiling-analysis', 'mindspeed-llm-train-profiler', 'mindspeed-mm-train-profiler', 'pytorch-profiling-collection', 'training-mfu-calculator']),
  }),
  'ascend-ops': Object.freeze({
    purpose: 'AscendC 算子开发与调优、torch_npu 接入、Triton-Ascend 迁移、单算子基准',
    skills: Object.freeze(['ascendc', 'ascend-opplugin', 'triton-ascend-migration', 'npu-op-benchmark']),
  }),
});

/**
 * DSH skill roots, in discovery order (rank 100 → 500).
 *
 * The user root falls back to `$DSH_HOME` and then `~/.dsh`, so the inventory is
 * still meaningful when the plugin runs outside a DSH-spawned process (tests, CLI
 * tooling) instead of silently listing no user root at all.
 *
 * @param {object} [options] - `{ dshHome, agentsHome, workspace, customDirs }`.
 * @returns {Array<{source: string, path: string}>} ordered roots.
 */
export function skillRoots({ dshHome, agentsHome, workspace, customDirs = [] } = {}) {
  const roots = [];
  if (workspace !== undefined) {
    roots.push({ source: 'project-dsh', path: join(workspace, '.dsh', 'skills') });
    roots.push({ source: 'project-agents', path: join(workspace, '.agents', 'skills') });
  }
  for (const dir of customDirs) roots.push({ source: 'custom', path: resolve(String(dir)) });
  const userHome = dshHome ?? process.env.DSH_HOME ?? join(homedir(), '.dsh');
  roots.push({ source: 'user-dsh', path: join(userHome, 'skills') });
  const userAgents = agentsHome ?? process.env.AGENTS_HOME;
  if (userAgents !== undefined) roots.push({ source: 'user-agents', path: join(userAgents, 'skills') });
  return roots;
}

/**
 * Which skill is in which bundle (a skill appears in exactly one bundle here).
 *
 * @param {string} skill - skill name.
 * @returns {string|undefined} bundle name.
 */
export function bundleOfSkill(skill) {
  for (const [bundle, entry] of Object.entries(SKILL_BUNDLES)) {
    if (entry.skills.includes(skill)) return bundle;
  }
  return undefined;
}

/**
 * Probe the skill roots for the named skills.
 *
 * A skill counts as installed when `<root>/<name>/SKILL.md` exists — the same
 * shape DSH itself discovers, so the page never claims a skill is available when
 * the runner would not find it.
 *
 * @param {string[]} names - skill names to look for.
 * @param {Array<{source: string, path: string}>} roots - candidate roots.
 * @returns {{installed: object[], missing: object[]}} probe result.
 */
export function probeSkills(names, roots) {
  const installed = [];
  const missing = [];
  for (const name of [...new Set(names)]) {
    const found = roots.find((root) => existsSync(join(root.path, name, 'SKILL.md')));
    const bundle = bundleOfSkill(name);
    const entry = { name, bundle, source: found?.source, path: found === undefined ? undefined : join(found.path, name) };
    if (found === undefined) missing.push(entry);
    else installed.push(entry);
  }
  return { installed, missing };
}

/**
 * How to install the bundles when the probe comes back empty.
 *
 * @returns {{repo: string, npx: string, manual: string[]}} install instructions.
 */
export function installHint() {
  return {
    repo: `https://github.com/${SKILLS_REPO}`,
    npx: `npx @ascend-ai-coding/ascend-skills@latest install ascend-base ascend-profiling ascend-ops`,
    manual: [
      `git clone --depth 1 https://github.com/${SKILLS_REPO}.git`,
      `# 把 skills/base、skills/profiling、skills/ops 下的目录复制到 DSH 的 skills 根，例如：`,
      `cp -r awesome-ascend-skills/skills/ops/ascendc "$DSH_HOME/skills/"`,
    ],
  };
}

/**
 * Advice id → the skill chain that executes it.
 *
 * `focus` decides which operators are attached (see {@link pickOperators}) and
 * `primary`/`support` decide the order the skills are invoked in: the primary
 * skill produces the change, the support skills supply evidence, environment
 * checks and the acceptance measurement.
 */
export const ADVICE_SKILLS = Object.freeze({
  'host.enable-graph-mode': { focus: 'graph', primary: ['torch_npu'], support: ['profiling-analysis', 'pytorch-profiling-collection'] },
  'host.async-sampling': { focus: 'graph', primary: ['torch_npu'], support: ['profiling-analysis', 'pytorch-profiling-collection'] },
  'host.smooth-sampling-path': { focus: 'graph', primary: ['torch_npu'], support: ['profiling-analysis'] },
  'host.reduce-dispatch': { focus: 'dispatch', primary: ['torch_npu', 'ascendc'], support: ['npu-op-benchmark', 'profiling-analysis'] },
  'compute.quantize': { focus: 'kernel', primary: ['ascendc'], support: ['ascend-opplugin', 'npu-op-benchmark', 'profiling-analysis'] },
  'compute.fuse-small-ops': { focus: 'kernel', primary: ['ascendc'], support: ['ascend-opplugin', 'npu-op-benchmark'] },
  'compute.decode-increase-batch': { focus: 'kernel', primary: ['ascendc'], support: ['npu-op-benchmark', 'profiling-analysis'] },
  'compute.tune-chunked-prefill': { focus: 'config', primary: ['torch_npu'], support: ['profiling-analysis', 'pytorch-profiling-collection'] },
  'comm.enable-overlap-fusion': { focus: 'comm', primary: ['ascendc'], support: ['profiling-analysis', 'npu-op-benchmark'] },
  'comm.batch-amortize': { focus: 'comm', primary: ['torch_npu'], support: ['profiling-analysis'] },
  'comm.parallel-strategy': { focus: 'comm', primary: ['torch_npu'], support: ['profiling-analysis', 'remote-server-guide'] },
  'copy.async-d2h': { focus: 'copy', primary: ['torch_npu'], support: ['profiling-analysis', 'ascendc'] },
  'copy.kv-locality': { focus: 'copy', primary: ['ascendc'], support: ['profiling-analysis', 'npu-op-benchmark'] },
  'copy.reduce-profiling-overhead': { focus: 'profile', primary: ['profiling-analysis'], support: ['pytorch-profiling-collection'] },
  'common.phase-isolation': { focus: 'profile', primary: ['pytorch-profiling-collection'], support: ['profiling-analysis'] },
  'common.collect-richer-profile': { focus: 'profile', primary: ['pytorch-profiling-collection'], support: ['profiling-analysis', 'ascend-dmi'] },
});

/** Human-readable purpose per skill, used in the generated instruction. */
const SKILL_PURPOSE = Object.freeze({
  'ascendc': '设计/生成/调优 AscendC 算子（tiling、op_host/op_kernel、精度与性能）',
  'ascend-opplugin': '把算子接入 torch_npu（注册、构建、Python 暴露与冒烟测试）',
  'npu-op-benchmark': '单算子基准：改造前后同口径对比时延与吞吐',
  'triton-ascend-migration': 'Triton/PyTorch 算子改写为 Triton-Ascend 实现',
  'profiling-analysis': '复核 Profiling 证据，确认瓶颈与门限判定',
  'pytorch-profiling-collection': '按同一负载重新采集 Profiling（验收测量）',
  'mindspeed-llm-train-profiler': 'MindSpeed-LLM 训练采集（若是训练负载）',
  'mindspeed-mm-train-profiler': 'MindSpeed-MM 训练采集（若是多模态负载）',
  'training-mfu-calculator': 'MFU/吞吐换算，判断收益是否落在合理区间',
  'npu-smi': '设备健康与占用检查（排除设备侧干扰）',
  'ascend-dmi': '算力/带宽基准与故障诊断（环境是否达标）',
  'torch_npu': 'torch_npu 图模式/异步下发/编译配置（Host 侧改造）',
  'remote-server-guide': '远程机器/容器连接与命令执行',
  'ascend-docker': '容器内环境（镜像挂载 NPU 设备）',
  'npu-docker-launcher': '启动带 NPU 的容器',
  'ascend-avi-vnpu': '虚拟化实例（vNPU）相关检查',
});

/** Category to look at for each focus, and how to rank the candidates. */
const FOCUS_SELECTION = Object.freeze({
  kernel: { category: 'compute', rankBy: 'totalUs', why: '设备侧计算算子' },
  comm: { category: 'comm', rankBy: 'totalUs', why: '通信算子' },
  copy: { category: 'copy', rankBy: 'totalUs', why: '数据拷贝算子' },
  dispatch: { category: 'schedule', rankBy: 'count', why: 'Host 侧下发/同步算子' },
  graph: { category: 'schedule', rankBy: 'count', why: 'Host 侧下发/同步算子' },
  config: { category: 'compute', rankBy: 'totalUs', why: '设备侧计算算子' },
  profile: undefined,
});

/**
 * Pick the operators the advice should act on.
 *
 * @param {object} dataset - built dataset (has `operators`).
 * @param {object} adviceSkills - entry from {@link ADVICE_SKILLS}.
 * @param {number} [limit] - how many candidates to return.
 * @returns {object[]} operator candidates with the reason they were picked.
 */
export function pickOperators(dataset, adviceSkills, limit = 3) {
  const selection = FOCUS_SELECTION[adviceSkills?.focus];
  if (selection === undefined) return [];
  const totalUs = dataset.operators.reduce((sum, operator) => sum + (operator.totalUs ?? 0), 0);
  const candidates = dataset.operators
    .filter((operator) => operator.category === selection.category)
    .sort((left, right) => (right[selection.rankBy] ?? 0) - (left[selection.rankBy] ?? 0))
    .slice(0, limit);
  return candidates.map((operator) => ({
    name: operator.name,
    label: operator.label ?? operator.name,
    group: operator.group,
    category: operator.category,
    subtype: operator.subtype,
    count: operator.count,
    totalUs: round(operator.totalUs ?? 0, 2),
    avgUs: round(operator.avgUs ?? 0, 2),
    maxUs: round(operator.maxUs ?? 0, 2),
    p95Us: Number.isFinite(operator.p95Us) ? round(operator.p95Us, 2) : undefined,
    waitUs: Number.isFinite(operator.waitUs) ? round(operator.waitUs, 2) : undefined,
    sharePct: totalUs === 0 ? undefined : round(((operator.totalUs ?? 0) / totalUs) * 100, 1),
    utilization: operator.utilization,
    why: `${selection.why} · 累计 ${String(round(operator.totalUs ?? 0, 2))}µs · 调用 ${String(operator.count ?? 0)} 次 · 单次 ${String(round(operator.avgUs ?? 0, 2))}µs`,
  }));
}

/**
 * Build the task package for one advice item.
 *
 * @param {object} input - `{ adviceId, dataset, analysis, datasetId, label, roots, createdAt }`.
 * @returns {object} task package (also serialisable to JSON).
 */
export function buildHandoff({ adviceId, dataset, analysis, datasetId, label, roots = [], createdAt } = {}) {
  const advice = analysis?.steps?.actions?.items?.find((item) => item.id === adviceId);
  if (advice === undefined) {
    return { ok: false, error: `未知的优化项：${String(adviceId)}`, knownIds: analysis?.steps?.actions?.items?.map((item) => item.id) ?? [] };
  }
  const mapping = ADVICE_SKILLS[advice.id] ?? { focus: 'config', primary: [], support: [] };
  const skillNames = [...mapping.primary, ...mapping.support];
  const probe = probeSkills(skillNames.length > 0 ? skillNames : Object.values(SKILL_BUNDLES).flatMap((bundle) => bundle.skills), roots);
  const metrics = describeMetrics(analysis);
  const metricOf = (key) => metrics.find((metric) => metric.key === key);
  const expectedPct = advice.expectedGain?.estimatePct;
  const targets = adviceTargets(advice.id).map(metricOf).filter((metric) => metric !== undefined);

  const packages = {
    adviceId: advice.id,
    title: advice.title,
    priority: advice.priority,
    phase: advice.phases ?? [advice.phase],
    bottleneck: {
      id: analysis.bottleneck.id,
      label: analysis.bottleneck.label,
      short: analysis.bottleneck.short,
      score: round(analysis.bottleneck.score, 1),
      scope: analysis.bottleneck.scope,
    },
    rationale: advice.rationale,
    actions: advice.actions,
    verification: advice.verification,
    risk: advice.risk,
    linkedRootCauses: advice.linkedRootCauses,
    expectedGain: advice.expectedGain,
    focus: mapping.focus,
    operators: pickOperators(dataset, mapping),
    skills: {
      bundles: [...new Set(skillNames.map(bundleOfSkill).filter((name) => name !== undefined))],
      chain: skillNames.map((name) => ({
        name,
        role: mapping.primary.includes(name) ? 'primary' : 'support',
        purpose: SKILL_PURPOSE[name] ?? '',
        bundle: bundleOfSkill(name),
        installed: probe.installed.some((entry) => entry.name === name),
      })),
      installed: probe.installed,
      missing: probe.missing,
    },
    acceptance: targets.map((metric) => ({
      key: metric.key,
      label: metric.label,
      unit: metric.unit,
      direction: metric.direction,
      current: metric.value,
      expectedPct,
      thresholdPct: acceptanceThresholdPct(expectedPct),
    })),
    acceptanceHowTo: '改造完成后，用同一负载、同一阶段口径重新采集 Profiling，在第 6 步与本次数据集对比；判定标准同上（阈值 = 预期收益的一半，下限 1%）。',
    evidence: {
      rationale: advice.rationale,
      indicators: {
        avgStepUs: analysis.indicators.avgStepUs,
        deviceBusyPct: analysis.indicators.deviceBusyPct,
        hostExclusivePerStepUs: analysis.indicators.hostExclusivePerStepUs,
        dispatchPerStep: analysis.indicators.dispatchPerStep,
        commExposedPct: analysis.indicators.commExposedPct,
      },
      thresholds: analysis.evidence?.items?.slice(0, 8)?.map((item) => ({
        key: item.key,
        label: item.label,
        value: item.value,
        threshold: item.threshold,
        pass: item.pass,
      })) ?? [],
    },
    dataset: {
      id: datasetId,
      label: label ?? dataset?.meta?.label,
      eventCount: analysis.indicators.eventCount,
      stepCount: analysis.indicators.stepCount,
      wallUs: round(analysis.indicators.wallUs, 2),
      phaseOverride: analysis.options?.phaseOverride ?? 'auto',
      files: (dataset?.meta?.files ?? []).map((file) => (typeof file === 'string' ? file : file.name)),
    },
    installHint: probe.missing.length > 0 ? installHint() : undefined,
    createdAt: createdAt ?? new Date().toISOString(),
    status: probe.missing.length > 0 ? 'needs-skills' : 'ready',
  };
  packages.prompt = handoffPrompt(packages);
  return { ok: true, ...packages };
}

/**
 * The instruction an agent (or a person) executes.
 *
 * Deliberately a single self-contained block: it names the skills, the operators,
 * the evidence and the bar, so pasting it into a session is enough to start.
 *
 * @param {object} task - task package.
 * @returns {string} Markdown instruction.
 */
export function handoffPrompt(task) {
  const lines = [];
  lines.push(`# 算子优化任务 · ${task.title}`);
  lines.push('');
  lines.push(`来源：vLLM-Ascend Profiler Analyzer 第 5 步 · 优化项 \`${task.adviceId}\`（优先级 ${task.priority}，阶段 ${(task.phase ?? []).join('/')}）`);
  lines.push(`数据集：${task.dataset.label ?? task.dataset.id} · ${String(task.dataset.eventCount)} 事件 · ${String(task.dataset.stepCount)} 步 · 窗口 ${String(task.dataset.wallUs)}µs`);
  lines.push(`瓶颈：${task.bottleneck.label}（${String(task.bottleneck.score)}/100，范围 ${task.bottleneck.scope}）`);
  lines.push('');
  lines.push('## 1. 目标算子');
  if (task.operators.length === 0) {
    lines.push('- （本优化项不针对单个算子：属于采集/口径类改造）');
  } else {
    for (const operator of task.operators) lines.push(`- \`${operator.name}\` — ${operator.why}`);
  }
  lines.push('');
  lines.push('## 2. 证据');
  lines.push(`- 判定依据：${task.rationale}`);
  lines.push(`- 该优化项要改善：${task.expectedGain?.metric ?? '（未声明指标）'}（估算 ${String(task.expectedGain?.estimatePct ?? '—')}%，区间 ${(task.expectedGain?.rangePct ?? []).join('–')}%，置信度 ${String(task.expectedGain?.confidence ?? '—')}）`);
  lines.push('- 采集时的实测值：');
  for (const [key, value] of Object.entries(task.evidence.indicators)) {
    lines.push(`  - ${key} = ${String(value)}`);
  }
  lines.push('');
  lines.push('## 3. 使用哪些 skills（按顺序执行）');
  task.skills.chain.forEach((skill, index) => {
    lines.push(`${String(index + 1)}. \`${skill.name}\`（${skill.role === 'primary' ? '主' : '支撑'}${skill.bundle === undefined ? '' : ` · ${skill.bundle}`}）— ${skill.purpose}`);
  });
  if (task.skills.missing.length > 0) {
    lines.push('');
    lines.push(`> 缺少 skills：${task.skills.missing.map((entry) => entry.name).join('、')}。安装：${task.installHint?.npx ?? ''}`);
  }
  lines.push('');
  lines.push('## 4. 验收标准（与第 6 步对比口径一致）');
  if (task.acceptance.length === 0) {
    lines.push('- 本优化项没有可直接量化的目标指标：以"证据是否补齐"为验收（见验证方法）。');
  } else {
    lines.push('| 指标 | 当前值 | 预期收益 | 判定阈值 | 方向 |');
    lines.push('| --- | --- | --- | --- | --- |');
    for (const row of task.acceptance) {
      lines.push(`| ${row.label} | ${String(row.current)} ${row.unit} | ${String(row.expectedPct ?? '—')}% | ≥ ${String(row.thresholdPct)}% | ${row.direction === 'higher' ? '越高越好' : row.direction === 'lower' ? '越低越好' : '仅背景'} |`);
    }
  }
  lines.push('');
  lines.push(`验收方法：${task.acceptanceHowTo}`);
  lines.push('');
  lines.push('## 5. 风险与验证方法');
  lines.push(`- 验证：${task.verification}`);
  lines.push(`- 风险：${task.risk}`);
  lines.push('');
  lines.push('## 6. 执行要求');
  lines.push('- 先加载上面的 skills（`ascend-profiling` 用于复核证据，`ascend-ops` 用于算子实现与基准，`ascend-base` 用于环境检查）；');
  lines.push('- 改动前先跑一次基准/采集作为对照，改动后用同一命令复测，把两次结果填进上面的验收表；');
  lines.push('- 没有 NPU 的环境请在目标机器上执行硬件相关步骤，不要用估算值代替实测；');
  lines.push('- 完成后回填：改了什么、实测值、是否达到阈值、下一步建议。');
  return lines.join('\n');
}

/**
 * Write the package into the workspace as Markdown + JSON.
 *
 * @param {object} task - package from {@link buildHandoff}.
 * @param {object} options - `{ workspace, allowOutside }`.
 * @returns {{ok: boolean, dir?: string, markdownPath?: string, jsonPath?: string, error?: string}} write result.
 */
export function writeHandoff(task, { workspace, allowOutside = false } = {}) {
  if (task?.ok !== true) return { ok: false, error: task?.error ?? '没有可写入的任务包' };
  const base = resolve(workspace ?? process.cwd());
  const dir = join(base, TASK_DIR);
  if (!allowOutside) {
    const root = resolve(base);
    if (dir !== root && !dir.startsWith(root + sep)) {
      return { ok: false, error: `任务目录超出会话工作区：${dir}` };
    }
  }
  const stamp = String(task.createdAt ?? new Date().toISOString()).replace(/[:.]/g, '-');
  const stem = `${stamp}-${task.adviceId.replace(/[^\w.-]+/g, '-')}`;
  const markdownPath = join(dir, `${stem}.md`);
  const jsonPath = join(dir, `${stem}.json`);
  mkdirSync(dir, { recursive: true });
  writeFileSync(markdownPath, `${task.prompt}\n`, 'utf8');
  writeFileSync(jsonPath, `${JSON.stringify(task, null, 2)}\n`, 'utf8');
  return { ok: true, dir, markdownPath, jsonPath };
}

/**
 * Read back a previously written handoff (used by tests and the report).
 *
 * @param {string} jsonPath - path written by {@link writeHandoff}.
 * @returns {object} parsed package.
 */
export function readHandoff(jsonPath) {
  return JSON.parse(readFileSync(jsonPath, 'utf8'));
}
