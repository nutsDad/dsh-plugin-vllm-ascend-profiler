import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';
import { analyzeDataset, COMPARE_ADVICE_IDS } from '../lib/analysis/index.js';
import {
  ADVICE_SKILLS,
  SKILL_BUNDLES,
  TASK_DIR,
  buildHandoff,
  bundleOfSkill,
  pickOperators,
  probeSkills,
  readHandoff,
  skillRoots,
  writeHandoff,
} from '../lib/handoff.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, 'fixtures');

/** Load one scenario fixture directory through the real pipeline. */
async function loadScenario(scenario) {
  const directory = join(fixtures, scenario);
  const inputs = readdirSync(directory).map((name) => ({ name, buffer: readFileSync(join(directory, name)) }));
  const parse = await parseProfileSet({ inputs });
  assert.equal(parse.ok, true, `${scenario}: ${JSON.stringify(parse.errors)}`);
  const dataset = buildDataset(parse);
  return { dataset, analysis: analyzeDataset(dataset), datasetId: scenario, label: scenario };
}

/** A DSH skill root: `<root>/skills/<skill>/SKILL.md` (the `user-dsh` layout). */
function skillRoot(names) {
  const root = mkdtempSync(join(tmpdir(), 'vap-skills-'));
  for (const name of names) {
    const directory = join(root, 'skills', name);
    mkdirSync(directory, { recursive: true });
    writeFileSync(join(directory, 'SKILL.md'), `---\nname: ${name}\n---\n`, 'utf8');
  }
  return root;
}

/**
 * A compute-bound capture plus the advice that targets it must produce a handoff
 * aimed at the real hot operators, with the kernel-development skill chain.
 */
test('a compute advice hands off to the AscendC chain with its hot operators', async () => {
  const { dataset, analysis, datasetId, label } = await loadScenario('prefill-compute-bound');
  const task = buildHandoff({ adviceId: 'compute.quantize', dataset, analysis, datasetId, label, roots: [] });

  assert.equal(task.ok, true);
  assert.equal(task.focus, 'kernel');
  // MatMulV2 dominates the device compute time in this fixture.
  assert.equal(task.operators[0].name, 'MatMulV2');
  assert.ok(task.operators[0].sharePct > 40, `share ${String(task.operators[0].sharePct)}%`);
  assert.deepEqual(task.skills.chain.map((entry) => entry.name).slice(0, 3), ['ascendc', 'ascend-opplugin', 'npu-op-benchmark']);
  assert.equal(task.skills.chain[0].role, 'primary');
  assert.deepEqual(task.skills.bundles, ['ascend-ops', 'ascend-profiling']);
  // Acceptance uses the same metric the comparison will judge later.
  assert.equal(task.acceptance.length, 1);
  assert.equal(task.acceptance[0].key, 'computeUs');
  assert.equal(task.acceptance[0].current, analysis.indicators.computeUs);
  assert.equal(task.acceptance[0].thresholdPct, 15); // half of the projected 30%
  assert.match(task.prompt, /## 3\. 使用哪些 skills/);
  assert.match(task.prompt, /`MatMulV2`/);
});

/**
 * A host-dispatch advice is not a kernel problem: it must point at the host ops
 * that are dispatched per step and drive torch_npu first.
 */
test('a host dispatch advice hands off to torch_npu and the dispatch hot spots', async () => {
  const { dataset, analysis, datasetId, label } = await loadScenario('host-schedule-bound');
  const task = buildHandoff({ adviceId: 'host.reduce-dispatch', dataset, analysis, datasetId, label, roots: [] });

  assert.equal(task.ok, true);
  assert.equal(task.focus, 'dispatch');
  assert.equal(task.skills.chain[0].name, 'torch_npu');
  assert.ok(task.operators.length > 0);
  assert.ok(['host', 'device'].includes(task.operators[0].group), `group ${String(task.operators[0].group)}`);
  // Dispatch ranking is by call count, so the attached operator is a hot one.
  assert.ok(task.operators[0].count >= 100, `count ${String(task.operators[0].count)}`);
  assert.deepEqual(task.acceptance.map((row) => row.key), ['dispatchPerStep', 'hostScheduleUs']);
});

test('an unknown advice id is refused with the known ids', async () => {
  const { dataset, analysis, datasetId, label } = await loadScenario('host-schedule-bound');
  const task = buildHandoff({ adviceId: 'nope.nope', dataset, analysis, datasetId, label, roots: [] });
  assert.equal(task.ok, false);
  assert.match(task.error, /未知的优化项/);
  assert.ok(task.knownIds.includes('host.enable-graph-mode'));
});

/**
 * The skill probe must report what DSH would actually discover, and the package
 * must say so instead of failing at run time.
 */
test('the skill probe reports installed and missing bundles', async () => {
  const { dataset, analysis, datasetId, label } = await loadScenario('host-schedule-bound');
  const partialRoot = skillRoot(['torch_npu', 'profiling-analysis']);
  const partialRoots = skillRoots({ dshHome: partialRoot, workspace: partialRoot });
  const probe = probeSkills(['torch_npu', 'profiling-analysis', 'ascendc'], partialRoots);
  assert.deepEqual(probe.installed.map((entry) => entry.name), ['torch_npu', 'profiling-analysis']);
  assert.deepEqual(probe.missing.map((entry) => entry.name), ['ascendc']);
  assert.equal(probe.missing[0].bundle, 'ascend-ops');

  // A partially installed machine still produces a package — it just says what is
  // missing (here: the re-collection skill the graph-mode advice needs).
  const partial = buildHandoff({ adviceId: 'host.enable-graph-mode', dataset, analysis, datasetId, label, roots: partialRoots });
  assert.equal(partial.status, 'needs-skills');
  assert.deepEqual(partial.skills.missing.map((entry) => entry.name), ['pytorch-profiling-collection']);
  assert.match(partial.installHint.repo, /awesome-ascend-skills/);

  const blocked = buildHandoff({ adviceId: 'host.reduce-dispatch', dataset, analysis, datasetId, label, roots: partialRoots });
  assert.equal(blocked.status, 'needs-skills');
  assert.ok(blocked.skills.missing.some((entry) => entry.name === 'ascendc'));
  assert.match(blocked.prompt, /缺少 skills/);

  // With the whole chain installed the same request comes back executable.
  const fullRoot = skillRoot([...new Set([...partial.skills.chain, ...blocked.skills.chain].map((entry) => entry.name))]);
  const fullRoots = skillRoots({ dshHome: fullRoot, workspace: fullRoot });
  const ready = buildHandoff({ adviceId: 'host.enable-graph-mode', dataset, analysis, datasetId, label, roots: fullRoots });
  assert.equal(ready.status, 'ready');
  assert.equal(ready.installHint, undefined);
  assert.ok(ready.skills.chain.every((entry) => entry.installed));

  rmSync(partialRoot, { recursive: true, force: true });
  rmSync(fullRoot, { recursive: true, force: true });
});

/** Every advice the analysis can emit must map to a skill chain (no silent gaps). */
test('every comparable advice id has a skill mapping and an operator focus', () => {
  for (const adviceId of COMPARE_ADVICE_IDS) {
    const mapping = ADVICE_SKILLS[adviceId];
    assert.ok(mapping !== undefined, `no skill mapping for ${adviceId}`);
    assert.ok(mapping.primary.length > 0, `no primary skill for ${adviceId}`);
    for (const skill of [...mapping.primary, ...mapping.support]) {
      assert.ok(bundleOfSkill(skill) !== undefined, `${skill} is not in any bundle`);
    }
  }
  // Bundles carry the upstream names, so an install hint can name them.
  assert.deepEqual(Object.keys(SKILL_BUNDLES), ['ascend-base', 'ascend-profiling', 'ascend-ops']);
  assert.ok(SKILL_BUNDLES['ascend-ops'].skills.includes('ascendc'));
});

/** pickOperators ranks by the focus: time for kernels, call count for dispatch. */
test('operator selection follows the advice focus', async () => {
  const { dataset } = await loadScenario('decode-comm-bound');
  const comm = pickOperators(dataset, ADVICE_SKILLS['comm.batch-amortize']);
  assert.ok(comm.length > 0);
  assert.ok(comm.every((operator) => operator.category === 'comm'));
  const schedule = pickOperators(dataset, ADVICE_SKILLS['host.enable-graph-mode'], 2);
  assert.ok(schedule.length <= 2);
  assert.ok(schedule.every((operator) => operator.category === 'schedule'));
  assert.equal(pickOperators(dataset, ADVICE_SKILLS['common.phase-isolation']).length, 0);
});

/** The package is written twice (Markdown for reading, JSON for tooling). */
test('the handoff is written into the workspace and reads back', async () => {
  const { dataset, analysis, datasetId, label } = await loadScenario('prefill-compute-bound');
  const task = buildHandoff({ adviceId: 'compute.quantize', dataset, analysis, datasetId, label, roots: [], createdAt: '2026-01-02T03:04:05.000Z' });
  const workspace = mkdtempSync(join(tmpdir(), 'vap-ws-'));
  const written = writeHandoff(task, { workspace });

  assert.equal(written.ok, true);
  assert.equal(dirname(written.markdownPath), join(workspace, TASK_DIR));
  assert.ok(existsSync(written.markdownPath));
  assert.ok(existsSync(written.jsonPath));
  const markdown = readFileSync(written.markdownPath, 'utf8');
  assert.match(markdown, /算子优化任务/);
  assert.match(markdown, /ascendc/);
  const back = readHandoff(written.jsonPath);
  assert.equal(back.adviceId, 'compute.quantize');
  assert.equal(back.artifacts, undefined); // the route adds paths, the builder does not
  assert.equal(back.acceptance[0].key, 'computeUs');

  rmSync(workspace, { recursive: true, force: true });
});
