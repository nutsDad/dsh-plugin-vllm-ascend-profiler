import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';
import { analyzeDataset } from '../lib/analysis/index.js';
import { renderMarkdownReport } from '../lib/report/markdown.js';
import { renderPrintableReport } from '../lib/report/print.js';
import { buildScenario, writeFixtures } from './make-fixture.mjs';

const here = dirname(fileURLToPath(import.meta.url));
const fixtureRoot = join(here, 'fixtures');

/**
 * Scenario fixtures are generated artifacts: create them on first use so the
 * suite runs from a clean checkout (`node test/make-fixture.mjs` does the same
 * thing explicitly).
 */
if (!existsSync(join(fixtureRoot, 'decode-comm-bound'))) {
  writeFixtures(fixtureRoot);
}

/** Parse a scenario directory into a dataset. */
async function loadScenario(scenario) {
  const directory = join(fixtureRoot, scenario);
  const inputs = readdirSync(directory).map((name) => ({
    name,
    buffer: readFileSync(join(directory, name)),
  }));
  const parse = await parseProfileSet({ inputs });
  assert.equal(parse.ok, true, `${scenario}: ${JSON.stringify(parse.errors)}`);
  const dataset = buildDataset(parse);
  return { parse, dataset, analysis: analyzeDataset(dataset) };
}

test('scenario fixtures exist and are complete', () => {
  for (const scenario of ['decode-comm-bound', 'prefill-compute-bound', 'host-schedule-bound']) {
    const files = readdirSync(join(fixtureRoot, scenario));
    for (const required of ['trace_view.json', 'kernel_details.csv', 'operator_details.csv', 'op_statistic.csv', 'step_trace_time.csv']) {
      assert.ok(files.includes(required), `${scenario} is missing ${required}`);
    }
  }
});

test('all three scenarios parse with full evidence', async () => {
  for (const scenario of ['decode-comm-bound', 'prefill-compute-bound', 'host-schedule-bound']) {
    const { parse, dataset } = await loadScenario(scenario);
    assert.ok(parse.events.length > 500, `${scenario}: events`);
    assert.equal(parse.validation.profileType, 'vllm-ascend', `${scenario}: vLLM markers must be recognized`);
    assert.ok(dataset.laneGroups.length === 2, `${scenario}: both host and device lanes`);
    assert.equal(dataset.utilization.available, true, `${scenario}: utilization collected from CSV`);
    assert.ok(dataset.phases.steps.length >= 3, `${scenario}: steps derived`);
    assert.ok(dataset.categories.totalUs > 0);
  }
});

test('decode scenario is located as a communication bottleneck', async () => {
  const { analysis } = await loadScenario('decode-comm-bound');
  const overall = analysis.steps.locate.perScope[0];
  const comm = overall.candidates.find((candidate) => candidate.id === 'comm');
  const host = overall.candidates.find((candidate) => candidate.id === 'host');
  assert.ok(comm.score >= 40, `comm score ${String(comm.score)} should clear the gate`);
  assert.ok(comm.score > host.score, `comm ${String(comm.score)} should outrank host ${String(host.score)}`);
  assert.ok(analysis.steps.cause.items.some((item) => item.bottleneck === 'comm'));
  assert.ok(analysis.steps.actions.items.some((item) => item.bottleneck === 'comm'));
  const decodeScope = analysis.steps.locate.perScope.find((scope) => scope.scope === 'decode');
  assert.ok(decodeScope !== undefined, 'decode phase must be analyzed separately');
});

test('prefill scenario is located as an NPU compute bottleneck', async () => {
  const { analysis } = await loadScenario('prefill-compute-bound');
  const overall = analysis.steps.locate.perScope[0];
  const compute = overall.candidates.find((candidate) => candidate.id === 'compute');
  assert.ok(compute.score >= 40, `compute score ${String(compute.score)} should clear the gate`);
  assert.equal(compute.id, overall.primaryCandidate.id);
  assert.equal(compute.subtype, 'compute-bound', 'high MAC ratio must be reported as compute bound');
  assert.ok(analysis.steps.cause.items.some((item) => item.id.startsWith('compute.')));
});

test('host scheduling scenario is located as a host bottleneck', async () => {
  const { analysis } = await loadScenario('host-schedule-bound');
  const overall = analysis.steps.locate.perScope[0];
  const host = overall.candidates.find((candidate) => candidate.id === 'host');
  assert.ok(host.score >= 40, `host score ${String(host.score)} should clear the gate`);
  assert.equal(overall.primaryCandidate.id, 'host');
  assert.ok(analysis.steps.cause.items.some((item) => item.id.startsWith('host.')));
  const actions = analysis.steps.actions.items.map((item) => item.id);
  assert.ok(actions.includes('host.enable-graph-mode'), `expected graph-mode advice, got ${actions.join(',')}`);
});

test('every recommendation carries a rationale, actions, and a labeled gain', async () => {
  const { analysis } = await loadScenario('host-schedule-bound');
  assert.ok(analysis.steps.actions.items.length > 0);
  for (const item of analysis.steps.actions.items) {
    assert.ok(item.rationale.length > 10, `${item.id}: rationale`);
    assert.ok(item.actions.length > 0, `${item.id}: actions`);
    assert.ok(['高', '中', '低'].includes(item.priority), `${item.id}: priority`);
    assert.ok(item.verification.length > 0, `${item.id}: verification`);
    assert.ok(item.risk.length > 0, `${item.id}: risk`);
    if (item.expectedGain !== undefined) {
      assert.ok(['high', 'medium', 'low'].includes(item.expectedGain.confidence), `${item.id}: confidence`);
      assert.ok(item.expectedGain.basis.length > 0, `${item.id}: basis`);
      assert.ok(item.expectedGain.assumption.length > 0, `${item.id}: assumption`);
    }
  }
});

test('markdown report renders the whole reasoning chain', async () => {
  const { dataset, analysis } = await loadScenario('decode-comm-bound');
  const markdown = renderMarkdownReport({ dataset, analysis });
  for (const heading of ['## ① 瓶颈类型定位', '## ② 量化证据', '## ③ 根因推断', '## ④ 可落地优化方案', '## ⑤ 预期收益汇总']) {
    assert.ok(markdown.includes(heading), `missing section ${heading}`);
  }
  assert.ok(markdown.includes('## 0. 数据来源与解析说明'));
  assert.ok(markdown.includes('## 附 E. 结论边界与数据质量'));
  assert.ok(markdown.length > 4000, `report too short: ${String(markdown.length)}`);
  assert.ok(!markdown.includes('undefined%'));
  assert.ok(!markdown.includes('NaN'));
});

test('printable report is a self-contained HTML document', async () => {
  const { dataset, analysis } = await loadScenario('prefill-compute-bound');
  const html = renderPrintableReport({ dataset, analysis, autoPrint: false });
  assert.ok(html.startsWith('<!DOCTYPE html>'));
  assert.ok(html.includes('vLLM-Ascend Profiling 性能分析报告'));
  assert.ok(html.includes('① 瓶颈类型定位'));
  assert.ok(!/<script src=|<link /.test(html), 'no external resources may be required');
  assert.ok(!html.includes('undefined'));
});

test('a capture without phase markers still yields phase-scoped advice', async () => {
  const { files } = buildScenario('decode-comm-bound');
  const parse = await parseProfileSet({
    inputs: Object.entries(files).map(([name, content]) => ({ name, buffer: Buffer.from(content, 'utf8') })),
  });
  const dataset = buildDataset(parse);
  const analysis = analyzeDataset(dataset);
  assert.ok(['step_trace_time', 'engine-events', 'idle-gaps', 'whole-window'].includes(dataset.phases.source));
  assert.ok(analysis.steps.locate.perScope.length >= 2, 'overall plus at least one phase scope');
  const labels = analysis.steps.locate.perScope.map((scope) => scope.scope);
  assert.ok(labels.includes('overall'));
});

test('an explicit phase override is honored and disclosed', async () => {
  const { parse } = await loadScenario('decode-comm-bound');
  const dataset = buildDataset(parse);
  const analysis = analyzeDataset(dataset, { phaseOverride: 'decode' });
  assert.equal(analysis.options.phaseOverride, 'decode');
  assert.equal(analysis.steps.locate.perScope.length, 1);
  assert.match(analysis.steps.locate.perScope[0].scopeLabel, /人工指定为 decode/);
});
