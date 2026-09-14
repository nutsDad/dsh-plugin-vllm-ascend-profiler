import test from 'node:test';
import assert from 'node:assert/strict';
import { readdirSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';
import { analyzeDataset, compareCaptures, COMPARE_ADVICE_IDS } from '../lib/analysis/index.js';

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

const metric = (comparison, key) => comparison.metrics.find((row) => row.key === key);

/**
 * `host-schedule-bound` and `host-schedule-bound-optimized` are the same workload
 * before/after the graph-mode + dispatch fixes, so the comparison must show a
 * large step-time win, a busier device and a satisfied graph-mode advice.
 */
test('the before/after comparison reports the optimisation that was applied', async () => {
  const before = await loadScenario('host-schedule-bound');
  const after = await loadScenario('host-schedule-bound-optimized');
  const comparison = compareCaptures({ before, after });

  // Headline: the step got much shorter and the device much busier.
  assert.ok(comparison.headline.stepPct < -30, `step change ${String(comparison.headline.stepPct)}%`);
  assert.ok(comparison.headline.busyDeltaPts > 10, `busy delta ${String(comparison.headline.busyDeltaPts)}pt`);
  assert.equal(comparison.headline.stepImproved, true);
  assert.match(comparison.headline.summary, /单步墙钟下降/);
  assert.match(comparison.headline.summary, /瓶颈类型未变/);

  // Per-metric direction handling: a falling "lower is better" metric improves,
  // while the rising exposed-communication share does not.
  assert.equal(metric(comparison, 'avgStepUs').improved, true);
  assert.equal(metric(comparison, 'deviceBusyPct').improved, true);
  assert.equal(metric(comparison, 'hostExclusivePerStepUs').improved, true);
  assert.ok(metric(comparison, 'hostExclusivePerStepUs').improvementPct > 50);
  assert.equal(metric(comparison, 'commExposedPct').improved, false, 'exposed comm rose, that is not an improvement');
  assert.equal(metric(comparison, 'commOverlapPct').improved, false);
  assert.equal(metric(comparison, 'commPerStep').improved, undefined, 'neutral metrics carry no verdict');
  assert.ok(comparison.warnings.some((warning) => /占比指标反向变化/.test(warning)), 'ratio metrics must be flagged');

  // Bottleneck: same type, lower score, and the fix is visible per category/operator.
  assert.equal(comparison.bottleneck.changed, false);
  assert.ok(comparison.bottleneck.scoreDelta < 0, `score delta ${String(comparison.bottleneck.scoreDelta)}`);
  const schedule = comparison.categories.find((row) => row.id === 'schedule');
  assert.ok(schedule.deltaUs < 0, 'host scheduling time must fall');
  assert.ok(comparison.operators.improved.length > 0, 'at least one operator saves time');
  assert.ok(comparison.operators.improved[0].improvementUs > 0);
  assert.ok(comparison.operators.improved.every((row) => row.deltaUs < 0));

  // Advice verification: graph mode targets the dispatch count, which halved.
  const graphMode = comparison.recommendations.find((item) => item.id === 'host.enable-graph-mode');
  assert.ok(graphMode !== undefined, 'the before capture recommends graph mode');
  assert.equal(graphMode.verdict, 'achieved', JSON.stringify({ observed: graphMode.observedPct, threshold: graphMode.thresholdPct }));
  assert.equal(graphMode.stillRecommended, false, 'the after capture no longer lists it');
  assert.ok(graphMode.targets.some((target) => target.key === 'dispatchPerStep'));
  assert.match(graphMode.note, /达到判定阈值/);

  // No NaN may survive into the JSON the page receives.
  const serialized = JSON.stringify(comparison);
  assert.ok(!serialized.includes('null,"deltaPct":null') && !/NaN/.test(serialized));
  assert.ok(serialized.length > 2000);
});

test('the comparison refuses to present unlike captures as an improvement', async () => {
  // 83% prefill vs 10% prefill, 6 steps vs 20 steps: provably not the same load.
  const before = await loadScenario('prefill-compute-bound');
  const after = await loadScenario('host-schedule-bound');
  const comparison = compareCaptures({ before, after });

  assert.equal(comparison.comparability.comparable, false);
  assert.equal(comparison.comparability.level, 'low');
  assert.ok(comparison.comparability.notes.some((note) => /推理步数相差/.test(note.text)), 'step-count mismatch must be reported');
  assert.ok(comparison.comparability.notes.some((note) => /Prefill\/Decode 时间构成不同/.test(note.text)), 'phase-mix mismatch must be reported');
  assert.ok(comparison.comparability.notes.some((note) => /主导瓶颈类型不同/.test(note.text)), 'a bottleneck type change must be flagged for review');
  assert.ok(comparison.warnings.length >= 3, comparison.warnings.join(' | '));
  assert.equal(comparison.bottleneck.changed, true);
  assert.match(comparison.headline.summary, /瓶颈类型由/);

  // A 20% step-count gap is enough to warn as well (same file set, same phases).
  const stepMismatch = compareCaptures({ before: await loadScenario('host-schedule-bound'), after: await loadScenario('decode-comm-bound') });
  assert.equal(stepMismatch.comparability.comparable, false);
  assert.ok(stepMismatch.comparability.notes.some((note) => /推理步数相差 20%/.test(note.text)));
});

test('the same capture compared with itself is a no-op', async () => {
  const before = await loadScenario('host-schedule-bound');
  const comparison = compareCaptures({ before, after: before });
  assert.equal(comparison.headline.stepPct, 0);
  assert.equal(comparison.bottleneck.changed, false);
  assert.equal(comparison.bottleneck.scoreDelta, 0);
  for (const row of comparison.metrics) {
    assert.equal(row.deltaAbs, 0, `${row.key} must show no change`);
    if (row.direction === 'neutral') assert.equal(row.improved, undefined, `${row.key} is context, not a verdict`);
    else assert.equal(row.improved, false, `${row.key} must not be reported as an improvement`);
  }
  assert.equal(comparison.operators.improved.length, 0);
  assert.equal(comparison.operators.regressed.length, 0);
  assert.equal(comparison.comparability.comparable, true);
});

test('every recommendation a scenario can produce has a comparison mapping', async () => {
  const seen = new Set();
  for (const scenario of ['decode-comm-bound', 'prefill-compute-bound', 'host-schedule-bound']) {
    const side = await loadScenario(scenario);
    for (const item of side.analysis.steps.actions.items) {
      seen.add(item.id);
      assert.ok(COMPARE_ADVICE_IDS.includes(item.id) || item.id.startsWith('common.'),
        `${item.id} is neither mapped in compare.js nor a confidence-only advice`);
    }
  }
  assert.ok(seen.size >= 3, `expected several advice ids, saw ${[...seen].join(', ')}`);
});

test('recommendations without a comparable metric are reported as unknown, not as failures', async () => {
  const before = await loadScenario('host-schedule-bound');
  const after = await loadScenario('host-schedule-bound-optimized');
  const comparison = compareCaptures({ before, after });
  for (const item of comparison.recommendations) {
    assert.ok(['achieved', 'partial', 'missed', 'unknown'].includes(item.verdict), item.verdict);
    if (item.id.startsWith('common.')) {
      assert.equal(item.verdict, 'unknown');
      assert.match(item.note, /采集质量|缺少可比数值/);
    }
  }
});
