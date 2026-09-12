import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixture = join(here, 'fixtures', 'ascend-trace_view.sample.json');

/**
 * The fixture is a prefix of the real 635 KB Ascend `trace_view.json` committed
 * in the Ascend/mstt repository, cut mid-event. It exercises three things at
 * once: a bare top-level JSON array (no `traceEvents` wrapper), `ts` as a
 * decimal string, and a file truncated by an interrupted export.
 */
test('parses a real truncated Ascend trace_view.json prefix', async () => {
  const buffer = readFileSync(fixture);
  const result = await parseProfileSet({ inputs: [{ name: 'trace_view.json', buffer }] });

  assert.equal(result.ok, true, JSON.stringify(result.errors));
  assert.ok(result.events.length > 100, `expected many events, got ${String(result.events.length)}`);
  assert.equal(result.validation.profileType, 'ascend');
  assert.ok(result.warnings.some((warning) => /截断/.test(warning)), 'truncation must be disclosed');

  const host = result.events.filter((event) => event.device === 'host');
  const device = result.events.filter((event) => event.device === 'device');
  assert.ok(host.length > 0, 'host events must be recognized');
  assert.equal(device.length, 0, 'this prefix contains only host (cpu_op) events');

  const first = result.events[0];
  assert.equal(first.name, 'aten::empty');
  assert.equal(first.tsUs, 1704161511420306.5);
  assert.equal(first.lane.startsWith('host:'), true);
  assert.equal(result.tables.length, 0);
});

test('builds a dataset from the real trace prefix', async () => {
  const buffer = readFileSync(fixture);
  const parse = await parseProfileSet({ inputs: [{ name: 'trace_view.json', buffer }] });
  const dataset = buildDataset(parse, { maxLanesPerGroup: 40 });

  assert.ok(dataset.events.length > 100);
  assert.equal(dataset.laneGroups.length, 1, 'only the host group exists in this prefix');
  assert.equal(dataset.laneGroups[0].id, 'host');
  assert.ok(dataset.lanes[0].rows.length > 5);
  assert.equal(dataset.meta.window.start, 0, 'the axis is rebased onto the first event');
  assert.ok(dataset.meta.window.end > 0);
  assert.ok(dataset.meta.totalsSource.length > 0);
  assert.equal(dataset.categories.items.length, 5);
  const sum = dataset.categories.items.reduce((total, item) => total + item.totalUs, 0);
  assert.ok(Math.abs(sum - dataset.categories.totalUs) < 1);
  assert.ok(dataset.ranking.byTotal.length > 0);
  assert.equal(dataset.ranking.byTotal[0].rank, 1);
  assert.ok(dataset.ranking.byTotal[0].shareOfOpsPct > 0);
  assert.ok(dataset.phases.steps.length >= 1);
});

test('rejects an unrelated file with an actionable error', async () => {
  const buffer = Buffer.from('id,name,value\n1,foo,2\n', 'utf8');
  const result = await parseProfileSet({ inputs: [{ name: 'random.csv', buffer }] });
  assert.equal(result.ok, false);
  assert.ok(result.errors.length > 0);
  assert.ok(result.errors.some((error) => /vLLM-Ascend/.test(error) || /未识别/.test(error) || /不受支持/.test(error)));
});

test('rejects msprof sqlite databases with guidance', async () => {
  const buffer = Buffer.concat([Buffer.from('SQLite format 3\0', 'utf8'), Buffer.alloc(64)]);
  const result = await parseProfileSet({ inputs: [{ name: 'analysis.db', buffer }] });
  assert.equal(result.ok, false);
  assert.ok(result.warnings.some((warning) => /msprof|SQLite|数据库/.test(warning)));
});
