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

/**
 * The same device work is described by several tables at once: `op_statistic.csv`
 * (CANN per-operator summary), `kernel_details.csv` (per kernel) and
 * `operator_details.csv` (per operator instance). Summing them counts that work
 * two or three times, which would inflate the totals whenever the CSV is the
 * authoritative source (a sampled trace) and turn the trace-vs-CSV cross-check
 * into nonsense. One table must win.
 */
test('one CSV table wins instead of summing the same work three times', async () => {
  const csv = (lines) => Buffer.from(`${lines.join('\n')}\n`, 'utf8');
  const inputs = [
    {
      name: 'op_statistic.csv',
      buffer: csv([
        'Device_id,OP Type,Core Type,Count,Total Time(us),Avg Time(us),Ratio(%)',
        '0,MatMulV2,AI_CORE,10,300,30,100',
      ]),
    },
    {
      name: 'kernel_details.csv',
      buffer: csv([
        'Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num',
        '0,MatMulV2,AI_CORE,AI_CORE,1000,10,1,24',
        '0,MatMulV2,AI_CORE,AI_CORE,1100,10,1,24',
        '0,MatMulV2,AI_CORE,AI_CORE,1200,10,1,24',
      ]),
    },
    {
      name: 'operator_details.csv',
      buffer: csv([
        'Device_id,Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us)',
        '0,MatMulV2,,,0,0,10,10',
        '0,MatMulV2,,,0,0,10,10',
      ]),
    },
  ];
  const parse = await parseProfileSet({ inputs });
  assert.equal(parse.ok, true, JSON.stringify(parse.errors));
  const dataset = buildDataset(parse);
  const row = dataset.operators.find((entry) => entry.name === 'MatMulV2');
  assert.ok(row !== undefined);

  // 300 (summary) — not 300 + 30 (kernels) + 20 (operator instances).
  assert.equal(row.totalUs, 300, 'the summary table is the source of truth');
  assert.equal(row.csvTotalUs, 300);
  assert.equal(row.count, 10, 'the call count must not be tripled either');
  // Every table it appears in stays visible as evidence.
  assert.deepEqual([...row.sources.csvTables].sort(), ['kernel_details', 'op_statistic', 'operator_details']);
  // A material disagreement between tables is reported, naming both sides.
  assert.ok(dataset.meta.warnings.some((warning) => /多张 CSV 表累计耗时相差较大/.test(warning) && /op_statistic 300\.0µs/.test(warning)),
    `expected a cross-table warning, got: ${dataset.meta.warnings.join(' | ')}`);
});

test('the shipped fixture cross-checks clean against the trace', async () => {
  // Regression guard for the bug above on real artifact shapes: an operator that
  // appears in all three tables must cross-check clean against the trace.
  const directory = join(here, 'fixtures', 'host-schedule-bound');
  const names = ['op_statistic.csv', 'kernel_details.csv', 'operator_details.csv', 'trace_view.json'];
  const inputs = names.map((name) => ({ name, buffer: readFileSync(join(directory, name)) }));
  const parse = await parseProfileSet({ inputs });
  const dataset = buildDataset(parse);
  const row = dataset.operators.find((entry) => entry.name === 'MatMulV2');
  assert.ok(row !== undefined && row.traceTotalUs > 0);
  assert.ok(row.sources.csvTables.includes('op_statistic') && row.sources.csvTables.includes('kernel_details'));
  assert.ok(Math.abs(row.csvTotalUs - row.traceTotalUs) / row.traceTotalUs < 0.01,
    `csv=${String(row.csvTotalUs)} trace=${String(row.traceTotalUs)}`);
  assert.ok(Math.abs(row.crossCheckPct ?? 100) < 5, `cross-check must be clean, got ${String(row.crossCheckPct)}%`);
  assert.equal(row.count, row.csvCount, 'the trace count and the summary count must agree');
});
