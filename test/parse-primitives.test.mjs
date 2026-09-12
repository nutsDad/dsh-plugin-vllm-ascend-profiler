import test from 'node:test';
import assert from 'node:assert/strict';

import { JsonStreamScanner, scanJsonText } from '../lib/parse/jsonstream.js';
import { CsvReader, buildHeaderIndex, decodeText, parseCsvText, unitOfHeader } from '../lib/parse/csv.js';

/** Collect every element of a document fed in fixed-size chunks. */
function collect(text, { chunkSize = text.length || 1, arrayKeys, maxElements } = {}) {
  const elements = [];
  const scanner = new JsonStreamScanner({
    onElement: (element) => elements.push(element),
    maxElements,
  });
  for (let at = 0; at < text.length; at += chunkSize) scanner.push(text.slice(at, at + chunkSize));
  scanner.end();
  const filtered = arrayKeys === undefined
    ? elements
    : elements.filter((element) => (element.key ?? element.path[element.path.length - 1]) === undefined
      || arrayKeys.includes(element.key ?? element.path[element.path.length - 1]));
  return { elements: filtered, truncated: scanner.truncated, error: scanner.error };
}

test('scanner emits traceEvents elements and ignores object scalars', () => {
  const doc = JSON.stringify({
    traceEvents: [{ ph: 'X', name: 'a' }, { ph: 'X', name: 'b' }],
    displayTimeUnit: 'ms',
    otherData: { version: 1 },
  });
  const { elements } = collect(doc);
  assert.equal(elements.length, 2);
  assert.deepEqual(elements.map((element) => JSON.parse(element.text).name), ['a', 'b']);
  assert.equal(elements[0].key, 'traceEvents');
});

test('scanner handles a bare top-level array', () => {
  const doc = '[{"ph":"X","name":"a"},{"ph":"X","name":"b","args":{"i":[1,2,3]}}]';
  const { elements } = collect(doc);
  assert.equal(elements.length, 2);
  const second = JSON.parse(elements[1].text);
  assert.deepEqual(second.args.i, [1, 2, 3]);
});

test('nested arrays inside one element do not become elements themselves', () => {
  const doc = '[{"name":"matmul","args":{"input_shapes":[[1,2],[3,4]],"output_shapes":[[1,4]]}}]';
  const { elements } = collect(doc);
  assert.equal(elements.length, 1);
  assert.deepEqual(JSON.parse(elements[0].text).args.input_shapes, [[1, 2], [3, 4]]);
});

test('scanner survives arbitrary chunk boundaries and escapes', () => {
  const events = [
    { ph: 'X', name: 'AllReduce', args: { note: 'quote " inside', path: 'C:\\a\\b' } },
    { ph: 'X', name: 'MatMul', args: { shape: '[(1, 1024), (1024, 4096)]' } },
    { ph: 'M', name: 'process_name', args: { name: 'Device 0' } },
  ];
  const doc = JSON.stringify({ traceEvents: events });
  for (const chunkSize of [1, 2, 3, 7, 13, 64]) {
    const { elements, error } = collect(doc, { chunkSize });
    assert.equal(error, undefined, `chunk ${String(chunkSize)}: ${String(error)}`);
    assert.equal(elements.length, events.length, `chunk ${String(chunkSize)}`);
    assert.deepEqual(JSON.parse(elements[0].text), events[0]);
    assert.deepEqual(JSON.parse(elements[2].text), events[2]);
  }
});

test('scanner emits scalar array elements', () => {
  const doc = '{"a":[1,2,"three",true,null]}';
  const { elements } = collect(doc);
  assert.equal(elements.length, 5);
  assert.deepEqual(elements.map((element) => JSON.parse(element.text)), [1, 2, 'three', true, null]);
});

test('scanner stops at the element budget and reports it', () => {
  const doc = JSON.stringify({ traceEvents: Array.from({ length: 10 }, (_, index) => ({ name: `op${String(index)}` })) });
  let elements = 0;
  const scanner = new JsonStreamScanner({ onElement: () => { elements += 1; }, maxElements: 4 });
  scanner.push(doc);
  scanner.end();
  assert.equal(elements, 4);
  assert.equal(scanner.truncated, true);
  assert.match(String(scanner.truncateReason), /上限/);
});

test('scanner reports an oversized element without buffering it', () => {
  const doc = JSON.stringify({ traceEvents: [{ name: 'big', args: { blob: 'x'.repeat(5000) } }, { name: 'small' }] });
  const seen = [];
  const scanner = new JsonStreamScanner({ onElement: (element) => seen.push(element.kind), maxElementBytes: 100 });
  scanner.push(doc);
  scanner.end();
  assert.deepEqual(seen, ['oversized', 'container']);
});

test('scanJsonText parses values and filters by array key', () => {
  const doc = JSON.stringify({ metadata: [{ pid: 1, name: 'Host' }], traceEvents: [{ name: 'op' }] });
  const values = [];
  const stats = scanJsonText(doc, { arrayKeys: ['traceEvents'], onValue: (value) => values.push(value) });
  assert.equal(stats.delivered, 1);
  assert.deepEqual(values, [{ name: 'op' }]);
});

test('CSV reader parses quoted fields, CRLF, and detects delimiters', () => {
  const text = 'a,b,c\r\n"x,1","he said ""hi""",3\r\n,,5\r\n';
  const table = parseCsvText(text);
  assert.deepEqual(table.header, ['a', 'b', 'c']);
  assert.deepEqual(table.rows[0], ['x,1', 'he said "hi"', '3']);
  assert.deepEqual(table.rows[1], ['', '', '5']);
});

test('CSV reader keeps records split across chunks', () => {
  const text = 'name,dur\n"MatMul, fused",12.5\nAdd,3\n';
  const rows = [];
  const reader = new CsvReader({ onRecord: (record) => rows.push(record) });
  for (const char of text) reader.push(char);
  reader.end();
  assert.deepEqual(rows[1], ['MatMul, fused', '12.5']);
  assert.deepEqual(rows[2], ['Add', '3']);
});

test('CSV reader honors the row budget', () => {
  const rows = [];
  const reader = new CsvReader({ onRecord: (record) => rows.push(record), maxRows: 3 });
  reader.push('h1,h2\n1,2\n3,4\n5,6\n7,8\n');
  reader.end();
  assert.equal(rows.length, 3);
  assert.equal(reader.truncated, true);
});

test('duration units are read from headers', () => {
  assert.equal(unitOfHeader('Duration(us)'), 'us');
  assert.equal(unitOfHeader('Total Time(ms)'), 'ms');
  assert.equal(unitOfHeader('Task Duration(us)'), 'us');
  assert.equal(unitOfHeader('Duration'), undefined);
});

test('header aliases tolerate CANN column drift', () => {
  const aliases = { name: ['Op Name', 'Kernel Name'], durationUs: ['Duration(us)', 'Task Duration(us)'] };
  const { index, unmatched } = buildHeaderIndex(['Device_id', 'Kernel Name', 'Task Duration(us)', 'Extra'], aliases);
  assert.equal(index.name, 1);
  assert.equal(index.durationUs, 2);
  assert.deepEqual(unmatched, ['Device_id', 'Extra']);
});

test('text decoding strips a BOM and reports the encoding', () => {
  const decoded = decodeText(Buffer.from('\uFEFFname,dur\n', 'utf8'));
  assert.equal(decoded.text, 'name,dur\n');
  assert.equal(decoded.encoding, 'utf-8');
});

test('operator classification prefers the name over the trace category', async () => {
  const { classifyOperator, normalizeOperatorName } = await import('../lib/model/classify.js');

  // A D2H copy mislabelled `HostToDevice` in `cat` must stay a D2H copy: the copy
  // bottleneck keys on the direction, and direction comes from the name.
  assert.equal(classifyOperator({ name: 'aclrtMemcpyAsync_D2H', cat: 'HostToDevice', device: 'device' }).subtype, 'd2h');
  assert.equal(classifyOperator({ name: 'aclrtMemcpyAsync_H2D', cat: 'DeviceToHost', device: 'device' }).subtype, 'h2d');
  // With no direction in the name, the category decides.
  assert.equal(classifyOperator({ name: 'aten::_to_copy', cat: 'HostToDevice', device: 'host' }).category, 'copy');
  assert.equal(classifyOperator({ name: 'aten::copy_', cat: 'cpu_op', device: 'host' }).category, 'schedule');

  // Collections are recognized in every spelling the tooling emits.
  for (const name of ['AllReduce', 'HcclAllreduce', 'hcom_allReduce__428_0_1', 'ReduceScatter']) {
    assert.equal(classifyOperator({ name, device: 'device' }).category, 'comm', name);
  }
  assert.equal(normalizeOperatorName('hcom_allReduce__428_0_1'), 'AllReduce');
  assert.equal(normalizeOperatorName('HcclAllreduce'), 'AllReduce');
  assert.equal(normalizeOperatorName('aten::addmm'), 'addmm');
  assert.equal(normalizeOperatorName('MatMulV2_1_0'), 'MatMulV2');

  // Device vs host decides compute vs framework, and the `cpu_op` category alone
  // must never turn the whole host timeline into CPU kernels.
  assert.equal(classifyOperator({ name: 'MatMulV2', device: 'device' }).category, 'compute');
  assert.equal(classifyOperator({ name: 'aten::empty', device: 'host', cat: 'cpu_op' }).subtype, 'aten');
  assert.equal(classifyOperator({ name: 'aten::empty', device: 'host', cat: 'cpu_op' }).role, 'framework');
});
