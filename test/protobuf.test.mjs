import test from 'node:test';
import assert from 'node:assert/strict';

import { decodeProtoMessage, extractProtoEvents, splitDelimitedStream } from '../lib/parse/protobuf.js';
import { parseProfileSet } from '../lib/parse/index.js';

/** Minimal protobuf wire encoder, so the decoder is tested against real bytes. */
function encodeVarint(value) {
  const bytes = [];
  let rest = value;
  while (rest > 0x7f) {
    bytes.push((rest & 0x7f) | 0x80);
    rest = Math.floor(rest / 128);
  }
  bytes.push(rest);
  return Buffer.from(bytes);
}

function fieldVarint(field, value) {
  return Buffer.concat([encodeVarint((field << 3) | 0), encodeVarint(value)]);
}

function fieldString(field, text) {
  const body = Buffer.from(text, 'utf8');
  return Buffer.concat([encodeVarint((field << 3) | 2), encodeVarint(body.length), body]);
}

function fieldMessage(field, body) {
  return Buffer.concat([encodeVarint((field << 3) | 2), encodeVarint(body.length), body]);
}

/** One Ascend-style record: name (1), start (2), duration (3), plus a nested op type (4.1). */
function sampleRecord({ name = 'MatMulV2', start = 1704161511420306, dur = 13 } = {}) {
  return Buffer.concat([
    fieldString(1, name),
    fieldVarint(2, start),
    fieldVarint(3, dur),
    fieldMessage(4, fieldString(1, 'AI_CORE')),
  ]);
}

test('decodes the protobuf wire format generically', () => {
  const buffer = sampleRecord();
  const { fields, warnings } = decodeProtoMessage(buffer);
  assert.deepEqual(warnings, []);
  const byField = new Map(fields.map((field) => [field.field, field]));
  assert.equal(byField.get(1).value, 'MatMulV2');
  assert.equal(byField.get(2).value, 1704161511420306);
  assert.equal(byField.get(3).value, 13);
  assert.equal(Array.isArray(byField.get(4).nested), true);
});

test('rejects malformed buffers instead of guessing', () => {
  const truncated = Buffer.concat([encodeVarint((1 << 3) | 2), encodeVarint(64), Buffer.from('abc', 'utf8')]);
  const { warnings } = decodeProtoMessage(truncated);
  assert.ok(warnings.length > 0, 'a length prefix past the end must be reported');
});

/** Frame messages the way Ascend writes per-event record streams. */
function frame(...messages) {
  return Buffer.concat(messages.flatMap((message) => [encodeVarint(message.length), message]));
}

test('the exact field map turns the decode into a precise mapping', () => {
  const buffer = frame(
    sampleRecord({ name: 'FusedInferAttentionScore', start: 1704161511500000, dur: 842 }),
    sampleRecord({ name: 'AllReduce', start: 1704161511600000, dur: 48 }),
  );
  const result = extractProtoEvents({
    name: 'kernel_details.proto',
    buffer,
    fieldMap: { name: 1, start: 2, duration: 3 },
  });
  assert.equal(result.confidence, 'mapped');
  assert.equal(result.events.length, 2);
  assert.equal(result.events[0].name, 'FusedInferAttentionScore');
  assert.equal(result.events[0].tsUs, 1704161511500000);
  assert.equal(result.events[0].durUs, 842);
  assert.equal(result.events[1].normalizedName, 'AllReduce');
  assert.equal(result.events[1].device, 'device');
  assert.equal(result.warnings.some((warning) => warning.includes('启发式')), false);
});

test('the heuristic recognizer finds records without a schema and says so', () => {
  const buffer = frame(
    sampleRecord({ name: 'MatMulV2', start: 1704161511420306, dur: 13 }),
    sampleRecord({ name: 'RmsNorm', start: 1704161511421000, dur: 24 }),
  );
  const result = extractProtoEvents({ name: 'device0.bin', buffer });
  assert.equal(result.confidence, 'low');
  assert.equal(result.events.length, 2);
  assert.equal(result.events[0].name, 'MatMulV2');
  assert.equal(result.events[0].tsUs, 1704161511420306);
  assert.equal(result.events[0].durUs, 13);
  assert.ok(result.warnings.some((warning) => warning.includes('启发式')), 'low confidence must be disclosed');
});

test('an unframed buffer is decoded as one message and says so', () => {
  const buffer = sampleRecord({ name: 'MatMulV2', start: 1704161511420306, dur: 13 });
  const result = extractProtoEvents({ name: 'single.bin', buffer });
  assert.equal(result.events.length, 1);
  assert.equal(result.events[0].name, 'MatMulV2');
  assert.ok(
    result.warnings.some((warning) => warning.includes('未检测到长度前缀消息流')),
    'unframed input must be disclosed instead of silently mis-segmented',
  );
});

test('an end timestamp is accepted in place of a duration', () => {
  const buffer = frame(
    Buffer.concat([
      fieldString(1, 'AllGather'),
      fieldVarint(2, 1704161511420306),
      fieldVarint(3, 1704161511420906),
    ]),
  );
  const result = extractProtoEvents({ name: 'comm.pb', buffer });
  assert.equal(result.events.length, 1);
  assert.equal(result.events[0].durUs, 600);
});

test('a message with no timestamp yields nothing', () => {
  const buffer = Buffer.concat([fieldString(1, 'some-metadata'), fieldVarint(2, 42)]);
  const result = extractProtoEvents({ name: 'meta.bin', buffer });
  assert.equal(result.events.length, 0);
  assert.ok(result.warnings.some((warning) => warning.includes('未识别出可用事件记录')));
});

test('length-delimited streams are split into messages', () => {
  const first = sampleRecord({ name: 'MatMulV2' });
  const second = sampleRecord({ name: 'Add' });
  const stream = Buffer.concat([
    encodeVarint(first.length), first,
    encodeVarint(second.length), second,
  ]);
  const { messages } = splitDelimitedStream(stream);
  assert.equal(messages.length, 2);
  assert.equal(decodeProtoMessage(messages[1]).fields[0].value, 'Add');
});

test('a proto-only upload is accepted and its events reach the dataset', async () => {
  // Ascend vocabulary in a string field provides the content evidence weight.
  const record = () => Buffer.concat([
    fieldString(1, 'MatMulV2'),
    fieldString(5, 'Ascend AI_CORE kernel, torch_npu'),
    fieldVarint(2, 1704161511420306),
    fieldVarint(3, 130),
  ]);
  const buffer = frame(record(), record(), record());
  const result = await parseProfileSet({ inputs: [{ name: 'kernel_details.proto', buffer }] });
  assert.equal(result.ok, true, JSON.stringify(result.errors));
  assert.equal(result.proto.length, 1);
  assert.equal(result.proto[0].confidence, 'low');
  assert.equal(result.events.length, 3);
  assert.equal(result.events[0].heuristic, true);
  assert.ok(result.warnings.some((warning) => warning.includes('启发式识别')));
});

test('an unrelated binary is rejected with an actionable error', async () => {
  const buffer = Buffer.from([0x00, 0x01, 0x02, 0x03, 0xff, 0xfe, 0x7f, 0x80]);
  const result = await parseProfileSet({ inputs: [{ name: 'random.bin', buffer }] });
  assert.equal(result.ok, false);
  assert.ok(result.errors.length > 0);
});
