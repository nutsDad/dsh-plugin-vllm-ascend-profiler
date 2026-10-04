import test from 'node:test';
import assert from 'node:assert/strict';

import { crc32, createZip } from '../lib/zip.js';

/** Extract the entry names from a stored ZIP, using only its own headers. */
function readNames(archive) {
  const names = [];
  let offset = 0;
  while (offset + 30 <= archive.length && archive.readUInt32LE(offset) === 0x04034b50) {
    const nameLength = archive.readUInt16LE(offset + 26);
    const extraLength = archive.readUInt16LE(offset + 28);
    const size = archive.readUInt32LE(offset + 18);
    names.push(archive.subarray(offset + 30, offset + 30 + nameLength).toString('utf8'));
    offset += 30 + nameLength + extraLength + size;
  }
  return names;
}

/** Read one entry back out of a stored ZIP. */
function readEntry(archive, wanted) {
  let offset = 0;
  while (offset + 30 <= archive.length && archive.readUInt32LE(offset) === 0x04034b50) {
    const crc = archive.readUInt32LE(offset + 14);
    const size = archive.readUInt32LE(offset + 18);
    const nameLength = archive.readUInt16LE(offset + 26);
    const extraLength = archive.readUInt16LE(offset + 28);
    const name = archive.subarray(offset + 30, offset + 30 + nameLength).toString('utf8');
    const start = offset + 30 + nameLength + extraLength;
    if (name === wanted) return { data: archive.subarray(start, start + size), crc };
    offset = start + size;
  }
  return undefined;
}

test('crc32 matches the standard test vectors', () => {
  assert.equal(crc32(Buffer.from('')), 0);
  assert.equal(crc32(Buffer.from('123456789')), 0xcbf43926);
  assert.equal(crc32(Buffer.from('The quick brown fox jumps over the lazy dog')), 0x414fa339);
});

test('createZip writes a readable archive with correct CRCs', () => {
  const files = [
    { name: 'csrc/ops/demo/op_kernel/demo.cpp', data: '// kernel\nint main() { return 0; }\n' },
    { name: 'csrc/ops/demo/design.md', data: '# design\n' },
    { name: 'README.md', data: Buffer.from('二进制也算一份', 'utf8') },
  ];
  const archive = createZip(files, { date: new Date('2026-01-02T03:04:06Z') });

  assert.equal(archive.readUInt32LE(0), 0x04034b50, 'starts with a local file header');
  assert.deepEqual(readNames(archive), files.map((file) => file.name));
  for (const file of files) {
    const entry = readEntry(archive, file.name);
    assert.ok(entry !== undefined, `${file.name} must be present`);
    assert.equal(entry.crc, crc32(Buffer.from(file.data)), `${file.name} CRC`);
    assert.equal(entry.data.toString('utf8'), Buffer.from(file.data).toString('utf8'));
  }
  // End-of-central-directory record carries the entry count.
  const end = archive.subarray(archive.length - 22);
  assert.equal(end.readUInt32LE(0), 0x06054b50);
  assert.equal(end.readUInt16LE(8), files.length);
  assert.equal(end.readUInt16LE(10), files.length);
  // Stored, not deflated: sizes match and the method field is 0.
  assert.equal(archive.readUInt16LE(8), 0);
});

test('createZip handles an empty archive and long paths', () => {
  const empty = createZip([]);
  assert.equal(empty.length, 22, 'an empty archive is just the end record');
  assert.equal(readNames(empty).length, 0);

  const deep = 'a/'.repeat(40) + 'deep.cpp';
  const archive = createZip([{ name: deep, data: 'x' }]);
  assert.deepEqual(readNames(archive), [deep]);
});
