/**
 * In-memory expansion of the archive formats profiling bundles arrive in.
 *
 * Ascend tooling and users hand over profiles as directories, but the same
 * content is frequently packed as `.zip` (MindStudio Insight export) or
 * `.tar.gz` (`msprof` result tarball). Both are unpacked here with `node:zlib`
 * only — no third-party archive dependency — and every entry is bounded so a
 * decompression bomb cannot exhaust memory.
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/archive
 */

import { gunzipSync, inflateRawSync } from 'node:zlib';

/** Guard against archives that claim an implausible expansion. */
export const DEFAULT_ENTRY_LIMIT = 512 * 1024 * 1024;
/** Guard against an archive with an absurd number of entries. */
export const DEFAULT_ENTRY_COUNT_LIMIT = 256;

const EOCD_SIGNATURE = 0x06054b50;
const EOCD64_LOCATOR_SIGNATURE = 0x07064b50;
const CENTRAL_SIGNATURE = 0x02014b50;
const LOCAL_SIGNATURE = 0x04034b50;

/**
 * Detect an archive format from its magic bytes (falling back to the name).
 * @param {string} name - file name.
 * @param {Buffer|Uint8Array} buffer - file bytes.
 * @returns {'zip'|'gzip'|'tar'|'plain'} archive format.
 */
export function archiveFormatOf(name, buffer) {
  const bytes = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  if (bytes.length >= 4 && bytes[0] === 0x50 && bytes[1] === 0x4b) return 'zip';
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) return 'gzip';
  if (/\.tar$/i.test(name)) return 'tar';
  return 'plain';
}

/**
 * Expand one artifact into its member files.
 *
 * @param {object} input - expansion input.
 * @param {string} input.name - artifact name.
 * @param {Buffer|Uint8Array} input.buffer - artifact bytes.
 * @param {number} [input.entryLimit] - per-entry byte ceiling.
 * @param {number} [input.entryCountLimit] - entry count ceiling.
 * @returns {{ files: {name: string, buffer: Buffer}[], warnings: string[], skipped: string[] }} expansion result.
 */
export function expandArchive({ name, buffer, entryLimit = DEFAULT_ENTRY_LIMIT, entryCountLimit = DEFAULT_ENTRY_COUNT_LIMIT }) {
  const bytes = Buffer.isBuffer(buffer) ? buffer : Buffer.from(buffer);
  const format = archiveFormatOf(name, bytes);
  const warnings = [];
  const skipped = [];
  if (format === 'zip') return expandZip(bytes, { entryLimit, entryCountLimit, warnings, skipped });
  if (format === 'gzip') {
    try {
      const inflated = gunzipSync(bytes, { maxOutputLength: entryLimit });
      const inner = name.replace(/\.gz$/i, '');
      if (archiveFormatOf(inner, inflated) === 'tar') {
        const tar = expandTar(inflated, { entryLimit, entryCountLimit, warnings, skipped });
        return { files: tar.files, warnings: [...warnings, ...tar.warnings], skipped: [...skipped, ...tar.skipped] };
      }
      return { files: [{ name: inner, buffer: inflated }], warnings, skipped };
    } catch (error) {
      warnings.push(`gzip 解压失败：${error instanceof Error ? error.message : String(error)}`);
      return { files: [], warnings, skipped };
    }
  }
  if (format === 'tar') return expandTar(bytes, { entryLimit, entryCountLimit, warnings, skipped });
  return { files: [{ name, buffer: bytes }], warnings, skipped };
}

/**
 * Read a ZIP archive from its central directory.
 * @param {Buffer} bytes - archive bytes.
 * @param {object} options - limits and diagnostics sinks.
 * @returns {{ files: {name: string, buffer: Buffer}[], warnings: string[], skipped: string[] }} expansion result.
 */
function expandZip(bytes, { entryLimit, entryCountLimit, warnings, skipped }) {
  const files = [];
  const eocd = findEocd(bytes);
  if (eocd === undefined) {
    warnings.push('ZIP 结构不完整（未找到中央目录），已跳过该压缩包。');
    return { files, warnings, skipped };
  }
  let entryCount = bytes.readUInt16LE(eocd + 10);
  let centralOffset = bytes.readUInt32LE(eocd + 16);
  if (entryCount === 0xffff || centralOffset === 0xffffffff) {
    const zip64 = findZip64(bytes, eocd);
    if (zip64 !== undefined) {
      entryCount = Number(zip64.entries);
      centralOffset = Number(zip64.offset);
    }
  }
  if (entryCount > entryCountLimit) {
    warnings.push(`ZIP 内含 ${String(entryCount)} 个成员文件，超过上限 ${String(entryCountLimit)}，仅读取前 ${String(entryCountLimit)} 个。`);
  }
  let at = centralOffset;
  for (let index = 0; index < entryCount && index < entryCountLimit; index += 1) {
    if (at + 46 > bytes.length || bytes.readUInt32LE(at) !== CENTRAL_SIGNATURE) break;
    const method = bytes.readUInt16LE(at + 10);
    const compressedSize = bytes.readUInt32LE(at + 20);
    const uncompressedSize = bytes.readUInt32LE(at + 24);
    const nameLength = bytes.readUInt16LE(at + 28);
    const extraLength = bytes.readUInt16LE(at + 30);
    const commentLength = bytes.readUInt16LE(at + 32);
    let localOffset = bytes.readUInt32LE(at + 42);
    const name = bytes.subarray(at + 46, at + 46 + nameLength).toString('utf8');
    const extra = bytes.subarray(at + 46 + nameLength, at + 46 + nameLength + extraLength);
    let size = uncompressedSize;
    if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff || localOffset === 0xffffffff) {
      const zip64 = parseZip64Extra(extra);
      size = zip64.uncompressedSize ?? size;
      localOffset = zip64.offset ?? localOffset;
    }
    at += 46 + nameLength + extraLength + commentLength;
    if (name.endsWith('/')) continue;
    if (size > entryLimit) {
      skipped.push(`${name}（解压后 ${formatBytes(size)}，超过单文件上限）`);
      continue;
    }
    const inflated = readZipEntry(bytes, localOffset, method, size, warnings, name);
    if (inflated === undefined) {
      skipped.push(name);
      continue;
    }
    files.push({ name, buffer: inflated });
  }
  if (files.length === 0 && skipped.length === 0) warnings.push('ZIP 内没有可读的成员文件。');
  return { files, warnings, skipped };
}

function readZipEntry(bytes, localOffset, method, size, warnings, name) {
  if (localOffset + 30 > bytes.length || bytes.readUInt32LE(localOffset) !== LOCAL_SIGNATURE) {
    warnings.push(`${name}：ZIP 本地头无效，已跳过。`);
    return undefined;
  }
  const nameLength = bytes.readUInt16LE(localOffset + 26);
  const extraLength = bytes.readUInt16LE(localOffset + 28);
  const dataStart = localOffset + 30 + nameLength + extraLength;
  const data = bytes.subarray(dataStart);
  try {
    if (method === 0) return Buffer.from(data.subarray(0, size));
    if (method === 8) return inflateRawSync(data, { maxOutputLength: Math.max(size, 1024) * 2 + 1024 });
    warnings.push(`${name}：不支持的压缩方法 ${String(method)}，已跳过。`);
    return undefined;
  } catch (error) {
    warnings.push(`${name}：解压失败（${error instanceof Error ? error.message : String(error)}）。`);
    return undefined;
  }
}

function findEocd(bytes) {
  const minimum = Math.max(0, bytes.length - 66000);
  for (let at = bytes.length - 22; at >= minimum; at -= 1) {
    if (bytes.readUInt32LE(at) === EOCD_SIGNATURE) return at;
  }
  return undefined;
}

function findZip64(bytes, eocd) {
  for (let at = eocd - 20; at >= 0 && at >= eocd - 128; at -= 1) {
    if (bytes.readUInt32LE(at) !== EOCD64_LOCATOR_SIGNATURE) continue;
    const recordOffset = Number(bytes.readBigUInt64LE(at + 8));
    if (recordOffset + 56 > bytes.length) return undefined;
    return {
      entries: bytes.readBigUInt64LE(recordOffset + 32),
      offset: bytes.readBigUInt64LE(recordOffset + 48),
    };
  }
  return undefined;
}

function parseZip64Extra(extra) {
  let at = 0;
  while (at + 4 <= extra.length) {
    const id = extra.readUInt16LE(at);
    const size = extra.readUInt16LE(at + 2);
    const body = extra.subarray(at + 4, at + 4 + size);
    if (id === 0x0001) {
      let cursor = 0;
      const uncompressedSize = body.length >= 8 ? body.readBigUInt64LE(cursor) : undefined;
      cursor += 8;
      const compressedSize = body.length >= 16 ? body.readBigUInt64LE(cursor) : undefined;
      cursor += 8;
      const offset = body.length >= 24 ? body.readBigUInt64LE(cursor) : undefined;
      return { uncompressedSize, compressedSize, offset };
    }
    at += 4 + size;
  }
  return {};
}

/**
 * Read a POSIX tar archive.
 * @param {Buffer} bytes - archive bytes.
 * @param {object} options - limits and diagnostics sinks.
 * @returns {{ files: {name: string, buffer: Buffer}[], warnings: string[], skipped: string[] }} expansion result.
 */
function expandTar(bytes, { entryLimit, entryCountLimit, warnings, skipped }) {
  const files = [];
  let at = 0;
  let count = 0;
  while (at + 512 <= bytes.length && count < entryCountLimit) {
    const header = bytes.subarray(at, at + 512);
    if (header.every((byte) => byte === 0)) break;
    const name = readTarString(header, 0, 100);
    const prefix = readTarString(header, 345, 155);
    const sizeText = readTarString(header, 124, 12).trim();
    const typeFlag = String.fromCharCode(header[156] ?? 0);
    const size = Number.parseInt(sizeText, 8);
    const fullName = prefix === '' ? name : `${prefix}/${name}`;
    at += 512;
    if (!Number.isFinite(size) || size < 0) {
      warnings.push(`tar 成员 ${fullName} 的长度字段非法，停止解析。`);
      break;
    }
    if (typeFlag === '0' || typeFlag === '\0' || typeFlag === '') {
      if (size > entryLimit) {
        skipped.push(`${fullName}（解压后 ${formatBytes(size)}，超过单文件上限）`);
      } else {
        files.push({ name: fullName, buffer: Buffer.from(bytes.subarray(at, at + size)) });
        count += 1;
      }
    }
    at += Math.ceil(size / 512) * 512;
  }
  if (files.length === 0 && skipped.length === 0) warnings.push('tar 内没有可读的普通文件。');
  return { files, warnings, skipped };
}

function readTarString(header, offset, length) {
  const slice = header.subarray(offset, offset + length);
  const end = slice.indexOf(0);
  return slice.subarray(0, end === -1 ? slice.length : end).toString('utf8');
}

/**
 * Format a byte count for messages.
 * @param {number} bytes - byte count.
 * @returns {string} human-readable size.
 */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '未知大小';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)}${units[unit]}`;
}
