/**
 * HTTP helpers for the analyzer's routes: bounded body reads, JSON responses,
 * static file serving with traversal rejection, and request logging that never
 * floods the log.
 *
 * The plugin owns a URL prefix on the DSH webserver and answers every request
 * under it, so these helpers are deliberately small and dependency-free.
 *
 * @module dsh-plugin-vllm-ascend-profiler/http
 */

import { createReadStream } from 'node:fs';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';

/** Content types for the assets the page loads. */
const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.map': 'application/json; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
};

/**
 * Send a JSON response.
 * @param {import('node:http').ServerResponse} res - response.
 * @param {number} status - HTTP status.
 * @param {unknown} body - JSON-serializable body.
 * @param {Record<string, string>} [headers] - extra headers.
 */
export function sendJson(res, status, body, headers = {}) {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json; charset=utf-8',
    'cache-control': 'no-store',
    'content-length': Buffer.byteLength(payload),
    ...headers,
  });
  res.end(payload);
}

/**
 * Send a text response (used for Markdown export).
 * @param {import('node:http').ServerResponse} res - response.
 * @param {number} status - HTTP status.
 * @param {string} body - text body.
 * @param {Record<string, string>} [headers] - extra headers.
 */
export function sendText(res, status, body, headers = {}) {
  res.writeHead(status, {
    'content-type': 'text/plain; charset=utf-8',
    'cache-control': 'no-store',
    ...headers,
  });
  res.end(body);
}

/**
 * Send an HTML response.
 * @param {import('node:http').ServerResponse} res - response.
 * @param {number} status - HTTP status.
 * @param {string} body - HTML body.
 */
export function sendHtml(res, status, body) {
  res.writeHead(status, { 'content-type': 'text/html; charset=utf-8', 'cache-control': 'no-store' });
  res.end(body);
}

/**
 * Read a request body with a byte ceiling.
 *
 * The ceiling is enforced while reading, so an oversized upload cannot exhaust
 * memory before being rejected: the stream is destroyed as soon as the limit is
 * crossed and `tooLarge` is reported.
 *
 * @param {import('node:http').IncomingMessage} req - request.
 * @param {number} maxBytes - byte ceiling.
 * @returns {Promise<{buffer: Buffer, tooLarge: boolean, bytes: number}>} the body.
 */
export async function readBody(req, maxBytes) {
  const chunks = [];
  let bytes = 0;
  let tooLarge = false;
  try {
    for await (const chunk of req) {
      bytes += chunk.length;
      if (bytes > maxBytes) {
        tooLarge = true;
        req.destroy();
        break;
      }
      chunks.push(chunk);
    }
  } catch {
    // A destroyed socket is an expected outcome of the size guard.
  }
  return { buffer: Buffer.concat(chunks), tooLarge, bytes };
}

/**
 * Read and parse a JSON body.
 * @param {import('node:http').IncomingMessage} req - request.
 * @param {number} [maxBytes] - byte ceiling.
 * @returns {Promise<{ok: boolean, value?: unknown, error?: string, tooLarge?: boolean}>} parse result.
 */
export async function readJsonBody(req, maxBytes = 4 * 1024 * 1024) {
  const body = await readBody(req, maxBytes);
  if (body.tooLarge) return { ok: false, tooLarge: true, error: `请求体超过 ${String(maxBytes)} 字节上限` };
  if (body.bytes === 0) return { ok: true, value: {} };
  try {
    return { ok: true, value: JSON.parse(body.buffer.toString('utf8')) };
  } catch (error) {
    return { ok: false, error: `JSON 解析失败：${error instanceof Error ? error.message : String(error)}` };
  }
}

/**
 * Serve one file from a static root with traversal rejection.
 *
 * @param {import('node:http').ServerResponse} res - response.
 * @param {string} root - absolute static root.
 * @param {string} pathname - decoded request path (relative to the root).
 * @param {object} [options] - `fallback` file served for a directory hit.
 * @returns {Promise<boolean>} whether a file was served.
 */
export async function serveStaticFile(res, root, pathname, options = {}) {
  const resolvedRoot = resolve(root);
  let target = resolve(normalize(join(resolvedRoot, pathname)));
  if (target !== resolvedRoot && !target.startsWith(resolvedRoot + sep)) {
    res.writeHead(403);
    res.end();
    return true;
  }
  let info = await stat(target).catch(() => undefined);
  if (info?.isDirectory() === true) {
    target = join(target, options.fallback ?? 'index.html');
    info = await stat(target).catch(() => undefined);
  }
  if (info === undefined || !info.isFile()) return false;
  const type = MIME[extname(target).toLowerCase()] ?? 'application/octet-stream';
  res.writeHead(200, {
    'content-type': type,
    'cache-control': type.startsWith('text/html') ? 'no-store' : 'no-cache',
    'content-length': info.size,
  });
  await new Promise((resolvePromise, rejectPromise) => {
    const stream = createReadStream(target);
    stream.on('error', rejectPromise);
    stream.on('end', resolvePromise);
    stream.pipe(res);
  });
  return true;
}

/**
 * Read a text asset from a static root.
 * @param {string} root - absolute root.
 * @param {string} relativePath - path inside the root.
 * @returns {Promise<string>} file contents.
 */
export async function readStaticText(root, relativePath) {
  return readFile(join(root, relativePath), 'utf8');
}

/**
 * Build a rate-limited logger so a busy page cannot flood the DSH log with one
 * line per request.
 *
 * @param {{info: Function, warn: Function}} logger - cordis logger.
 * @param {number} [windowMs] - aggregation window.
 * @returns {(message: string, key?: string) => void} log function.
 */
export function createThrottledLogger(logger, windowMs = 5000) {
  const seen = new Map();
  return (message, key = message) => {
    const now = Date.now();
    const last = seen.get(key);
    if (last !== undefined && now - last < windowMs) {
      seen.set(`${key}:suppressed`, (seen.get(`${key}:suppressed`) ?? 0) + 1);
      return;
    }
    const suppressed = seen.get(`${key}:suppressed`) ?? 0;
    seen.set(key, now);
    seen.delete(`${key}:suppressed`);
    logger.info(suppressed > 0 ? `${message}（忽略 ${String(suppressed)} 条同类日志）` : message);
  };
}

/**
 * Format a byte count for diagnostics messages.
 * @param {number} bytes - byte count.
 * @returns {string} human-readable size.
 */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)}${units[unit]}`;
}
