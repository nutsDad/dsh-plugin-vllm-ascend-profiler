/**
 * API client for the analyzer page: collection upload with progress, job
 * polling, dataset retrieval, re-analysis, and report export.
 *
 * Uploads use XHR because the page shows real byte-level upload progress, which
 * `fetch` cannot report.
 */
(function attachApi(global) {
  'use strict';

  /** Route prefix of the page this script was loaded from. */
  const PREFIX = (function resolvePrefix() {
    const path = global.location.pathname;
    const cleaned = path.endsWith('/') ? path.slice(0, -1) : path;
    return cleaned === '' ? '/vllm-ascend-profiler' : cleaned;
  })();

  /**
   * Upload one file into a collection job, reporting progress.
   * @param {string} groupId - collection job id.
   * @param {File} file - the file.
   * @param {AbortSignal} [signal] - abort signal.
   * @param {(loaded: number, total: number) => void} [onProgress] - progress sink.
   * @returns {Promise<object>} the API response.
   */
  function uploadFile(groupId, file, signal, onProgress) {
    return new Promise((resolve, reject) => {
      const request = new XMLHttpRequest();
      const url = `${PREFIX}/api/jobs?group=${encodeURIComponent(groupId)}&name=${encodeURIComponent(file.name)}`;
      request.open('POST', url, true);
      request.setRequestHeader('content-type', 'application/octet-stream');
      if (onProgress !== undefined) {
        request.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) onProgress(event.loaded, event.total);
        });
      }
      request.addEventListener('load', () => {
        let payload;
        try {
          payload = JSON.parse(request.responseText);
        } catch {
          reject(new Error(`上传响应无法解析（HTTP ${String(request.status)}）`));
          return;
        }
        if (request.status >= 200 && request.status < 300) resolve(payload);
        else reject(new Error(payload.error ?? `上传失败（HTTP ${String(request.status)}）`));
      });
      request.addEventListener('error', () => reject(new Error('上传失败：网络错误')));
      request.addEventListener('abort', () => reject(new Error('上传已取消')));
      if (signal !== undefined) {
        signal.addEventListener('abort', () => request.abort());
      }
      request.send(file);
    });
  }

  /** POST JSON and parse the response, surfacing server errors verbatim. */
  async function postJson(path, body) {
    const response = await fetch(`${PREFIX}/api${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body ?? {}),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error ?? `请求失败（HTTP ${String(response.status)}）`);
    return payload;
  }

  /** GET JSON and parse the response. */
  async function getJson(path) {
    const response = await fetch(`${PREFIX}/api${path}`);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error ?? `请求失败（HTTP ${String(response.status)}）`);
    return payload;
  }

  /**
   * Poll a job until it settles.
   * @param {string} jobId - job id.
   * @param {(job: object) => void} onTick - called on every poll.
   * @param {number} [intervalMs] - poll interval.
   * @returns {Promise<object>} the finished job.
   */
  function waitForJob(jobId, onTick, intervalMs = 400) {
    return new Promise((resolve, reject) => {
      let stopped = false;
      const tick = async () => {
        if (stopped) return;
        try {
          const job = await getJson(`/jobs/${jobId}`);
          onTick(job);
          if (job.state === 'done' || job.state === 'error') {
            stopped = true;
            if (job.state === 'error') reject(Object.assign(new Error((job.errors ?? ['解析失败']).join('；')), { job }));
            else resolve(job);
            return;
          }
          setTimeout(tick, intervalMs);
        } catch (error) {
          stopped = true;
          reject(error);
        }
      };
      tick();
    });
  }

  global.VAP = global.VAP ?? {};
  Object.assign(global.VAP, {
    PREFIX,
    api: {
      uploadFile,
      postJson,
      getJson,
      waitForJob,
      health: () => getJson('/health'),
      docs: () => getJson('/docs'),
      listDatasets: () => getJson('/datasets'),
      getDataset: (id) => getJson(`/datasets/${id}`),
      analyze: (id, options) => postJson(`/datasets/${id}/analyze`, options),
      postCharts: (id, charts) => postJson(`/datasets/${id}/charts`, { charts }),
      deleteDataset: async (id) => {
        const response = await fetch(`${PREFIX}/api/datasets/${id}`, { method: 'DELETE' });
        return response.ok;
      },
      reportUrl: (id, format) => `${PREFIX}/api/datasets/${id}/${format}`,
    },
  });
})(window);
