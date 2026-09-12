/**
 * Operator classification for Ascend NPU profiles.
 *
 * Every event and every statistical row is mapped to one of four top-level
 * classes the analyzer reasons about — compute (计算), communication (通信),
 * data copy (数据拷贝), and host scheduling (调度) — plus a `other` bucket for
 * synchronisation primitives that belong to none of them.
 *
 * A second dimension, `role`, records *what the class means physically*, so the
 * advice engine can separate "the NPU is busy doing math" from "the host is
 * busy launching math":
 *
 * | role          | side   | meaning                                              |
 * |---------------|--------|------------------------------------------------------|
 * | `npu-kernel`  | device | AI Core / AI Vector kernel execution                 |
 * | `communication`| device| HCCL collective on the NPU                           |
 * | `transfer`    | device | H2D / D2H / D2D memcpy                               |
 * | `framework`   | host   | python / ATen / engine dispatch, sampler, scheduler  |
 * | `cpu-kernel`  | host   | real CPU-side compute or CPU memcpy                  |
 *
 * @module dsh-plugin-vllm-ascend-profiler/model/classify
 */

/** The four reasoning classes plus the fallback bucket. */
export const CATEGORIES = Object.freeze({
  compute: { id: 'compute', label: '计算算子', short: '计算', color: '#4f7cff' },
  comm: { id: 'comm', label: '通信算子', short: '通信', color: '#f2994a' },
  copy: { id: 'copy', label: '数据拷贝算子', short: '拷贝', color: '#27ae60' },
  schedule: { id: 'schedule', label: '调度算子', short: '调度', color: '#9b59b6' },
  other: { id: 'other', label: '其他/同步', short: '其他', color: '#7f8c9b' },
});

/** Category display order used by every chart. */
export const CATEGORY_ORDER = ['compute', 'comm', 'copy', 'schedule', 'other'];

/**
 * Collective communication patterns, checked before everything else.
 * Ascend emits both the HCCL entry point (`AllReduce`) and the torch_npu
 * wrapper (`hcom_allReduce_...`), so both spellings are listed.
 */
const COMM_RULES = [
  { subtype: 'allreduce', pattern: /(?:^|[^a-z])(?:all_?reduce|hcclallreduce|hcom_allreduce|reduce_?scatter|hcclreducescatter|hcom_reduce_scatter)(?:[^a-z]|$)/i },
  { subtype: 'allgather', pattern: /(?:^|[^a-z])(?:all_?gather|hcclallgather|hcom_allgather|_allgather_base)(?:[^a-z]|$)/i },
  { subtype: 'alltoall', pattern: /(?:^|[^a-z])(?:all_?to_?all|all2all|hcclalltoall|hcom_alltoall|alltoallv|dispatch|combine)(?:[^a-z]|$)/i },
  { subtype: 'broadcast', pattern: /(?:^|[^a-z])(?:broadcast|hcclbroadcast|hcom_broadcast)(?:[^a-z]|$)/i },
  { subtype: 'sendrecv', pattern: /(?:^|[^a-z])(?:send|recv|isend|irecv|p2p|batchsendrecv|hcclsend|hcclrecv)(?:[^a-z]|$)/i },
  { subtype: 'barrier', pattern: /(?:^|[^a-z])(?:barrier|hcclbarrier)(?:[^a-z]|$)/i },
  { subtype: 'notify', pattern: /notify[_\s-]?wait|notify[_\s-]?record|wait[_\s-]?notify|hcom_notify/i },
  { subtype: 'generic', pattern: /(?:^|[^a-z])(?:hccl|hcom_|mc2|all_?reduce_?mc2|moe_?distribute)(?:[^a-z]|$)/i },
];

/** Data movement patterns, checked after communication. */
const COPY_RULES = [
  { subtype: 'h2d', pattern: /(?:h2d|host[_\s-]?to[_\s-]?device|memcpy_?host2device|aclrtmemcpyh2d)/i },
  { subtype: 'd2h', pattern: /(?:d2h|device[_\s-]?to[_\s-]?host|memcpy_?device2host|aclrtmemcpyd2h)/i },
  { subtype: 'd2d', pattern: /(?:d2d|device[_\s-]?to[_\s-]?device|memcpy_?device2device)/i },
  { subtype: 'p2p', pattern: /(?:p2p|peer[_\s-]?to[_\s-]?peer|kv[_\s-]?transfer|pull[_\s-]?kv|push[_\s-]?kv)/i },
  { subtype: 'other', pattern: /(?:^|[^a-z])(?:memcpy|memcopy|aclnnmemcpy|aclrtmemcpy|copy_?async|_to_copy|movement|dma_copy)(?:[^a-z]|$)/i },
];

/** Device compute subtypes. */
const COMPUTE_RULES = [
  { subtype: 'attention', pattern: /(?:flash[_\s-]?attention|\bfa\b|attention|paged_attention|unified_attention|mla|incremental|prompt_flash|decode_attention|attn)/i },
  { subtype: 'matmul', pattern: /(?:matmul|mat_?mul|gemm|batchmatmul|linear|addmm|mm\b|fc\b|grouped_?matmul|gmm|quant_?matmul)/i },
  { subtype: 'moe', pattern: /(?:moe|expert|topk|top_k|gate|router|shared_expert)/i },
  { subtype: 'norm', pattern: /(?:rms_?norm|layer_?norm|norm\b|batchnorm)/i },
  { subtype: 'activation', pattern: /(?:silu|gelu|relu|sigmoid|tanh|swiglu|swish|activation|softmax|log_?softmax)/i },
  { subtype: 'quant', pattern: /(?:quant|dequant|dynamic_?quant|per_?token|int8|int4|fp8|w8a8|antiquant)/i },
  { subtype: 'reduce', pattern: /(?:reduce_?(?:sum|max|mean|min)|cumsum|argmax|argmin)/i },
  { subtype: 'elementwise', pattern: /(?:^|[^a-z])(?:add|sub|mul|div|pow|exp|log|sqrt|rsqrt|cast|transpose|permute|reshape|slice|concat|split|pad|gather|scatter|index|select|where|masked|clip|clamp|neg|abs|sign|copy|fill|zeros|ones|embedding|rope|rotary)/i },
  { subtype: 'conv', pattern: /(?:conv\d?d?|convolution|pool)/i },
];

/** Host-side framework patterns: dispatcher, engine, python, samplers. */
const FRAMEWORK_RULES = [
  { subtype: 'engine', pattern: /(?:execute_model|model_runner|model\.forward|forward\b|scheduler|prepare_input|prepare_inputs|profile_run|warmup|dummy_run)/i },
  { subtype: 'sampler', pattern: /(?:sampler|sampling|logits|_to_list|\.item\(|tolist|argmax_sample|penalties|top_?p|top_?k_?sample)/i },
  { subtype: 'python', pattern: /(?:^python|py::|python_function|__call__|nn\.module|\bmodule\b)/i },
  { subtype: 'aten', pattern: /^(?:aten::|torch_npu::|torch::|vllm::|aclnn)/i },
  { subtype: 'sync', pattern: /(?:synchronize|stream_?sync|device_?sync|aclrt_?synchronize|wait_?event|event_?record)/i },
  { subtype: 'memory', pattern: /(?:block_manager|block_table|kv_cache|allocate|malloc|free\b|cache_engine)/i },
];

/** Longest suffix chain of instance ids: `MatMul_1_0`, `AllReduce__1234_0_1`. */
const INSTANCE_SUFFIX = /(?:_+\d+){1,4}$/;
/** Torch scope prefixes stripped from host operator names. */
const SCOPE_PREFIX = /^(?:aten::|torch_npu::|torch::|vllm::|nn\.modules?\.|c10::|aclnn)/i;
/** Hash-like suffixes some Ascend kernels carry. */
const HASH_SUFFIX = /_[0-9a-f]{8,}$/i;

/**
 * Classify one operator.
 *
 * @param {object} input - classification input.
 * @param {string} input.name - operator name.
 * @param {'host'|'device'|undefined} [input.device] - owning side when known.
 * @param {string} [input.opType] - profiling `OP Type` / `Op Type` column.
 * @param {string} [input.taskType] - profiling `Task Type` column.
 * @param {string} [input.coreType] - profiling `Core Type` column.
 * @param {string} [input.cat] - trace `cat` field.
 * @returns {{ category: string, subtype: string, role: string }} classification.
 */
export function classifyOperator({ name, device, opType, taskType, coreType, cat }) {
  const raw = String(name ?? '');
  const op = String(opType ?? '');
  const task = String(taskType ?? '');
  const core = String(coreType ?? '');
  const traceCat = String(cat ?? '');
  const named = `${raw} ${op}`;
  const haystack = `${named} ${task} ${core} ${traceCat}`;

  if (/hccl|communication|collective/i.test(`${op} ${task} ${traceCat}`)) {
    return { category: 'comm', subtype: commSubtypeOf(raw), role: 'communication' };
  }
  for (const rule of COMM_RULES) {
    if (rule.pattern.test(haystack)) return { category: 'comm', subtype: rule.subtype, role: 'communication' };
  }
  // Data movement: the operator name is authoritative for the *direction*, and
  // `cat` is only a fallback. A D2H copy whose cat reads `HostToDevice` (which
  // happens as soon as one layer of tooling mislabels it) must still be a D2H
  // copy, because that direction is what the copy-bottleneck reasoning keys on.
  for (const rule of COPY_RULES) {
    if (rule.pattern.test(named)) return { category: 'copy', subtype: rule.subtype, role: 'transfer' };
  }
  for (const rule of COPY_RULES) {
    if (rule.pattern.test(haystack)) return { category: 'copy', subtype: rule.subtype, role: 'transfer' };
  }
  const side = device ?? (/(?:aicore|ai_core|aicpu|kernel|npu)/i.test(`${task} ${core} ${traceCat}`) ? 'device' : 'host');
  if (side === 'device') {
    if (/aicpu|ai_cpu/i.test(`${task} ${core}`)) {
      return { category: 'schedule', subtype: 'aicpu', role: 'framework' };
    }
    return { category: 'compute', subtype: computeSubtypeOf(raw), role: 'npu-kernel' };
  }
  // Host side: anything that is not communication or a transfer is framework
  // work — dispatcher, python, engine bookkeeping, or a launch call. A genuine
  // CPU-side kernel is labelled separately, but only when the operator name says
  // so: the `cpu_op` trace category describes *every* host event, so matching on
  // it would classify the whole host timeline as CPU kernels.
  if (/(?:^|[^a-z])(?:cpu_?(?:kernel|op|fallback|impl)|_cpu|cpukernel)(?:[^a-z]|$)/i.test(`${raw} ${op} ${core} ${task}`)) {
    return { category: 'schedule', subtype: 'cpu-kernel', role: 'cpu-kernel' };
  }
  for (const rule of FRAMEWORK_RULES) {
    if (rule.pattern.test(haystack)) return { category: 'schedule', subtype: rule.subtype, role: 'framework' };
  }
  return { category: 'schedule', subtype: 'host-other', role: 'framework' };
}

function commSubtypeOf(name) {
  for (const rule of COMM_RULES) {
    if (rule.pattern.test(name)) return rule.subtype;
  }
  return 'generic';
}

function computeSubtypeOf(name) {
  for (const rule of COMPUTE_RULES) {
    if (rule.pattern.test(name)) return rule.subtype;
  }
  return 'other';
}

/**
 * Normalize an operator name into an aggregation key.
 *
 * Removes torch scope prefixes, trailing instance ids, and hash suffixes, and
 * canonicalizes HCCL spellings so `hcom_allReduce__1234_0_1`, `HcclAllReduce`
 * and `AllReduce` aggregate into one row.
 *
 * @param {string} name - raw operator name.
 * @returns {string} aggregation key.
 */
export function normalizeOperatorName(name) {
  let value = String(name ?? '').trim();
  if (value === '') return '(unnamed)';
  value = value.replace(SCOPE_PREFIX, '');
  value = value.replace(HASH_SUFFIX, '');
  value = value.replace(INSTANCE_SUFFIX, '');
  // `hcom_allReduce__1234` style wrappers map onto the collective name.
  const hcom = /^(?:hcom[_-]|hccl[_-]?)/i.exec(value);
  if (hcom !== null) {
    const rest = value.slice(hcom[0].length);
    value = /^all[_-]?reduce/i.test(rest) ? 'AllReduce'
      : /^all[_-]?gather/i.test(rest) ? 'AllGather'
        : /^reduce[_-]?scatter/i.test(rest) ? 'ReduceScatter'
          : /^all[_-]?to[_-]?all/i.test(rest) ? 'AlltoAll'
            : /^broadcast/i.test(rest) ? 'Broadcast'
              : rest;
  }
  // `MatMul.default` / `MatMul(x2)` spellings collapse onto the base name.
  value = value.replace(/\.(?:default|out|input|Tensor|Scalar)$/, '');
  value = value.replace(/\((?:x\d+|\d+)\)$/, '');
  value = value.replace(/[,\s]+$/, '');
  return value === '' ? '(unnamed)' : value;
}

/**
 * Cheap "never sample this away" test used while streaming a huge trace:
 * communication and data-movement operators, and mstx markers, are always kept.
 *
 * @param {string} name - operator name.
 * @returns {boolean} whether the event must survive sampling.
 */
export function looksImportantOperator(name) {
  const value = String(name ?? '');
  if (value === '') return false;
  for (const rule of COMM_RULES) {
    if (rule.pattern.test(value)) return true;
  }
  for (const rule of COPY_RULES) {
    if (rule.pattern.test(value)) return true;
  }
  return /^(?:mstx|marker|step|iteration|prefill|decode)/i.test(value) || /prefill|decode/i.test(value);
}

/**
 * Human-readable subtype label for the UI.
 * @param {string} category - top-level category id.
 * @param {string} subtype - subtype id.
 * @returns {string} Chinese label.
 */
export function subtypeLabel(category, subtype) {
  const table = {
    comm: {
      allreduce: 'AllReduce 规约', allgather: 'AllGather 聚集', reducescatter: 'ReduceScatter 散射',
      alltoall: 'AllToAll 交换/MoE 分发', broadcast: 'Broadcast 广播', sendrecv: '点对点收发',
      barrier: 'Barrier 同步', notify: 'Notify/Wait 通知', generic: '其他 HCCL 通信',
    },
    copy: {
      h2d: 'Host→Device', d2h: 'Device→Host', d2d: 'Device→Device', p2p: '跨卡点对点搬运', other: '其他拷贝',
    },
    compute: {
      matmul: '矩阵乘/线性层', attention: '注意力', moe: 'MoE/专家路由', norm: '归一化', activation: '激活函数',
      quant: '量化/反量化', reduce: '归约', elementwise: '逐元素运算', conv: '卷积/池化', other: '其他计算',
    },
    schedule: {
      engine: '引擎调度', sampler: '采样', python: 'Python 层', aten: 'ATen/aclnn 派发', sync: '同步等待',
      memory: 'KV 缓存/内存管理', aicpu: 'AI CPU 算子', 'cpu-kernel': 'CPU 侧计算', 'host-other': '其他 Host 开销',
    },
    other: { default: '其他' },
  };
  return table[category]?.[subtype] ?? table[category]?.default ?? subtype;
}
