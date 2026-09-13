/**
 * Generate realistic vLLM-Ascend profiling bundles for the three bottleneck
 * scenarios the analyzer must distinguish.
 *
 * Layout per scenario (mirroring a torch_npu export):
 *
 * ```
 * test/fixtures/<scenario>/
 *   trace_view.json            bare top-level array, ts as decimal strings
 *   kernel_details.csv         Device_id,Name,Type,Accelerator Core,Start Time(us),...
 *   operator_details.csv       Device Total Duration(us) + Host Total Duration(us)
 *   op_statistic.csv           OP Type,Count,Total Time(us),Ratio(%)
 *   step_trace_time.csv        Device_id,Step,Computing,Communication(Not Overlapped),...
 *   profiler_info_{rank}.json  rank metadata
 *   communication.json         HCCL op summaries
 * ```
 *
 * Run with `node test/make-fixture.mjs [outputRoot]`. The scenarios are:
 *
 * * `decode-comm-bound` — TP=8 decode: many short steps, a small-message
 *   AllReduce per layer that cannot overlap, high device busy, low MAC.
 * * `prefill-compute-bound` — long chunked-prefill steps dominated by MatMul
 *   and FlashAttention, high MAC utilisation, negligible communication.
 * * `host-schedule-bound` — eager decode where the host is the bottleneck:
 *   many host ops per step, device idle gaps between them, low NPU busy.
 *
 * @module dsh-plugin-vllm-ascend-profiler/test/make-fixture
 */

import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

/** Deterministic PRNG so fixtures are reproducible. */
function makeRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

/**
 * Build one scenario.
 *
 * @param {'decode-comm-bound'|'prefill-compute-bound'|'host-schedule-bound'} scenario - scenario id.
 * @param {{steps?: number, rank?: number}} [options] - overrides for short samples.
 * @returns {{files: Record<string, string>, summary: object}} generated files.
 */
export function buildScenario(scenario, options = {}) {
  const random = makeRandom(scenario.length * 7919 + 13);
  const trace = [];
  const hostStartUs = 1704161511420000;
  let cursor = hostStartUs;
  const kernelRows = [];
  const operatorRows = [];
  const opStats = new Map();
  const stepRows = [];
  const commOps = [];

  trace.push(meta('process_name', 1001, { name: 'Python' }));
  trace.push(meta('process_name', 2002, { name: 'Ascend Hardware' }));
  trace.push(meta('process_name', 'HCCL', { name: 'Communication' }));
  trace.push(meta('thread_name', 1001, { name: 'MainThread' }, 1001));

  const layers = 32;
  const config = SCENARIOS[scenario];
  const steps = options.steps ?? config.steps;
  const rank = options.rank ?? config.rank;

  for (let step = 0; step < steps; step += 1) {
    const stepStart = cursor;
    let deviceCursor = cursor + config.deviceOffsetUs;
    const deviceEvents = [];
    const hostEvents = [];
    const stepMetrics = { computing: 0, comm: 0, commNotOverlapped: 0, overlapped: 0, free: 0 };

    // ── host side: engine + dispatch ────────────────────────────────────────
    const hostStepStart = cursor;
    hostEvents.push(hostEvent('execute_model', hostStepStart, config.engineUs, { 'Sequence number': step }));
    hostEvents.push(hostEvent('_prepare_inputs', hostStepStart + 40, config.prepareUs, {}));
    hostEvents.push(hostEvent('vllm::unified_attention_metadata', hostStepStart + 60, config.metadataUs, {}));
    for (let index = 0; index < config.hostOpsPerStep; index += 1) {
      const name = pickHostOp(random);
      const ts = hostStepStart + 60 + random() * config.hostOpWindowUs;
      const dur = 3 + random() * config.hostOpJitterUs;
      hostEvents.push(hostEvent(name, ts, dur, { 'Sequence number': step }));
      if (name === 'aten::item' || name === 'aten::_local_scalar_dense') {
        hostEvents.push(hostEvent('aclrtSynchronizeStream', ts + dur, 25 + random() * 30, {}));
      }
    }
    hostEvents.push(hostEvent('Sampler.forward', hostStepStart + config.hostOpWindowUs, config.samplerUs, {}));

    // ── device side ─────────────────────────────────────────────────────────
    for (let layer = 0; layer < layers; layer += 1) {
      const matmulDur = config.matmulUs * (0.9 + random() * 0.2);
      deviceEvents.push(deviceEvent('MatMulV2', deviceCursor, matmulDur, { 'Task Type': 'AI_CORE', 'OP Type': 'MatMulV2', stream: layer % 4 }));
      stepMetrics.computing += matmulDur;
      recordOp(opStats, 'MatMulV2', 'AI_CORE', matmulDur);
      kernelRows.push(kernelRow('MatMulV2', 'AI_CORE', 'AI_CORE', deviceCursor, matmulDur, config.macRatio, config.mte2Ratio));
      deviceCursor += matmulDur;

      const attentionDur = config.attentionUs * (0.9 + random() * 0.2);
      deviceEvents.push(deviceEvent('FusedInferAttentionScore', deviceCursor, attentionDur, { 'Task Type': 'AI_CORE', 'OP Type': 'FusedInferAttentionScore', stream: layer % 4 }));
      stepMetrics.computing += attentionDur;
      recordOp(opStats, 'FusedInferAttentionScore', 'AI_CORE', attentionDur);
      kernelRows.push(kernelRow('FusedInferAttentionScore', 'AI_CORE', 'AI_CORE', deviceCursor, attentionDur, config.macRatio, config.mte2Ratio));
      deviceCursor += attentionDur;

      if (config.vectorOpsPerLayer > 0) {
        const vectorDur = config.vectorUs * (0.8 + random() * 0.4);
        deviceEvents.push(deviceEvent('RmsNorm', deviceCursor, vectorDur, { 'Task Type': 'AI_VECTOR_CORE', 'OP Type': 'RmsNorm', stream: layer % 4 }));
        stepMetrics.computing += vectorDur;
        recordOp(opStats, 'RmsNorm', 'AI_VECTOR_CORE', vectorDur);
        kernelRows.push(kernelRow('RmsNorm', 'AI_VECTOR_CORE', 'AI_VECTOR_CORE', deviceCursor, vectorDur, 0.05, 0.3));
        deviceCursor += vectorDur;
      }

      if (config.commUsPerLayer > 0) {
        // HCCL runs on its own "stream": with overlap it is hidden under the
        // next layer's compute, without overlap it extends the step.
        const commStart = config.commOverlapped ? deviceCursor + matmulDur * 0.2 : deviceCursor;
        const commDur = config.commUsPerLayer * (0.9 + random() * 0.2);
        deviceEvents.push(deviceEvent('AllReduce', commStart, commDur, { 'Task Type': 'HCCL', 'OP Type': 'AllReduce', 'Group Name': 'tp-8', stream: 'HCCL' }));
        commOps.push({ name: 'AllReduce', group: 'tp-8', durationUs: commDur, sizeBytes: config.commSizeBytes });
        recordOp(opStats, 'AllReduce', 'HCCL', commDur);
        stepMetrics.comm += commDur;
        if (config.commOverlapped) stepMetrics.overlapped += Math.min(commDur, matmulDur * 0.8);
        else stepMetrics.commNotOverlapped += commDur;
        if (!config.commOverlapped) deviceCursor += commDur;
      }

      if (config.copyPerStep > 0 && layer === 0) {
        const copyDur = config.copyPerStep;
        deviceEvents.push({
          ...deviceEvent('aclrtMemcpyAsync_D2H', deviceCursor, copyDur, { 'Task Type': 'AI_CPU', stream: 9 }),
          // The trace category must agree with the operator name: D2H is a
          // `DeviceToHost` copy, H2D is `HostToDevice`.
          cat: 'DeviceToHost',
        });
        recordOp(opStats, 'aclrtMemcpyAsync', 'AI_CPU', copyDur);
        deviceCursor += copyDur;
      }
    }
    stepMetrics.free = config.freeUs;

    const stepEnd = Math.max(hostStepStart + config.hostStepUs, deviceCursor + config.freeUs);
    stepMetrics.bubble = 0;
    cursor = stepEnd;

    for (const event of hostEvents) trace.push(event);
    for (const event of deviceEvents) trace.push(event);
    for (const event of deviceEvents) {
      operatorRows.push(operatorRow(event, config));
    }

    stepRows.push({
      device: 0,
      step,
      computingMs: stepMetrics.computing / 1000,
      commNotOverlappedMs: stepMetrics.commNotOverlapped / 1000,
      overlappedMs: stepMetrics.overlapped / 1000,
      commMs: stepMetrics.comm / 1000,
      freeMs: Math.max(0, (stepEnd - stepStart - stepMetrics.computing - stepMetrics.commNotOverlapped) / 1000),
      stage: step < config.prefillSteps ? 'Prefill' : 'Decode',
    });
  }

  const files = {
    'trace_view.json': `[${trace.map((event) => JSON.stringify(event)).join(',\n')}]`,
    'kernel_details.csv': toCsv(
      ['Device_id', 'Name', 'Type', 'Accelerator Core', 'Start Time(us)', 'Duration(us)', 'Wait Time(us)', 'Block Num', 'Input Shapes', 'Output Shapes', 'mac_ratio', 'mte2_ratio'],
      kernelRows,
    ),
    'operator_details.csv': toCsv(
      ['Device_id', 'Name', 'Input Shapes', 'Call Stack', 'Host Self Duration(us)', 'Host Total Duration(us)', 'Device Self Duration(us)', 'Device Total Duration(us)', 'mac_ratio', 'mte2_ratio'],
      operatorRows,
    ),
    'op_statistic.csv': toCsv(
      ['Device_id', 'OP Type', 'Core Type', 'Count', 'Total Time(us)', 'Avg Time(us)', 'Ratio(%)'],
      [...opStats.entries()].map(([name, entry]) => ({
        device: 0,
        opType: name,
        core: entry.core,
        count: entry.count,
        totalUs: entry.totalUs,
        avgUs: entry.totalUs / entry.count,
        ratio: 0,
      })),
    ),
    'step_trace_time.csv': toCsv(
      ['Device_id', 'Step', 'Computing', 'Communication(Not Overlapped)', 'Overlapped', 'Communication', 'Free', 'Stage', 'Bubble', 'Preparing'],
      stepRows,
    ),
    [`profiler_info_${rank}.json`]: JSON.stringify({
      'Device ID': rank,
      'Device Type': 'Ascend910B4',
      'Rank ID': rank,
      'World Size': 8,
      'Parallel': 'tp8',
      'Profiler Level': 'Level1',
      'AI Core Metrics': 'PipeUtilization',
      'vllm version': '0.9.1',
      'vllm-ascend version': '0.9.1rc1',
      'torch_npu version': '2.5.1',
      'CANN version': '8.1.RC1',
    }, null, 2),
    'communication.json': JSON.stringify({
      'AllReduce': {
        count: steps * layers,
        totalTimeUs: [...opStats.values()].length === 0 ? 0 : (opStats.get('AllReduce')?.totalUs ?? 0),
        op: commOps.slice(0, 200).map((op) => ({ name: op.name, group: op.group, durationUs: op.durationUs, sizeBytes: op.sizeBytes })),
      },
    }, null, 2),
  };

  return {
    files,
    summary: {
      scenario,
      description: config.description,
      steps,
      eventCount: trace.length,
      deviceEvents: trace.filter((event) => event.pid === 2002 || event.pid === 'HCCL').length,
      hostEvents: trace.filter((event) => event.pid === 1001).length,
    },
  };
}

/** Scenario knob sets. */
const SCENARIOS = {
  'decode-comm-bound': {
    description: 'TP=8 Decode：每层小消息 AllReduce 且未与计算重叠，通信占设备耗时高、未掩盖比例高、MAC 利用率低。',
    steps: 24,
    prefillSteps: 2,
    rank: 0,
    matmulUs: 170,
    attentionUs: 90,
    vectorUs: 22,
    vectorOpsPerLayer: 2,
    commUsPerLayer: 48,
    commOverlapped: false,
    commSizeBytes: 96 * 1024,
    copyPerStep: 180,
    freeUs: 120,
    engineUs: 320,
    prepareUs: 180,
    metadataUs: 140,
    samplerUs: 420,
    hostOpsPerStep: 260,
    hostOpWindowUs: 2200,
    hostOpJitterUs: 26,
    hostStepUs: 12000,
    deviceOffsetUs: 260,
    macRatio: 0.21,
    mte2Ratio: 0.58,
  },
  'prefill-compute-bound': {
    description: 'Chunked Prefill：长步、MatMul/FlashAttention 主导，MAC 利用率高，通信占比低。',
    steps: 6,
    prefillSteps: 5,
    rank: 0,
    matmulUs: 4200,
    attentionUs: 2600,
    vectorUs: 210,
    vectorOpsPerLayer: 2,
    commUsPerLayer: 180,
    commOverlapped: true,
    commSizeBytes: 32 * 1024 * 1024,
    copyPerStep: 60,
    freeUs: 200,
    engineUs: 900,
    prepareUs: 700,
    metadataUs: 1500,
    samplerUs: 300,
    hostOpsPerStep: 900,
    hostOpWindowUs: 9000,
    hostOpJitterUs: 40,
    hostStepUs: 60000,
    deviceOffsetUs: 900,
    macRatio: 0.72,
    mte2Ratio: 0.38,
  },
  'host-schedule-bound': {
    description: 'Eager Decode：Host 逐步派发大量算子与同步，设备频繁空闲，NPU 忙碌率低。',
    steps: 20,
    prefillSteps: 2,
    rank: 0,
    matmulUs: 120,
    attentionUs: 70,
    vectorUs: 24,
    vectorOpsPerLayer: 2,
    commUsPerLayer: 34,
    commOverlapped: true,
    commSizeBytes: 96 * 1024,
    copyPerStep: 260,
    freeUs: 2600,
    engineUs: 2600,
    prepareUs: 1400,
    metadataUs: 900,
    samplerUs: 1800,
    hostOpsPerStep: 1500,
    hostOpWindowUs: 12000,
    hostOpJitterUs: 30,
    hostStepUs: 22000,
    deviceOffsetUs: 4200,
    macRatio: 0.18,
    mte2Ratio: 0.44,
  },
};

function meta(name, pid, args, tid) {
  return { ph: 'M', name, pid, ...(tid === undefined ? {} : { tid }), args };
}

function hostEvent(name, ts, dur, args) {
  return {
    ph: 'X',
    name,
    pid: 1001,
    tid: 1001,
    ts: ts.toFixed(3),
    dur: Number(dur.toFixed(3)),
    cat: 'cpu_op',
    args: { 'Fwd thread id': 0, ...args },
  };
}

function deviceEvent(name, ts, dur, args) {
  return {
    ph: 'X',
    name,
    pid: 2002,
    tid: args.stream === 'HCCL' ? 'HCCL' : String(args.stream ?? 0),
    ts: ts.toFixed(3),
    dur: Number(dur.toFixed(3)),
    cat: name.startsWith('All') || args['Task Type'] === 'HCCL' ? 'hccl' : 'kernel',
    args: {
      'Device Id': 0,
      'Stream Id': args.stream ?? 0,
      'Task Type': args['Task Type'] ?? 'AI_CORE',
      ...(args['OP Type'] === undefined ? {} : { 'OP Type': args['OP Type'] }),
      ...(args['Group Name'] === undefined ? {} : { 'Group Name': args['Group Name'] }),
      'Input Dims': [[1, 1024], [1024, 4096]],
      'Call stack': 'vllm/worker/model_runner.py:1234;vllm/attention/backends/ascend.py:88',
    },
  };
}

function kernelRow(name, type, core, startUs, durUs, macRatio, mte2Ratio) {
  return {
    device: 0,
    name,
    type,
    core,
    startUs: startUs.toFixed(3),
    durUs: durUs.toFixed(3),
    waitUs: (durUs * 0.08).toFixed(3),
    blockNum: 24,
    shapesIn: '[[1,1024],[1024,4096]]',
    shapesOut: '[[1,4096]]',
    macRatio,
    mte2Ratio,
  };
}

function operatorRow(event, config) {
  return {
    device: 0,
    name: event.name,
    shapesIn: '[[1,1024],[1024,4096]]',
    callStack: 'vllm/worker/model_runner.py:1234',
    hostSelfUs: 6.5,
    hostTotalUs: 9.8,
    deviceSelfUs: Number(event.dur.toFixed(3)),
    deviceTotalUs: Number(event.dur.toFixed(3)) + 4,
    macRatio: config.macRatio,
    mte2Ratio: config.mte2Ratio,
  };
}

function recordOp(opStats, name, core, durUs) {
  const entry = opStats.get(name) ?? { core, count: 0, totalUs: 0 };
  entry.count += 1;
  entry.totalUs += durUs;
  opStats.set(name, entry);
}

/** Host-side operator mix for the eager-dispatch scenarios. */
function pickHostOp(random) {
  const value = random();
  if (value < 0.28) return 'aten::empty';
  if (value < 0.44) return 'aten::view';
  if (value < 0.56) return 'aten::as_strided';
  if (value < 0.66) return 'aclnnInplaceCopy';
  if (value < 0.74) return 'aten::_to_copy';
  if (value < 0.8) return 'aten::item';
  if (value < 0.84) return 'aten::_local_scalar_dense';
  if (value < 0.9) return 'vllm::reshape_and_cache';
  if (value < 0.95) return 'vllm::block_table_gather';
  return 'aclnnMatmul';
}

/** Serialize rows to CSV with the canonical column order given. */
function toCsv(header, rows) {
  const lines = [header.join(',')];
  for (const row of rows) {
    lines.push(header.map((column) => {
      const value = cellOf(row, column);
      if (value === undefined) return '';
      const text = String(value);
      return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }).join(','));
  }
  return `${lines.join('\n')}\n`;
}

const COLUMN_KEYS = {
  'Device_id': 'device',
  'Name': 'name',
  'Type': 'type',
  'Accelerator Core': 'core',
  'Start Time(us)': 'startUs',
  'Duration(us)': 'durUs',
  'Wait Time(us)': 'waitUs',
  'Block Num': 'blockNum',
  'Input Shapes': 'shapesIn',
  'Output Shapes': 'shapesOut',
  'OP Type': 'opType',
  'Core Type': 'core',
  'Count': 'count',
  'Total Time(us)': 'totalUs',
  'Avg Time(us)': 'avgUs',
  'Ratio(%)': 'ratio',
  'Host Self Duration(us)': 'hostSelfUs',
  'Host Total Duration(us)': 'hostTotalUs',
  'Device Self Duration(us)': 'deviceSelfUs',
  'Device Total Duration(us)': 'deviceTotalUs',
  'Call Stack': 'callStack',
  'Step': 'step',
  'Computing': 'computingMs',
  'Communication(Not Overlapped)': 'commNotOverlappedMs',
  'Overlapped': 'overlappedMs',
  'Communication': 'commMs',
  'Free': 'freeMs',
  'Stage': 'stage',
  'Bubble': 'bubble',
  'Preparing': 'preparing',
  'mac_ratio': 'macRatio',
  'mte2_ratio': 'mte2Ratio',
};

function cellOf(row, column) {
  if (COLUMN_KEYS[column] !== undefined) return row[COLUMN_KEYS[column]];
  return row[column.toLowerCase()] ?? row[column];
}

/** Write every scenario under `root`. */
export function writeFixtures(root) {
  const written = [];
  for (const scenario of Object.keys(SCENARIOS)) {
    const { files, summary } = buildScenario(scenario);
    const directory = join(root, scenario);
    mkdirSync(directory, { recursive: true });
    for (const [name, content] of Object.entries(files)) {
      writeFileSync(join(directory, name), content, 'utf8');
    }
    written.push(summary);
  }
  return written;
}

/** True when this module is the process entry point. */
const invokedDirectly = process.argv[1] !== undefined
  && fileURLToPath(import.meta.url) === (await import('node:path')).resolve(process.argv[1]);
if (invokedDirectly) {
  const root = process.argv[2] ?? join(here, 'fixtures');
  const written = writeFixtures(root);
  for (const summary of written) {
    process.stdout.write(`${summary.scenario}: ${String(summary.eventCount)} events (${String(summary.deviceEvents)} device / ${String(summary.hostEvents)} host), ${String(summary.steps)} steps — ${summary.description}\n`);
  }
  process.stdout.write(`fixtures written to ${root}\n`);
}
