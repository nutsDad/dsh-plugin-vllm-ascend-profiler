/**
 * Ascend profiler CSV table parsing.
 *
 * The analyzer accepts the whole family of tables CANN, `torch_npu`, and
 * `msprof-analyze` emit, mapping each one onto a canonical row shape. Column
 * spelling drifts between CANN releases (`OP Type` / `Op Type` / `op_type`,
 * `Duration(us)` / `Task Duration(us)`), so every field is resolved through an
 * alias table rather than by fixed position, and unrecognised columns are kept
 * in `extra` instead of being dropped.
 *
 * Recognised table kinds:
 *
 * | kind                | produced by                        | granularity      |
 * |---------------------|------------------------------------|------------------|
 * | `op_statistic`      | msprof-analyze / torch_npu export  | per operator type|
 * | `api_statistic`     | msprof-analyze / torch_npu export  | per host API     |
 * | `op_summary`        | msprof / MindStudio Insight export | per op instance  |
 * | `kernel_details`    | torch_npu Ascend profiler export   | per kernel       |
 * | `operator_details`  | torch_npu Ascend profiler export   | per op instance  |
 * | `step_trace_time`   | torch_npu Ascend profiler export   | per step         |
 * | `communication_statistic` | msprof-analyze export        | per collective   |
 * | `generic`           | anything else with the right shape | per row          |
 *
 * @module dsh-plugin-vllm-ascend-profiler/parse/ascendcsv
 */

import { buildHeaderIndex, CsvReader, cell, decodeText, num, unitOfHeader } from './csv.js';

/** Canonical field aliases: canonical name → accepted header spellings. */
const ALIASES = {
  deviceId: ['Device_id', 'Device ID', 'Device Id', 'device_id', 'Rank ID', 'Rank_id'],
  name: [
    // `Name` is the torch_npu header; `Op Name`/`Kernel Name` are msprof and
    // MindStudio exports. `API Name` covers api_statistic.csv.
    'Name', 'Op Name', 'OP Name', 'Operator Name', 'Kernel Name', 'Kernel_Name', 'API Name',
    'kernel_name', 'op_name',
  ],
  opType: ['OP Type', 'Op Type', 'op_type', 'OPType', 'Operator Type', 'Type'],
  coreType: ['Accelerator Core', 'Core Type', 'core_type', 'Core_Type', 'Accelerator'],
  taskType: ['Task Type', 'task_type', 'Task_Type'],
  opState: ['OP State', 'Op State', 'op_state'],
  count: ['Count', 'count', 'Times', 'Times(us)', 'Call Count', 'Total Count'],
  totalUs: ['Total Time(us)', 'Total Time(ms)', 'Total Duration(us)', 'Total Duration(ms)', 'Total Time', 'total_time'],
  avgUs: ['Avg Time(us)', 'Avg Time(ms)', 'Average Time(us)', 'Avg Duration(us)', 'Avg Time', 'avg_time'],
  minUs: ['Min Time(us)', 'Min Time(ms)', 'Min Duration(us)'],
  maxUs: ['Max Time(us)', 'Max Time(ms)', 'Max Duration(us)'],
  ratioPct: ['Ratio(%)', 'Ratio', 'Proportion(%)', 'Percentage(%)', 'Ratio(%) of Total'],
  startUs: [
    // Class B op_summary uses `Task Start Time(us)`; Class A kernel_details is
    // a rename-projection of it and uses `Start Time(us)`. Both spellings must
    // resolve to the same canonical field.
    'Start Time(us)', 'Start Time(ms)', 'Task Start Time(us)', 'Task Start Time(ms)',
    'Start_Time(us)', 'Start_Time(ms)', 'Start Time', 'Start_Timestamp(us)',
    'Timestamp(us)', 'start_time',
  ],
  durUs: [
    // torch_npu operator_details.csv reports host and device durations
    // separately; the device total is what the analyzer treats as operator cost.
    'Device Total Duration(us)', 'Device Self Duration(us)', 'Duration(us)', 'Duration(ms)',
    'Task Duration(us)', 'Task Duration(ms)', 'Duration', 'Task_Duration(us)',
    'Elapse Time(us)', 'Total Time(us)',
  ],
  hostTotalUs: ['Host Total Duration(us)', 'Host Total Duration(ms)'],
  hostSelfUs: ['Host Self Duration(us)', 'Host Self Duration(ms)'],
  deviceSelfUs: ['Device Self Duration(us)', 'Device Self Duration(ms)'],
  waitUs: ['Wait Time(us)', 'Wait Time(ms)', 'Task Wait Time(us)', 'Wait_Time(us)', 'Wait Time'],
  aicoreUs: [
    'Device Self Duration With AICore(us)', 'Device Total Duration With AICore(us)',
    'Aicore Time(us)', 'Aicore Time(ms)', 'AI Core Time(us)', 'aicore_time(us)',
  ],
  blockDim: ['Block Dim', 'Block_Dim', 'Block Num', 'block_dim'],
  streamId: ['Stream ID', 'Stream_Id', 'Stream Id', 'stream_id'],
  taskId: ['Task ID', 'Task_Id', 'task_id'],
  step: ['Step', 'Step Id', 'Step_ID', 'step', 'Iteration', 'Iteration Id'],
  shapesIn: ['Input Shapes', 'Input_Shapes', 'input_shapes', 'Input Shape', 'Input Dims'],
  shapesOut: ['Output Shapes', 'Output_Shapes', 'output_shapes', 'Output Shape'],
  callStack: ['Call Stack', 'Call_Stack', 'call_stack', 'Callstack'],
  commGroup: ['Group Name', 'Group_Name', 'Comm Group', 'Group'],
  commSize: ['Size(MB)', 'Size(Bytes)', 'Data Size(MB)', 'Size'],
  commBandwidth: ['Bandwidth(GB/s)', 'Bandwidth', 'SDMA Bandwidth(GB/s)'],
  commOp: ['Collective Type', 'Communication Type', 'OP Type', 'Op Type'],
  // step_trace_time.csv is written without spaces before the parentheses.
  stepComputing: ['Computing', 'Computing(us)', 'Computing Time(us)'],
  stepComm: ['Communication', 'Communication(us)', 'Total Communication(us)'],
  stepCommNotOverlapped: [
    'Communication(Not Overlapped)', 'Communication (Not Overlapped)',
    'Communication Not Overlapped', 'Not Overlapped Communication',
  ],
  stepOverlapped: ['Overlapped', 'Communication(Overlapped)', 'Overlapped Communication'],
  stepFree: ['Free', 'Free(us)', 'Idle(us)'],
  stepTotal: ['Total', 'Total(us)', 'Duration(us)', 'Step Total(us)'],
  stepStage: ['Stage'],
  stepBubble: ['Bubble'],
  stepPreparing: ['Preparing'],
};

/** Table kinds recognized from a file name. */
const KIND_BY_NAME = [
  { kind: 'op_statistic', pattern: /op_?statistic/i },
  { kind: 'api_statistic', pattern: /api_?statistic/i },
  { kind: 'op_summary', pattern: /op_?summary/i },
  { kind: 'kernel_details', pattern: /kernel_?details/i },
  { kind: 'operator_details', pattern: /operator_?details|op_?details/i },
  { kind: 'step_trace_time', pattern: /step_?trace_?time|step_?time/i },
  { kind: 'communication_statistic', pattern: /communication_?(?:statistic|statics)|hccl_?statistic|comm_?statistic/i },
  { kind: 'pipe_utilization', pattern: /pipe_?utilization|aic_?metrics|aicore_?metrics/i },
];

/** Table kinds whose rows describe one instance in time. */
export const INSTANCE_KINDS = new Set(['op_summary', 'kernel_details', 'operator_details', 'generic']);

/** Utilization column patterns mapped onto canonical metric names. */
const UTILIZATION_COLUMNS = [
  { key: 'mac', pattern: /(?:^|_)mac(?:_|$)|cube_?ratio/i, label: 'MAC（Cube 乘加）利用率' },
  { key: 'mte1', pattern: /mte1/i, label: 'MTE1（L1→L0 搬运）利用率' },
  { key: 'mte2', pattern: /mte2/i, label: 'MTE2（GM→L1 搬运）利用率' },
  { key: 'mte3', pattern: /mte3/i, label: 'MTE3（L0→GM 写回）利用率' },
  { key: 'vec', pattern: /(?:^|_)vec(?:_|$)|vector_?ratio/i, label: 'Vector（向量单元）利用率' },
  { key: 'scalar', pattern: /scalar_?ratio|(?:^|_)scalar(?:_|$)/i, label: 'Scalar（标量单元）利用率' },
  { key: 'fixpipe', pattern: /fixpipe/i, label: 'FixPipe 利用率' },
  { key: 'aicore', pattern: /aicore_?utilization|aic_?utilization|core_?utilization/i, label: 'AI Core 利用率' },
  { key: 'icacheMiss', pattern: /icache_?miss/i, label: 'ICache 缺失率' },
  { key: 'hbm', pattern: /hbm_?(?:ratio|utilization|bandwidth)|ddr_?(?:ratio|utilization)/i, label: 'HBM 带宽利用率' },
];

/**
 * Read the canonical table kind from a file name.
 * @param {string} name - artifact file name (path stripped by the caller).
 * @returns {string} table kind, or `generic`.
 */
export function tableKindOf(name) {
  for (const entry of KIND_BY_NAME) {
    if (entry.pattern.test(name)) return entry.kind;
  }
  return 'generic';
}

/**
 * Parse one Ascend CSV artifact.
 *
 * @param {object} input - parse input.
 * @param {string} input.name - artifact name (used for kind detection).
 * @param {Buffer|Uint8Array} input.buffer - raw bytes.
 * @param {number} [input.maxRows] - row budget.
 * @param {string} [input.kind] - override the detected table kind.
 * @returns {{
 *   kind: string, name: string, rows: object[], header: string[], unmatchedColumns: string[],
 *   truncated: boolean, warnings: string[], encoding: string, totals: object
 * }} parsed table.
 */
export function parseAscendCsv({ name, buffer, maxRows = 400000, kind }) {
  const decoded = decodeText(buffer);
  const warnings = [];
  if (decoded.warning !== undefined) warnings.push(decoded.warning);
  const resolvedKind = kind ?? tableKindOf(name);

  let headerIndex;
  let header;
  /** @type {object[]} */
  const rows = [];
  const unmatchedColumns = [];
  let unitMap = {};

  const reader = new CsvReader({
    maxRows,
    onRecord: (record, rowIndex) => {
      if (rowIndex === 0) {
        header = record;
        const built = buildHeaderIndex(record, ALIASES);
        headerIndex = built.index;
        // Utilization columns are matched by pattern rather than by alias, so
        // they must be excluded from the "unrecognised column" report.
        unmatchedColumns.push(...built.unmatched.filter((cellText) => !isUtilizationColumn(cellText)));
        unitMap = resolveUnits(record, built.index);
        return;
      }
      const row = canonicalRow(record, headerIndex, header, unitMap, resolvedKind);
      if (row !== undefined) rows.push(row);
    },
  });

  reader.push(decoded.text);
  reader.end();
  if (reader.truncated) {
    warnings.push(`表格行数超过上限 ${String(maxRows)}，仅保留前 ${String(rows.length)} 行。`);
  }
  if (header === undefined) {
    warnings.push('未找到 CSV 表头行，文件可能为空或不是表格产物。');
  } else if (headerIndex.name === undefined && headerIndex.opType === undefined && !isStepTable(headerIndex)) {
    warnings.push('表头中既没有算子名称列，也没有算子类型列，无法归类。');
  }
  if (unmatchedColumns.length > 0 && rows.length > 0) {
    warnings.push(`未识别的列已保留在 extra 中：${unmatchedColumns.slice(0, 8).join(', ')}${unmatchedColumns.length > 8 ? ' …' : ''}`);
  }
  const unitNote = normalizeStepTableUnits(resolvedKind, rows);
  if (unitNote !== undefined) warnings.push(unitNote);

  return {
    kind: resolvedKind,
    name,
    rows,
    header: header ?? [],
    unmatchedColumns,
    truncated: reader.truncated,
    warnings,
    encoding: decoded.encoding,
    totals: {
      rows: rows.length,
      totalUs: rows.reduce((sum, row) => sum + (row.durUs ?? 0), 0),
      count: rows.reduce((sum, row) => sum + (row.count ?? (row.durUs === undefined ? 0 : 1)), 0),
    },
  };
}

/**
 * Resolve the unit each time column declares.
 * @param {string[]} header - header cells.
 * @param {Record<string, number>} index - canonical → column position.
 * @returns {Record<string, 'us'|'ms'|'ns'|'s'|undefined>} unit per canonical time field.
 */
function resolveUnits(header, index) {
  const map = {};
  for (const field of ['totalUs', 'avgUs', 'minUs', 'maxUs', 'startUs', 'durUs', 'waitUs', 'aicoreUs', 'hostSelfUs', 'hostTotalUs', 'deviceSelfUs']) {
    const at = index[field];
    map[field] = at === undefined ? undefined : unitOfHeader(header[at]);
  }
  return map;
}

/**
 * Convert one CSV record into the canonical row shape.
 * @param {string[]} record - split record.
 * @param {Record<string, number>} index - header lookup.
 * @param {string[]} header - header cells.
 * @param {Record<string, string|undefined>} unitMap - per-field units.
 * @param {string} kind - table kind.
 * @returns {object|undefined} canonical row, or `undefined` when unusable.
 */
function canonicalRow(record, index, header, unitMap, kind) {
  const stepFieldsPresent = index.stepComputing !== undefined || index.stepFree !== undefined || index.stepTotal !== undefined;
  const stepNumber = num(cell(record, index, 'step'));
  let name = cell(record, index, 'name') ?? cell(record, index, 'opType');
  const opType = cell(record, index, 'opType');
  // `step_trace_time.csv` has no operator column at all: its rows are steps, so
  // a synthetic name keeps them addressable instead of dropping the table.
  if (name === undefined && stepFieldsPresent) name = stepNumber === undefined ? '(step)' : `Step ${String(stepNumber)}`;
  if (name === undefined && opType === undefined) return undefined;
  const durationUs = toUs(cell(record, index, 'durUs'), unitMap.durUs ?? 'us');
  const totalUs = toUs(cell(record, index, 'totalUs'), unitMap.totalUs ?? 'us');
  const count = num(cell(record, index, 'count'));
  const consumed = new Set(Object.values(index));
  const row = {
    kind,
    deviceId: cell(record, index, 'deviceId'),
    name: name ?? opType ?? '(unnamed)',
    opType,
    coreType: cell(record, index, 'coreType'),
    taskType: cell(record, index, 'taskType'),
    opState: cell(record, index, 'opState'),
    count: count ?? (durationUs === undefined ? undefined : 1),
    durUs: durationUs,
    totalUs: totalUs ?? durationUs,
    avgUs: toUs(cell(record, index, 'avgUs'), unitMap.avgUs ?? 'us'),
    minUs: toUs(cell(record, index, 'minUs'), unitMap.minUs ?? 'us'),
    maxUs: toUs(cell(record, index, 'maxUs'), unitMap.maxUs ?? 'us'),
    ratioPct: num(cell(record, index, 'ratioPct')),
    // Raw timestamps stay raw here; the normalizer rebases them per source so
    // that CSV (absolute device time) and trace (relative) coordinate systems
    // can be compared without losing the original values.
    startRawUs: toUs(cell(record, index, 'startUs'), unitMap.startUs ?? 'us'),
    waitUs: toUs(cell(record, index, 'waitUs'), unitMap.waitUs ?? 'us'),
    aicoreUs: toUs(cell(record, index, 'aicoreUs'), unitMap.aicoreUs ?? 'us'),
    hostSelfUs: toUs(cell(record, index, 'hostSelfUs'), unitMap.hostSelfUs ?? 'us'),
    hostTotalUs: toUs(cell(record, index, 'hostTotalUs'), unitMap.hostTotalUs ?? 'us'),
    deviceSelfUs: toUs(cell(record, index, 'deviceSelfUs'), unitMap.deviceSelfUs ?? 'us'),
    blockDim: num(cell(record, index, 'blockDim')),
    streamId: cell(record, index, 'streamId'),
    taskId: cell(record, index, 'taskId'),
    step: num(cell(record, index, 'step')),
    shapesIn: cell(record, index, 'shapesIn'),
    shapesOut: cell(record, index, 'shapesOut'),
    callStack: cell(record, index, 'callStack'),
    utilization: {},
    extra: {},
  };
  captureUtilization(record, header, row, consumed);
  captureStepFields(record, index, header, row, consumed);
  captureCommFields(record, index, row, consumed);
  captureExtra(record, consumed, row);
  return row;
}

/**
 * Map utilization-ish columns onto the canonical utilization bag.
 *
 * Iteration is over the *header* rather than the alias index: utilization
 * columns (`mac_ratio`, `mte2_ratio`, `aicore_utilization`, …) are open-ended
 * and are deliberately not part of the fixed alias table, so matching them by
 * alias would find nothing at all.
 */
function captureUtilization(record, header, row, consumed) {
  for (let at = 0; at < header.length; at += 1) {
    const column = header[at];
    if (column === undefined || column === '') continue;
    for (const rule of UTILIZATION_COLUMNS) {
      if (!rule.pattern.test(column)) continue;
      const raw = num(record[at]);
      consumed.add(at);
      if (raw === undefined) break;
      row.utilization[rule.key] = normalizeRatio(raw);
      row.utilizationLabels ??= {};
      row.utilizationLabels[rule.key] = rule.label;
      break;
    }
  }
}

/** Step-level totals used by `step_trace_time.csv`. */
function captureStepFields(record, index, header, row, consumed) {
  const fields = {
    stepComputing: 'computingUs',
    stepComm: 'commUs',
    stepCommNotOverlapped: 'commNotOverlappedUs',
    stepOverlapped: 'commOverlappedUs',
    stepFree: 'freeUs',
    stepTotal: 'stepTotalUs',
  };
  let hit = false;
  for (const [canonical, target] of Object.entries(fields)) {
    const at = index[canonical];
    if (at === undefined) continue;
    const value = toUs(record[at], unitOfHeader(header[at]) ?? 'us');
    if (value === undefined) continue;
    row[target] = value;
    consumed.add(at);
    hit = true;
  }
  const extraStepFields = { stepBubble: 'stepBubbleUs', stepPreparing: 'stepPreparingUs' };
  for (const [canonical, target] of Object.entries(extraStepFields)) {
    const at = index[canonical];
    if (at === undefined) continue;
    const value = toUs(record[at], unitOfHeader(header[at]) ?? 'us');
    if (value === undefined) continue;
    row[target] = value;
    consumed.add(at);
  }
  // `Stage` is the one column that names the phase (Prefill / Decode) outright;
  // it is text, so it is copied verbatim.
  const stageAt = index.stepStage;
  if (stageAt !== undefined) {
    const stage = record[stageAt];
    if (stage !== undefined && stage !== '') {
      row.stage = stage;
      consumed.add(stageAt);
      hit = true;
    }
  }
  if (hit) row.isStepRow = true;
}

/** Collective-specific columns used by `communication_statistic.csv`. */
function captureCommFields(record, index, row, consumed) {
  const pairs = [
    ['commGroup', 'commGroup', undefined],
    ['commSize', 'commSizeMb', undefined],
    ['commBandwidth', 'commBandwidthGBps', undefined],
    ['commOp', 'commType', undefined],
  ];
  for (const [canonical, target] of pairs) {
    const at = index[canonical];
    if (at === undefined) continue;
    const value = record[at];
    if (value === undefined || value === '') continue;
    row[target] = canonical === 'commGroup' || canonical === 'commOp' ? value : num(value);
    consumed.add(at);
  }
}

/** Keep every remaining column so nothing is silently lost. */
function captureExtra(record, consumed, row) {
  for (let at = 0; at < record.length; at += 1) {
    if (consumed.has(at)) continue;
    const value = record[at];
    if (value === undefined || value === '') continue;
    row.extra[`col${String(at)}`] = value;
  }
}

/** Utilization values arrive as a fraction (0–1) or a percentage (0–100). */
function normalizeRatio(value) {
  if (value > 1.5) return value / 100;
  return value;
}

/**
 * `step_trace_time.csv` declares no unit in its headers, and CANN writes
 * milliseconds there. Values that small cannot be microseconds for an inference
 * step (a decode step is milliseconds, a prefill step tens to hundreds of
 * milliseconds), so a table whose step totals are all below
 * {@link STEP_MS_CEILING} is rescaled and the rescale is disclosed.
 *
 * @param {string} kind - table kind.
 * @param {object[]} rows - canonical rows.
 * @returns {string|undefined} a warning describing the rescale, when it happened.
 */
function normalizeStepTableUnits(kind, rows) {
  if (kind !== 'step_trace_time' || rows.length === 0) return undefined;
  const totals = rows
    .map((row) => row.stepTotalUs ?? sumStepFields(row))
    .filter((value) => Number.isFinite(value) && value > 0);
  if (totals.length === 0) return undefined;
  const max = Math.max(...totals);
  const median = [...totals].sort((left, right) => left - right)[Math.floor(totals.length / 2)];
  if (max >= STEP_MS_CEILING || median <= 0) return undefined;
  const fields = ['stepTotalUs', 'computingUs', 'commUs', 'commNotOverlappedUs', 'commOverlappedUs', 'freeUs', 'stepBubbleUs', 'stepPreparingUs'];
  for (const row of rows) {
    for (const field of fields) {
      if (Number.isFinite(row[field])) row[field] *= 1000;
    }
    row.unitRescaledFromMs = true;
  }
  return `step_trace_time 的步长数值最大仅 ${max.toFixed(3)}，按毫秒解释（换算为微秒，×1000）；该表表头未声明单位。`;
}

/** Sum the step fields that describe one step's wall time. */
function sumStepFields(row) {
  const parts = [row.computingUs, row.commUs, row.freeUs].filter((value) => Number.isFinite(value));
  if (parts.length === 0) return undefined;
  return parts.reduce((sum, value) => sum + value, 0);
}

/** Step totals below this cannot be microseconds for a real inference step. */
const STEP_MS_CEILING = 2000;

/** Whether the header describes step rows (`step_trace_time.csv`). */
function isStepTable(index) {
  return index.stepComputing !== undefined || index.stepFree !== undefined || index.stepTotal !== undefined;
}

/** Whether a header cell is one of the pattern-matched utilization columns. */
function isUtilizationColumn(cellText) {
  return UTILIZATION_COLUMNS.some((rule) => rule.pattern.test(String(cellText)));
}

/**
 * Convert a cell to microseconds using a declared unit.
 * @param {string|undefined} value - raw cell.
 * @param {'us'|'ms'|'ns'|'s'|undefined} unit - declared unit.
 * @returns {number|undefined} microseconds.
 */
function toUs(value, unit) {
  const parsed = num(value);
  if (parsed === undefined) return undefined;
  switch (unit) {
    case 'ms': return parsed * 1000;
    case 'ns': return parsed / 1000;
    case 's': return parsed * 1e6;
    default: return parsed;
  }
}
