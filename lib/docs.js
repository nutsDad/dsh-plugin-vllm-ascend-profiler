/**
 * The plugin's built-in documentation: what every metric means, how it is
 * computed, and what each vLLM-Ascend / Ascend profiling artifact and column
 * contains.
 *
 * It is served to the page's "说明" panel and rendered in the report appendix,
 * so an operator reading a number can always find its definition and its
 * known limitations in the same place.
 *
 * @module dsh-plugin-vllm-ascend-profiler/docs
 */

/** Metric definitions, grouped by the module that displays them. */
export const METRIC_DOCS = [
  {
    group: '模块一 · 泳道时序图',
    metrics: [
      {
        name: '时间轴（ms）',
        definition: '以本次采集窗口第一个事件为原点的时间轴，单位毫秒。',
        formula: 't = (ts − min(ts)) / 1000',
        caveats: 'trace 中的 ts 为微秒（可能是十进制字符串），CSV 的 Start Time 为设备绝对时间；两者不做混轴，泳道图只画 trace 事件。',
      },
      {
        name: '算子条（bar）',
        definition: '一个算子的一次执行：起点为开始时间，宽度为持续时长。',
        formula: 'start = ts，width = dur',
        caveats: '同一泳道行内并行执行（多 stream）的算子条会重叠显示；行统计使用区间并集，不会重复计数。',
      },
      {
        name: '泳道分组',
        definition: 'Host（CPU）与 Device（昇腾 NPU）两组，每组内一个算子一行。',
        formula: '分组依据：trace 的 process_name 元数据（Python / CANN → Host；Ascend Hardware → Device），缺失时回退到 device id / stream id 参数或事件 cat。',
        caveats: '未标注 process_name 的 trace 会按参数与 cat 回退推断，推断结果在页面提示中列出。',
      },
      {
        name: '算子分类',
        definition: '计算 / 通信 / 数据拷贝 / 调度 四类，另有"其他/同步"。',
        formula: '按算子名、OP Type、Task Type、Core Type 与 cat 的规则匹配，通信与拷贝优先于计算。',
        caveats: 'Host 侧非通信非拷贝的算子统一计入"调度"（框架开销）；Host 侧 aten::_to_copy 等仍按"拷贝"归类，分析时会区分设备侧与 Host 侧拷贝。',
      },
      {
        name: '视图抽样',
        definition: '当时间线事件数超过视图预算时，按行分配预算并优先保留耗时最长的算子条。',
        formula: '每行配额 ∝ 该行累计耗时占比；行内按 dur 降序取样后恢复时间顺序',
        caveats: '页面会明确显示"显示 N/M 个算子条"。累计耗时统计不受视图抽样影响。',
      },
    ],
  },
  {
    group: '模块二 · 耗时占比',
    metrics: [
      {
        name: '累计总耗时',
        definition: '某算子所有调用的耗时之和。',
        formula: 'totalUs = Σ durUs',
        caveats: '多流并行时各算子累计耗时之和会大于墙钟时间；占比的分母是"算子总耗时"，不是墙钟。',
      },
      {
        name: '单次执行耗时',
        definition: '某算子单次调用的平均耗时，用于识别"单次很慢"的算子。',
        formula: 'avgUs = totalUs / count',
        caveats: '在 Prefill/Decode 混采时，同一算子的均值会被两类负载混合；请切换阶段维度查看。',
      },
      {
        name: '占总时间百分比',
        definition: '该算子累计耗时 ÷ 全部算子累计耗时。',
        formula: 'shareOfOpsPct = totalUs / ΣtotalUs × 100%',
        caveats: '与"占墙钟比例"不同：前者是算子构成占比，后者是时间线占用率。报告同时给出两者。',
      },
      {
        name: '总计来源（totalsSource）',
        definition: '该行的累计耗时来自 trace 事件聚合还是 CANN 统计表。',
        formula: 'trace 采样时优先取 CSV；否则取 trace；CSV 侧按「统计表 > 实例表」只取一张表',
        caveats: '同一个算子常同时出现在 op_statistic.csv（按算子汇总）、kernel_details.csv（按 kernel）与 operator_details.csv（按算子实例）中，'
          + '三张表描述的是同一份设备耗时，因此只取一张表（优先统计表）作为 CSV 口径，绝不相加 —— 相加会把耗时与调用次数成倍放大，并让 cross-check 偏差失去意义。'
          + '两者都会保留（traceTotalUs / csvTotalUs）并给出 cross-check 偏差；偏差过大意味着 trace 采样或过滤丢了算子。'
          + '若两张 CSV 表的累计耗时相差超过 20%，会明确告警并说明取值来源，用于发现重复导出或产物不完整。',
      },
      {
        name: '通信与计算重叠',
        definition: '通信算子执行区间与计算算子区间重合的部分。',
        formula: 'overlap = |union(comm) ∩ union(compute)|',
        caveats: '重叠率低说明通信暴露在关键路径上；这是通信优化的首要判据，而不是通信总时长。',
      },
    ],
  },
  {
    group: '模块三 · 性能分析',
    metrics: [
      {
        name: 'NPU 忙碌率',
        definition: '设备侧事件覆盖的时间占采集窗口的比例（区间并集，不重复计数）。',
        formula: 'deviceBusyPct = |union(device intervals)| / wall',
        caveats: '多流并行不会使其超过 100%；该值低说明设备在等待（Host、通信依赖或数据）。',
      },
      {
        name: 'Host 独占（设备空等）',
        definition: 'Host 忙碌而设备空闲的时间。',
        formula: 'hostOnlyUs = |union(host)| − |union(host) ∩ union(device)|',
        caveats: '这是"设备在等主机"的下界：若设备同时也在等通信，该值会低估调度问题。',
      },
      {
        name: '通信未掩盖',
        definition: '通信时间中没有被计算覆盖的部分，即真正进入关键路径的通信。',
        formula: 'commExposedUs = |union(comm)| − |union(comm) ∩ union(compute)|',
        caveats: '小消息集合通信（≤60µs）通常受 HCCL 固定开销限制，扩大 buffer 无效，需减少次数或提高重叠。',
      },
      {
        name: '瓶颈得分',
        definition: '四类候选瓶颈的 0–100 分，用于比较"离瓶颈有多近"。',
        formula: '得分 = Σ 权重 × 门限化严重度；未达门限的项得 0（门限是闸门而非梯度）',
        caveats: '≥40 视为达到瓶颈门限；得分不是性能损失比例，而是证据强度。',
      },
      {
        name: 'MAC / MTE2 利用率',
        definition: 'CANN 算子级流水指标按耗时加权的平均值：MAC 反映 Cube 计算单元占用，MTE2 反映 GM→L1 搬运占用。',
        formula: 'weighted = Σ(ratio × dur) / Σdur',
        caveats: '仅有开启了 AI Core 指标采集（PipeUtilization）的产物才有该列；缺少时"算力受限/访存受限"只能给条件性结论。',
      },
      {
        name: 'Duration 与 AICore Time',
        definition: 'CANN 的 Duration 包含调度等待时间；Device Self Duration With AICore 更接近纯计算时间。',
        formula: '—',
        caveats: '当算子 Duration 远大于 AICore 时间时，说明时间花在等待而非计算上，属于调度/依赖问题。',
      },
      {
        name: '预期收益',
        definition: '针对某项优化给出的量化收益估算。',
        formula: '由本数据集推导（如 host 独占/步 ÷ 平均步长）或标注为经验区间',
        caveats: '置信度 low 的项为经验区间，不得当作承诺值；多项优化不可简单相加。',
      },
    ],
  },
];

/** Artifact reference: which file carries what. */
export const ARTIFACT_DOCS = [
  {
    name: 'trace_view.json',
    producer: 'torch_npu Ascend PyTorch Profiler（*_ascend_pt/ASCEND_PROFILER_OUTPUT）',
    content: 'Chrome trace 事件流：Host 的 python/ATen/aclnn 事件与 Device 的 kernel / HCCL / memcpy 事件。',
    notes: [
      '**是一个裸 JSON 数组**，不是 {"traceEvents": [...]} 包装形式；两者都会被解析。',
      'ts 为微秒，可能是十进制字符串（如 "1704161511420306.491"）；dur 为数字。',
      'Host/Device 由 ph:"M" 的 process_name 元数据区分：Python / CANN → Host，Ascend Hardware → Device。',
      '导出中断时文件可能缺少收尾括号；解析器容忍截断并给出告警。',
    ],
  },
  {
    name: 'kernel_details.csv',
    producer: 'torch_npu Ascend PyTorch Profiler',
    content: '设备侧每个 kernel 的明细，含起止时间、时长、等待时间与 AI Core 流水指标。',
    notes: [
      '基础表头：Device_id, Name, Type, Accelerator Core, Start Time(us), Duration(us), Wait Time(us), Block Num。',
      'Step Id / Task ID / Stream ID / Input Shapes 等列为条件输出，按列名解析、不按列号。',
      '开启 AI Core 指标后会出现 mac_ratio / mte1_ratio / mte2_ratio / mte3_ratio / vec_ratio / scalar_ratio 等列。',
      'Start Time 未必有序，时间线重建前需排序。',
    ],
  },
  {
    name: 'operator_details.csv',
    producer: 'torch_npu Ascend PyTorch Profiler',
    content: 'Host 与 Device 两侧的算子级耗时：Host Self/Total 与 Device Self/Total（含 AICore）。',
    notes: [
      '表头：Name, Input Shapes, Call Stack, Host Self Duration(us), Host Total Duration(us), Device Self Duration(us), Device Total Duration(us), Device Self Duration With AICore(us), Device Total Duration With AICore(us)。',
      'Host Total 包含子算子，做"框架开销"统计时应使用 Host Self。',
    ],
  },
  {
    name: 'op_statistic.csv',
    producer: 'torch_npu / msprof-analyze',
    content: '按算子类型聚合的调用次数与累计/平均耗时、占比。',
    notes: ['常用表头：Device_id, OP Type, Core Type, Count, Total Time(us), Avg Time(us), Ratio(%)。', '用作 trace 聚合的交叉校验来源。'],
  },
  {
    name: 'op_summary.csv',
    producer: 'msprof / MindStudio Insight（PROF_*/mindstudio_profiler_output）',
    content: '算子实例明细，含 Task Type / Accelerator Core / Block Num 与流水指标列。',
    notes: [
      '常用表头：Device_id, Op Name, OP Type, Task Type, Task Start Time(us), Task Duration(us), Task Wait Time(us), Block Num。',
      '注意这里是 `Task Start Time(us)`，而 kernel_details 用 `Start Time(us)`：解析器同时兼容两种拼写。',
    ],
  },
  {
    name: 'step_trace_time.csv',
    producer: 'torch_npu Ascend PyTorch Profiler',
    content: '逐步的 Computing / Communication / Overlapped / Free 分解，可带 Stage 列。',
    notes: [
      '表头：Device_id, Step, Computing, Communication(Not Overlapped), Overlapped, Communication, Free, Stage, Bubble, Preparing。',
      '表头未声明单位，CANN 写入的是毫秒；解析器按数值量级判断并换算为微秒（会在告警中说明）。',
      'Stage 列（Prefill / Decode）是唯一由工具直接给出的阶段标签，解析器优先采用。',
    ],
  },
  {
    name: 'api_statistic.csv',
    producer: 'torch_npu / msprof-analyze',
    content: 'Host 侧 API 调用统计（含 aclnn/runtime 接口）。',
    notes: ['计入 Host 调度开销；其时间与设备算子时间不可直接相加。'],
  },
  {
    name: 'communication.json / communication_matrix.json',
    producer: 'msprof-analyze（PROF_*/analyze 或 ASCEND_PROFILER_OUTPUT）',
    content: '集合通信的耗时、带宽与通信矩阵。',
    notes: ['用于确认通信算子分组、消息量与跨卡流量分布。'],
  },
  {
    name: 'profiler_info_{Rank_ID}.json',
    producer: 'torch_npu Ascend PyTorch Profiler（*_ascend_pt 根目录）',
    content: '采集元数据：设备型号、rank、并行配置、CANN/torch_npu/vLLM 版本。',
    notes: ['用于确认产物来源与分析上下文。'],
  },
  {
    name: 'analysis.db / msprof_*.db',
    producer: 'torch_npu / msprof',
    content: 'SQLite 数据库，含 StepTraceTime、CommAnalyzer* 等表。',
    notes: [
      '**不直接解析**：请用 msprof-analyze 或 MindStudio Insight 导出为 CSV 后再上传。',
      '数据库中的时间戳基准与 JSON 不同（DB 为纳秒级 Unix 时间，JSON 为相对微秒），不可混用。',
    ],
  },
  {
    name: 'mstx / msprof_tx 产物',
    producer: 'torch_npu mstx 插件或 msprof',
    content: '用户打点（如 dataloader / forward / step），可作为阶段与区间标记。',
    notes: ['Ascend 默认不开启 mstx；若存在阶段标记，本插件会优先用于 Prefill/Decode 划分。'],
  },
];

/** Usage guidance shown in the page's empty state. */
export const USAGE_DOCS = {
  quickStart: [
    '在 vLLM-Ascend 服务上开启 profiling（torch_npu profiler 或 /start_profile、/stop_profile），并确保导出了 ASCEND_PROFILER_OUTPUT 目录。',
    '把 trace_view.json、kernel_details.csv、operator_details.csv、op_statistic.csv、step_trace_time.csv 一起上传（可多选，也可直接打包成 zip/tar.gz）。',
    '大文件建议使用"按路径分析"：把 *_ascend_pt 目录或其中的文件放到 DSH 工作区，然后在页面填写路径，避免上传 GB 级文件。',
    '解析完成后依次查看三大模块，最后导出 Markdown/PDF 报告。',
  ],
  phaseAdvice: [
    'Ascend 的 trace 与 CSV **不包含 Prefill/Decode 阶段标签**（除 step_trace_time.csv 的 Stage 列以外）。',
    '要得到精确的阶段结论，最可靠的做法是分别采集 prefill-only 与 decode-only 两个窗口，并在页面的"阶段口径"中选择对应阶段。',
    '若无法分开采集，本插件会按步长分布推断，并在报告中标注置信度与推断依据。',
  ],
  performanceTips: [
    'trace_view.json 可能达到数 GB：默认按事件预算做等距采样（通信/拷贝/长耗时算子全量保留），并在报告与页面中明确标注抽样情况。',
    '关闭不需要的采集项（内存、FLOPS、算子级流水）可以显著降低 profiling 自身对被测负载的干扰；分析完成后再决定是否开启更高采集级别。',
    '同一时间只解析一个大 trace，解析完成后再上传下一个，避免内存峰值叠加。',
  ],
};

/**
 * The whole documentation bundle served to the page.
 * @param {object} [context] - plugin context (version, config) to include.
 * @returns {object} documentation payload.
 */
export function documentationBundle(context = {}) {
  return {
    plugin: {
      name: 'vllm-ascend-profiler-analyzer',
      title: 'vLLM-Ascend Profiler Analyzer',
      version: context.version ?? '1.0.0',
      routePrefix: context.routePrefix ?? '/vllm-ascend-profiler',
      config: context.config ?? {},
    },
    metricDocs: METRIC_DOCS,
    artifactDocs: ARTIFACT_DOCS,
    usage: USAGE_DOCS,
  };
}
