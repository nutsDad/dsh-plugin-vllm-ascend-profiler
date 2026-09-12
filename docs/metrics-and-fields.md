# 指标口径与 vLLM-Ascend Profiling 字段说明

本文档解释插件页面与报告中出现的每一个指标，以及它来自哪个产物的哪个字段、怎么算出来的、有什么坑。页面内「说明文档」面板展示同一份内容（由 `lib/docs.js` 提供），因此报告中的数字总能就近找到定义。

---

## 一、时间与坐标口径

| 概念 | 定义 | 计算 | 注意事项 |
| --- | --- | --- | --- |
| 采集窗口（wall） | 时间线覆盖的总时长 | `max(ts+dur) − min(ts)` | 已按首个事件**归零**；多 rank 时按各 rank 首个事件分别对齐 |
| 时间单位 | trace 时间戳单位 | 前 2000 个事件推断（`<0.05` 的时长占比 > 30% → 毫秒），可配置固定 | `trace_view.json` 的 `ts` 常为**十进制字符串**（如 `"1704161511420306.491"`），`dur` 为浮点数 |
| 绝对时间 | 未被归零的原始时间戳 | `tsUs` | CSV 的 `Start Time(us)` 是设备绝对时间，trace 的 `ts` 是相对时间，**两者不做同轴绘制** |
| 相对时间 | 页面 X 轴使用的值 | `relTsUs = ts − rank 起点` | 多 rank 时各卡时钟起点不同属正常现象；报告中会列出各 rank 起点 |
| 区间并集 | 同侧并行执行的时间去重 | `union(intervals)` | 多流并行不会让忙碌时间超过墙钟；这是"忙碌率"不会 >100% 的原因 |
| 区间重叠 | 两组区间同时覆盖的时间 | `|union(a) ∩ union(b)|` | 两侧都先合并再扫描，否则并行算子会**重复计数** |

---

## 二、模块一 · 泳道时序图

| 指标 | 定义 | 计算 | 注意事项 |
| --- | --- | --- | --- |
| 算子条（bar） | 一次算子调用 | `start = ts`，`width = dur` | 同一行内多 stream 并行时算子条会重叠显示；行统计用区间并集，不重复计数 |
| 泳道分组 | Host（CPU）/ Device（昇腾 NPU） | 由 `ph:"M"` 的 `process_name` 判定：`Python` / `CANN` → Host；`Ascend Hardware` → Device | `pid` 本身异构（OS pid、打包整数、字符串 `"HCCL"`），不作为主要判据；缺失元数据时回退到 `device id` / `stream id` 参数与 `cat` |
| 算子分类 | 计算 / 通信 / 数据拷贝 / 调度 / 其他 | 依 `name`、`OP Type`、`Task Type`、`Core Type`、`cat` 规则匹配，通信与拷贝优先 | Host 侧非通信非拷贝的算子统一计为「调度（框架开销）」；Host 侧 `aten::_to_copy` 仍归「拷贝」，分析时会区分设备侧与 Host 侧拷贝 |
| 行排序 | 组内算子行的顺序 | 累计耗时 / 调用次数 / 算子名 / 单次最长 | 默认按累计耗时降序 |
| 视图抽样 | 时间线下发到浏览器的事件预算 | 每行配额 ∝ 该行累计耗时占比；行内保留**耗时最长**的事件后恢复时间序 | 页面显示「本行显示 N/M」，累计耗时统计**不受**视图抽样影响 |
| 解析采样 | 解析阶段的等距采样 | `stride = ceil(已见事件数 / 预算)`，通信/拷贝/长耗时算子**全量保留** | 采样时累计耗时优先取 CANN 统计表（`totalsSource`） |

---

## 三、模块二 · 耗时占比

| 指标 | 定义 | 计算 | 注意事项 |
| --- | --- | --- | --- |
| 累计总耗时 | 某算子所有调用的耗时之和 | `Σ durUs` | 多流并行时全部算子累计之和会大于墙钟；占比的分母是**算子总耗时**而非墙钟 |
| 单次执行耗时 | 平均单次耗时 | `totalUs / count` | Prefill/Decode 混采时均值会被两类负载混合，应切换阶段维度查看 |
| p50 / p95 / 最长 | 单次耗时分布 | 最近秩分位数 | p95 远大于 p50 说明该算子存在长尾（常见于变长序列） |
| 占总时间百分比 | 该算子占全部算子累计耗时的比例 | `totalUs / ΣtotalUs` | 与"占墙钟比例"不同：前者是算子构成占比，后者是时间线占用率 |
| 总计来源 | 该行累计耗时的来源 | trace 采样时优先取 CSV，否则取 trace；CSV 侧只取一张表（优先 `op_statistic`，其次 `kernel_details` / `operator_details`） | 同一个算子的耗时常同时出现在三张 CSV 表里，**不能相加**（相加会成倍放大耗时与调用次数）；两者都保留（`traceTotalUs` / `csvTotalUs`）并给出偏差；两张 CSV 表相差 >20% 会告警 |
| 通信与计算重叠 | 通信区间与计算区间重合的时间 | `|union(comm) ∩ union(compute)|` | 通信优化的首要判据是**未掩盖**的通信，而不是通信总时长 |
| 通信未掩盖 | 通信时间中未被计算覆盖的部分 | `|union(comm)| − overlap` | 小消息集合通信（≤60µs）受 HCCL 固定开销限制，扩大 buffer 无效 |
| 大类占比（构成条 / 耗时分布图） | 按计算/通信/拷贝/调度汇总 | 口径可切「设备侧 / 全部 / Host 侧」 | 分母是所选口径下的算子总耗时，页面会显示该数值；构成条按大类切分，耗时分布图按算子切分（面积 = 占比） |

---

## 四、模块三 · 分析指标

| 指标 | 定义 | 计算 | 注意事项 |
| --- | --- | --- | --- |
| NPU 忙碌率 | 设备侧事件覆盖窗口的比例 | `|union(device)| / wall` | 低 → 设备在等待（Host、通信依赖或数据） |
| Host 忙占比 | Host 事件覆盖窗口的比例 | `|union(host)| / wall` | 只作参考，不单独构成门限 |
| Host 独占（设备空等） | Host 忙而设备空闲的时间 | `|union(host)| − overlap(host, device)` | 是"设备在等主机"的**下界**；设备同时也在等通信时会低估调度问题 |
| NPU 空闲 | 设备未覆盖的时间 | `wall − |union(device)|`（忽略 <50µs 的抖动） | 结合空闲段数量与平均长度判断"缺一个大算子"还是"下发跟不上" |
| 推理步（step） | 一次前向的执行区间 | 优先 `step_trace_time.csv`；否则 mstx 标记 → 引擎算子重复出现 → 空闲间隔切分 → 整窗 | 步长表未声明单位，按数值量级判定为毫秒并换算（会给出告警） |
| Prefill / Decode | 阶段归属 | ① `Stage` 列 → ② mstx 标记 → ③ 步长 log 空间双峰（长步=Prefill，短步=Decode） | 无标签时置信度下降，报告中给出推断依据；步长高度一致时会判为纯 Decode 负载并标注低置信度 |
| 每步 Host 独占 | 每步设备空等主机的时间 | `hostOnlyUs / stepCount` | Decode 瓶颈的核心指标之一 |
| Host 派发算子数/步 | 每步 CPU 侧下发算子数量 | `host 调度类事件数 / stepCount` | 超过 ~1200 个/步说明逐步派发开销已成固定成本 |
| 通信占设备耗时 | 通信在设备忙碌时间中的占比 | `commUnion / deviceBusy` | 与"占墙钟"一起看，避免用错分母 |
| 单次通信时长中位数 | 集合通信单次耗时 | `median(dur)` | ≤60µs → 小消息**延迟受限**（固定开销主导）；≥16MB → **带宽受限** |
| 设备侧拷贝 / Host 侧拷贝 | 拷贝类算子耗时占比 | 分别按设备与 Host 分母计算 | 推理稳态下设备侧拷贝应接近 0；Host 侧拷贝高通常意味着同步式 D2H |
| D2H / H2D 每步耗时 | 每步拷贝耗时 | `Σdur / stepCount` | 持续 H2D 常见于 PD 分离 KV 传输、前缀缓存换入或权重重载 |
| 同步类算子 | `.item()` / `tolist` / `synchronize` 等 | 名称匹配计数与累计耗时 | 每次同步都会排空设备队列，直接叠加到 TPOT |
| MAC / MTE1 / MTE2 / MTE3 / Vector / Scalar 利用率 | 算子级流水占用 | CANN 指标列按 `dur` 加权平均 | 只有开启 AI Core 指标采集（`PipeUtilization`）的产物才有；缺列时"算力受限 / 访存受限"只能给条件性结论 |
| Duration 与 AICore Time | CANN 的 `Duration` 含调度等待；`Device Self Duration With AICore` 更接近纯计算 | — | `Duration` 远大于 AICore 时间 → 时间花在等待而非计算 |
| 瓶颈得分 | 四类候选的 0–100 分 | 门限化严重度加权（未达门限得 0） | ≥40 视为达到门限；得分是**证据强度**，不是性能损失比例 |
| 预期收益 | 某项优化的量化收益 | 由本数据集推导或标注经验区间 | 置信度 `low` 的项为经验区间；多项优化不可简单相加（上限截断 70%） |

---

## 五、vLLM-Ascend / 昇腾 profiling 产物与字段

### 5.1 目录结构

```
<host>_<pid>_<YYYYmmddHHMMSSmmm>_ascend_pt/          # 每个 worker 进程一个目录
├── profiler_info_<Rank_ID>.json                     # 采集元数据（在根目录）
├── FRAMEWORK/                                       # 框架侧原始数据
├── PROF_<id>_<ts>_<hash>/                           # msprof 原生结果（嵌套在此）
│   └── analyze/                                     # communication.json 等
└── ASCEND_PROFILER_OUTPUT/                          # 主要分析产物
    ├── trace_view.json          analyse.done
    ├── kernel_details.csv       operator_details.csv
    ├── op_statistic.csv         api_statistic.csv
    ├── step_trace_time.csv      communication.json
    ├── communication_matrix.json memory_record.csv
    ├── npu_module_mem.csv       analysis.db
    └── task_time.csv / soc_pmu.csv / l2_cache.csv / nic.csv / pcie.csv / roce.csv / hccs.csv
```

注意：vLLM-Ascend 采集使用 `data_simplification=True` 时会删除 `PROF_*/analyze`、`mindstudio_profiler_output` 等目录，因此**通常只有 `ASCEND_PROFILER_OUTPUT/`**，插件不依赖任何被删除的目录。

### 5.2 `trace_view.json`

* 是**裸 JSON 数组**（不是 `{"traceEvents": [...]}`）；插件同时兼容两种形态，也兼容按通用数组回退；
* 事件字段：`ph`（`X` 完整事件 / `M` 元数据 / `s`、`f` 流事件 / `C` 计数器）、`name`、`pid`、`tid`、`ts`（**微秒，可能是十进制字符串**）、`dur`、`cat`、`args`；
* `cat` 常见值：`cpu_op`、`python_function`、`kernel`、`hccl`、`HostToDevice`、`dequeue`、`enqueue`、`async_npu`、`fwdbwd`、`GC`；
* `args` 常见键：`Device Id`、`Stream Id`、`Task Type`、`OP Type`、`Input Dims`、`Input type`、`Call stack`、`Module Hierarchy`、`Sequence number`、`Fwd thread id`、`flops`（小写）；
* `ph:"M"` 的 `process_name` 是 Host/Device 判定的**权威来源**：`Python` / `CANN` → Host，`Ascend Hardware` → Device；
* 导出中断时文件可能缺少收尾括号：插件按流式解析保留全部完整事件，并给出截断告警。

### 5.3 `kernel_details.csv`（torch_npu）

规范基础表头（其余列按采集项条件输出）：

```
Device_id, Name, Type, Accelerator Core, Start Time(us), Duration(us), Wait Time(us), Block Num
```

* 该表是 msprof `op_summary_*.csv` 的**改名投影**：`Op Name→Name`、`OP Type→Type`、`Task Type→Accelerator Core`、`Task Start Time(us)→Start Time(us)`、`Task Duration(us)→Duration(us)`、`Task Wait Time(us)→Wait Time(us)`；插件两种拼写都认；
* 开启 AI Core 指标后会出现 `mac_ratio` / `mte1_ratio` / `mte2_ratio` / `mte3_ratio` / `vec_ratio` / `scalar_ratio` / `icache_miss_rate` 等列，插件按列名模式匹配并归一（>1.5 视为百分数）；
* `Step Id` / `Task ID` / `Stream ID` / `Input Shapes` 等列在 vLLM-Ascend 采集下**通常不存在**（未传 `schedule=`），因此解析器一律按列名查找，不做位置假设；
* `Start Time` **未必有序**，重建时间线前必须排序。

### 5.4 `operator_details.csv`（torch_npu）

```
Name, Input Shapes, Call Stack,
Host Self Duration(us), Host Total Duration(us),
Device Self Duration(us), Device Total Duration(us),
Device Self Duration With AICore(us), Device Total Duration With AICore(us)
```

* 本插件把 **Device Total Duration** 作为算子耗时（`durUs`），同时保留 Host Self/Total 与 AICore 版本；
* 统计"框架开销"时应使用 **Host Self**（Host Total 含子算子）。

### 5.5 `op_statistic.csv` / `api_statistic.csv`

```
op_statistic:  Device_id, OP Type, Core Type, Count, Total Time(us), Avg Time(us), Ratio(%)
api_statistic: Device_id, API Name, Count, Total Time(us), Avg Time(us), Min Time(us), Max Time(us), Ratio(%)
```

* 作为 trace 聚合的**交叉校验**来源（报告中给出 cross-check 偏差）；
* `api_statistic` 计的是 Host 侧 API 调用时间，**不能**与设备算子时间直接相加，插件会在告警中说明。

### 5.6 `op_summary*.csv`（msprof / MindStudio）

```
Device_id, Op Name, OP Type, Task Type, Task Start Time(us), Task Duration(us), Task Wait Time(us), Block Num
（+ Mix Block Num / Input Shapes / Input Data Types / Input Formats / Output Shapes / ... 条件列）
```

* 列集合与顺序随 `--task-time` / `--aic-mode` / 产品形态变化，务必按列名解析；
* 数值列中 `N/A` 是合法值（插件将其视为缺失，避免污染泳道与统计）。

### 5.7 `step_trace_time.csv`（torch_npu）

```
Device_id, Step, Computing, Communication(Not Overlapped), Overlapped,
Communication, Free, Stage, Bubble, Communication(Not Overlapped and Exclude Receive), Preparing
```

* 表头**不带单位**，CANN 写入的是**毫秒**；插件按量级判定（步长最大值 < 2000 视为毫秒）并换算为微秒，同时给出告警；
* `Stage` 列（`Prefill` / `Decode`）是唯一由工具直接给出的阶段标签，阶段划分优先级最高；
* 该表没有算子列，插件为每行生成 `Step N` 作为名称；步骤总时长在没有总计列时按 `Computing + Communication(Not Overlapped) + Free` 重建。

### 5.8 `analysis.db` / `msprof_*.db`

* **不直接解析**：提示改用 msprof-analyze / MindStudio Insight 导出 CSV；
* 数据库时间基准与 JSON 不同（DB 为纳秒级 Unix 时间、部分表的时长为毫秒），不可与 trace 混用。

### 5.9 `.proto` / `.bin`

* Ascend proto 无自描述 schema，插件做**:通用 wire 解码（varint / fixed32 / fixed64 / length-delimited，递归）→ 识别"名称字符串 + 两个大整数（起始 + 时长或结束）"**的候选记录，标注 `confidence: low`；
* 若已知 schema，可在插件配置中用 `protoFieldMap` 指定字段号（`{name, start, duration|end, opType, device}`），即变为精确映射。

---

## 六、常见解析陷阱与插件的处理方式

| 陷阱 | 处理 |
| --- | --- |
| `trace_view.json` 是裸数组 | 同时支持裸数组与 `traceEvents` 包装；元素按"是否含 `ph`/`ts`"区分事件与文档元数据 |
| `ts` 是十进制字符串 | 数值化解析，保留小数精度 |
| 文件被截断（无收尾括号） | 流式解析保留全部完整元素 + 截断告警 |
| `pid` 异构、Host/Device 混淆 | 以 `process_name` 为准，`device id` / `stream id` / `cat` 仅作回退 |
| CSV 表头版本漂移 | 别名表 + 按列名解析；未识别列保留在 `extra` 且只报告一次 |
| `Duration` 含调度等待 | 同时保留 `Wait Time` 与 AICore 时间，分析时可区分"等待"与"计算" |
| `N/A` 出现在数值列 | 统一视为缺失 |
| 编码不确定（UTF-8 / GBK） | UTF-8 优先，替换字符比例异常时回退 GBK 并给出提示 |
| 引号内含逗号（Call Stack / Input Shapes） | RFC4180 词法解析，不使用 `split(',')` |
| 单位混用（µs / ms） | 按表头声明的单位换算；未声明单位的表按量级判定并告警 |
| 多 rank 时钟起点不同 | 按各 rank 首个事件对齐，并列出各自起点；可用 `multiRankMode: raw` 保留原始时钟 |
| 多流并行导致累计耗时 > 墙钟 | 忙碌时间用区间并集；占比分母明确标注为"算子总耗时" |
| 大文件卡死页面 | 解析采样 + 视图预算 + 服务端流式读取（按路径分析） |
