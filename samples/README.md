# 示例 profiling 产物

这里的示例模拟 torch_npu 导出的 `*_ascend_pt` 目录，用来快速试用 **vLLM-Ascend Profiler Analyzer**：
把某个场景的 zip 导进页面，就能走完「导入 → 概览 → 时序取证 → 占比归因 → 优化行动」五步，再按下面第 6 步的说明做一次前后对比。

示例是**生成物**（确定性随机种子，可重复生成），不是真实硬件采集结果 —— 真实产物的字段说明见
[`docs/metrics-and-fields.md`](../docs/metrics-and-fields.md) 与页面顶部的「说明文档」。
仓库里唯一一份真实数据是 [`test/fixtures/ascend-trace_view.sample.json`](../test/fixtures/ascend-trace_view.sample.json)
（Ascend/mstt，Apache-2.0 的 trace 截断前缀，只含 Host 侧事件）。

## 直接可用的 zip（已随仓库提供）

| zip | 大小 | 用途 |
| --- | --- | --- |
| [`quickstart.zip`](quickstart.zip) | 46 KB | 2 步的最小产物：先确认"能导入、能出图" |
| [`host-schedule-bound.zip`](host-schedule-bound.zip) | 429 KB | 优化前（Host 调度受限，20 步） |
| [`host-schedule-bound-optimized.zip`](host-schedule-bound-optimized.zip) | 231 KB | **优化后**（与上一个配对：每步墙钟 −45%、NPU 忙碌率 33% → 60%） |
| [`decode-comm-bound.zip`](decode-comm-bound.zip) | 180 KB | 跨卡通信受限（TP=8 decode，24 步） |
| [`prefill-compute-bound.zip`](prefill-compute-bound.zip) | 96 KB | NPU 计算受限（chunked prefill，6 步） |

每个 zip 解包后就是一个完整的产物目录（`trace_view.json` + 4 张 CSV + `profiler_info_0.json` + `communication.json`），
页面里可以直接拖入 zip，也可以解包后多选文件，或放进 DSH 工作区用"按路径分析"。

## 生成

```powershell
node test/make-fixture.mjs D:\tmp\vllm-ascend-samples   # 四个场景（24 / 6 / 20 / 20 步）
```

命令行只接受输出目录；要生成**短样例**（例如 2 步、约 0.6 MB 的冒烟样例）用生成器 API 覆盖步数与 rank：

```js
import { buildScenario } from './test/make-fixture.mjs';
import { writeFileSync, mkdirSync } from 'node:fs';

const { files } = buildScenario('host-schedule-bound', { steps: 2 });
mkdirSync('D:/tmp/quickstart', { recursive: true });
for (const [name, content] of Object.entries(files)) writeFileSync(`D:/tmp/quickstart/${name}`, content, 'utf8');
```

## 目录结构（每个场景一份）

| 文件 | 内容 |
| --- | --- |
| `trace_view.json` | Chrome-trace 风格的**裸 JSON 数组**（无 `traceEvents` 包装），`ts` 为十进制字符串；`ph:"M"` 的 `process_name` 区分 Host（Python）与 Device（Ascend Hardware） |
| `kernel_details.csv` | 每个 device kernel 一行：`Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num,Input Shapes,Output Shapes,mac_ratio,mte2_ratio` |
| `operator_details.csv` | 每个算子实例一行，含 `Host/Device Self/Total Duration(us)` |
| `op_statistic.csv` | 按算子汇总的 CANN 统计：`OP Type,Count,Total Time(us),Avg Time(us),Ratio(%)` |
| `step_trace_time.csv` | 逐步时序：`Computing / Communication(Not Overlapped) / Overlapped / Communication / Free`，含 `Stage`（Prefill/Decode） |
| `profiler_info_0.json` | rank 元信息（设备型号、并行策略、CANN / torch_npu / vllm-ascend 版本） |
| `communication.json` | HCCL 通信算子汇总（名称、通信组、单次耗时、消息大小） |

## 四个场景与预期结论

四个场景故意做成**互相可区分**，用来验证"结论确实随产物变化"；其中前两个是一对**优化前 / 优化后**：

| 场景 | 步数 | 生成规模 | 预期瓶颈 | 形态 |
| --- | --- | --- | --- | --- |
| `host-schedule-bound` | 20 | 约 36k 事件 / 6.4 MB | **Host 调度瓶颈 96.5** | eager decode：Host 每步派发 ~1500 个算子与同步，设备频繁空闲，NPU 忙碌率低（~33%） |
| `host-schedule-bound-optimized` | 20 | 约 16k 事件 / 3.4 MB | **Host 调度瓶颈 87.6** | **与上一行配对**的"优化后"采集：图模式把逐步下发合并成一次回放，每步墙钟 −45%、NPU 忙碌率 33% → 60%、派发算子数/步 −59% |
| `decode-comm-bound` | 24 | 约 10k 事件 / 2.6 MB | **跨卡通信瓶颈 90.0** | TP=8 decode：每层小消息 AllReduce（96 KB）且**未与计算重叠**，通信未掩盖比例高 |
| `prefill-compute-bound` | 6 | 约 6.7k 事件 / 1.3 MB | **NPU 计算瓶颈 94.8** | chunked prefill：长步、MatMul/FlashAttention 主导，MAC 利用率高、通信占比低 |

> 上表数值由生成器产出后经插件自身 `parseProfileSet → buildDataset → analyzeDataset` 实测。
> 想调成贴近自己负载的形状：改 `test/make-fixture.mjs` 顶部 `SCENARIOS` 里的每步算子数、MatMul/Attention 时长、
> 通信是否重叠、消息大小、Host 每步派发数等参数，再重新生成即可。

### 试第 6 步的前后对比

先导入 `host-schedule-bound`（优化前），再到第 6 步导入 `host-schedule-bound-optimized`（优化后），页面会：

* 第 3 步给出左右两张泳道图（左 before / 右 after）与一排 delta 徽标；
* 第 4 步给出两条大类构成条 + 两张耗时分布图，以及大类 / 算子的变化表；
* 第 6 步给出结论句、四张 `前 → 后` 对比卡、可比性说明，以及第 5 步每条建议的 `已达成 / 部分达成 / 未达成 / 无法判定`。

预期结果（`test/compare.test.mjs` 就是按这组数值断言的）：每步墙钟 −45%、NPU 忙碌率 +27pt、Host 独占/步 −79%、派发算子数/步 −59%，"启用图模式 / 减少逐步下发 / 消除同步点"三条建议达成；由于墙钟缩短，**通信未掩盖占比**从 0.31% 升到 0.57%（绝对值几乎不变），页面会把它解释为"占比类指标反向变化"，而不是性能退化。

## 导入方式

1. **打包上传**：把场景目录压成 zip 拖入导入区（页面内存内解压，成员大小与数量有上限）；
2. **多选上传**：选中该目录里的全部文件一起拖入 —— 它们会作为**一个数据集**；
3. **按路径分析**：把目录放进 DSH 会话工作区，在页面填写路径；解析为流式读取，适合 GB 级 `trace_view.json`。

导入后建议按顺序核对：第 2 步的结论是否与上表一致 → 第 3 步泳道图 → 第 4 步构成条 / 耗时分布图 → 第 5 步 ①→⑤ 推理链。
