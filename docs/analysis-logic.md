# 分析推理链、门限与收益推算

本文档说明模块三给出的结论是怎么推出来的：每一步的输入、判据、公式与边界。目标是**任何人都能用同一份数据复现同一个结论**，并且能明确指出"我不认同这一步，因为它的门限/假设不适用于我的场景"。

代码对应关系：

```
① 定位   lib/analysis/bottleneck.js  scoreBottlenecks() + lib/analysis/thresholds.js
② 证据   lib/analysis/bottleneck.js  computeIndicators() / push() 生成的 evidence 行
③ 根因   lib/analysis/rootcause.js   HYPOTHESES（机理库 + 触发条件 + 确认方法）
④ 方案   lib/analysis/recommend.js   TEMPLATES（适用条件 + 动作 + 验证 + 风险）
⑤ 收益   lib/analysis/recommend.js   estimate() + lib/analysis/index.js combineGains()
编排     lib/analysis/index.js        analyzeDataset()
```

---

## 一、总体结构

分析在多个**作用域**上分别进行，互不覆盖：

| 作用域 | 数据范围 | 说明 |
| --- | --- | --- |
| `overall` | 整个采集窗口 | 无单一阶段时，按**主导阶段**（墙钟占比最大的阶段）评估，并在摘要中标明"按主导阶段 X 评估" |
| `prefill` | `Stage` / 标记 / 推断为 Prefill 的步骤区间 | 阶段专属结论 |
| `decode` | 同上（Decode） | 阶段专属结论 |
| 人工指定 | 整窗，但按指定阶段解释 | 页面「阶段口径」选择后重新分析；适用于分开采集的窗口 |

每个作用域独立完成：定位 → 证据 → 根因 → 方案 → 收益；最后跨作用域**按 id 去重**合并（保留证据最全的一条并列出适用作用域）。

**为什么必须分阶段**：同一个采集里 Prefill 是算力受限、Decode 是通信/延迟受限的情况非常常见，混在一起做平均会把两个瓶颈都抹掉，得到"哪都不明显"的结论。

---

## 二、① 瓶颈类型定位

四类候选各自打分 0–100：

```
score = Σ (权重 × severity(指标, 门限))
severity = 0                                 当指标未达门限（门限是闸门，不是梯度）
         = 0.5 + 0.5 × clamp01((v − gate)/(saturation − gate))   当达到门限
```

得分是**证据强度**，不是性能损失比例。`≥40` 视为达到瓶颈门限（`triggerScore` 可配置）；未达门限的候选同样展示，并逐条列出"未达门限项"，便于判断"离瓶颈还有多远"。

### 2.1 Host 调度瓶颈（权重 45 / 35 / 20）

| 门限 | 默认值 | 取值依据 |
| --- | --- | --- |
| `hostOnlyGate` Host 独占占墙钟 | ≥ 45%（主门限之一） | Host 从"发起者"变成"瓶颈候选"的分界；图模式/流水充分的推理负载通常在 10%–30% |
| `deviceIdleSupport` NPU 空闲占比 | ≥ 8%（主门限之一） | 更小的空隙属于正常 launch 延迟与流切换 |
| `dispatchPerStep` 派发算子数/步 | ≥ 1200（支撑条件） | 超过此值，逐 op 派发（Python + ATen + aclnn 包装）成为每步固定成本 |
| `hostExclusivePerStepUs` 每步独占 | ≥ 800µs（支撑条件） | Decode 每步 5–30ms 时，800µs 已占显著比例 |
| `hostBusyGate` Host 忙占比 | ≥ 45%（加分项） | 单独不构成门限，只放大得分 |

判定式为：`主门限之一成立` 且（`支撑条件之一成立` 或 `Host 忙占比门限成立`）。证据行同时给出同步类算子（`.item()`/`tolist`/`synchronize`）的次数与累计耗时。

### 2.2 NPU 计算瓶颈（权重 55 / 25 / 20 / 12）

| 门限 | 默认值 | 取值依据 |
| --- | --- | --- |
| `deviceBusyGate` NPU 忙碌率 | ≥ 82% | 设备几乎全程有任务，瓶颈在设备内部（优化方向是"算得更快"） |
| `computeShareGate` 计算算子占墙钟 | ≥ 55% | 计算确实是主要构成 |
| `commExposedLow` 通信未掩盖占比 | < 12%（加分条件） | 通信不是主瓶颈时才把注意力放在计算上 |
| `macRatioComputeBound` MAC 利用率 | ≥ 0.45 → `compute-bound`（算力受限） | Cube 计算单元是主要受限资源 |
| `mte2RatioMemoryBound` MTE2 利用率 | ≥ 0.45 → `memory-bound`（访存受限） | 受 HBM/搬运带宽限制 |
| `lowUtilizationGate` MAC 利用率 | < 0.3 → `low-efficiency`（忙而不算） | 设备忙但算力单元空转 |

三者都不满足时不给子类型（只报"计算占比高"），避免过度解读。

### 2.3 跨卡通信瓶颈（权重 55 / 20 / 15 / 15）

| 门限 | 默认值 | 取值依据 |
| --- | --- | --- |
| `commShareGate` 通信占设备耗时 | ≥ 12% | 张量并行 8 卡场景的典型观察线 |
| `commExposedGate` 通信未掩盖占墙钟 | ≥ 5% | 只有未被掩盖的通信才在关键路径上 |
| `commLatencyBoundUs` 单次时长中位数 | ≤ 60µs → 延迟受限 | 该区间由 HCCL 固定启动开销主导，扩大消息无效 |
| `commBandwidthBoundMb` 报文大小中位数 | ≥ 16MB → 带宽受限 | 互联带宽成为限制 |
| 通信重叠率 | < 40% 时 +15 分 | 重叠率低说明通信暴露 |

**两个门限必须同时成立**（占比 + 未掩盖），否则只算"通信很重但被藏住了"。

### 2.4 数据拷贝瓶颈（权重 45 / 12 / 10）

| 门限 | 默认值 | 取值依据 |
| --- | --- | --- |
| `copyShareGate` 设备侧拷贝占墙钟 | ≥ 4% | 推理稳态下权重与 KV 应常驻显存，拷贝本应接近 0 |
| `hostCopyShareGate` Host 侧拷贝占 Host 忙 | ≥ 8% | 通常意味着同步式 D2H（采样、日志、指标） |
| `d2hPerStepUs` 每步 D2H | ≥ 200µs | 常见于 logits/统计量回传，且往往伴随流同步 |
| 同步类算子存在 | +12 × min(1, 同步次数/步) | 同步把拷贝变成停顿 |
| 每步 H2D ≥ 200µs | +10 × severity | 稳态 H2D 说明有东西每步都在重载 |

若四类都未达门限：结论为「未发现单一主导瓶颈」，同时列出各候选的接近程度与缺失证据，**不编造瓶颈**。

---

## 三、② 量化证据

每个候选的每条证据行包含五个字段，页面与报告都完整展示：

| 字段 | 含义 |
| --- | --- |
| `metric` | 指标名（中文） |
| `value` | 实测值（数值来自解析结果，保留 2 位小数） |
| `threshold` / `relation` / `comparedTo` | 门限与比较方向 |
| `comparison` | 人类可读的比较式，含**门限取值依据** |
| `source` | 该数值由哪些产物字段、哪种算法得出 |
| `passed` | 是否达门限 |

此外全量窗口与每个阶段都有一张指标表（时间分布 / 算子构成 / 步粒度 / 算力与流水利用率），并附**口径说明**（区间并集、通信未掩盖、Host 独占的定义）。

---

## 四、③ 根因推断

每个根因假设包含：`mechanism`（作用机理）、`vllmBehaviour`（vLLM-Ascend 侧的实现原因）、`triggers`（触发该推断的数据）、`checks`（现场确认方法）。

只有**达到门限的候选**才会进入根因推断；假设还必须通过 `select()` 的数据条件（例如"Decode 阶段 + 派发算子数 ≥ 400 + NPU 未打满"），否则不会出现在报告里。若某瓶颈证据充分但没有可细分的机理，会给出通用的 `*.generic` 假设，明确写出"需要现场确认"。

机理库覆盖（节选）：

| id | 瓶颈 | 触发条件 |
| --- | --- | --- |
| `host.eager-python-per-token` | Host | Decode + 派发 ≥ 400/步 + NPU < 95% |
| `host.sampler-sync` | Host | 存在同步类算子且 ≥ 0.5 次/步或 ≥ 3% 墙钟 |
| `host.prefill-metadata` | Host | Prefill + Host 独占 ≥ 15% + 派发 ≥ 300/步 |
| `host.launch-gaps` | Host | 空闲段 ≥ 5 个且平均 ≥ 80µs 且空闲 ≥ 5% |
| `compute.prefill-attention-quadratic` | 计算 | Prefill + 计算占比 ≥ 55% + 注意力类算子存在 |
| `compute.decode-memory-bound` | 计算 | Decode + NPU ≥ 70% + MAC < 0.35 + MTE2 ≥ 0.25 |
| `compute.fragmented-small-ops` | 计算 | NPU ≥ 70% + 计算 ≥ 40% + 计算算子时长中位数 < 30µs |
| `comm.decode-tp-small-message` | 通信 | Decode + 通信占比 ≥ 12% + 单次中位数 ≤ 120µs |
| `comm.prefill-bandwidth` | 通信 | Prefill + 通信占比 ≥ 12% + 报文 ≥ 1MB |
| `comm.overlap-disabled` | 通信 | 通信占比 ≥ 8% + 重叠率 < 40% |
| `copy.d2h-sync-sampling` | 拷贝 | D2H ≥ 150µs/步 或存在同步类算子 |
| `copy.h2d-kv-or-weight` | 拷贝 | H2D ≥ 200µs/步 |

---

## 五、④ 可落地优化方案

每项包含：适用阶段、优先级（高/中/低）、依据（引用实测指标）、执行动作（`config` / `code` / `measure` / `verify` 分类）、预期收益、验证方法、风险、关联根因。

排序分数：

```
priorityScore = 20 × 优先级权重(高3/中2/低1) + min(40, 0.4 × 瓶颈得分) + min(20, 0.4 × 估算收益)
```

涉及具体版本开关的建议（例如通信计算融合、融合算子集合、图模式配置项）会标记为**"需版本确认"**，措辞为"确认当前 vLLM-Ascend 版本支持后再启用"，不把未验证的开关当作既成事实。

---

## 六、⑤ 预期收益

### 6.1 由本数据集推导（置信度 medium / high）

| 优化项 | 公式 | 说明 |
| --- | --- | --- |
| 图模式（Decode） | `min(0.8, 0.75 × hostExclusivePerStepUs / avgStepUs)` | 每步可消除的 host 独占时间占步长比例；区间按 60%–100% 取 |
| 采样同步消除 | `0.8 × (hostSyncUs / stepCount) / avgStepUs` | 同步开销移出关键路径 |
| 通信重叠/融合 | `0.6 × (commExposedUs / stepCount) / avgStepUs` | 未掩盖通信再藏起 60% |
| 通信 batch 摊薄 | `0.7 × (commUsPerStep / avgStepUs)` | 固定开销被摊薄 |
| D2H 异步化 | `0.7 × d2hPerStepUs / avgStepUs` | 拷贝与同步不再阻塞 |
| 提高 Decode 并发 | `min(60%, (100 − deviceBusyPct) + 15%)` | 以"设备未饱和 + 权重读取受限"为前提 |

每条都写出 `basis`（推算依据，含具体数值）与 `assumption`（前提假设），并给出区间（下限为保守估计）。

### 6.2 经验区间（置信度 low）

量化加速比、chunk 调优、并行策略调整、算子融合收益等**无法从本数据集推出**的项目，明确标注为经验区间：

| 优化项 | 经验区间 | 说明 |
| --- | --- | --- |
| 量化（W8A8/FP8） | 20%–45% | 取决于模型结构与算子覆盖率 |
| 小算子融合 | 10%–30% | 取决于融合覆盖率 |
| chunked prefill 调优 | 5%–25% | 需结合业务序列长度分布实测 |
| 并行策略调整 | 5%–30% | 强依赖模型结构与卡数 |

报告中会写明："经验区间而非本数据集推导；请以实际测量为准"。

### 6.3 合并口径

```
conservative = Σ 各项区间下限
optimistic   = min(70%, Σ 各项估算)
```

并附说明：多项优化之间存在重叠与相互制约，**不可简单相加**；上限按 70% 截断。

---

## 七、可信度与边界

分析结果自带告警（页面与报告均展示）：

* 阶段划分为低置信度时说明原因（缺少 `Stage` 列、缺少 mstx 标记、步长分布无双峰特征）；
* trace 采样时说明采样策略、步长与保留数，并指出累计耗时来源已切换为 CANN 统计表；
* 缺少流水/算力利用率列时说明"算力受限 vs 访存受限"只能给条件性结论；
* 只识别到 1 个推理步时说明逐步指标可信度有限；
* 所有作用域都未达门限时说明"仅给出接近程度与缺失证据"。

### 如何提高结论可信度（按性价比排序）

1. **分开采集 Prefill / Decode**：这是消除阶段推断不确定性的唯一根治办法；
2. **开启 AI Core 指标（PipeUtilization）**：让"算力受限 / 访存受限 / 忙而不算"从条件性结论变成定论；
3. **补齐 `op_statistic.csv` / `operator_details.csv` / `step_trace_time.csv`**：让累计耗时与阶段指标有 CANN 侧交叉校验；
4. **多卡场景补充通信统计**（`communication.json` / `communication_matrix.json`）：确认通信分组、消息量与跨卡分布；
5. **在同负载下做变量对照**（并发、chunk 大小、并行策略各改一项）：把估算收益变成实测收益。

---

## 八、门限全部可覆盖

所有门限都可用插件配置覆盖，便于适配不同芯片与业务：

```yaml
- id: vllm-ascend-profiler
  name: dsh-plugin-vllm-ascend-profiler
  config:
    thresholds:
      hostOnlyGate: 35          # Host 独占门限（%）
      commShareGate: 10         # 通信占设备耗时门限（%）
      deviceBusyGate: 85        # NPU 忙碌率门限（%）
    peakTflops: 320             # 单卡峰值算力，用于把 FLOPs 换算成算力利用率
    topN: 30                    # 排行条数
    viewEventBudget: 200000     # 下发给浏览器的算子条预算
```

门限名字与默认值见 `lib/analysis/thresholds.js`（每个门限都带 `rationale` 字段，说明取值依据），页面「关于 / 健康检查」会回显当前生效的限制值。
