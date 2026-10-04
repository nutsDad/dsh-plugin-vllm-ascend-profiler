# 从 profiling 分析到算子开发：端到端交接

第 5 步的「执行」按钮把一条优化建议交给昇腾算子 skills 去执行。这份文档记录**这条链路
实际跑通的证据**、每一步的产物，以及必须到昇腾机器上才能完成的部分。

复现一条命令即可：

```powershell
node .tools/verify-e2e.mjs --scenario prefill-compute-bound --advice compute.quantize `
  --project D:\00_deepseekharness\ascend-kernel --operator rms_norm
```

（脚本在工作区 `D:\00_deepseekharness\.tools\`，用真实解析/分析/交接模块，不打桩。）

## 链路与实测结果

| 环节 | 结果 | 证据 |
| --- | --- | --- |
| ① profiling 解析 | PASS | `prefill-compute-bound` · 6742 事件 · 6 步 · 瓶颈 NPU 计算瓶颈 95/100 |
| ② 「执行」生成算子任务 | PASS | `compute.quantize` → 目标算子 `MatMulV2(52.5%)`、`FusedInferAttentionScore(32%)`、`RmsNorm(2.6%)` |
| ③ skills 可见性 | PASS | 16/16 已安装（`ascend-base` 7 + `ascend-profiling` 5 + `ascend-ops` 4）；链条 `ascendc → ascend-opplugin → npu-op-benchmark → profiling-analysis` |
| ④ 任务包落盘 | PASS | `.dsh-vap-tasks/<ts>-compute.quantize.md`（2850 B）+ 同名 `.json` |
| ⑤ 算子工程脚手架 | PASS | `ascend-kernel/csrc/ops/rms_norm/{op_host,op_kernel,design.md}`（模板来自 `ascendc` skill 的 `templates/ascend-kernel`） |
| ⑥ 设计文档门禁 | PASS | `design.md` 9/9 项齐全（签名、实现路径、伪代码、两级 tiling、UB 分配表、bufferCoefficient、升精度路径、workspace、验收标准），9135 字符 |
| ⑦ 编译 / 精度 / 性能 | **BLOCKED** | `ASCEND_HOME_PATH` 与 `CONDA_DEFAULT_ENV` 均为空，本机无 CANN / torch_npu / npu-smi → Phase 5/7/8 必须在昇腾机器执行 |

第 ⑦ 行是**环境门禁**，不是链路故障：`ascendc` skill 的 Phase 0 要求先确认 CANN 与 conda
环境，本机两样都没有，因此按 skill 的规定停在编写阶段（Phase 1–4 是作者工作，不需要硬件）。

## 为什么选 `RmsNorm` 而不是占比最高的 `MatMulV2`

任务包按"建议焦点"给出三个候选算子，落地时按 skill 的适用范围挑：

* `MatMulV2`（52.5% 设备耗时）是 **cube** 算子，属于 CATLASS / genop 路线，`ascendc` skill 明确不覆盖；
  它应该走量化配置与 aclnn 接入（`ascend-ops` 的另一条路径），不是新写一个 kernel；
* `RmsNorm` 是 **row/reduction 向量**算子 —— 正是该 skill 模板覆盖的类型，且它同样出现在这份
  capture 的命中列表里（192 次调用、单次 210 µs），所以作为自定义 kernel 的切入点是正确的。

这个取舍本身写进了 `design.md` 的 Provenance 段，避免"换了算子但不知道换了"。

## 产物

| 产物 | 位置 |
| --- | --- |
| 算子优化任务（人读） | `D:\00_deepseekharness\.dsh-vap-tasks\<ts>-compute.quantize.md` |
| 算子优化任务（结构化） | 同名 `.json`（含算子清单、skill 链、验收表、指标快照） |
| 设计文档 | `D:\00_deepseekharness\ascend-kernel\csrc\ops\rms_norm\design.md` |
| 工程脚手架 | `D:\00_deepseekharness\ascend-kernel\`（`build.sh`、`csrc/`、`python/`、`tests/`） |
| 页面截图 | `docs/screenshots/16-execute-handoff.png`（第 5 步 ④ 优化行动 · 执行后的任务面板） |

## 验收标准怎么回到插件

任务包里的验收表与第 6 步对比**同口径**（同一个指标键、同一个阈值规则：预期收益的一半，
下限 1%）：

| 指标 | 当前值 | 预期收益 | 判定阈值 | 方向 |
| --- | --- | --- | --- | --- |
| 设备计算耗时 `computeUs` | 1 349 191.78 µs | −30% | ≥ 15% | 越低越好 |

因此算子改完之后的闭环是：**用同一负载、同一阶段口径重新采集 → 第 6 步与本次数据集对比 →
「已达成 / 部分达成 / 未达成」直接给出**，不需要人工对表。

## 在昇腾机器上继续（Phase 5/7/8）

```bash
# 0) 环境
source ${CANN_PATH}/*/set_env.sh && conda activate <env>

# 1) 补齐算子实现（Phase 3/4）：测试用例 + op_host/op_kernel + 三处注册点
#    依据 design.md；模板见 ascendc skill 的 templates/code-gen/

# 2) 编译安装并跑功能/精度测试（Phase 5）
cd ascend-kernel && chmod +x build.sh && bash build.sh
pip install output/ascend_kernel*.whl --force-reinstall --no-deps
python tests/test_rms_norm.py && pytest -v

# 3) 单算子基准（Phase 8，双路对比）
python -m npu_op_benchmark --op rms_norm --shapes "2048x4096" --dtypes fp16,bf16 --warmup 5 --active 5

# 4) 端到端复测：重新采集 → 第 6 步对比 → 看 computeUs 是否越过 15% 阈值
```

## 已知边界

* 本机没有昇腾硬件，所以**没有任何实测性能数字**：`design.md` 里的 tiling 取值是按平台 API
  公式推导的示例，真实值必须由 `op_host` 在运行时查询设备得到（skill 明令禁止硬编码）。
* 任务包与页面显示的阈值来自**当前数据集**的预期收益（经验区间，置信度 low），不是实测承诺；
  `compute.quantize` 的收益依赖模型结构与量化算子覆盖率。
* skills 由上游仓库 `ascend-ai-coding/awesome-ascend-skills` 维护，本插件只做可见性探测与调用
  编排，不内置其内容；升级 skills 只需替换 `$DSH_HOME/skills` 下的目录。
