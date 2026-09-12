# HCCL (COMMUNICATION) & MEMCPY in Ascend NPU Profiling Output — Verified Field/Name Reference

Research for writing a faithful parser. Every claim carries a source URL and a confidence tag.
Confidence legend: **[DOC]** official Huawei/CANN/MindStudio doc · **[SRC]** vendor source code (msprof / mstt / msprof-analyze) ·
**[TEST]** real fixture inside vendor test code · **[COMM]** community guide/blog · **[UNCONFIRMED]** could not verify.

Versions where stated are noted inline; several enums changed between CANN 8.x and 9.x/26.0, and between the
`msprof` output tree (`mindstudio_profiler_output/`) and the torch_npu output tree (`*_ascend_pt/ASCEND_PROFILER_OUTPUT/`).

---

## 0. Two different output trees — do not mix columns

| Tree | Produced by | Key files |
|---|---|---|
| `PROF_xxx/mindstudio_profiler_output/` | `msprof` CLI | `msprof_*.json`, `op_summary_*.csv`, `op_statistic_*.csv`, `api_statistic_*.csv`, `task_time_*.csv`, `communication_statistic_*.csv`, `step_trace_*.csv` |
| `*_ascend_pt/ASCEND_PROFILER_OUTPUT/` | Ascend PyTorch Profiler / `torch_npu.profiler.analyse()` | `trace_view.json`, `kernel_details.csv`, `op_statistic.csv`, `api_statistic.csv`, `operator_details.csv`, `step_trace_time.csv`, `analysis.db`, `ascend_pytorch_profiler_{n}.db`, **`communication.json`**, **`communication_matrix.json`**, `analyse.done` |

- The `ASCEND_PROFILER_OUTPUT` file list is confirmed twice: by a vLLM-Ascend guide ([docs.vllm.com.cn service profiling guide](https://docs.vllm.com.cn/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html), uses `torch_npu.profiler.profiler.analyse`) **[DOC]**, and by the Ascend-Inference wiki ([ascend-profiling-analysis.md](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)) **[COMM]**. The wiki explicitly lists `communication.json` / `communication_matrix.json` under `ASCEND_PROFILER_OUTPUT`.
- MindSpore's `ASCEND_PROFILER_OUTPUT` additionally contains `communication_analyzer.db`, `hccs.csv`, `pcie.csv`, `nic.csv`, `roce.csv` ([MindSpore profiler.md, gitee](https://raw.giteeusercontent.com/mindspore/docs/raw/85b861084d7e17b1042bf40eea4074f47188d637/tutorials/source_zh_cn/debug/profiler.md)) **[DOC]**: "`communication.json` 文件记录通信类算子的通信耗时、带宽等详细信息" and "`communication_matrix.json` 文件记录通信小算子基本的信息，包含通信size、通信带宽、通信rank等信息". Both are generated only when multi-card/cluster **and** `profiler_level=Level1`/`Level2`.
- **`communication_statistic*.csv` does NOT exist in `ASCEND_PROFILER_OUTPUT`.** It is an `msprof`-tree file. In the torch tree the equivalent summary exists only inside `communication_analyzer.db` / `analysis.db`.

---

## A1. HCCL collective operator naming

### A1.1 Confirmed literal operator-name strings

| Literal | Where it appears | Source | Confidence |
|---|---|---|---|
| `hcom_allReduce__428_0_1` | `COMMUNICATION_OP.opName` in `msprof_{ts}.db` — given as the doc's own example | [msprof DB reference (EN)](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references_db.md), [ZH 26.0.0](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md) | **[DOC]** |
| `hcom_broadcast_` | `COMMUNICATION_OP.opType` — doc's own example | same | **[DOC]** |
| `hcom_allReduce__844_2_1@13681369207305868844`, `hcom_broadcast__844_1_1@13681369207305868844` | keys inside a real `communication.json` fixture | [msprof-analyze test_bandwidth_contention_advice.py](https://raw.gitcode.com/Ascend/msprof-analyze/raw/master/test/ut/advisor/communication_advice/test_bandwidth_contention_advice.py) | **[TEST]** |
| `hcom_send`, `hcom_receive` | `HCOM_SEND` / `HCOM_RECEIVE` constants used to classify p2p comm ops | [mstt constant.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_common/constant.py) | **[SRC]** |
| `Memcpy` | small-op event name inside an HCCL op whose `transport_type == SDMA`; `SDMA_TRANSIT_ITEMS = ["Memcpy", "Reduce_Inline"]` | [msprof str_constant.py](https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d76ab6ff3be8901962298196bf67cd61b/analysis/common_func/ms_constant/str_constant.py) and [communication_parser.py](https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d76ab6ff3be8901962298196bf67cd61b/analysis/msparser/cluster/communication_parser.py) | **[SRC]** |
| `Reduce_Inline` | same list as above | same | **[SRC]** |
| `RDMASend` | `RDMA_SEND`; second task of an RDMA triplet, followed by `Notify_Wait` | str_constant.py + meta_parser.py | **[SRC]** |
| `Reduce TBE` | `REDUCE_TBE` | str_constant.py | **[SRC]** |
| `RDMA_PAYLOAD_PREPARE`, `RDMA_SEND_PAYLOAD`, `RDMA_PAYLOAD_ACK` | `rdma_type` values | str_constant.py; parser tests `event.rdma_type == 'RDMA_SEND_PAYLOAD'` | **[SRC]** |
| `Notify_Wait` | `NOTIFY_WAIT` — the small-op name whose `duration` is folded into **Synchronization Time(ms)** and **Wait Time(ms)** | str_constant.py, communication_parser.py | **[SRC]** |
| `Notify_Record` | `NOTIFY_RECORD = "Notify_Record"` (note underscore, **not** a space) | [mstt constant.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_common/constant.py) | **[SRC]** |
| `MemcpyAsync` | `AYNC_MEMCPY = "MemcpyAsync"` (typo in identifier is upstream's); the `task_time` doc says the `Receive Time`/`Start Time`/`End Time`/`size(B)`/`bandwidth(GB/s)`/`operation` fields are "仅MemcopyAsync接口展示" | str_constant.py; [msprof ZH doc](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md) | **[SRC]** + **[DOC]** |
| `HcomAllReduce` | given as an example kernel `Name` value in `kernel_details.csv` | [cann-recipes-infer kernel_data_guide.md](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/kernel_data_guide.md) | **[COMM]** |
| `HcclAllGather...` | real trace op name shown in the MindStudio Insight detail panel for a comm op | [Ascend-Inference wiki](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md) (figure 28 caption) | **[COMM]** |
| `AllgatherMatmul`, `AllgatherMatmulAicpu`, `MatmulAllReduce`, `MatmulAllReduceAddRmsNormAicpu` | MC2 fused compute-comm operators; the comm-stream twin of the fused op is `<fused name>` + `Aicpu` | [msprof Profile Data File Reference (EN)](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references.md) | **[DOC]** |
| `AivKernel`, `AicpuKernel` | `AIV_KERNEL` / `AICPU_KERNEL` | str_constant.py | **[SRC]** |

**Naming grammar (partially confirmed).** Large-op identity is `<hcom_*>` (or `Hccl*`/`Hcom*` for torch-side kernels),
optionally suffixed `@<20-digit group/connection hash>` in `communication.json` keys. Group names look like
`10.170.22.98%enp67s0f5_60000_0_1708156014257149` (doc example) **[DOC]**. `algType` looks like `HD-MESH` **[DOC]**, with
component algorithms `MESH`/`RING`/`NB`/`HD`/`NHR`/`PIPELINE`/`PAIRWISE`/`STAR` **[DOC]**.

### A1.2 Refuted / unconfirmed for your candidate list

| Candidate | Verdict |
|---|---|
| `AllReduce`, `AllGather`, `ReduceScatter`, `AlltoAll`, `AlltoAllV`, `Broadcast`, `Reduce`, `Send`, `Recv` | **UNCONFIRMED as literal cell values.** Prose-only sightings: wiki prose uses "AllGather", "ReduceScatter", "alltoallv", "alltoall" (part of `MoeDistributeDispatchV2` = permute1+alltoallv+permute2); the `step_trace` doc uses `Reduce` as a **field name**. No source shows these as a `Name`/`opType`/`OP Type` cell. |
| `HcclAllReduce` | **UNCONFIRMED.** Only `HcclAllGather...` was observed (truncated in the figure caption). |
| `hcom_allReduce_` | **CONFIRMED** (doc example is `hcom_allReduce__428_0_1` — note the double underscore before the numeric fields). |
| `hcom_all_gather_` | **UNCONFIRMED** (only `hcom_broadcast_` and `hcom_allReduce_` verified). |
| `hccl_allreduce_` | **UNCONFIRMED / likely wrong** — the observed prefix pattern is `hcom_`. |
| `Notify Wait` (with a space) | **UNCONFIRMED as a literal.** The MindStudio best-practices doc says "等待(例如Notify Wait事件)" in prose, but the code constant is `Notify_Wait`. |
| `Wait`, `Barrier` | **UNCONFIRMED**, no source at all. |
| `FFTS` / `ffts_plus` | Partially: numeric `TASK_TYPE_MAPPING` has `"51": "ffts task"` and `"52": "ffts plus task"` (internal descriptions, not CSV values); `FFTS_PLUS` appears as a **Task Type** value (community guide). |
| `Memcpy` / `MemcpyAsync` | **CONFIRMED** but as *small-op* names, not collective names. See A3. |
| `aclrtMemcpy`, `aclrtMemcpyAsync` | **UNCONFIRMED as literal strings in any CSV.** `api_statistic.csv` has an `API Name` column at the `AscendCL` level, so these are plausible values, but I found no source showing the actual row. |
| `MEMCPY_H2D`, `MEMCPY_D2H`, `MEMCPY_D2D` | **NOT FOUND ANYWHERE — treat as refuted.** No doc, no source, no test. The real host↔device copy vocabulary is (a) `ENUM_MEMCPY_OPERATION` names `host to device` / `device to host` / `device to device` / … and (b) msPTI `copyKind` `HTOD`/`DTOH`/`DTOD`. |

---

## A2. `Task Type` / `TaskType` column

### A2.1 Where the column actually lives

- **`op_summary_*.csv`** (msprof tree) has a **`Task Type`** column. Official wording, CANN Commercial 8.5.0 EN:
  *"Task Type | Type of the accelerator that executes the task. Valid values include: `AI_CORE`, `AI_VECTOR_CORE`, `AI_CPU, CCU`, and `DPU`. If `task_time` is set to `l0`, this field is not collected and `N/A` is displayed."*
  ([msprof Profile Data File Reference EN](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references.md)) **[DOC]**
  Chinese 26.0.0 wording is looser: *"执行该Task的加速器类型，包含AI_CORE、AI_VECTOR_CORE、AI_CPU等"* ([ZH doc](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md)) **[DOC]**
- Same doc, op_summary notes: *"Operators with the `communication` task type usually consist of a sequence of communication tasks, each with an independent `Task ID` and `Stream ID`. Since these individual identifiers are not displayed here, the `Task ID` and `Stream ID` for this type of operator are marked as `N/A`."* and *"Communication operators do not have this state, so `N/A` is displayed"* for `OP State`. **[DOC]** — **important for your parser: expect literal `N/A` in `Task ID`/`Stream ID`/`OP State` on every communication row of `op_summary.csv`.**
- **`kernel_details.csv`** (torch tree) does **not** have a `Task Type` column in the versions I could inspect. Its column is **`Accelerator Core`** (newer) / **`Accelerator`** (older). See A2.4.
- The DB equivalent is `TASK.taskType` (INTEGER) and `COMPUTE_TASK_INFO.taskType` (INTEGER → `STRING_IDS(taskType)`) **[DOC]** — i.e. in the `.db` it is a **numeric** enum whose string is resolved through `STRING_IDS`.
- `task_time_*.csv` uses lowercase `kernel_type` with values `KERNEL_AICORE`, `KERNEL_AICPU`, … **[DOC]** — a *different* vocabulary; do not conflate.

### A2.2 Confirm/refute of your candidate Task Type enum values

| Candidate | Verdict | Source |
|---|---|---|
| `AI_CORE` | **CONFIRMED** | official op_summary Task Type **[DOC]**; also `kernel_details.csv` `Accelerator Core` **[TEST]** |
| `AI_CPU` | **CONFIRMED** | official op_summary **[DOC]**; blog op_type mapping (`Core = AI_CPU`) **[COMM]** |
| `AI_VECTOR_CORE` | **CONFIRMED** | official op_summary **[DOC]**; `kernel_details.csv` row `'AI_VECTOR_CORE'` **[TEST]** |
| `AI_VECTOR` | **UNCONFIRMED** as a Task Type (the confirmed spelling carries `_CORE`) | — |
| `MIX_AIC` | **CONFIRMED** | official note: "the **Task Type** will change from **AI_Core** to `MIX_AIC`" **[DOC]**; community guide **[COMM]** |
| `MIX_AIV` | **CONFIRMED** | community guide Task Type table **[COMM]** |
| `MIX_AICORE` | **UNCONFIRMED** | — |
| `HCCL` | **CONFIRMED as a Task Type value** | community guide Task Type table: `HCCL` = "Collective communication" **[COMM]** |
| `FFTS_PLUS` | **CONFIRMED as a Task Type value** | community guide Task Type table **[COMM]**; internal `TASK_TYPE_MAPPING["52"] = "ffts plus task"` **[SRC]** |
| `DVPP` | **CONFIRMED as a Task Type value** | community guide Task Type table **[COMM]**; `DVPP_DATA = "dvpp"` **[SRC]**; `DVPP_ENGINE_TYPE` map **[SRC]** |
| `CCU` | **CONFIRMED as a Task Type value** (official) | op_summary EN doc **[DOC]** |
| `DPU` | **CONFIRMED as a Task Type value** (official, Ascend 950) | op_summary EN doc **[DOC]** |
| `AIV`, `AIC` | **NOT Task Type values.** They are `coreType` values (`STRING_IDS(coreType)`, "AIC or AIV") in `SAMPLE_PMU_TIMELINE`/`SAMPLE_PMU_SUMMARY` **[DOC]**, and `StrConstant.AIC = "aic"` / `AIV = "aiv"` are raw msvp data-type keys **[SRC]** |
| `MEMCPY`, `MEMCPY_ASYNC`, `SDMA`, `PCIE_DMA`, `ROCE`, `UB`, `HMCCS`, `MC2`, `SYSTEM`, `Memset`, `Event`, `Barrier`, `RTS` | **ALL UNCONFIRMED as Task Type values.** Closest real evidence is the numeric `TASK_TYPE_MAPPING` in msprof, which maps device task-type ints to *descriptions*, not to these tokens: `"5": 'memory copy task'`, `"14": 'notify wait task'`, `"15": 'notify record task'`, `"16": 'HCCL rdma cpy task'`, `"17": 'L2 SDMA memory copy task'`, `"51": 'ffts task'`, `"52": 'ffts plus task'`, `"66": "AI vector task"`, `"2": 'event record task'`, `"50": "stars common task"`, `"68": "AICPU schedule task"` **[SRC]**. Note the *description* strings for 5 and 17 contain "memory copy" — but capitalised `MEMCPY`/`SDMA` as Task Type tokens were not observed in any file. |

### A2.3 `kernel_details.csv` `Accelerator Core` / `Accelerator` values

| Value | Source |
|---|---|
| `AI_CORE` | **[TEST]** msprof-analyze fixture row |
| `AI_VECTOR_CORE` | **[TEST]** msprof-analyze fixture row |
| `AI_CPU` | **[COMM]** blog op_type mapping |
| `MIX_AIC`, `MIX_AIV` | **[COMM]** blog + community guide |
| `COMMUNICATION` | **[COMM]** blog: "`communication` \| Core = COMMUNICATION 且 AIV = 0 \| 纯 HCCL 集合通信" and "`mix_comm_aiv` \| Core = COMMUNICATION 且 AIV > 0 \| 通信+AIV 融合（DispatchFFNCombine）". **This is the key signal: HCCL collective kernels surface in `kernel_details.csv` with `Accelerator Core = COMMUNICATION`.** |
| `MIXED_AIC` | **[COMM]** Ascend-Inference wiki (older CANN/torch_npu version) — note the conflict with `MIX_AIC`; version-dependent, handle both |

`Accelerator Core` is a real header in the torch_npu `kernel_details.csv` — confirmed by the msprof-analyze fixture
header and by a second community skill doc (`kernel-fields-lookup.md`, which also lists `Mix Block Dim`,
`aicore_time(us)`, `aic_*_ratio`, `aiv_*_ratio`, `cube_utilization(%)`) **[TEST]** **[COMM]**.

### A2.4 Full real `kernel_details.csv` header (best evidence)

From the msprof-analyze unit test, which physically writes this header and rows **[TEST]**
([source](https://raw.gitcode.com/Ascend/msprof-analyze/raw/master/test/ut/advisor/communication_advice/test_bandwidth_contention_advice.py)):

```
Step Id, Model ID, Task ID, Stream ID, Name, Type, Accelerator Core, Start Time(us), Duration(us), Wait Time(us),
Block Num, Mix Block Num, Input Shapes, Input Data Types, Input Formats, Output Shapes, Output Data Types,
Output Formats, Context ID, aicore_time(us), aic_total_cycles, aic_mac_ratio, aic_mac_int8_ratio, aic_cube_fops,
aic_vector_fops, aiv_time(us), aiv_total_cycles, aiv_vec_fp32_ratio, aiv_vec_fp16_ratio, aiv_vec_int32_ratio,
aiv_vec_misc_ratio, aiv_cube_fops, aiv_vector_fops
```

Real sample rows from that fixture (verbatim):

```
1, 4294967295, 1265, 16, 'MatMul56', 'MatMul', 'AI_CORE',        "172317\t", 21.2, 261.56, 9, 0, '4,1025','INT64','FORMAT_ND','4,1025','INT32','FORMAT_ND','N/A', 0,0,0,0,0,0, 1.77,29508,0,0,0.0062,0,0,5856
1, 4294967295, 1265, 16, 'Add2',    'Add',    'AI_VECTOR_CORE', "183317\t", 1.5,  261.56, 9, 0, ...
```

Notes: `Model ID = 4294967295` is the "not collected" sentinel (`INVALID_RANK_NUM` / `0xFFFFFFFF` is used broadly as an
invalid sentinel in this stack) **[SRC]**. `Step Id` is present only when a `schedule`/`step` was used — MindSpore doc:
"若用户前端调用了 `schedule` 进行 `step` 打点，则会增加 `Step Id` 字段" **[DOC]**. Header variants seen elsewhere add
`Device_id` before `Model ID` (**[COMM]** wiki) and use `OP State` (capital OP) or omit `Block Num`/`Mix Block Num` —
**your parser should be header-driven, not positional.**

Full authoritative column list (with units/meaning) for the current torch_npu `kernel_details.csv`:
[community kernel_data_guide.md](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/kernel_data_guide.md) **[COMM]**.
Note it lists `Wait Time(us)`, `Duration(us)`, `Start Time(us)` in newer versions vs the wiki's `Wait Time`,
`Duration`, `Start Time` (**[COMM]**) — again version-dependent.

---

## A3. How H2D / D2H / D2D copies appear

### A3.1 As device tasks in the timeline (`msprof_*.json` / `trace_view.json`)

The `task_time` section of the official msprof doc defines these timeline detail fields, valid **only for MemcpyAsync** **[DOC]**:

| Field | Meaning (verbatim ZH) |
|---|---|
| `Receive Time` | "Device收到内存拷贝Task的信息接收时间，单位us。仅MemcopyAsync接口展示。" |
| `Start Time` | "内存拷贝Task开始拷贝的时间，单位us。仅MemcopyAsync接口展示。" |
| `End Time` | "内存拷贝Task结束拷贝的时间，单位us。仅MemcopyAsync接口展示。" |
| `size(B)` | "拷贝的数据量，单位B。仅MemcopyAsync接口展示。" |
| `bandwidth(GB/s)` | "拷贝的带宽，单位GB/s。仅MemcopyAsync接口展示。" |
| `operation` | "拷贝类型，host to device或device to host等。仅MemcopyAsync接口展示。" |

So on the timeline, a device copy is an event named **`MemcpyAsync`** carrying an **`operation`** arg whose values are the
lowercase prose forms **`host to device` / `device to host` / …**.

### A3.2 Canonical copy-direction enum — `ENUM_MEMCPY_OPERATION` (numeric in DB, prose in CSV)

**[DOC]** ([EN](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references_db.md), [ZH](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md)):

```
0  host to host
1  host to device
2  device to host
3  device to device
4  managed memory
5  addr device to device
6  host to device ex
7  device to host ex
65535 other
```

Consumed by the `MEMCPY_INFO` table: `globalTaskId`, `size`, `memcpyOperation` (→ `STRING_IDS(memcpyOperation)`,
controlled by `--runtime-api`) **[DOC]**. **This is the most likely thing you were thinking of as "MEMCPY_H2D" —
but the literal token `MEMCPY_H2D` does not exist; the values are the lowercase strings above (or their integer ids in the `.db`).**

### A3.3 msPTI API-level copy direction

`MSPTI_ACTIVITY_KIND_MEMCPY` → `msptiActivityMemcpy{ kind, msptiActivityMemcpyKind copyKind, bytes, start, end, deviceId, streamId, correlationId, isAsync }`,
where `copyKind` is documented only in prose as *"Copy type (HTOD / DTOH / DTOD, and so on)"* **[DOC]**
([mspti activity_api.md](https://raw.githubusercontent.com/mindstudio-docs/master/master/mspti/docs/en/user_guide/activity_api.md)).
The exact `MSPTI_ACTIVITY_MEMCPY_KIND_*` literal spellings: **UNCONFIRMED** (docs page 404s).

### A3.4 Internal device task-type integers (msprof)

`TASK_TYPE_MAPPING` **[SRC]**: `"5": 'memory copy task'`, `"16": 'HCCL rdma cpy task'`,
`"17": 'L2 SDMA memory copy task'`, `"25": "HCCL rdma db cpy task"`, `"14": 'notify wait task'`,
`"15": 'notify record task'`, `"51": 'ffts task'`, `"52": 'ffts plus task'`, `"66": 'AI vector task'`,
`"50": "stars common task"`. These are the *device-side* numeric task types behind the `TASK.taskType` column of the `.db`.
They are descriptions, not the CSV `Task Type` strings.

### A3.5 Host-side API rows (`api_statistic.csv`)

Header (ZH 26.0.0 official) **[DOC]**:

```
Device_id, Level, API Name, Time(us), Count, Avg(us), Min(us), Max(us), Variance
```

- `Device_id`: "采集到的数据来源于Host侧时，显示值为host" — **literal `host`** for host-side rows.
- `Level` values: `AscendCL`, `Runtime`, `Node`, `Model`, `Communication` **[DOC]**; the underlying
  `ENUM_API_TYPE` ids are `20000 acl`, `15000 model`, `10000 node`, `5500 communication`, `5000 runtime`,
  `50001 op`, `50002 queue`, `50003 trace`, `50004 mstx` **[DOC]**. `msprof`'s own `LEVEL_MAP` is
  `{"acl": "AscendCL", "runtime": "Runtime", "model": "Model", "node": "Node"}` **[SRC]**.
- `Mode` (in the timeline view of the same data) can be `ACL_OP` / `ACL_MODEL` / `ACL_RTS` **[DOC]**.
- **`aclrtMemcpy` / `aclrtMemcpyAsync` as literal `API Name` values: UNCONFIRMED.** The column exists and the layer is
  right, but I found no source exhibiting those exact rows. Do not hard-code them; match on the actual `API Name` cells.

### A3.6 In `communication.json` — local copies appear as `Memcpy` with SDMA transport

Inside an HCCL op, host/device local copies are represented by small-op name **`Memcpy`** with
`transport_type == SDMA` and a `link_type` that selects the bandwidth bucket **[SRC]**:

```
get_communication_bandwidth_info_type():
  link_type == HCCS_SW                    -> "HCCS"
  link_type in [PCIE, HCCS, SIO]          -> that value
  else (RESERVED / INVALID_TYPE)          -> "SDMA"
```
`SDMA` totals are computed as the **sum** of the `PCIE + HCCS + SIO` buckets **[SRC]**.

---

## A4. HCCL artifacts — exact structure

### A4.1 `communication.json` — REAL sample (verbatim, from a vendor test fixture)

Source: [msprof-analyze test_bandwidth_contention_advice.py](https://raw.gitcode.com/Ascend/msprof-analyze/raw/master/test/ut/advisor/communication_advice/test_bandwidth_contention_advice.py) **[TEST]**. This is written verbatim as `communication.json` into `ASCEND_PROFILER_OUTPUT/`.

```json
{"step1": {"collective": {
  "hcom_broadcast__844_1_1@13681369207305868844": {
    "Communication Time Info": {
      "Start Timestamp(us)": 171317.0,
      "Elapse Time(ms)": 10.6086,
      "Transit Time(ms)": 0.00126,
      "Wait Time(ms)": 0.014939999999999998,
      "Synchronization Time(ms)": 0.00714,
      "Idle Time(ms)": 0.044660000000000005,
      "Wait Time Ratio": 0.9222,
      "Synchronization Time Ratio": 0.85
    },
    "Communication Bandwidth Info": {
      "RDMA": {"Transit Size(MB)": 0, "Transit Time(ms)": 0, "Bandwidth(GB/s)": 0, "Large Packet Ratio": 0, "Size Distribution": {}},
      "HCCS": {"Transit Size(MB)": 0.28575999999999997, "Transit Time(ms)": 1.8620000000000001,
               "Bandwidth(GB/s)": 13.3151, "Large Packet Ratio": 0.0,
               "Size Distribution": {"0.004224": [6, 0.00736], "0.003232": [1, 0.00126]}},
      "PCIE": {"Transit Size(MB)": 0, "Transit Time(ms)": 0, "Bandwidth(GB/s)": 0, "Large Packet Ratio": 0, "Size Distribution": {}},
      "SDMA": {"Transit Size(MB)": 0.28575999999999997, "Transit Time(ms)": 1.8620000000000001,
               "Bandwidth(GB/s)": 3.3151, "Large Packet Ratio": 0, "Size Distribution": {}},
      "SIO":  {"Transit Size(MB)": 0, "Transit Time(ms)": 0, "Bandwidth(GB/s)": 0, "Large Packet Ratio": 0, "Size Distribution": {}}
    }
  },
  "hcom_allReduce__844_2_1@13681369207305868844": { "Communication Time Info": {...}, "Communication Bandwidth Info": {...} }
}}}
```

**Structure (confirmed from the producer code too):**
- Top level: keyed by **step** — `"step1"`, `"step2"`, … (this sample uses `step1`).
- Second level: `"collective"` / `"p2p"` / `"total"` — `P2P = "p2p"`, `COLLECTIVE = "collective"`, `TOTAL = "total"` **[SRC]**.
  Note the `CommAnalyzerTime.type` DB column carries the same tri-state (`'collective'`/`'p2p'`/`'total'`) **[SRC]**.
- Per-op key = HCCL op name, optionally `@<hash>`.
- Per-op value has exactly two keys: **`"Communication Time Info"`** and **`"Communication Bandwidth Info"`**
  (`CommunicationParser.parse_ops` writes exactly these) **[SRC]**.
- `"Communication Time Info"` fields, verbatim string constants (`OpAnalysisType` + `Constant`) **[SRC]**:
  `Start Timestamp(us)`, `Elapse Time(ms)`, `Transit Time(ms)`, `Wait Time(ms)`, `Synchronization Time(ms)`,
  `Idle Time(ms)`, `Wait Time Ratio`, `Synchronization Time Ratio`.
- `"Communication Bandwidth Info"` is keyed by exactly **five** transport types —
  `TRANSIT_TYPE = ["RDMA", "HCCS", "PCIE", "SDMA", "SIO"]` **[SRC]** — each with
  `Transit Size(MB)`, `Transit Time(ms)`, `Bandwidth(GB/s)`, `Large Packet Ratio`, `Size Distribution`
  (`OpBandWidthType`) **[SRC]**.
- `Size Distribution` is a dict `{"<sizeMB>": [count, total_time_ms]}` **[SRC]** + sample above **[TEST]**.
- The aggregate bucket is named **`"Total HCCL Operators"`** (`StrConstant.TOTAL`), not `"total"`, as the op key **[SRC]**
  (the *step-level* `"total"` group key is separate).

**Derivation formulas (so your parser can validate) [SRC]:**
- `Idle Time(ms) = Elapse Time(ms) − Transit Time(ms) − Wait Time(ms)`
- `Wait Time Ratio = Wait Time / (Wait Time + Transit Time)`, rounded to 4 dp
- `Synchronization Time Ratio = Synchronization Time / (Synchronization Time + Transit Time)`, rounded to 4 dp
- `Bandwidth(GB/s) = Transit Size(MB)/1000 / (Transit Time(ms)/1000)` when `Transit Time != 0`, else 0 (rounded 4 dp)
- `Bandwidth(Utilization) = Bandwidth(GB/s) / standard_bandwidth[transport]` (chip-dependent), rounded 4 dp
- `Large Packet Ratio = large_packet_count / packet_count` where "large" ⟺ `size > MessageSizeThreshold[transport]`

**Field names you asked about that are NOT in `communication.json`:** `op_type`, `count`, `group_name`, `group_id`,
`rank`, `src_rank`, `dst_rank`, `transport_type`, `bandwidth`, `transit_time`, `wait_time` — these snake_case names
belong to the **DB** (`COMMUNICATION_OP` / `COMMUNICATION_TASK_INFO`) and to `communication_matrix.json`, **not** to
`communication.json`'s human-readable form. There is no `group_id` in either artifact; group identity is the
`groupName` string. `src_rank`/`dst_rank`/`transport_type`/`transit_time` (snake_case) are **UNCONFIRMED** for
`communication.json`.

### A4.2 `communication_matrix.json` — structure from the producer

Source: [communication_matrix_parser.py](https://raw.githubusercontent.com/kali20gakki/msprof/master/analysis/msparser/cluster/communication_matrix_parser.py) **[SRC]**.
It is a **JSON array**, one entry per HCCL op plus a trailing total entry:

```json
[
  {"op_name": "<hccl op name>",
   "link_info": [
     {"Src Rank": "<local rank>", "Dst Rank": "<remote rank>",
      "Transport Type": <int: 0 HCCS | 1 PCIE | 2 RDMA | 3 LOCAL | 4 SIO>,
      "Transit Size(MB)": <float>,
      "Transit Time(ms)": <float>,
      "Bandwidth(GB/s)": <float>,
      "Bandwidth(Utilization)": <float>,
      "Large Packet Ratio": <float>}
   ]},
  {"op_name": "Total HCCL Operators", "link_info": [ ... ]}
]
```

- Top-level keys: exactly **`op_name`** and **`link_info`** (`StrConstant.OP_NAME = "op_name"`, `LINK_INFO = "link_info"`) **[SRC]**.
- Row keys come from `CommunicationMatrixInfo` **[SRC]**: `Src Rank`, `Dst Rank`, `Transport Type`, `Transit Size(MB)`,
  `Transit Time(ms)`, `Bandwidth(GB/s)`, `Bandwidth(Utilization)`, `Large Packet Ratio`.
- Rank pair key internally is `"{local_rank}-{remote_rank}"`; if `remote_rank == 0xffffffff` the remote is replaced by
  `local_rank` (local on-chip op) **[SRC]**.
- `Transport Type` is the **integer** `TransportType` enum: `HCCS=0, PCIE=1, RDMA=2, LOCAL=3, SIO=4` **[SRC]**.
  Mapping from link type: `HCCS`/`HCCS_SW`→0, `PCIE`→1, `RDMA`→2, `LOCAL`→3, `SIO`→4, else `-1`. For matrix rows
  specifically: `ON_CHIP → LOCAL`, `HCCS_SW → HCCS`, otherwise the raw `link_type` **[SRC]**.
- **No `group_id`, `count`, or `group_name`** at the top level — the matrix is per-op, per-rank-pair **[SRC]**.
  An empty `link_info` list is possible.
- Cluster-level variant file `cluster_communication_matrix.json` exists **[SRC]**.

### A4.3 `communication_statistic_*.csv` (msprof tree) — header confirmed

**It does exist.** Official **[DOC]** ([EN](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references.md), [ZH](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md)):

```
Device_id, OP Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)
```

- `OP Type` = "Type of the collective communication operator" / "集合通信算子类型" — so **yes, it aggregates by op type**.
- `Ratio(%)` = that op type's total ÷ overall collective-communication total.
- Only produced in multi-rank/multi-server/cluster scenarios, from `--task-time` + `--hccl`.
- There is also a doc page titled `hccl_statistic（集合通信算子统计信息）`
  ([atlasprofiling_16_0066.html](https://www.hiascend.com/document/detail/zh/mindstudio/700/T&ITools/Profiling/atlasprofiling_16_0066.html)) that I could **not** fetch (JS-rendered) — **its exact header is UNCONFIRMED.** In mstt the recipe named `hccl_sum` reads `COMMUNICATION_OP` and emits `OpName`, `OpType`, `Duration`, `GroupName` **[SRC]** ([hccl_sum_export.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/hccl_sum_export.py)) — this is the closest thing to an `hccl_statistic` and it is **not** the same shape as `communication_statistic_*.csv`.

### A4.4 `communication_analyzer.db` tables (torch/MindSpore tree) — real column names

**[SRC]** ([communicaion_info_export.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/communicaion_info_export.py)):

- `CommAnalyzerTime`: `hccl_op_name`, `group_name`, `start_timestamp`, `elapse_time`, `step`, `type`, `rank_id`
- `CommAnalyzerBandwidth`: `hccl_op_name`, `transport_type`, `transit_time`, `transit_size`, `bandwidth`, `large_packet_ratio`
  (queried with `transport_type IN ('SDMA','RDMA')`)
- `ClusterCommunicationTime`: `hccl_op_name`, `group_name`, `start_timestamp`, `elapsed_time`, `step`, `rank_id` (note **`elapsed_time`** vs `elapse_time`)
- `ClusterCommunicationBandwidth`: `hccl_op_name`, `band_type`, `transit_time`, `transit_size`, `bandwidth`, `large_packet_ratio`
- `CommunicationGroupMapping` / `CommunicationGroup`: `group_name`, `rank_set`
- `ClusterStepTraceTime`

Also in `Constant`: `DB_COMMUNICATION_ANALYZER = "communication_analyzer.db"`, `DB_CLUSTER_COMMUNICATION_ANALYZER = "cluster_analysis.db"`,
`DB_MS_COMMUNICATION_ANALYZER = "communication_analyzer.db"`; output files `communication_group.json`,
`cluster_communication.json`, `cluster_communication_matrix.json` **[SRC]**.

### A4.5 `comm_op_*` key names used by mstt when reading these files

`comm_op_type` (`COMM_OP_TYPE`), `comm_op_name`, `comm_op_info`, `Total Op Info`, `type`, `data_type`,
`Transport Type`, `group_name`, `rank_id`, `step_id` **[SRC]** ([mstt constant.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_common/constant.py)).

### A4.6 `ENUM_HCCL_*` — numeric enums (`.db` side)

**[DOC]** ([EN](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references_db.md), [ZH 26.0.0](https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md) — the ZH release lists *fewer* members, so version-gate these):

```
ENUM_HCCL_LINK_TYPE      0 ON_CHIP | 1 HCCS | 2 PCIE | 3 ROCE | 4 SIO | 5 HCCS_SW | 6 STANDARD_ROCE
                         (EN 8.5+ adds 7 UB, 8 UBoE) | 255 RESERVED | 65534 N/A | 65535 INVALID_TYPE
ENUM_HCCL_TRANSPORT_TYPE 0 SDMA | 1 RDMA | 2 LOCAL | (EN 8.5+ adds 3 UB, 4 ROCE)
                         | 255 RESERVED | 65534 N/A | 65535 INVALID_TYPE
ENUM_HCCL_RDMA_TYPE      0 RDMA_SEND_NOTIFY | 1 RDMA_SEND_PAYLOAD | 255 RESERVED | 65534 N/A | 65535 INVALID_TYPE
ENUM_HCCL_DATA_TYPE      0 INT8 | 1 INT16 | 2 INT32 | 3 FP16 | 4 FP32 | 5 INT64 | 6 UINT64 | 7 UINT8
                         | 8 UINT16 | 9 UINT32 | 10 FP64 | 11 BFP16 | 12 INT128
                         (EN 8.5+ adds 14 HIF8, 15 FP8E4M3, 16 FP8E5M2, 17 FP8E8M0) | 255 RESERVED | 65534 N/A | 65535 INVALID_TYPE
ENUM_OVERLAP_ANALYSIS_TYPE 0 COMPUTE | 1 COMMUNICATION | 2 COMM_NOT_OVERLAP_COMP | 3 FREE | 65535 RESERVE
ENUM_API_TYPE            20000 acl | 15000 model | 10000 node | 5500 communication | 5000 runtime
                         | 50001 op | 50002 queue | 50003 trace | 50004 mstx
ENUM_MODULE              3 HCCL | 7 RUNTIME | 28 HCCP | 29 ROCE | 62 FFTS | 6 DVPP | 36 AICPU | 56 AIVECTOR | 48 ASCENDCL | ...
```

### A4.7 DB tables carrying communication/memcpy data (exact columns)

**`COMMUNICATION_OP`** (large ops; `--task-time` + `--hccl`) **[DOC]**: `opName`, `startNs`, `endNs`, `connectionId`,
`groupName`, `opId`, `relay`, `retry`, `dataType`, `algType`, `count`, `opType`, `deviceId`.
Doc examples: `opName = "hcom_allReduce__428_0_1"`, `opType = "hcom_broadcast_"`,
`groupName = "10.170.22.98%enp67s0f5_60000_0_1708156014257149"`, `algType = "HD-MESH"`.

**`COMMUNICATION_TASK_INFO`** (small ops) **[DOC]**: `timestampNs`, `name`, `globalTaskId`, `taskType`, `planeId`,
`groupName`, `notifyId`, `rdmaType`, `srcRank`, `dstRank`, `transportType`, `size`, `dataType`, `linkType`, `opId`,
`isMaster` (0 = secondary stream, 1 = primary stream), `bandwidth` (Byte/s).

**`COMMUNICATION_SCHEDULE_TASK_INFO`** (AICPU comm ops only) **[DOC]**: `name`, `globalTaskId`, `taskType`, `opType`.

**`MEMCPY_INFO`** (`--runtime-api`) **[DOC]**: `globalTaskId`, `size`, `memcpyOperation`.

**`CANNN_API` / `CANN_API`** (`--ascendcl`) **[DOC]**: `startNs`, `endNs`, `type`, `globalTid` (high 32 = PID, low 32 = TID),
`connectionId`, `name`. `connectionId` is the join key between `TASK`, `CANNN_API` and `COMMUNICATION_OP`.

**`TASK`** **[DOC]**: `startNs`, `endNs`, `deviceId`, `connectionId`, `globalTaskId`, `globalPid`, `taskType`,
`contextId`, `streamId`, `taskId`, `modelId`. `contextId` = "用于区分子图小算子，常见于MIX算子和FFTS+任务".

**`OVERLAP_ANALYSIS`** **[DOC]**: `id`, `deviceId`, `startNs`, `endNs`, `type` (→ `ENUM_OVERLAP_ANALYSIS_TYPE`).

---

## A5. Communication / HCCL category events in `trace_view.json` and `msprof_*.json`

### A5.1 What is confirmed

- The lane/track is named **`Communication`**, "旧称HCCL泳道" (formerly the HCCL lane), and it *"记录NPU层通信事件，与Ascend Hardware的通信子泳道一一对应，此处由HCCL等组件上报"* — i.e. it mirrors the communication sub-lanes of `Ascend Hardware` and is reported by HCCL etc. **[DOC]** ([MindStudio msinsight Timeline泳道介绍](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/best_practices/Timeline_Common_Lanes_and_Interface.md))
- Internal bar/track constant names used by the mstt parser: `NPU_BAR = "Ascend Hardware"`, `COMM_BAR = "Communication"`, `OVERLAP_BAR = "Overlap Analysis"` **[SRC]**.
- Overlap Analysis row/arg names: `Computing`, `Free`, `Communication(Not Overlapped)`, plus `Communication` **[SRC]** + **[DOC]**.
- Trace flow `cat` values (flow-event categories in `msprof_*.json`): `async_npu`, `MsTx`, `async_task_queue`,
  `HostToDevice`, `fwdbwd` **[DOC]**. Note the doc spells the mstx flow **`MsTx`**, while the source constant is
  `MSTX = "MsTx"` **[SRC]**.
- **`args` of a communication event** — the official field table for the `Communication` track **[DOC]**:
  - common: `Group * Communication` (communication-group name, as reported), `Plane ID`, `Title`, `Start`,
    `Wall Duration`, `Self Time`, `model id`
  - large-operator info: `connection_id`, `model id`, `data_type`, `alg_type`, `count`, `relay` (`yes`/`no`), `retry` (`yes`/`no`)
  - small-operator info: `notify id` (invalid ⇒ **`18446744073709551615`**), `duration estimated(us)`, `stream id`,
    `task id`, **`task type`**, `src rank`, `dst rank` (local on-chip ⇒ **`4294967295`**; on Ascend 950 it equals `src rank`),
    **`transport type`** (`LOCAL`/`SDMA`/`RDMA`, EN 8.5 adds `UB`/`RoCE`), `size(Byte)`, `data type`,
    **`link type`** (`HCCS`/`PCIE`/`RoCE`, EN 8.5 adds `UBoE`/`SIO`/`HCCS_SW`/`STANDARD_ROCE`/`UB`/`ON_CHIP`),
    `bandwidth(GB/s)`, `model id`
- The `msptiActivityHccl` C struct is the canonical source of the HCCL trace payload: `{kind, start, end, ds{deviceId, streamId}, double bandWidth, const char *name, const char *commName}` — `name` = comm op name, `commName` = **communication group name** **[DOC]** ([mspti activity_api.md](https://raw.githubusercontent.com/mindstudio-docs/master/master/mspti/docs/en/user_guide/activity_api.md)).
- `msptiActivityCommunication` adds `dataType`, `count`, `algType`, `correlationId` **[DOC]**.
- MC² note: *"The communication part of the timeline displays only level-0 data."* **[DOC]**

### A5.2 What is NOT confirmed

- **The literal `cat` string for HCCL events is UNCONFIRMED.** I could not fetch any doc or source that shows
  `"cat": "hccl"` (or any other value) on a communication trace event. `mstt` has an internal event-type constant
  `HCCL_EVENT = "hccl_event"` **[SRC]** but that is mstt's own key, not necessarily the trace `cat`.
  The `cat` values that *are* confirmed are the five flow categories listed above (those are flow events, not slices).
  **Recommendation for your parser: key off `name`/`args` (group name, `task type`, `transport type`, `link type`,
  `bandwidth(GB/s)`, `notify id`, `planeId`) and the containing track name, and treat `cat` as opaque.**
- `group name` / `rank` / `connection_id` / `transport_type` / `bandwidth` / `notify id` as they appear in the actual
  `args` JSON: the **field names** are confirmed by the official table, but the exact JSON key spelling and nesting
  (e.g. whether it is `group name` with a space, or `groupName`, or `Group * Communication`) is **UNCONFIRMED** —
  the doc table is prose describing the UI detail panel, and the UI renames things.
  For the **DB** the key is definitively `groupName` **[DOC]/[SRC]**; `planeId` in the DB is `Plane ID` in the UI.
- **`Barrier` / `FFTS` slices in the config/`RTS` layer**: **UNCONFIRMED** (no literal found). `Mstx`, `HostToDevice`,
  `async_npu`, `async_task_queue`, `fwdbwd` are the confirmed flow names.

### A5.3 `mstx` and communication

- DB table **`MSTX_EVENTS`** (host-side tx data; device-side aggregates into `TASK`), columns **[DOC]**:
  `startNs`, `endNs`, `eventType`, `rangeId`, `category`, `message`, `globalTid`, `endGlobalTid`, `domainId`, `connectionId`.
- **`ENUM_MSTX_EVENT_TYPE`** **[DOC]**: `0 marker | 1 push/pop | 2 start/end | 3 marker_ex`.
- `connectionId` "maps to `TASK(connectionId)`" **[DOC]**.
- **This is how communication MSTX ranges carry op metadata:** mstt's `Mstx2CommopExport` joins
  `MSTX_EVENTS.connectionId = TASK.connectionId` and filters `STRING_IDS.value LIKE '%"streamId":%' AND '%"count":%'
  AND '%"dataType":%' AND '%"groupName":%' AND '%"opName":%'` **[SRC]**
  ([mstx2commop_export.py](https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/mstx2commop_export.py)).
  ⟹ **the `mstx` `message` payload for a communication op is a JSON string containing at least
  `opName`, `groupName`, `count`, `dataType`, `streamId`.** (I did not find a file literally named `mstx.json`;
  `msprof_*.json` + the `MSTX_EVENTS` table + `MsTx` flow events are the carrier. A literal `mstx.json` /
  `mstx_*.json` artifact: **UNCONFIRMED**.)
- MindSpore adds built-in instrumentation: *"同时支持通信算子的内置打点，用户开启轻量化打点功能，通讯算子前后将自动实现打点"* — with `mstx=True`, communication operators are auto-instrumented **[DOC]** ([MindSpore profiler.md](https://raw.giteeusercontent.com/mindspore/docs/raw/85b861084d7e17b1042bf40eea4074f47188d637/tutorials/source_zh_cn/debug/profiler.md)).

---

## A6. Supporting facts worth encoding in the parser

- **Invalid sentinels.** `4294967295` (`0xFFFFFFFF`) is the "not collected / local on-chip" sentinel for rank and
  model-id-like fields **[DOC]** + `INVALID_RANK_NUM = 4294967295` **[SRC]**. `18446744073709551615` (`0xFFFFFFFFFFFFFFFF`)
  is the invalid `notify id` sentinel **[DOC]**. `N/A` is used freely in CSV cells.
- **Notify semantics.** Small-op `Notify_Wait` duration is what makes a comm op look slow: the *first* `Notify_Wait`
  in a large op contributes to `Synchronization Time(ms)`; *all* of them contribute to `Wait Time(ms)` **[SRC]**.
  MindStudio's own guidance: *"当出现某些卡存在时长较长的通信算子，且通信算子主要时长来源于等待(例如Notify Wait事件)时，优先考虑是否出现了快慢卡问题"* **[DOC]**.
- **Which stream to trust.** 5/6 of the analysis uses only the **primary** stream: `isMaster == 1` in the DB,
  `event.is_master == 1` in the source; primary selection is by `plane_id` of the master event **[SRC]**.
- **`communication.json` is only produced when there IS communication** and profiler level ≥ Level1; single-rank runs
  have no Communication track and no overlap comm data **[DOC]**.
- **Op-count/distribution semantics.** `Communication Time Info` accumulation: all keys sum except
  `Wait Time Ratio`, `Synchronization Time Ratio`, `Start Timestamp(us)` **[SRC]**.
- Naming of the CSV header for the same concept differs across trees: `op_summary.csv` → `Task Type`;
  `kernel_details.csv` → `Accelerator Core` (new) / `Accelerator` (old); `op_statistic.csv` → `Core Type`
  (**[COMM]** wiki); `task_time.csv` → `kernel_type` (`KERNEL_AICORE`/`KERNEL_AICPU`).
- `op_statistic.csv` header (torch tree, **[COMM]** wiki): `Device_id, OP Type, Core Type, Count, Total Time, Min Time, Avg Time, Max Time, Ratio(%)`.
  The msprof-tree `op_statistic_*.csv` has the same spirit but is documented as "AI Core和AI CPU算子调用次数及耗时统计".

---

## B. Real sample rows / fragments collected

1. **`communication.json`** — full verbatim fixture (A4.1) **[TEST]**.
2. **`kernel_details.csv`** — full header + 4 real rows including `AI_CORE` and `AI_VECTOR_CORE` (A2.4) **[TEST]**.
3. **`COMMUNICATION_OP` doc examples** — `hcom_allReduce__428_0_1`, `hcom_broadcast_`,
   `10.170.22.98%enp67s0f5_60000_0_1708156014257149`, `HD-MESH` **[DOC]**.
4. **MC² timeline phases** — `StartServer`, `TaskWaitRequest`, `TaskOrchestration`, `TaskLaunch`, `TaskExecute`, `Finalize` **[DOC]**.
5. **DPU track fields** (Ascend 950) — `Thread Id`, `Physic Stream Id`, `Task Id`, `OP Type`, `AI CPU Device Id`,
   `AI CPU Task Id`, `Plane Id`, `Notify Id`, `Duration Estimated(us)`, `Src Rank`, `Dst Rank`, `Transport Type`,
   `Size(Byte)`, `Bandwidth(GB/s)`, `Data Type`, `Link Type`, `Rdma Type` **[DOC]**.
6. Community `Task Type` table (the source of `HCCL`/`FFTS_PLUS`/`DVPP`/`MIX_AIV`): `AI_CORE`, `AI_CPU`, `HCCL`,
   `MIX_AIC`, `MIX_AIV`, `FFTS_PLUS`, `DVPP` **[COMM]**.

---

## C. Source index

- msprof Profile Data File Reference (EN): https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references.md
- msprof Profile Data File Reference (DB, EN): https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references_db.md
- msprof 性能数据文件参考 (ZH, 26.0.0, includes DB enums): https://raw.gitcode.com/Ascend/msprof/raw/br_release_MindStudio_26.0.0_O1_20270430/docs/zh/profile_data_file_references.md
- msprof `str_constant.py` (HCCL constants, matrix info, task-type map, api levels): https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d76ab6ff3be8901962298196bf67cd61b/analysis/common_func/ms_constant/str_constant.py
- msprof `communication_parser.py`: https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d76ab6ff3be8901962298196bf67cd61b/analysis/msparser/cluster/communication_parser.py
- msprof `communication_matrix_parser.py`: https://raw.githubusercontent.com/kali20gakki/msprof/master/analysis/msparser/cluster/communication_matrix_parser.py
- msprof `meta_parser.py` (`HcclAnalysisTool`): https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d76ab6ff3be8901962298196bf67cd61b/analysis/msparser/cluster/meta_parser.py
- mstt `constant.py`: https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_common/constant.py
- mstt `communicaion_info_export.py`: https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/communicaion_info_export.py
- mstt `hccl_sum_export.py`: https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/hccl_sum_export.py
- mstt `mstx2commop_export.py`: https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_exports/mstx2commop_export.py
- msprof-analyze communication.json + kernel_details.csv fixture: https://raw.gitcode.com/Ascend/msprof-analyze/raw/master/test/ut/advisor/communication_advice/test_bandwidth_contention_advice.py
- MindStudio Insight Communication design doc: https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Communication.md
- MindStudio Insight Timeline lanes (Communication = former HCCL lane): https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/best_practices/Timeline_Common_Lanes_and_Interface.md
- msPTI Activity API (HCCL/MEMCPY/COMMUNICATION records): https://raw.githubusercontent.com/mindstudio-docs/master/master/mspti/docs/en/user_guide/activity_api.md
- MindSpore profiler (ASCEND_PROFILER_OUTPUT inventory): https://raw.giteeusercontent.com/mindspore/docs/raw/85b861084d7e17b1042bf40eea4074f47188d637/tutorials/source_zh_cn/debug/profiler.md
- vLLM-Ascend service profiling guide: https://docs.vllm.com.cn/projects/ascend/en/latest/developer_guide/performance_and_debug/service_profiling_guide.html
- Ascend-Inference wiki profiling analysis: https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md
- Community `kernel_details.csv` field + Task Type guide: https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/kernel_data_guide.md
- Community kernel field lookup (multi-stream): https://raw.gitcode.com/cann/cannbot-skills/raw/master/model/model-infer-multi-stream/references/kernel-fields-lookup.md
- Blog with `Accelerator Core = COMMUNICATION`: https://pillumina.github.io/posts/aiinfra/ascend-profiling-analysis-skill/
