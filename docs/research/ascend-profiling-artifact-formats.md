# Ascend NPU / vLLM-Ascend Profiling Artifact Formats — Faithful-Parser Reference

**Scope.** Two distinct producers exist. Almost every ambiguity in a parser comes from conflating them.

| | **Class A — Ascend PyTorch Profiler** | **Class B — native msprof / CANN Profiling** |
|---|---|---|
| Producer | `torch_npu.profiler.profile(...)` + `tensorboard_trace_handler`; also MindSpore Profiler (`_ascend_ms`) | `msprof` CLI / AscendCL Profiling API |
| Top dir | `<worker>_<timestamp>_ascend_pt` (PyTorch) / `_ascend_ms` (MindSpore) | `PROF_<id>_<ts>_<hash>` |
| Kernel CSV | `kernel_details.csv` | `op_summary_*.csv` |
| Timeline | `trace_view.json` | `msprof_*.json` |
| CSV suffix | none (bare names) | `_<timestamp>` suffix |

The convention "`kernel_details.csv` / `trace_view.json` = torch_npu.profiler" vs "`op_summary_*.csv` / `msprof_*.json` = native msprof" is stated explicitly in [CANN's own `prof_layouts.md` reference](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/prof_layouts.md). That same source warns: **team wrappers commonly rename Class B files to Class A names** — decide by field content, not file name.

Primary sources used throughout:
- [Ascend PyTorch Profiler User Guide (official, `Ascend/pytorch` v2.7.1-26.1.0)](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)
- [msprof Profile Data File Reference (official, `Ascend/msprof`)](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)
- [msprof Profile Data File Reference (DB tables)](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references_db.md)
- [MindStudio Insight Timeline design doc](https://raw.githubusercontent.com/Ascend/msinsight/master/docs/zh/development_guide/design/Timeline.md)
- [vLLM-Ascend Service Profiling Guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)
- [MindSpore Profiler tutorial (full `_ascend_ms` tree)](https://raw.giteeusercontent.com/mindspore/docs/raw/master/tutorials/source_zh_cn/debug/profiler.md)

---

## 1. vLLM-Ascend profiling workflow, env vars, and directory layout

### 1.1 Enabling collection

**Current (mainline) form — `--profiler-config`.** The `VLLM_TORCH_PROFILER_DIR` environment variable is **deprecated in the vLLM mainline as of January 19, 2026**. vLLM-Ascend's own guide instructs using `--profiler-config` (online) or the `profiler_config` argument (offline) instead ([vLLM-Ascend Service Profiling Guide](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md), referencing vllm-ascend PR #5928):

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --port 8080 --model "facebook/opt-125m" --tensor-parallel-size 1 \
  --max-num-seqs 128 \
  --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile", "torch_profiler_with_stack": false}' \
  --dtype bfloat16 --max-model-len 256
```

JSON keys confirmed in that guide: `profiler`, `torch_profiler_dir`, `torch_profiler_with_stack`. The object is validated by upstream [`vllm/config/profiler.py::ProfilerConfig`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/profiler.py). Complete accepted key set: `profiler` (`Literal["torch","cuda","proton"] | None`), `torch_profiler_dir`, `proton_profiler_dir`, `proton_context`, `proton_data`, `proton_backend`, `proton_mode`, `proton_hook`, `proton_output_format`, `torch_profiler_with_stack` (default **True**), `torch_profiler_with_flops` (False), `torch_profiler_use_gzip` (True), `torch_profiler_dump_cuda_time_total` (True), `torch_profiler_record_shapes` (False), `torch_profiler_with_memory` (False), `capture_torch_profiler`, `detailed_trace_annotation`, `ignore_frontend`, `delay_iterations`, `max_iterations`, `warmup_iterations`, `active_iterations` (5), `wait_iterations`. There is **no** `torch_profiler_with_modules` and **no** `torch_profiler_with_shapes` key. `torch_profiler_dir` is normalised via `os.path.abspath(os.path.expanduser(...))` unless it is a URI. Dot-form also works: `--profiler-config.profiler torch --profiler-config.torch_profiler_dir DIR`.

**⚠️ vllm-ascend consumes only four of those keys**: `profiler` (must be `"torch"`, else `RuntimeError(f"Unrecognized profiler: {profiler_config.profiler}")`), `torch_profiler_dir` (must be non-empty), `torch_profiler_with_memory` → `profile_memory`, and `torch_profiler_with_stack` → **`with_modules`** (not `with_stack`). `torch_profiler_with_flops`, `torch_profiler_record_shapes`, `use_gzip`, `dump_cuda_time_total`, and the `*_iterations`/`ignore_frontend`/`detailed_trace_annotation` fields are **accepted but ignored on Ascend**.

**Exact `torch_npu.profiler.profile(...)` call on vllm-ascend main** ([`vllm_ascend/profiler/torch_npu_profiler.py`](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/profiler/torch_npu_profiler.py)):
```python
experimental_config = torch_npu.profiler._ExperimentalConfig(
    export_type=torch_npu.profiler.ExportType.Text,
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    msprof_tx=False,
    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
    l2_cache=False,
    op_attr=False,
    data_simplification=True,
    record_op_args=False,
    gc_detect_threshold=None,
)
return torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU],
    with_stack=False,
    profile_memory=profiler_config.torch_profiler_with_memory,
    # NOTE: torch_npu.profiler.with_modules is equivalent to torch.profiler.with_stack.
    with_modules=profiler_config.torch_profiler_with_stack,
    experimental_config=experimental_config,
    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
        profiler_config.torch_profiler_dir, worker_name=trace_name),
)
```

Parser-relevant consequences:
- **`profiler_level=Level1` ⇒ `op_statistic.csv` and `api_statistic.csv` ARE produced** for vllm-ascend runs (they are Level1/Level2-gated in the general rules).
- **`aic_metrics=PipeUtilization` ⇒ the `*_vec_ratio` / `*_mac_ratio` / `*_mte*_ratio` / `*_icache_miss_rate` / `memory_bound` / `cube_utilization(%)` columns ARE appended to `kernel_details.csv`** in vllm-ascend captures.
- **`export_type=Text`** ⇒ `analysis.db` / `ascend_pytorch_profiler_*.db` only appear if the user configured Db export or re-analysed with `export_type="db"`.
- **`msprof_tx=False` and `mstx` never set ⇒ mstx is OFF** in the default vllm-ascend path, so no mstx points unless the user re-analyses.
- **No `schedule=` is ever passed on Ascend** ⇒ **no `ProfilerStep#N` markers, no wait/warmup/active/repeat semantics**, and `activ`/`delay`/`max_iterations` are inert. It also means **`Step Id` in `kernel_details.csv` will not be present** for a default vllm-ascend capture (the column is added only when a step range was derived from schedule). Corroborated by the guide's example output list, which shows no `schedule` argument.
- `_stop()` is `self.profiler.stop()` with no `.key_averages()` dump ⇒ **no `profiler_out_<rank>.txt`** (that file is a CUDA-only artifact).
- `_profiler_step()` returns `True` without calling `profiler.step()`.
- **Version drift:** vllm-ascend v0.9.1 used `aic_metrics=AiCMetrics.AiCoreNone`, `data_simplification=False`, `with_stack=False`, `with_modules=False`, `profile_memory=False`, and `tensorboard_trace_handler(dir)` **without `worker_name`**. Current main uses `data_simplification=True`. **`data_simplification=True` deletes `mindstudio_profiler_output/`, `mindstudio_profiler_log/` and `analyze/` under `PROF_*`, and always removes `sqlite` + each `device_*`/`host`'s `summary` and `timeline`** — so a vllm-ascend capture may legitimately have **no** `mindstudio_profiler_output/` at all while an older one does. A parser must not rely on either.

> ⚠️ **Spelling:** it is `torch_npu.profiler.AiCMetrics` (capital A, lowercase i, capital C) — **not** `AICMetrics`. Members: `AiCoreNone, PipeUtilization, ArithmeticUtilization, Memory, MemoryL0, MemoryUB, ResourceConflictRatio, L2Cache, MemoryAccess`, backed by strings `ACL_AICORE_NONE`, `ACL_AICORE_PIPE_UTILIZATION`, `ACL_AICORE_ARITHMETIC_UTILIZATION`, `ACL_AICORE_MEMORY_BANDWIDTH`, `ACL_AICORE_L0B_AND_WIDTH`, `ACL_AICORE_MEMORY_UB`, `ACL_AICORE_RESOURCE_CONFLICT_RATIO`, `ACL_AICORE_L2_CACHE`, `ACL_AICORE_MEMORY_ACCESS` ([`torch_npu/profiler/experimental_config.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/experimental_config.py)). At `Level0`/`Level_none`, `_check_params` warns *"Please use level1 or level2 if you want to collect aic metrics, reset aic metrics to None!"* and forces `AiCoreNone` ⇒ **no PMU columns at Level0**.

**Historical form — env vars (now dead).** `vllm_ascend/envs.py` on main contains **no profiler env vars at all**; the variables came from upstream `vllm/envs.py` and were consumed by vllm-ascend until PR #5928. The authoritative upstream list ([vLLM v0.12.0 profiling doc](https://raw.githubusercontent.com/vllm-project/vllm/v0.12.0/docs/contributing/profiling.md)):

| Env var | Default | Real? |
|---|---|---|
| `VLLM_TORCH_PROFILER_DIR` | — | ✅ real; consumed by vllm-ascend pre-#5928 (`if envs.VLLM_TORCH_PROFILER_DIR:` … `torch_profiler_trace_dir = envs.VLLM_TORCH_PROFILER_DIR`) |
| `VLLM_TORCH_PROFILER_WITH_STACK` | **on** | ✅ real; passed through as `with_stack=envs_vllm.VLLM_TORCH_PROFILER_WITH_STACK` |
| `VLLM_TORCH_PROFILER_RECORD_SHAPES` | off | ✅ real |
| `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` | off | ✅ real; named as dead in the [vllm-ascend v0.19.1rc1 release notes](https://raw.githubusercontent.com/vllm-project/vllm-ascend/a5ec26693eec9af9a92b89a18e7111507819e45b/docs/source/user_guide/release_notes.md): *"The profiler envs, such as `VLLM_TORCH_PROFILER_DIR` and `VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY` do not work with vLLM Ascend now. Please use vLLM `--profiler-config` parameters instead. [#5928]"* |
| `VLLM_TORCH_PROFILER_WITH_FLOPS` | off | ✅ real **upstream-only**; never referenced in vllm-ascend source ⇒ **you cannot get FLOPs on Ascend via env var** |
| `VLLM_TORCH_PROFILER_USE_GZIP` | on | ✅ real (upstream) |
| `VLLM_TORCH_PROFILER_DUMP_CUDA_TIME_TOTAL` | on | ✅ real (upstream) |
| `VLLM_TORCH_PROFILER_WITH_MODULES` | — | ❌ **not found anywhere** — treat as non-existent |
| `VLLM_TORCH_PROFILER_WITH_SHAPES` | — | ❌ **not found**; the real name is `..._RECORD_SHAPES` |

vllm-ascend's guide notes *"vLLM enables **python stack** by default, which can significantly inflate the collected performance data. If you do not wish to collect python stack, you can disable it using `torch_profiler_with_stack=false`"* — stack defaults **on** ([source](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)).

**vllm-ascend-only profiler-related env var: `MSMONITOR_USE_DAEMON`** — *"MSMONITOR_USE_DAEMON and torch profiler cannot be both enabled at the same time."* ([PR #5928 diff](https://patch-diff.githubusercontent.com/raw/vllm-project/vllm-ascend/pull/5928.diff)). On current main this is `additional_config.msmonitor_use_daemon`, raising *"additional_config.msmonitor_use_daemon and torch profiler cannot be enabled at the same time."*

**Endpoints.** `POST /start_profile` and `POST /stop_profile`. Critically: **these routes are only registered when the server is launched with `--profiler-config` set (non-empty `profiler` field). If you forget it, the endpoints do not exist and `curl` returns 404 Not Found** ([source](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)).

**No body, no query params, no response payload.** Registration logic ([vLLM main `vllm/entrypoints/serve/profile/api_router.py`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/serve/profile/api_router.py)):
```python
@router.post("/start_profile")
async def start_profile(raw_request: Request): ...
@router.post("/stop_profile")
async def stop_profile(raw_request: Request): ...

def attach_router(app: FastAPI):
    profiler_config = getattr(app.state.args, "profiler_config", None)
    assert profiler_config is None or isinstance(profiler_config, ProfilerConfig)
    if profiler_config is not None and profiler_config.profiler is not None:
        logger.warning_once("Profiler with mode '%s' is enabled in the API server. This should ONLY be used for local development!" % ...)
        app.include_router(router)
```
In the env-var era the condition was `if envs.VLLM_TORCH_PROFILER_DIR:`. **There is no `ProfileRequest` pydantic model and no `profile_prefix` HTTP field** in v0.6.6, v0.8.5, v0.9.1 or main — `profile_prefix` exists only as a **Python** argument (`NPUWorker.profile(is_start, profile_prefix)`, `EngineClient.start_profile(profile_prefix)`).

**PD-disaggregated deployments.** Prefiller and Decoder are separate vLLM instances; each must be launched with its own `--profiler-config` pointing at a **different directory**, and the main PD proxy (`load_balance_proxy_server_example.py`) does **not** forward `/start_profile` or `/stop_profile` — you must curl each node directly. The EPD proxy (`epd_load_balance_proxy_layerwise_server_example.py`) **does** broadcast to all E/P/D instances ([source](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)).

### 1.2 Parsing step

**This is the most commonly missed step:** the `*_ascend_pt` folder is **raw** and must be analysed before its data is meaningful.

```python
from torch_npu.profiler.profiler import analyse
analyse("./vllm_profile/localhost.localdomain_*_ascend_pt/")
```

vLLM-Ascend's guide phrases this as: *"locate the generated `*ascend_pt` folder. This folder needs to be analyzed before profiling data can be examined."* ([source](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)).

### 1.3 Class A directory layout (source-verbatim + official doc)

The top-level name comes from [`torch_npu/profiler/_profiler_path_creator.py::ProfPathCreator.create_prof_dir`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/_profiler_path_creator.py):
```python
worker_name = "{}_{}".format(self._worker_name or socket.gethostname(), str(os.getpid()))
current_time = datetime.now(tz=timezone.utc).astimezone()
span_name = "{}_{}_ascend_pt".format(worker_name, current_time.strftime("%Y%m%d%H%M%S%f")[:-3])
self._prof_path = os.path.join(dir_path, span_name)
```
⇒ `<hostname|worker_name>_<pid>_<YYYYmmddHHMMSSmmm>_ascend_pt`. If `dir_name` is falsy, the base is `os.getenv("ASCEND_WORK_PATH") + "/profiling_data"`, else cwd. `MAX_WORKER_NAME_LENGTH = 226`; an over-long `worker_name` is dropped in favour of the hostname form.

**vllm-ascend ≥ PR #6968 (commit `1a7f845`, 2026-03-05, RFC #6954, vLLM v0.16.0)** passes `worker_name=trace_name`, where
```
trace_name = [{profile_prefix}_]dp{dp}_pp{pp}_tp{tp}_dcp{dcp}_ep{ep}_rank{global_rank}
```
from [`vllm/distributed/utils.py::get_worker_rank_suffix`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/utils.py) (which falls back to `f"rank{global_rank}"` if parallel state is uninitialised, and to `""` if there is no rank at all). Resulting example: `dp0_pp0_tp0_dcp0_ep0_rank0_20260305161834123_ascend_pt` (or `warmup_dp0_...` with a prefix). **A parser must glob BOTH shapes**: `<host>*_<pid>_<ts>_ascend_pt` and `<rank-suffix>_<ts>_ascend_pt`. The official guide's own example uses the hostname form (`analyse("./vllm_profile/localhost.localdomain_*_ascend_pt/")`).

Full tree — **note that `PROF_*` is a child *inside* the `*_ascend_pt` directory**, sibling of `FRAMEWORK/` and `ASCEND_PROFILER_OUTPUT/`. That nesting is exactly how torch_npu detects a profiler directory ([`ProfilerPathManager.get_cann_path`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_path_manager.py) scans `os.listdir(profiler_path)` for `^PROF_\d+_\d+_[0-9a-zA-Z]+`; `get_fwk_path` looks for `FRAMEWORK` in the same dir):

```
<torch_profiler_dir>/
├── PROF_<...>/                              # CANN raw data may ALSO appear at this level
├── <hostname|rank-suffix>_<pid>_<ts>_ascend_pt/
│   ├── profiler_info.json  OR  profiler_info_<rank_id>.json
│   ├── profiler_metadata.json
│   ├── ASCEND_PROFILER_OUTPUT/               # analysis products (+ analyse.done)
│   ├── FRAMEWORK/                            # framework-side raw binary data
│   ├── PLATFORM_<ts>/                        # only when platform analysis produced data
│   └── PROF_<digits>_<digits>_<alnum>/       # CANN dir, NESTED INSIDE *_ascend_pt
│       ├── host/{info.json, host_start.log, start_info}
│       ├── device_<id>/{start_info.<id>, sqlite/, summary/, timeline/}
│       ├── mindstudio_profiler_output/       # CANN parsed text data (may be DELETED)
│       ├── mindstudio_profiler_log/
│       ├── analyze/
│       └── msprof_<n>.db, msprof_analysis_<n>.log, communication_analyzer.db
└── logs/                                     # ProfilerLogger output (community-documented)
```

Structure documented officially as `PROF_{number}_{timestamp}_{string}` (e.g. `PROF_000001_20230628101446_FKFLNPEPPRRCFCBA`) with `analyze/`, `device_{Rank_ID}/`, `host/`, `mindstudio_profiler_log/`, `mindstudio_profiler_output/` ([official guide](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md), [MindSpore doc](https://raw.giteeusercontent.com/mindspore/docs/raw/master/tutorials/source_zh_cn/debug/profiler.md)). `data_simplification=True` deletes `analyze/`, `mindstudio_profiler_log/`, `mindstudio_profiler_output/` under `PROF_*` plus every `msprof_analysis_\d+\.log`, and always removes `sqlite` and each `device_*`/`host`'s `summary` and `timeline` ([`_profiling_parser.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/profiling_parser.py)).

One `*_ascend_pt` per worker **process** — flat siblings, never nested per rank. One `PROF_*` per `*_ascend_pt` (when NPU activity is on). One `device_<id>` per device seen.

### 1.4 Multi-rank layout (important, often wrong in parsers)

Each worker process gets its **own independent top-level directory**, created as a **flat sibling** — never nested per rank:

```
<torch_profiler_dir>/
├── dp0_pp0_tp0_dcp0_ep0_rank0_20260305161834123_ascend_pt/   # rank 0
│   ├── profiler_info_0.json
│   ├── profiler_metadata.json
│   ├── ASCEND_PROFILER_OUTPUT/
│   ├── FRAMEWORK/
│   └── PROF_<...>/
├── dp0_pp0_tp0_dcp0_ep0_rank1_20260305161834567_ascend_pt/   # rank 1
│   └── ...
└── localhost.localdomain_12345_20260119103000123_ascend_pt/  # pre-PR#6968 / no worker_name form
```

- **Pre-PR #6968** the rank is **not** in the name at all: `<hostname>_<pid>_<ts>_ascend_pt`. Ranks are then distinguishable only by `pid`, or by the `profiler_info_<rank_id>.json` inside (`rank_id` comes from `os.environ["RANK"]` or `torch.distributed.get_rank()`).
- **≥ PR #6968** the rank is in the name via `worker_name=trace_name`.
- `get_profiler_path_list()` returns **all** valid subdirs and `analyse()` forks one parser per dir — torch_npu's own model is one dir per rank.
- ⚠️ The `<ts>_host_<pid>_rank<N>_ascend_pt` shape shown in [CANN's `prof_layouts.md`](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/prof_layouts.md) **does not match the torch_npu source pattern** — treat it as a wrapper/rename convention, not authoritative. That source's own advice stands: *"top-level 一卡一目录，目录名末尾 `_rank<N>_ascend_pt`"* and `kernel_details.csv` is **already per-card**, so do not additionally split by `Device_id`.
- **Practical rule: treat the top-level directory name as opaque.** Glob for `*_ascend_pt`, then read `profiler_info*.json` to identify the rank, rather than parsing the name.

### 1.5 MS Service Profiler (the other vLLM-Ascend solution)

Distinct tool, distinct artifacts. `SERVICE_PROF_CONFIG_PATH=ms_service_profiler_config.json`, `PROFILING_SYMBOLS_PATH=service_profiling_symbols.yaml`. Config defaults to `${HOME}/.ms_server_profiler`; per-run dir is a timestamp name. Then `msserviceprofiler parse --input-path=./ --output-path output` produces: `chrome_tracing.json`, `profiler.db`, `request.csv`, `kvcache.csv`, `batch.csv` ([source](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/docs/source/developer_guide/performance_and_debug/service_profiling_guide.md)). **Do not confuse `chrome_tracing.json` (MS Service Profiler) with `trace_view.json` (Ascend PyTorch Profiler).**

vLLM-Ascend also ships a symbol config generator writing `~/.config/vllm_ascend/service_profiling_symbols.<vllm_version>.yaml` ([`vllm_ascend/profiling_config.py`](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/profiling_config.py)) — relevant only to MS Service Profiler, not to `_ascend_pt` parsing.

---

## 2. File inventory with one-line purposes

### 2.1 Class A — `ASCEND_PROFILER_OUTPUT/`

Verbatim tree with annotations, from the [official guide](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md):

| File | Purpose / condition |
|---|---|
| `trace_view.json` | Chrome-trace timeline of the whole AI task. Always (CPU and/or NPU activity). |
| `kernel_details.csv` | Every operator executed on the NPU. Generated when `activities` includes NPU. |
| `operator_details.csv` | Torch-operator-level detail (host + device durations). Generated by default. |
| `op_statistic.csv` | AI Core / AI CPU operator call counts and durations. |
| `api_statistic.csv` | API duration statistics (AscendCL/Runtime/Node/Model/Communication). Level1/Level2. |
| `step_trace_time.csv` | Computation/communication time statistics per iteration. |
| `task_time.csv` | AI-task scheduling durations (Ascend Hardware track summary). NPU activity + Level0/1/2. |
| `memory_record.csv` | HBM usage records for PTA / GE / APP / WORKSPACE. `profile_memory=True`. |
| `npu_module_mem.csv` | Component-level NPU memory usage. Auto-collected. |
| `operator_memory.csv` | Per-operator memory allocation/release details. `profile_memory=True`. |
| `communication.json` | Communication-operator time/bandwidth detail. Multi-card/cluster + Level1/Level2. |
| `communication_matrix.json` | Small-communication-operator basics (size, bandwidth, ranks). Same conditions. |
| `analysis.db` | Communication-scenario aggregate DB. Default in multi-card/cluster. |
| `ascend_pytorch_profiler_{Rank_ID}.db` | All profile data in DB format. Default in PyTorch. |
| `data_preprocess.csv` | AI CPU operator data. `profiler_level=Level2`. |
| `l2_cache.csv` | L2 cache hit rates. `l2_cache=True`. |
| `soc_pmu.csv` | TLB / SoC PMU. NPU activity + `l2_cache`. |
| `pcie.csv`, `hccs.csv` | PCIe / HCCS bandwidth. `sys_interconnection=True`. |
| `nic.csv`, `roce.csv` | NIC / RoCE bandwidth. `sys_io=True`. |

Parent-level: `profiler_info_{Rank_ID}.json` (Profiler metadata), `profiler_metadata.json` (user `add_metadata` + env-var metadata).

> An independent practitioner guide additionally lists an **`analyse.done`** completion marker file in `ASCEND_PROFILER_OUTPUT` ([Ascend-Inference-wiki](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)) — useful as a "parsing finished" sentinel, but **not** in the official file list, so treat as version-dependent.

### 2.2 Class B — `PROF_*` directory

Top level ([msprof reference](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md), verbatim structure):

```
PROF_XXX
├── host   // Host-side raw profile data
│   └── data
├── device_{id}   // Device-side raw profile data
│   └── data
├── msprof_{timestamp}.db                     // all profile data, DB format
├── mindstudio_profiler_output/               // text format
│   ├── msprof_{timestamp}.json
│   ├── op_summary_{timestamp}.csv
│   └── ...
PLATFORM_{timestamp}/                         // NUMA data (sibling of PROF_*, not inside it)
├── platform.db
└── metrics.csv
```

`PROF_*` naming: found by torch_npu with the regex `^PROF_\d+_\d+_[0-9a-zA-Z]+` ([`_path_manager.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_path_manager.py)) → `PROF_<digits>_<digits>_<alnum>`, e.g. `PROF_000001_20230628101435646_FKFLNPEPPRRCFCBA` (index/prof-id, timestamp `YYYYMMDDHHMMSSfff`, random suffix). `device_\d` dirs; `PLATFORM_\d{1,20}`.

Other confirmed subdirectories / conditions:
- `msprof --parse=on` creates `sqlite/` (with DB files) under `PROF_XXX/device_{id}` **and** `PROF_XXX/host`.
- `msprof --clear=on` deletes the `sqlite` dir.
- `msprof --analyze=on --rule=communication[,communication_matrix]` creates **`PROF_XXX/analyze/communication.json`**, **`PROF_XXX/analyze/communication_matrix.json`** (text) and **`PROF_XXX/analyze/communication_analyzer.db`** (always). ⚠️ **So the Class B `communication*.json` files live in `analyze/`, NOT in `mindstudio_profiler_output/`.**
- `mindstudio_profiler_log/` also exists (torch_npu deletes it in simplification mode).
- `FRAMEWORK/` holds raw framework binary data.
- Multi-process: **one `PROF_XXX` per collecting process**, each holding only its own `device_{id}`; post-`--export` tree includes `device_0/`, `device_1/`, `host/`, `msprof_*.db`, `mindstudio_profiler_output/{msprof_*.json, step_trace_*.json, xx_*.csv, README.txt}` ([`msprof_parsing_instruct.md`](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/zh/user_guide/msprof_parsing_instruct.md)).

**File-name suffix pattern: `_<digits>` (timestamp) before the extension**, with an `_slice_<digits>` variant for many of them. Exact regexes from [`CANNFileParser.CANN_DATA_MATCH`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py) — each also has a `*_slice_<digits>` form unless noted:

| Regex | Content |
|---|---|
| `^op_summary_\d{1,20}.*\.csv` | AI Core / AI Vector Core / AI CPU per-op summary |
| `^op_statistic_\d{1,20}.*\.csv` | op-count/duration statistics |
| `^api_statistic_\d{1,20}.*\.csv` | CANN API stats (AscendCL/Runtime/Node/Model/Communication) |
| `^task_time_\d{1,20}.*\.csv` | task scheduler |
| `^step_trace_\d{1,20}.*\.csv`, `^step_trace_\d{1,20}.*\.json` | iteration trace summary / timeline |
| `^ge_memory_record_\d{1,20}.*\.csv`, `^memory_record_\d{1,20}.*\.csv` | CANN/GE memory records |
| `^ge_operator_memory_…`, `^operator_memory_\d{1,20}.*\.csv` | per-op memory |
| `^npu_mem_\d{1,20}.*\.csv`, `^npu_module_mem_\d{1,20}.*\.csv` | NPU / component memory |
| `^l2_cache_\d{1,20}.*\.csv` | L2 cache |
| `^aicpu_\d{1,20}.*\.csv` | AICPU (torch_npu exports it as `data_preprocess.csv`) |
| `^soc_pmu_\d{1,20}.*\.csv` | SoC PMU |
| `^nic_…`, `^roce_…`, `^pcie_…`, `^hccs_…` | sys IO / interconnect |
| `^msprof_\d{1,20}.*\.json` | **timeline master table** (the chrome-trace array) |
| `^msprof_\d{1,20}\.db` (at `PROF_*` **root**) | unified DB of all profile data |
| `^communication\.json`, `^communication_matrix\.json` | (in `analyze/`) |
| `communication_analyzer\.db` | (in `analyze/`) |
| `prof_rule_0_{timestamp}.json`, `fusion_op_*.csv` | tuning-rule results / op fusion before-after (**offline-inference doc table only**) |
| `README.txt` | present in the dir |

**`kernel_details_*` is NOT in torch_npu's CANN regex list** — `kernel_details.csv` is a *framework-side projection* of `op_summary_*.csv`, produced in `ASCEND_PROFILER_OUTPUT/`.

Files documented in `mindstudio_profiler_output/`:

| File | Purpose |
|---|---|
| `msprof_*.json` | Full timeline report; areas = application layer / CANN layer / underlying NPU (`Ascend Hardware`, `Communication`, `Overlap Analysis`) / per-op detail. |
| `msprof_tx_*.json` | msproftx (user instrumentation) timeline; subset of `msprof_*.json`. |
| `msprof_tx_*.csv` | msproftx summary, concatenated by thread. |
| `op_summary_*.csv` | Per-operator detail + PMU columns. |
| `op_statistic_*.csv` | Operator call counts and total durations. |
| `kernel_details` (note: documented as `kernel_details.csv` in the [older MindStudio 7.0 PDF](https://www.hiascend.com/doc_center/source/zh/mindstudio/70RC1/mscommandtoolug/mscommandug/MindStudio%207.0.RC1%20%E5%91%BD%E4%BB%A4%E8%A1%8C%E5%BC%80%E5%8F%91%E5%B7%A5%E5%85%B7%E6%8C%87%E5%8D%97%2001.pdf)) | In CANN docs the torch_npu `kernel_details.csv` fields are described by reference to `op_summary` (see §3.3). |
| `api_statistic_*.csv` | CANN API duration statistics across AscendCL / Runtime / Node / Model / Communication. |
| `task_time_*.csv` | Task scheduler (Ascend Hardware) durations. |
| `step_trace_*.csv` / `step_trace_*.json` | Iteration trace. **"This profile data file does not exist in single-operator scenarios (such as the PyTorch scenario)."** |
| `communication_statistic_*.csv` | Collective-communication operator statistics + overlap. |
| `memory_record_*.csv` | CANN-operator memory records (GE component). |
| `operator_memory_*.csv` | Per-operator memory allocation details. |
| `npu_mem_*.csv`, `npu_module_mem_*.csv` | NPU memory / component memory. |
| `ai_core_utilization_*.csv` | AI Core instruction-ratio metrics; content varies by `--aic-metrics`. |
| `ai_vector_core_utilization_*.csv` | AI Vector Core ratios. |
| `aicpu_*.csv`, `aicpu_mi_*.csv` | AI CPU operator durations / data-prep queues. |
| `l2_cache_*.csv` | L2 hit/victim rates per op. |
| `dp_*.csv` | Data-augmentation info (training only, Atlas training products). |
| `summary/`, `timeline/` | **Legacy** locations: older versions/modes put CSVs in `device_<id>/summary/op_summary.csv` and JSON in `device_<id>/timeline/msprof_*.json`. Prefer `mindstudio_profiler_output/` when both exist. ([CANN `prof_layouts.md`](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304789/.agents/skills/model-infer-perf-breakdown/references/prof_layouts.md)) |

### 2.3 Confirm / correct the specific list you asked about

| Name you listed | Verdict |
|---|---|
| `trace_view.json` | ✅ Class A, `ASCEND_PROFILER_OUTPUT/`. **Bare JSON array, not `{"traceEvents":...}`** — see §4.1. |
| `kernel_details.csv` | ✅ Class A only. Class B analogue = `op_summary_*.csv`. |
| `operator_details.csv` | ✅ Class A. |
| `op_statistic.csv` | ✅ Class A. Class B = `op_statistic_*.csv`. |
| `op_summary.csv` | ✅ **Class B only** (`op_summary_*.csv`). Not a Class A filename. |
| `api_statistic.csv` | ✅ Class A. Class B = `api_statistic_*.csv`. |
| `step_trace_time.csv` | ✅ Class A. Class B = `step_trace_*.csv` (+ `step_trace_*.json`); absent in PyTorch single-operator scenarios. |
| `communication.json` / `communication_matrix.json` | ✅ Class A `ASCEND_PROFILER_OUTPUT/`; multi-card/cluster + Level1/Level2. Class B equivalents live in `PROF_*/analyze/`, not `mindstudio_profiler_output/`. |
| `mstx.json` | ❌ **No such file exists.** mstx/msproftx data lands in `msprof_tx_*.json` / `msprof_tx_*.csv` (Class B — *"msproftx timeline data. It is a subset of `msprof_*.json`"*) and in the `MSTX_EVENTS` DB table. |
| `memory_record.csv` | ✅ Class A (`profile_memory=True`). Class B = `memory_record_*.csv` / `ge_memory_record_*.csv`. |
| `npu_module_mem.csv` | ✅ Class A (auto-collected). Class B = `npu_module_mem_*.csv`. |
| `profiler_info.json` | ✅ Class A, at the `*_ascend_pt` **root** (next to `ASCEND_PROFILER_OUTPUT/`, not inside it). Documented name is `profiler_info_{Rank_ID}.json`; `ProfilerPathManager.get_info_file_path` accepts **either** `profiler_info.json` **or** `profiler_info_<digits>.json`. Not a Class B file. |
| `analysis.db` | ✅ Class A `ASCEND_PROFILER_OUTPUT/` (communication scenarios). **Not** a Class B `mindstudio_profiler_output/` file. Beware three distinct DB names: `analysis.db` (torch_npu), `communication_analyzer.db` (msprof `--analyze`), `cluster_analysis.db` (msprof-analyze). |
| `msprof_*.db` | ✅ Class B, `PROF_*/msprof_{timestamp}.db` (top level of `PROF_*`, **not** inside `mindstudio_profiler_output/`). |
| `aicore_freq.csv` | ❌ **No such file.** AI Core frequency exists as the `AICORE_FREQ` DB table (`deviceId`, `timestampNs`, `freq`, `dieId`) and as the **`AI Core Freq`** timeline lane. |
| pipe-utilization / `aic_metrics` outputs | ⚠️ **Not standalone files.** Pipe ratios are *columns appended inside* `op_summary_*.csv` (Class B) and `kernel_details.csv` (Class A), selected by `--aic-metrics` / `experimental_config.aic_metrics`. The separately-named files are `ai_core_utilization_*.csv` and `ai_vector_core_utilization_*.csv`. |
| `msprof_*.json` inside `ASCEND_PROFILER_OUTPUT/` | ➕ **Exists** — MindStudio's "PyTorch training/inference data" table lists `trace_view.json`, `msprof_*.json`, `operator_details.csv`, `memory_record.csv`, `operator_memory.csv`, `kernel_details.csv`, `step_trace_time.csv`, `communication.json`, `communication_matrix.json`, `ascend_pytorch_profiler_{rank_id}.db`, `analysis.db` (`*` = a timestamp). |
| `analyse.done` | ✅ **Confirmed in source** — `generate_parser_done_file()` writes an empty `ASCEND_PROFILER_OUTPUT/analyse.done`. Canonical "analysis finished" marker; good readiness gate for a parser. |

### 2.4 Level → file availability (useful sanity table)

Documented mapping (MindStudio `system_tuning` 表8):
- **Level0** → `trace_view.json`, `msprof_*.json`, `operator_details.csv`, `kernel_details.csv` (**no** AI Core metrics), `memory_record.csv`, `operator_memory.csv`
- **Level1** adds → `communication.json`, `communication_matrix.json`
- **Level2** adds → memcpy data

**AI Core metrics in `kernel_details.csv` (`aicore_time`, `aic_mac_ratio`, …) require `aic_metrics=PipeUtilization` AND `profiler_level >= Level1`.**

---

## 3. CSV header columns

### 3.1 How much to trust column *order*

Official docs publish **field tables**, and screenshots of sample CSVs, but generally do not print the literal header line. Therefore:

- **Class A** headers below are given in the **documented table order**, which matches the order in the official sample screenshots (e.g. Figure 5 `kernel_details`). Treat order as **highest-confidence-but-not-byte-verified**.
- **Class B** tables in CANN docs are explicitly conditional: *"The content of the `op_summary_*.csv` file varies depending on the msProf collection parameters used"* and *"Supported fields may vary by product. Please refer to the actual result file for the final list of fields."* **Do not hard-code Class B column order — index by header name.**

A corroborating third-party parser report shows the **actual** `kernel_details.csv` header in the field uses `Accelerator`, not `Accelerator Core`, and `Block Dim`, not `Block Num` ([Ascend-Inference-wiki](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). Both facts are consistent with §3.3/§3.4 — real files differ from the doc table labels. **Parse by name.**

### 3.2 `op_statistic.csv` (Class A) / `op_statistic_*.csv` (Class B)

Class A — official field list order ([CANN `op_statistic` section](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)). Note this is the **Class B authoring** described under a name Class A also uses; Class A `op_statistic.csv` is produced by `IntegrateParser` copying/merging the CANN file:

```
Device_id, Model Name, OP Type, Core Type, Count, Total Time(us), Avg Time(us), Min Time(us), Max Time(us), Ratio(%)
```

| Column | Meaning |
|---|---|
| `Device_id` | Device ID |
| `Model Name` | Model name; empty if not collected; not displayed by default or in single-operator scenarios |
| `OP Type` | **Operator type** |
| `Core Type` | **Core type**: `AI_CORE`, `AI_VECTOR_CORE`, `AI_CPU` |
| `Count` | **Number of operator calls** |
| `Total Time(us)` | **Total duration (µs)** |
| `Avg Time(us)` / `Min Time(us)` / `Max Time(us)` | Average / min / max call durations (µs) |
| `Ratio(%)` | Percentage of total duration for this operator type in the model |

⚠️ `Avg Time(us), Min Time(us), Max Time(us)` are listed as a *single combined row* in the doc table — the literal header may be `Avg Time(us)` only, or all three, depending on version. **Verify against a real file.**

### 3.3 `kernel_details.csv` (Class A) — **source-verbatim base headers**

Verbatim from [`torch_npu/profiler/analysis/prof_common_func/_csv_headers.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_csv_headers.py):

```python
class CsvHeaders(object):
    # op_summary
    TASK_START_TIME = "Task Start Time(us)"
    OP_SUMMARY_SHOW_HEADERS = ["Device_id", "Op Name", "OP Type", "Task Type", TASK_START_TIME, "Task Duration(us)",
                               "Task Wait Time(us)", "Block Num"]
    OP_SUMMARY_ADDITIONAL_HEADERS = ["Mix Block Num", "Input Shapes", "Input Data Types", "Input Formats", "Output Shapes",
                                     "Output Data Types", "Output Formats"]
    OP_SUMMARY_KERNEL_BASE_HEADERS = ["Device_id", "Name", "Type", "Accelerator Core", "Start Time(us)", "Duration(us)",
                                      "Wait Time(us)", "Block Num"]
```

**This is the authoritative confirmation that `kernel_details.csv` is a *projection* of Class B `op_summary_*.csv`** — `OP_SUMMARY_KERNEL_BASE_HEADERS` is exactly the class-A kernel header set, and it renames the class-B fields:

| `op_summary` (Class B) | `kernel_details.csv` (Class A) |
|---|---|
| `Op Name` | `Name` |
| `OP Type` | `Type` |
| `Task Type` | `Accelerator Core` |
| `Task Start Time(us)` | `Start Time(us)` |
| `Task Duration(us)` | `Duration(us)` |
| `Task Wait Time(us)` | `Wait Time(us)` |
| `Block Num` | `Block Num` |
| `Device_id` | `Device_id` |

⇒ **The literal Class A base header line is:**
```
Device_id,Name,Type,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Num
```

⚠️ **Three important consequences:**

1. **Do not confuse `op_summary`'s timing column names with `kernel_details`'s.** `op_summary` uses `Task Start Time(us)` / `Task Duration(us)` / `Task Wait Time(us)`; `kernel_details` uses `Start Time(us)` / `Duration(us)` / `Wait Time(us)`. §3.4 below lists the Class B `op_summary` column names as they appear in CANN's own documentation tables (`Task Start Time(us)` etc.) — both are correct *for their own file*.
2. **The Class A base set does NOT include `Step Id`, `Model ID`, `Task ID`, `Stream ID`, `OP State`, `Mix Block Num`, `HF32 Eligible`, or the Input/Output triples.** Those are added conditionally: `Step Id` is prepended only when a step range was derived from `schedule` (§3.3 note below), and the rest come from the collection parameters. **A parser must treat every one of them as optional and index by header name.**
3. **`Accelerator Core` IS the real header** (not `Accelerator`). The third-party guide that reported `Accelerator` is describing a *rendered/shortened* view. Trust the source constant, but match tolerantly since display layers shorten it.

**Documented extended field list** ([official guide, Table 1](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)) — the full set when all collection options are on, prepending `Step Id` and appending to the base above:

```
[Step Id,] Device_id, Name, Type, [Model ID, Task ID, Stream ID,] [OP State,] Accelerator Core,
Start Time(us), Duration(us), Wait Time(us), Block Num, [Mix Block Num,] [HF32 Eligible,]
[Input Shapes, Input Data Types, Input Formats, Output Shapes, Output Data Types, Output Formats]
[...aic_metrics columns...]
```

| Column | Meaning |
|---|---|
| `Step Id` | Iteration ID. **Added only when a step range was derived** (i.e. `schedule` was used). With `warmup != 0` plus async ops after each step, ops from warmup may be collected with **no Step Id** (null). **vllm-ascend never passes `schedule=` ⇒ expect this column to be ABSENT in vllm-ascend captures.** |
| `Device_id` | Device ID |
| `Model ID` | Model ID |
| `Task ID` | Task ID |
| `Stream ID` | Stream the task resides on |
| `Name` | **Operator name** |
| `Type` | **Operator type** |
| `OP State` | `dynamic` / `static` / `N/A` (communication ops have no state). Reported only when `--task-time=l1`. |
| `Accelerator Core` | **AI accelerator core type** — `AI Core`, `AI CPU`, etc. (practice reports the literal value set `AI_CORE` / `AI_VECTOR_CORE` / `MIXED_AIC`) |
| `Start Time(us)` | **Start time (µs)** |
| `Duration(us)` | **Execution duration of this operator (µs)** |
| `Wait Time(us)` | **Wait time (µs)** |
| `Block Num` | Number of run splits = number of cores used |
| `Mix Block Num` | Block count of the secondary accelerator for MIX ops. `N/A` at `task_time=l0`. Atlas A2/A3 only. |
| `HF32 Eligible` | `YES` / `NO` |
| `Input Shapes` / `Input Data Types` / `Input Formats` | Input shape / dtype / format |
| `Output Shapes` / `Output Data Types` / `Output Formats` | Output shape / dtype / format |


**Critical note from the doc:** *"When the `aic_metrics` parameter of `experimental_config` is configured, the `kernel_details.csv` file will add corresponding fields... For detailed descriptions of related fields in the file, see [op_summary](...)"* — i.e. **`kernel_details.csv` is a `op_summary`-shaped table**. When `aic_metrics != AiCoreNone`, PMU/ratio columns are appended at the end. See §3.4 for those names.

### 3.4 `op_summary_*.csv` (Class B) — the aic_metrics / pipe-utilization home

**Source-verbatim "show" header** (the columns torch_npu exports/reads by default) — see §3.3 for the quoting source:
```
Device_id,Op Name,OP Type,Task Type,Task Start Time(us),Task Duration(us),Task Wait Time(us),Block Num
```
plus `OP_SUMMARY_ADDITIONAL_HEADERS` appended when the relevant collection options are on:
```
Mix Block Num,Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats
```

**Full documented field set** (superset; varies by `--task-time` / `--aic-mode` / `--aic-metrics` and by product). The doc's own tables list these, but note the doc is a *field reference*, not a header dump:
```
Device_id, Model Name, Model ID, Task ID, Stream ID, Infer ID, Op Name, OP Type, OP State,
Task Type, Task Start Time(us), Task Duration(us), Task Wait Time(us), Block Num, HF32 Eligible,
Mix Block Num, Input Shapes, Input Data Types, Input Formats, Output Shapes, Output Data Types,
Output Formats, Context ID, aiv_time(us), aicore_time(us), total_cycles, Register value
```
⚠️ The **source constants above are the authority** for what the header actually looks like; the wider doc list reflects all fields that *can* be collected. Do not assume `Infer ID`, `Model Name`, `Context ID`, `aiv_time(us)`, `aicore_time(us)`, `total_cycles` or `Register value` are present.

Mapping to your requested semantics:

| You asked for | Column |
|---|---|
| operator name | `Op Name` |
| OP type | `OP Type` |
| task type | `Task Type` — documented values: `AI_CORE`, `AI_VECTOR_CORE`, `AI_CPU`, `CCU`, `DPU`; docs elsewhere show `MIX_AIC` and `communication` |
| core type | Class B has no `Core Type`; Class A `op_statistic` does |
| device id | `Device_id` |
| start time | `Task Start Time(us)` |
| duration | `Task Duration(us)` ("includes scheduling the task to the accelerator, execution on the accelerator, and the completion response time") |
| wait time | `Task Wait Time(us)` ("interval between the end of the previous task and the start of the current task") |
| count | *not in `op_summary`* — `Count` lives in `op_statistic_*.csv` |
| total time | *not in `op_summary`* — `Total Time(us)` lives in `op_statistic_*.csv` |
| ratio | *not in `op_summary`* — `Ratio(%)` lives in `op_statistic_*.csv` |
| aicore time | **`aicore_time(us)`** — theoretical AI Core execution duration assuming all blocks scheduled simultaneously and equal duration; populated when `--task-time=l1` **and** `--aic-mode=task-based`. Doc warns it is inaccurate if AI Core frequency changes. |

**Pipe / hardware-utilization columns** (appended; each is `*_`-prefixed where `*` = `aic` or `aiv`). Generated when `--task-time=l1` and `--aic-mode=task-based`; selection controlled by `--aic-metrics`. `N/A` at `task_time=l0`.

*PipeUtilization group:*
```
*_vec_time(us), *_vec_ratio, *_mac_time(us), *_mac_ratio, *_scalar_time(us), *_scalar_ratio,
aic_fixpipe_time(us), aic_fixpipe_ratio, *_mte1_time(us), *_mte1_ratio, *_mte2_time(us), *_mte2_ratio,
*_mte3_time(us), *_mte3_ratio, *_icache_miss_rate, memory_bound, cube_utilization(%)
```
- `*_mac_ratio` = Cube instruction cycles / total cycles
- `*_mte1_ratio` = MTE1 (L1→L0A/L0B) cycles / total
- `*_mte2_ratio` = MTE2 (DDR→AI Core) cycles / total
- `*_mte3_ratio` = MTE3 (AI Core→DDR) cycles / total
- `*_vec_ratio` = Vector cycles / total; `*_scalar_ratio` = Scalar cycles / total
- `*_icache_miss_rate` = L2 instruction-cache miss rate
- `memory_bound` = `mte2_ratio / max(mac_ratio, vec_ratio)`; >1 ⇒ memory bound
- `cube_utilization(%)` = `total_cycles / (freq * core_num * task_duration)`

*ArithmeticUtilization group:* `*_mac_fp16_ratio`, `*_mac_int8_ratio`, `*_vec_fp32_ratio`, `*_vec_fp16_ratio`, `*_vec_int32_ratio`, `*_vec_misc_ratio`, `*_cube_fops`, `*_vector_fops`

*Memory group:* `*_ub_read_bw(GB/s)`, `*_ub_write_bw(GB/s)`, `*_l1_read_bw(GB/s)`, `*_l1_write_bw(GB/s)`, `*_l2_read_bw`, `*_l2_write_bw`, `*_main_mem_read_bw(GB/s)`, `*_main_mem_write_bw(GB/s)`

*MemoryL0 group:* `l0a_read_bw(GB/s)`, `l0a_write_bw(GB/s)`, `l0b_read_bw(GB/s)`, `l0b_write_bw(GB/s)`, `l0c_read_bw(GB/s)`, `l0c_write_bw(GB/s)`, `l0c_read_bw_cube(GB/s)`, `l0c_write_bw_cube(GB/s)`

*MemoryUB group:* `ub_read_bw_vector(GB/s)`, `ub_write_bw_vector(GB/s)`, `ub_read_bw_scalar(GB/s)`, `ub_write_bw_scalar(GB/s)`

*ResourceConflictRatio group:* `vec_bankgroup_cflt_ratio`, `vec_bank_cflt_ratio`, `vec_resc_cflt_ratio`

*L2Cache group:* `write_cache_hit`, `write_cache_miss_allocate`, `r*_read_cache_hit`, `r*_read_cache_miss_allocate`, `read_local_l2_hit`, `read_local_l2_miss`, `read_local_l2_victim`, `write_local_l2_hit`, `write_local_l2_miss`, `write_local_l2_victim`

*MemoryAccess group:* `read_main_memory_datas(KB)`, `write_main_memory_datas(KB)`, `gm_to_l1_datas(KB)`, `l0c_to_l1_datas(KB)`, `l0c_to_gm_datas(KB)`, `gm_to_ub_datas(KB)`, `ub_to_gm_datas(KB)`

### 3.5 `operator_details.csv` (Class A) — **source-verbatim header**

Verbatim from [`OperatorViewParser.OPERATOR_VIEW`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_view/operator_view_parser.py):

```
Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us),Device Self Duration With AICore(us),Device Total Duration With AICore(us)
```

⚠️ **Correction to the official doc table:** the doc writes the units as `(µs)` with U+00B5 MICRO SIGN; **the source writes `(us)` ASCII**. Trust the source for the header string. `Call Stack` is controlled by `with_stack` / `with_modules`. `operator_details.csv` is generated by default.

### 3.6 `api_statistic.csv` (Class A) / `api_statistic_*.csv` (Class B)

```
Device_id, Level, API Name, Time(us), Count, Avg(us), Min(us), Max(us), Variance
```

- `Device_id` — *"displayed as `host` for host-side data"* ⇒ **this column is not always numeric**
- `Level` — AscendCL / Runtime / Node / Model / Communication
- `API Name`, `Time(us)` = total duration, `Count`, `Avg(us)`, `Min(us)`, `Max(us)`, `Variance`

### 3.7 `step_trace_time.csv` (Class A)

```
Device_id, Step, Computing, Communication (Not Overlapped), Overlapped, Communication, Free,
Stage, Bubble, Communication (Not Overlapped and Exclude Receive), Preparing
```

All times in **µs** ([official guide, Table 6](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)). The last field, `Preparing` ("time from the start of the iteration to the execution of the first computation or communication operator"), was added in a later version than the first ten — older files end at `Communication (Not Overlapped and Exclude Receive)`.

Class B `step_trace_*.csv` uses **different names** ([msprof `step_trace`](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)):
```
Device_id, Iteration ID, FP Start(us), BP End(us), Iteration End(us), Iteration Time(us),
FP to BP Time(us), Iteration Refresh(us), Data Aug Bound(us), Model ID, Reduce Start(us), Reduce Duration(us)
```
In offline inference, FP/BP are not collected, so **`FP Start` and `BP End` are `N/A`**. In single-device scenarios no `Reduce` data is output.

### 3.8 `communication_statistic.csv` (Class B; no Class A equivalent)

```
Device_id, OP Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)
```
`Ratio(%)` = proportion of this op type's execution duration to the total collective-communication duration ([msprof `communication_statistic`](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)).

**Class A has no `communication_statistic.csv`** — communication aggregates live in `communication.json`, `communication_matrix.json`, and `analysis.db` (tables `CommAnalyzerBandwidth`, `CommAnalyzerTime`, `CommAnalyzerMatrix` — see §6.4).

### 3.9 `memory_record.csv` (Class A) vs `memory_record_*.csv` (Class B)

Class A ([official guide, Table 2](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)):
```
Component, Timestamp (µs), Total Allocated (MB), Total Reserved (MB), Total Active (MB), Stream Ptr, Device Type
```
Class B: `Device_id, Component, Timestamp(us), Total Allocated(KB), Total Reserved(KB), Device` — **KB vs MB. This is a real unit trap.**

### 3.10 `npu_module_mem.csv` (Class A) vs `npu_module_mem_*.csv` (Class B)

Class A: `Device_id, Component, Timestamp (µs), Total Reserved(MB), Device`
Class B: `Device_id, Component, Timestamp(us), Total Reserved(KB), Device` — again **MB vs KB**.

### 3.11 AI-core / AI-vector-core utilization CSVs (Class B)

`ai_core_utilization_*.csv` — content varies with `--aic-metrics`. PipeUtilization example field set:
```
vec_ratio, mac_ratio, scalar_ratio, mte1_ratio, mte2_ratio, mte3_ratio,
icache_miss_rate, fixpipe_ratio, memory_bound
```
`ai_vector_core_utilization_*.csv`:
```
vec_ratio, mac_ratio, scalar_ratio, mte1_ratio, mte2_ratio, mte3_ratio, icache_miss_rate, memory_bound
```
The corresponding timeline track is **`AI Core Utilization`**; fields are `Average`, `Core {ID}`, `utilization(%)` ([msprof](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)).

> ⚠️ Note the **lowercase** field names here (`vec_ratio`, not `*_vec_ratio`) versus the **`aic_`/`aiv_`-prefixed** names in `op_summary`/`kernel_details`. Both spellings exist in official docs — do not normalize blindly.

---

## 4. Chrome-trace (`trace_view.json`) event structure

### 4.1 ⚠️ Top-level shape — **it is a bare JSON ARRAY**

**There is NO `{"traceEvents": [...]}` wrapper, and therefore no sibling keys at all** — no `displayTimeUnit`, `baseTime`, `deviceId`, `distributedInfo`, `systemTraceEvents`, `vizViewer`, and no separate metadata JSON. Confirmed four ways:

1. **Source — the Ascend mstt toolchain selects its JSON parser prefix per backend:**
   `trace_event_item = {Constant.GPU: "traceEvents.item", Constant.NPU: "item"}`
   → CUDA traces are `{"traceEvents":[...]}`; **NPU/Ascend traces are a top-level array** ([`base_profiling_parser.py`](https://raw.githubusercontent.com/Ascend/mstt/master/profiler/msprof_analyze/compare_tools/compare_backend/profiling_parser/base_profiling_parser.py)).
2. **Source — torch_npu serializes a Python `list`**: `FileManager.create_json_file_by_path(self._trace_file_path, self._trace_data)` ([`_trace_view_parser.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_view/_trace_view_parser.py)).
3. **Source — torch_npu's CANN-timeline reader rejects non-lists**: `_json_load()` returns `[]` unless `isinstance(data, list)` ([`_cann_file_parser.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py)).
4. **Real committed fixture** — `profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json` (635 KB, in the official `Ascend/mstt` repo) literally begins:
   `[{"ph": "X", "name": "aten::empty", "pid": 437675, "tid": 437675, "ts": "1704161511420306.491", "dur": 13.08, "cat": "cpu_op", "args": {"Sequence number": -1, "Fwd thread id": 0}}, ...`
   ([raw fixture](https://raw.githubusercontent.com/Ascend/mstt/master/profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json))

Equivalent metadata lives in the sidecar files `profiler_metadata.json` and `profiler_info.json`, **not** in the trace.

### 4.2 ⚠️ How the file is assembled — interrupted parses are malformed JSON

The write is **two-phase**, and the intermediate state is deliberately invalid JSON ([`_fwk_pre_parser.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_view/prepare_parse/_fwk_pre_parser.py), [`_file_manager.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_file_manager.py)):

1. `TracePreParser` writes `ASCEND_PROFILER_OUTPUT/trace_view.json` containing the framework (CPU/torch-op) events via `create_prepare_trace_json_by_path()`, which writes **`data[:-1]` — i.e. it deliberately omits the trailing `]`.**
2. `CANNExportParser` runs `msprof --export=on --output=<PROF_*>` (plus `--type=db` for Db export).
3. `CANNTimelineParser` busy-waits until `PROF_*/mindstudio_profiler_output/*.csv` exists (text) or `PROF_*/msprof_<digits>.db` exists (db) — **this is the definition of "export finished"**, and also a useful readiness gate for a parser.
4. `TraceViewParser` appends `",{...}"` chunks via `append_trace_json_by_path()`: `data = f",{data[1:]}"` — it strips the leading `[` and **keeps the trailing `]`, thereby closing the array**.
5. If there is no `PROF_*` dir at all (framework-only profiling), `TraceViewParser` builds the entire array from `FRAMEWORK/` raw data instead.

**Consequence: a killed or interrupted parse leaves a `trace_view.json` with no closing `]`.** A robust parser must tolerate/repair a truncated array (e.g. append `]` when the tail is incomplete, or stream-parse and accept EOF).

### 4.3 `ph` values actually emitted

| `ph` | Emitted on Ascend? | Evidence |
|---|---|---|
| `X` | **Yes** — dominant | `TraceEventManager.create_x_event` → `{"ph":"X",...}`; real fixture |
| `M` | **Yes** — torch_npu writes them; CANN writes `process_name`/`thread_name` | `create_m_event`, `create_gc_m_event`; mstt `is_m_mode()`, `is_process_meta()=="process_name"`, `is_thread_meta()=="thread_name"`, `is_thread_sort_meta()=="thread_sort_index"` |
| `s` / `f` | **Yes** — flow start/finish | `Constant.FLOW_START_PH="s"`, `FLOW_END_PH="f"`; `create_torch_to_npu_flow`, `create_task_queue_flow`, `create_fwd_flow` |
| `C` | **Yes** — counters (AI Core Freq, HBM/LLC/DDR/NPU_MEM/SAMPLE_PMU/NIC/RoCE/PCIE/HCCS) | [MindStudio `Timeline.md`](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Timeline.md) §4.5.3.2 |
| `B` / `E` | **Not emitted on Ascend** — referenced nowhere in torch_npu, mstt, or MindStudio docs. Class B `step_trace_*.json` uses `ph:B`/`ph:E`, but that is a different file. |
| `i` | **Unconfirmed** — no evidence. |

**Non-standard field:** counter (`ph:"C"`) events carry a **top-level `"processName"` key** (e.g. `"APP/DDR"`, `"write_ost"`, `"AI Core Freq"`), not only `args`.

### 4.4 Field types (from the real fixture)

- `ts` is a **decimal string in microseconds**: `"1704161511420306.491"` — parse as float/Decimal, never `int`.
- `dur` is a **float in microseconds**: `13.08`.
- `pid` / `tid` are **JSON numbers** for host events (OS pid/tid, e.g. `437675`; a backward thread was `439228`).
- **`pid` is heterogeneous.** torch_npu also writes packed pids for its Python/GC lanes:
  `pid = (pid << 10) | (sort_index << 5) | device_id` (`PID_OFFSET=10`, `INDEX_OFFSET=5`) ([`_trace_event_manager.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py)). MindStudio's own query example uses a **string** pid `"HCCL"` with `tid` `["272_0"]`. ⇒ treat `pid`/`tid` as opaque strings/heterogeneous values.
- Microseconds for `ts` is corroborated by `EventBean.ts` → `convert_us2ns(self._origin_data.get("ts", 0))` ([`_event_bean.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_event_bean.py)).

### 4.5 Host vs Device discrimination — use `process_name` metadata, NOT `pid == "Host"`

There are **no `"Host"` / `"Device"` pid values**. Lanes are identified by `ph="M"`, `name="process_name"` events; the mapping name→pid then classifies every `X` event under that pid ([`npu_profiling_parser.py`](https://raw.githubusercontent.com/Ascend/mstt/master/profiler/msprof_analyze/compare_tools/compare_backend/profiling_parser/npu_profiling_parser.py), `_filter_meta_id()`):

| `args.name` of the `process_name` M event | Meaning |
|---|---|
| `"Python"` | Host: framework/PyTorch ops. mstt's advisor requires exactly `['Python', 'CANN', 'Ascend Hardware']`. |
| `"CANN"` | Host: ACL / GE / Runtime / Node / Model / Communication APIs |
| `"Ascend Hardware"` | **Device**: NPU task streams (`Constant.NPU_BAR`) |
| `"Communication"` or `"HCCL"` | Communication lane (helper `is_hccl_process_name()` accepts either spelling; `Constant.COMM_BAR="Communication"`) |
| `"Overlap Analysis"` | Compute/communication overlap lane (`Constant.OVERLAP_BAR`) |
| `"Python GC"` | GC lane |

Also documented as lanes: `AI Core Freq`, host-sys lanes (CPU Usage, CPU Freq, Memory Usage, Disk Usage, Network Usage, OS Runtime API) ([Timeline design doc](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Timeline.md), [Best practices: Timeline lanes](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/best_practices/Timeline_Common_Lanes_and_Interface.md)). `Process.hash() = args.get("name")`; `Thread.hash() = args.get("name")` ([`trace_view_json.py`](https://raw.githubusercontent.com/Ascend/mstt/09ff65f6b8a57c5e4307957770d020f02f4f17ed/profiler/advisor/advisor_backend/common_func_advisor/trace_view_json.py)).

For the DB scenario MindStudio maps the same concepts to `PROCESS_TYPE` enum values `ASCEND_HARDWARE`, `HCCL`, `OVERLAP_ANALYSIS`, `CANN_API`, `API`, `MS_TX`, `TEXT`.

**Logical vs physical stream:** *"The stream ID under Ascend Hardware is the complete logic stream ID of the task, and the stream ID attribute of each API in the timeline on the right is the physical stream ID of the API"* — so `Stream Id` on `Ascend Hardware` slices is the **logical** stream; CANN-API events expose `Physic Stream Id`.

### 4.6 `args` key names — confirmed verbatim, with corrections

| Key (exact) | Meaning | Source |
|---|---|---|
| `Input Dims` | Input shapes (string; `;`-separated) | `Constant.INPUT_SHAPES = "Input Dims"`; mstt `args.get("Input Dims")` |
| `Input type` | Input dtypes | `Constant.INPUT_DTYPES = "Input type"` |
| `Call stack` | Call stack (`;` → `;\r\n` rewritten) | `Constant.CALL_STACK = "Call stack"`; mstt `CALL_STACKS` |
| `Module Hierarchy` | Module path (with_modules) | `Constant.MODULE_HIERARCHY` |
| **`flops`** — lowercase | FLOPS estimate (with_flops) | `Constant.FLOPS = "flops"`. ⚠️ **`FLOPs` is NOT the spelling torch_npu writes.** |
| `Sequence number` | Step/seq id | `Constant.SEQUENCE_NUMBER`; real fixture |
| `Fwd thread id` | `0` = forward, non-zero = backward | `Constant.FORWARD_THREAD_ID`; real fixture |
| `Scope` | Autograd scope enum | `Constant.SCOPE` |
| `Task Type` | Accelerator type; `== "AI_CORE"` is how `EventBean.is_ai_core` decides | mstt `args.get('Task Type')` |
| `Stream Id` | Stream id | mstt `args.get('Stream Id')` |
| `Task Id` | Task id | mstt `args.get('Task Id')` |
| `Device Id` | Device id (`int(self._args.get('Device Id', -1))`) | mstt |
| `correlation_id` | Correlation id (snake_case, lowercase) | mstt `args.get('correlation_id')` |
| `connection_id` | Launch↔kernel correlation (snake_case) | Documented as a panel field; **raw-JSON spelling not code-proven** |
| `Total Reserved`, `Bytes`, `Addr` | Memory lane | mstt `args.get('Total Reserved', 0)`, `args.get("Bytes",0)/1024`, `args.get("Addr")` |
| `stream` (lowercase) | std::thread id fallback | mstt `args.get("stream")` |
| `name` (inside args) | Lane/process name for M events; also a prune key | `Process.hash()`, `Thread.hash()`; torch_npu `_prune_trace_by_level` uses `data["args"]["name"]` |
| `labels`, `sort_index` | Only in `process_labels` / `*_sort_index` M events | torch_npu `create_m_event` |
| `KB`, `value`, `acc_id` | Counter payloads | Timeline.md real snippets |
| `Input Tensors`, `Input TensorLists`, `Input Scalars` | Raw descriptors; **skipped** from `args` by torch_npu (`TorchOpBean.SKIP_FIELDS`) | `_constant.py` |

**Minimum `args` on a host CPU-op event** is exactly `{"Sequence number": <int>, "Fwd thread id": <int>}` (`-1` = unset).

**Keys that are NOT trace `args` (do not implement them as such):** `step`, `Actual Time`, `aicore time`/`aicore_time`, `count`, `bandwidth`, `transport_type`, `group name`/`group_name`, `rank`, `op_type`, `Thread id`, `Output Shapes`, `Output type`. Where they *do* live:
- `Output Shapes`/`Output Data Types`/`Output Formats`/`Input Formats`/`Input Data Types`/`Attr Info` are fields of the **MindStudio detail panel / `kernel_details.csv` / `op_summary_*.csv`**, not Ascend X-event trace args.
- `aicore_time` is a **CSV column** and a `TASK_PMU_INFO.value` DB column.
- `count`, `bandwidth`, `transport_type`, `group_name`, `rank` are **CANN DB columns** (`COMMUNICATION_OP.count/groupName`, `COMMUNICATION_TASK_INFO.groupName/transportType/bandwidth`, `TASK.deviceId/connectionId/globalPid/taskType/streamId/taskId/modelId/contextId`, `RANK_DEVICE_MAP.rankId/deviceId`).
- `step`: the trace analogue is a **slice name** `ProfilerStep#<n>` (plus `Optimizer.step#<...>.step`), not an arg (`step_name = "ProfilerStep#" + str(...)` in [`profiler.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/profiler.py); mstt `is_step_profiler()` → `name.find("ProfilerStep#") != -1`).
- `Bandwidth(GB/s)`, `Transit Time(ms)`, `Transit Size(MB)`, `Transport Type`, `Op Name`, `Total Op Info`, `Communication Time Info`, `Communication Bandwidth Info`, `Start Timestamp(us)`, `Wait Time(ms)`, `Synchronization Time(ms)`, `Size Distribution`, `Large Packet Ratio`, `Elapse Time(ms)`, `Idle Time(ms)` belong to **`communication.json` / `communication_matrix.json`**, *not* to trace args ([`_communication_parser.py`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_view/communication_parser.py), mstt `constant.py`).

### 4.7 `cat` values actually emitted

**torch_npu-written (verbatim strings):** `"cpu_op"`, `"dequeue"`, `"enqueue"`, `"async_npu"`, `"async_task_queue"`, `"fwdbwd"`, `"GC"`.

**CANN/msprof-written:** `"HostToDevice"` (CANN Node → NPU kernel, and Node → Communication; carries `id` and `ph` `s`/`f` — `CANNFileParser.HOST_TO_DEVICE`, `combine_acl_to_npu()`); `"kernel"` (kernel X events, matched case-insensitively).

**Also seen:** `"python_function"` — Python call-stack slices; `trace_view_json.py` builds new-version call stacks as `[event.name for event in python_dur_events if event.cat == "python_function"]`.

**Not `cat` values (they are slice *names* on the Overlap Analysis lane):** `"Computing"`, `"Communication(Not Overlapped)"`, `"Free"` (`COMPUTING_EVENT`/`FREE_EVENT`/`UNCOVERED_COMMUNICATION_EVENT`, `is_computing_event()`, `is_comm_not_overlap()`).

**Server-only, never in Ascend traces:** `"gpu_memcpy"`, `"nccl"` (CUDA branches of mstt's bean).

**Unconfirmed as `cat` values:** `"operator"`, `"npu_op"`, `"Kernel"` (capital K), `"user_annotation"`, `"HCCL"`, `"MSTX"`, `"Memory"`, `"cpu_launch"`, `"record_function"`, `"npu_memcpy"`, `"hccl"`, `"ac2g"`, `"default"`, `"trace"`. Several of those strings exist in Ascend code as **API-type enum names** (`ENUM_API_TYPE`: `acl`, `model`, `node`, `communication`, `runtime`, `op`, `queue`, `trace`, `mstx`) or as lane names — not as `cat`. Note `mstx_` does appear as a **name prefix** on torch ops (`torch_op.name.startswith("mstx_")`).

### 4.8 Metadata (`ph="M"`) events — exact emitted forms

Written by torch_npu verbatim ([`TraceEventManager.create_m_event` / `create_gc_m_event`](https://raw.githubusercontent.com/Ascend/pytorch/master/torch_npu/profiler/analysis/prof_common_func/_trace_event_manager.py)):
```json
{"ph":"M","name":"process_name",     "pid":pid,"tid":0,  "args":{"name":"Python"}}
{"ph":"M","name":"process_labels",   "pid":pid,"tid":0,  "args":{"labels":"CPU"}}
{"ph":"M","name":"process_sort_index","pid":pid,"tid":0, "args":{"sort_index":0}}
{"ph":"M","name":"thread_name",      "pid":pid,"tid":tid,"args":{"name":"Thread <tid>"}}
{"ph":"M","name":"thread_sort_index","pid":pid,"tid":tid,"args":{"sort_index":<tid or max(tid)+1>}}
```
GC variants use the same five names with `args.name = "Python GC"`, `args.labels = "CPU"`, and `process_sort_index.sort_index = 1` (`GC_SORT_INDEX=1`). `sort_index` 0 = framework, 1 = GC.

Only `process_name`, `thread_name`, `thread_sort_index` are actually inspected by Ascend tooling. **No evidence** for Ascend-specific metadata names such as `"Ascend Profile level"`, `"Device Index"`, `"Device ID"`, `"soc_version"`, or `"step"` — the nearest real analogues are `profiler_info.json`'s `config.experimental_config._profiler_level`, `PLATFORM_*/metrics.csv`, and the `META_DATA` DB table (`SCHEMA_VERSION*`).

**Practical gotcha:** if you emit `thread_sort_index` for only *some* lanes, the others get hidden — it must be emitted for every `tid` in TEXT mode ([Timeline issues doc](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/support/issue_feedback/Timeline_Issues.md)).

### 4.9 Flow events and launch→kernel correlation

Officially listed connection kinds ([msprof timeline section](https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/zh/user_guide/profile_data_file_references.md), "查看算子下发方向"):
- `async_npu` — app-layer op → Ascend Hardware NPU kernel
- `MsTx` — msproftx mark → Ascend Hardware mark task (via `aclprofMarkEx`)
- `async_task_queue` — app-layer Enqueue → Dequeue
- `HostToDevice` — CANN Node → NPU kernel
- `HostToDevice` — CANN Node → Communication op
- `fwdbwd` — forward API → backward API

Mechanics (source-verified):
- **Flow `id` on `async_npu` is the end event's `ts` in nanoseconds**: `{"ph":"s"|"f","bp":"e","name":"torch_to_npu","id":<kernel ts in ns>,...,"cat":"async_npu"}` (`flow_id = end_event.ts`).
- `{"ph":"s"|"f","bp":"e","name":"enqueue_to_dequeue","id":<corr_id>,...,"cat":"async_task_queue"}`
- `{"ph":"s"|"f","bp":"e","name":"fwdbwd","id":<sequence number>,...,"cat":"fwdbwd"}` (skipped when start and end share a `tid`)
- **`"bp": "e"` always** on Ascend flows.
- CANN timeline flows: `ph="s"`/`"f"` + `cat="HostToDevice"` + `id`. The pairing rule is: `id` matches **and** the end event's `(pid, tid, ts)` equals an `X` event's `(pid, tid, ts)` — torch_npu uses `unique_id = f"{pid}-{tid}-{ts}"`; mstt's `_update_kernel_dict` uses `self._all_kernels.get(f"{end_event.pid}-{end_event.tid}-{end_event.start_time}")`.
- DB scenario `Mstx` flow is `connectionId`-based, mstx lane → hardware lane.
- **`ph:"t"` unconfirmed** in Ascend data (the TEXT flow-point table mentions `s`, `f`, `t`, but only `s`/`f` are observed in generated traces).

### 4.10 Ascend-specific tracks

`Python`, `CANN`, `Ascend Hardware`, `Communication` (formerly `HCCL`), `Overlap Analysis`, `AI Core Freq`, `NPU MEM`, `AI Core Utilization`, `Memory`, `msproftx` upper-layer tracks, `SIO` (Atlas A3), `QoS`, `Voltage Info`, `DPU` (Ascend 950), `Fusion Task` (Ascend 950), plus host-sys lanes ([msprof reference](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)).

---

## 5. Prefill vs decode distinguishability

**Bottom line: there is no prefill/decode marker, column, or filename in either artifact class. Phase must be inferred from structure or supplied out-of-band.**

Evidence:

1. **Not in the directory name.** CANN's own skill reference states: *"**phase**（prefill / decode / encode / ...）**不在目录名里**，无论 A / B 类都得问用户"* — "phase is **not in the directory name**; for both class A and B you must ask the user." It further notes a single collection run may be prefill-only, or prefill and decode collected as two separate runs ([CANN `prof_layouts.md`](https://raw.gitcode.com/cann/cann-recipes-infer/raw/7abdb815a28f17c77493e37df0ac3587e3304785/.agents/skills/model-infer-perf-breakdown/references/prof_layouts.md)).
2. **Not in the CSVs.** `kernel_details.csv` has `Step Id` — an iteration counter, not a phase. `step_trace_time.csv` has `Step`. Neither distinguishes phase. No documented column carries "prefill"/"decode".
3. **No built-in mstx phase markers.** The Ascend PyTorch Profiler mstx feature *"collects performance data for communication operators, dataloader duration, and checkpoint saving interface duration by default"* ([official guide](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)). The optional `mstx_torch_plugin` adds only **`dataloader`, `forward`, `step`, `save_checkpoint`** — generic training phases, and it is *"not supported in PyTorch graph mode"* ([mstt README](https://raw.githubusercontent.com/Ascend/mstt/master/profiler/example/mstx_torch_plugin/README.md)). Neither emits prefill/decode.
4. **What does change per phase, and is usable:**
   - **Decode is visually periodic.** *"for inference tasks, one decode cycle is very obviously identifiable in the profile — operator blocks repeat periodically"* ([Ascend-Inference-wiki §4](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). A parser can detect phase by periodicity/batch-shape fingerprinting: prefill steps have large, variable `Input Shapes` on attention ops; decode steps repeat a near-identical op sequence with tiny shapes.
   - **`Input Shapes` / `Input Data Types` columns** in `kernel_details.csv` and `op_summary_*.csv` are the practical signal (e.g. `num_tokens` dimension collapsing to the batch size).
   - **Speculative decoding** produces distinguishable draft-model activity in the Service Profiler symbol set: domains `SpecDecode` with names `draft_propose`, `draft_model_forward`, `draft_compute_token_ids`, and `ModelForward` (`NPUModelRunner._model_forward`) ([`vllm_ascend/profiling_config.py`](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/profiling_config.py)). These are MS Service Profiler events (`chrome_tracing.json`), **not** `_ascend_pt` artifacts.
   - **vLLM framework markers** in the profiler, if you also collect the MS Service Profiler side: `EngineCore.step`, `EngineCore.step_with_batch_queue`, `batchFrameworkProcessing`, `modelExec`, `modelRunnerExec`, `NPUModelRunner._model_forward` with attributes `req_ids`, `dp_rank`, `npu_id` ([`vllm_ascend/profiling_config.py`](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/profiling_config.py)). Note this config also uses `vllm.v1.utils:record_function_or_nullcontext` for vLLM ≥ 0.15.0rc1 — the vLLM scheduler wraps work in `schedule: allocate_slots`, `schedule: make_cached_request_data`, `schedule: get_num_common_prefix_blocks`, `schedule: update_after_schedule` ([`scheduler_profiling_chunk.py`](https://raw.githubusercontent.com/vllm-project/vllm-ascend/main/vllm_ascend/core/scheduler_profiling_chunk.py)). **None of these name prefill or decode.**
5. **`Infer ID`** exists in Class B `op_summary_*.csv` but is documented as *"Inference iteration ID. (This field is not displayed by default or in single-operator scenarios.)"* — an iteration counter, not a phase label.
6. **Practical recommendation:** record the phase alongside the capture (e.g. two separate `--profiler-config` runs / two `start_profile`…`stop_profile` windows, one prefill-only, one decode-only) so the artifact set is labelled externally.

---

## 6. Communication (HCCL) and memcpy naming

### 6.1 Collective operator name strings — confirmed

| String | Where confirmed |
|---|---|
| `HcclAllreduce` | ✅ mstx data-format example: `{"streamId": "32","count": "25701386","dataType": "fp16","groupName": "group_name_43","opName": "HcclAllreduce"}` ([official guide mstx section](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)). This is the **mstx `opName`** for a collective. |
| `hcom_allReduce__428_0_1` | ✅ `COMMUNICATION_OP.opName` example ([msprof DB reference](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references_db.md)) |
| `hcom_broadcast_` | ✅ `COMMUNICATION_OP.opType` example — *"Operator type, which maps to `STRING_IDS(opType)`, such as `hcom_broadcast_`"* |
| `hcom_broadcast__303_1_1` | ✅ `CommAnalyzerBandwidth.hccl_op_name` example ([official guide §analysis.db](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)) |
| `HcclAllGather...` | ✅ Timeline op name in a real Insight screenshot description ([Ascend-Inference-wiki §6.3](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)) |
| `AllGather`, `ReduceScatter` | ✅ Timeline op names in real screenshots (`AllGather` in §5.4, `ReduceScatter` in §7.2) |
| `AllgatherMatmul`, `AllgatherMatmulAicpu`, `MatmulAllReduce`, `MatmulAllReduceAddRmsNormAicpu` | ✅ MC² fused operators: *"The operator name on the communication stream follows the format of fused operator name + **Aicpu**"* ([msprof](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references.md)) |
| `hcom_allReduce__123_0_1` + `group_dp_0` + `(45.0, 256.0, "RDMA")` | ✅ mock-data generator tuple ([AtomGit msagent](https://atomgit.com/Ascend/msagent/blob/9f983f9e0881cd532d253e0bcd37478a31143ea8/tests/model/generate_mock_data.py)) |
| `AllReduce`, `AlltoAll`/`AlltoAllV`, `Broadcast`, `Reduce`, `Send`, `Recv`, `hccl_allreduce_`, `hcom_all_gather_`, `Notify Wait`, `Notify Record`, `Memcpy`, `Wait`, `Barrier` | ❌ **Unconfirmed** as literal emitted strings in these artifacts. Real traces show `alltoall`/`alltoallv` as *sub-operations inside fused ops* (`dispatch` = permute1 + alltoallv + permute2; `combine` = unpermute1 + alltoallv + unpermute2) ([Ascend-Inference-wiki §5.2/§5.4](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). Treat as unconfirmed spellings. |
| `EVENT_WAIT`, `AivKernel`, `MoeDistributeDispatchV2` | ✅ Op names observed in real inference traces ([Ascend-Inference-wiki §5.4/§7.2](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). `EVENT_WAIT` appears abundantly and is partly graph-mode-generated, partly communication-generated. |

**Naming grammar.** Large/collective ops are named `hcom_<collective>_<suffix>` (e.g. `hcom_allReduce__428_0_1`); mstx reports the CamelCase form (`HcclAllreduce`); fused MC² ops concatenate compute+comm names and append `Aicpu` on the communication stream.

### 6.2 `Task Type` / `Type` / `Accelerator` values

| Value | Status |
|---|---|
| `AI_CORE`, `AI_VECTOR_CORE`, `AI_CPU`, `CCU`, `DPU` | ✅ Documented `Task Type` value set in `op_summary_*.csv` |
| `communication` | ✅ Documented: *"Operators with the `communication` task type usually consist of a sequence of communication tasks, each with an independent `Task ID` and `Stream ID`. Since these individual identifiers are not displayed here, the `Task ID` and `Stream ID` for this type of operator are marked as `N/A`."* ⇒ **a parser must tolerate `N/A` in numeric columns** |
| `AI_Core` (mixed case) | ✅ Documented in a `MatMul`→MIX note: *"the **Task Type** will change from **AI_Core** to **MIX_AIC**"* |
| `MIX_AIC` | ✅ Documented (CANN) |
| `AI_CORE`, `AI_VECTOR_CORE`, `MIXED_AIC` | ✅ For the `Accelerator` column of `kernel_details.csv` in a real DeepSeek trace — *"`AI_VECTOR_CORE` = pure VECTOR op; `AI_CORE` = CUBE op; `MIXED_AIC` = both CUBE and VECTOR"* ([Ascend-Inference-wiki](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). ⚠️ **`MIXED_AIC` here vs `MIX_AIC` in CANN docs — both spellings occur. Do not normalize.** |
| `aicore` (lowercase) | ✅ `SAMPLE_PMU_TIMELINE` / `SAMPLE_PMU_SUMMARY` `coreType` maps to `STRING_IDS(coreType)` with values "AIC or AIV" per the doc text |
| `aiv` / `aic` | ✅ `*_` prefix in `op_summary` PMU column names; `coreType` values |
| `HCCL`, `DVPP`, `FFTS`, `AICPU`, `AIVECTOR`, `CCU`, `ROCE`, `HCCP`, `TS`, `GE`, `CCE`, `SDMA`(no) | `HCCL`, `DVPP`, `FFTS`, `AICPU`, `AIVECTOR`, `CCU`, `ROCE`, `HCCP`, `TS`, `GE`, `CCE` are ✅ confirmed as members of the **`ENUM_MODULE`** table (component names), **not** `Task Type` values. Do not conflate. |
| `AI_VECTOR_CORE` for SIMT ops | ✅ *"For a SIMT operator, the value is fixed to `AI_VECTOR_CORE`"* (Ascend 950 `Task Type`); and `FUSION` for fusion tasks with `fusion_task_type` ∈ {`AICORE`, `AIVECTORCORE`, `AICPU`} |
| `MIX_AIV`, `MIX_AICORE`, `FFTS_PLUS`, `MEMCPY`, `MEMCPY_ASYNC`, `SDMA`, `PCIE_DMA`, `HMCCS`, `MC2`, `SYSTEM`, `Memset`, `Event`, `Barrier`, `RTS` | ❌ **Unconfirmed** as literal `Task Type` values in these artifacts |

### 6.3 H2D / D2H copies

Confirmed via the **`ENUM_MEMCPY_OPERATION`** enum table ([msprof DB reference](https://raw.githubusercontent.com/Ascend/msprof/master/docs/en/user_guide/profile_data_file_references_db.md)):

| id | name |
|---|---|
| 0 | `host to host` |
| 1 | `host to device` |
| 2 | `device to host` |
| 3 | `device to device` |
| 4 | `managed memory` |
| 5 | `addr device to device` |
| 6 | `host to device ex` |
| 7 | `device to host ex` |
| 65535 | `other` |

Consumed by the **`MEMCPY_INFO`** table (`globalTaskId`, `size`, `memcpyOperation`), controlled by `--runtime-api`, joined to `TASK` by `globalTaskId`. Access from the msprof CLI is `--runtime-api`.

- The copied-data-size/direction data therefore lives in **`MEMCPY_INFO` in `msprof_*.db` and the `ascend_pytorch_profiler_{Rank_ID}.db`** — as **integer ids** that must be resolved through `ENUM_MEMCPY_OPERATION`.
- ❌ The literal strings **`MEMCPY_H2D` / `MEMCPY_D2H`** were **not found in any official source**. They may be MindStudio Insight UI labels or a different CANN version's spelling — **do not assume they appear in files.**
- ❌ `aclrtMemcpy` / `aclrtMemcpyAsync` as row values in `api_statistic.csv`: **unconfirmed**, though plausible since `api_statistic` documents a `Runtime` level covering "CANN runtime APIs" and `api_statistic` covers `AscendCL`/`Runtime`/`Node`/`Model`/`Communication` layers.
- Memcpy as a timeline entity: `Ascend Hardware` stream tasks include memcpy tasks whose `args` carry `transport_type` ∈ {`LOCAL`, `SDMA`, `RDMA`, `UB`, `RoCE`} etc. per the DPU table; and `HcclRepo`/`HardWareRepo` both read the `TASK` table with `linkType` ∈ {`HCCS`, `PCIE`, `RoCE`, `UBoE`, `SIO`, `HCCS_SW`, `STANDARD_ROCE`, `UB`, `ON_CHIP`} for communication.

### 6.4 `communication.json` / `communication_matrix.json` structure

**Unconfirmed as JSON.** No official source fetched shows the literal JSON key layout of these two files. What *is* officially documented is the equivalent **`analysis.db` table schema**, which is the authoritative field model ([official guide, `analysis.db Data`](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)):

**`CommAnalyzerBandwidth`** (large/communication macro ops)
```
hccl_op_name TEXT, group_name TEXT, transport_type TEXT, transit_size NUMERIC (MB),
transit_time NUMERIC (ms), bandwidth NUMERIC (GB/s), large_packet_ratio NUMERIC,
package_size NUMERIC (MB), count NUMERIC, total_duration NUMERIC, step TEXT, type TEXT
```
`transport_type` ∈ `LOCAL`, `SDMA`, `RDMA`; `step` formatted like `step12`; `type` ∈ `Collective`, `P2P`.

**`CommAnalyzerTime`** (per-op timing)
```
hccl_op_name TEXT, group_name TEXT, start_timestamp NUMERIC (us), elapse_time NUMERIC (ms),
transit_time NUMERIC (ms), wait_time NUMERIC (ms), synchronization_time NUMERIC (ms),
idle_time NUMERIC (ms), step TEXT, type TEXT
```
`idle_time = elapse_time - transit_time - wait_time`. **`wait_time` is explicit here** — this is the authoritative source for communication wait time.

**`CommAnalyzerMatrix`** (the matrix view)
```
hccl_op_name TEXT, group_name TEXT, src_rank TEXT, dst_rank TEXT, transport_type TEXT,
transit_size NUMERIC (MB), transit_time NUMERIC (ms), bandwidth NUMERIC (GB/s),
step TEXT, type TEXT, op_name TEXT
```
`hccl_op_name` here is the *simplified* name after matrix analysis (e.g. `send-top1`); `op_name` is the original (e.g. `hcom_broadcast__303_1_1`). `group_name` is a *communication-domain hash ID* in the matrix table but a human-readable group name elsewhere.

⚠️ **Unit inconsistency to watch:** `start_timestamp` in **µs** while every other time field in the same table is in **ms**.

**`StepTraceTime`** (the DB form of `step_trace_time.csv`, all in **ms**):
```
deviceId INTEGER, step TEXT, computing NUMERIC, communication NUMERIC, overlapped NUMERIC,
communication_not_overlapped NUMERIC, free NUMERIC, stage NUMERIC, bubble NUMERIC,
communication_not_overlapped_and_exclude_receive NUMERIC
```
Note `step` is **TEXT** here (e.g. `12`) vs a numeric-looking `Step` column in the CSV.

### 6.5 `COMMUNICATION_OP` / `COMMUNICATION_TASK_INFO` (Class B DB)

`COMMUNICATION_OP` — large communication operators; `opName` e.g. `hcom_allReduce__428_0_1`, `opType` e.g. `hcom_broadcast_`, `groupName` e.g. `10.170.22.98%enp67s0f5_60000_0_1708156014257149`, plus `startNs`, `endNs`, `connectionId`, `opId`, `relay`, `retry`, `dataType`, `algType`, `count`, `deviceId`.

`algType` values: `MESH`, `RING`, `NB`, `HD`, `NHR`, `PIPELINE`, `PAIRWISE`, `STAR`; may be multi-phase, e.g. `HD-MESH`.

`COMMUNICATION_TASK_INFO` — small communication operators: `timestampNs`, `name`, `globalTaskId`, `taskType`, `planeId`, `groupName`, `notifyId`, `rdmaType` (`RDMASendNotify`/`RDMASendPayload`), `srcRank`, `dstRank`, `transportType`, `size`, `dataType`, `linkType`, `opId`, `isMaster` (0 = secondary stream, 1 = primary stream), `bandwidth` (bytes/s).

⇒ **`Notify` appears as `notifyId`, not as an operator named "Notify Wait"** — which is why `Notify Wait` remained unconfirmed.

---

## 7. Observed sample rows

### 7.1 ✅ A real, complete, committed trace fixture (use this as your primary test vector)

The official **`Ascend/mstt`** repository commits a real Ascend `trace_view.json` (≈635 KB):
`profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json`
— [raw](https://raw.githubusercontent.com/Ascend/mstt/master/profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json).

Its literal opening bytes (verbatim):
```json
[{"ph": "X", "name": "aten::empty", "pid": 437675, "tid": 437675, "ts": "1704161511420306.491", "dur": 13.08, "cat": "cpu_op", "args": {"Sequence number": -1, "Fwd thread id": 0}}, ...
```
This single line proves: bare array root; `ts` = µs **decimal string**; `dur` = µs **float**; numeric `pid`/`tid`; `cat: "cpu_op"`; and the exact minimal host-event `args` pair `Sequence number` / `Fwd thread id`. Backward-thread ops in the same file use a different `tid` (e.g. `439228`) with `Fwd thread id: 1`.

### 7.2 ✅ Real counter (`ph:"C"`) events — Class B

```json
{"processName": "APP/DDR", "ts": "1707359574357536.879", "pid": 1717664, "tid": 0, "args": {"KB": 0.0}, "ph": "C"}
{"processName": "APP/HBM", "ts": "1707359574357536.879", "pid": 1717664, "tid": 0, "args": {"KB": 9069036.0}, "ph": "C"}
{"processName": "write_ost", "ts": "1707359579320538.120", "pid": 512, "tid": 0, "args": {"value": 0, "acc_id": 2}, "ph": "C"}
```
Verbatim from [MindStudio Insight Timeline design doc §4.5.3.2](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Timeline.md).

### 7.3 ✅ Real metadata (`ph:"M"`) event shapes — torch_npu-emitted

```json
{"ph":"M","name":"process_name",      "pid":pid,"tid":0,  "args":{"name":"Python"}}
{"ph":"M","name":"process_labels",    "pid":pid,"tid":0,  "args":{"labels":"CPU"}}
{"ph":"M","name":"process_sort_index","pid":pid,"tid":0,  "args":{"sort_index":0}}
{"ph":"M","name":"thread_name",       "pid":pid,"tid":tid,"args":{"name":"Thread <tid>"}}
{"ph":"M","name":"thread_sort_index", "pid":pid,"tid":tid,"args":{"sort_index":<tid or max(tid)+1>}}
```
Plus flow-event shapes: `{"ph":"s"|"f","bp":"e","name":"torch_to_npu","id":<kernel ts in ns>,"ts":<µs str>,"cat":"async_npu"}`, `... "name":"enqueue_to_dequeue","id":<corr_id>,"cat":"async_task_queue"`, `... "name":"fwdbwd","id":<sequence number>,"cat":"fwdbwd"`.

### 7.4 ✅ Real mstx payload — Class A (communication instrumentation)

```
Format: {"streamId": "{pg streamId}","count": "{count}","dataType": "{dataType}",["srcRank": "{srcRank}"],["destRank": "{destRank}"],"groupName": "{groupName}","opName": "{opName}"}
Example: {"streamId": "32","count": "25701386","dataType": "fp16","groupName": "group_name_43","opName": "HcclAllreduce"}
```
([official guide](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md))

### 7.5 ✅ Real timeline-query payload — shows actual `pid`/`tid`/`metaType` values

```json
{
  "rankId": "ubuntu8438122216155992192_0 0",
  "tid": ["272_0"],
  "pid": "HCCL",
  "startTime": 1531153458,
  "endTime": 3248708207,
  "name": "Reduce_Inline",
  "wallDuration": 583860,
  "metaType": "HCCL",
  "count": 194,
  "field": "duration", "order": "descend",
  "total": 195, "current": 1, "pageSize": 10, "orderBy": "duration"
}
```
([Timeline design doc §4.6.1](https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Timeline.md))

### 7.6 ✅ Real counter `args` shapes per counter family (Class B)

```json
{"timestamp": 20534090, "value": {"Read(B/s)": 115726466}}
{"timestamp": 20534090, "value": {"freq(Mhz)": 115726466, "usage(%)": 32, "totalCycle": 115726466}}
{"timestamp": 20534090, "value": {"txThroughput(B/s)": 11572.6466, "rxThroughput(B/s)": 11572.6466}}
{"timestamp": 20534090, "value": {"Mhz": 115726466}}
```
Counter `processName` values: `APP/DDR`, `APP/HBM`, `APP/MEMORY`, `Device/DDR`, `Device/HBM`, `Device/MEMORY`, `{llcId} Read/Throughput`, `Read`, `Write`, `HCCS`, `PCIe_post`, `PCIe_nonpost`, `PCIe_cpl`, `PCIe_nonpost_latency`, `AI Core Freq`, `Port {funcId}/rx`, `Port {funcId}/tx`, `{hbmId}/Read`, `{hbmId}/Write`.

### 7.7 ⚠️ CSV data rows — still not recovered from a primary source

No verbatim CSV **data row** was recovered in this research. Official docs present sample CSVs only as **screenshots (PNG)** — `op_statistic_-csv.png`, `kernel_details.png`, `communication_statistic_-csv.png`, `op_summary_(example_only).png`, etc. — which are not machine-readable. Web search did not surface a public fixture with literal header+row CSV lines.

**Status of the headers themselves:** Class A headers for `operator_details.csv` (§3.5) and `step_trace_time.csv` (§3.7) are now **source-verbatim** from torch_npu's `_csv_headers.py` / parser modules; Class A `kernel_details.csv` is a **projection of Class B `op_summary_*.csv`** by `KernelViewParser` and is **not** listed among the files the official guide enumerates for `ASCEND_PROFILER_OUTPUT` (see §2.3). Class B column *sets* are documented but explicitly vary by collection parameters and product.

**Action for the parser author:** take a real `ASCEND_PROFILER_OUTPUT/` from your own CANN version, read the header line of each CSV, and assert it against a fixture you own. The documented column *sets* here tell you which columns to expect and which are optional; the literal header bytes must be verified empirically.

### 7.8 ✅ Confirmatory note on Class A field values from a real analyzer

`kernel_details.csv` in practice shows `Accelerator` (not `Accelerator Core`) and `Block Dim` (not `Block Num`), with `Accelerator` ∈ {`AI_CORE`, `AI_VECTOR_CORE`, `MIXED_AIC`}, and `Stream ID` values like `7` (shared-expert side stream), `8` (communication stream), `9` (main stream) ([Ascend-Inference-wiki](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)).

---

## 8. Real-world parsing pitfalls

### 8.1 Encoding and delimiters
- **BOM.** Class A CSVs are written by a Python toolchain and commonly carry a UTF-8 BOM. **A naive `line.split(',')[0]` then yields `\ufeffDevice_id`, silently breaking column lookup.** Read with `encoding="utf-8-sig"` and normalise the first header cell.
- **GBK / UTF-8.** Chinese-locale CANN toolchains may emit **GBK/GB18030** rather than UTF-8 (especially in Chinese-language Windows deployments and older versions). Official docs are Chinese-first; the toolchain is Chinese-authored. There is **no official statement of the CSV encoding** in the sources reviewed — treat encoding as *undetermined per installation*: sniff, and fall back GBK → UTF-8 (`errors="replace"`) rather than assuming one. Do **not** assume UTF-8 only.
- **Comma-in-field quoting.** Shape/dtype/format columns contain semicolons as intra-cell tensor separators (`Input Shapes` uses `;` per dimension **and** between tensors — the doc explicitly notes *"If the value of `Input Shapes` is empty (formatted as `; ; ; ;`), it indicates that the input is a scalar. The semicolon (`;`) serves as the delimiter for each dimension."*). `Call Stack` is captured Python source text and **will** contain commas and quotes. Use a real CSV parser (`csv`/`pandas`), never `split(',')`. Values like `N/A` appear in numeric columns.
- **`µ` vs `u` in unit names.** Most Class A CSVs are source-verified to use ASCII `(us)` — e.g. `operator_details.csv`, `step_trace_time.csv`, `kernel_details.csv`, `op_statistic.csv`. However the **official doc tables** transcribe some of these as `(µs)` with U+00B5 MICRO SIGN, and several Class A files (`memory_record.csv`, `npu_module_mem.csv`) use `(µs)` in the doc. **The two encodings coexist across files and across doc-vs-source**, so normalise header comparisons for both characters rather than trusting either spelling. Class B mostly uses `(us)`.
- **Inconsistent capitalisation.** `Device_id` (lowercase `id`) vs `Device ID` in prose vs `deviceId` in DB columns vs `Device Id` in trace `args`. `OP Type` vs `Op Name` vs `opName`. Match case-insensitively plus a normalised alias table.
- **Reserved-word-ish names.** `Name`, `Type`, `Step`, `Step Id`, `Count`, `Level`, `Device`, `Stage`, `Bubble`, `Free`, `Computing`, `Variance`, `Preparing` are all real column names that collide with common programming identifiers.

### 8.2 Size and JSON fragility
- `trace_view.json` for a multi-second, multi-rank LLM capture is **commonly 10²–10³ MB and can exceed several GB**. The official guide warns: *"Performance data occupies a certain amount of drive space, which may cause the risk of the server becoming unavailable due to a full drive... It is recommended that the performance data profiling time be within 5 minutes, and reserve at least 20 times the memory and drive space of the raw performance data size."* ([official guide, Data dump constraints](https://raw.githubusercontent.com/Ascend/pytorch/v2.7.1-26.1.0/docs/en/ascend_pytorch_profiler/ascend_pytorch_profiler_user_guide.md)). Practical consequence: **stream-parse the JSON (ijson/SAX) or `json.load` once with `mmap`; do not load it twice; do not retain every event as a Python dict if you only need aggregates.** `with_stack=True` (vllm-ascend's default) massively inflates the file because each event carries a `Call stack`.
- **The root is a bare array, and the file can be *truncated mid-array*.** `TracePreParser` deliberately writes the array **without its closing `]`**, and `TraceViewParser` appends chunks that eventually close it (§4.2). **A killed parse therefore leaves syntactically invalid JSON.** A parser must either tolerate EOF without `]`, repair by appending `]`, or report a clear diagnostic — a plain `json.load()` will throw and lose all partial data.
- `kernel_details.csv` is also routinely **hundreds of MB / millions of rows** (docs reference row indices like 4873/4874 for `operator_memory`, implying large tables). Use chunked readers.

### 8.3 Units
- **Microseconds dominate Class A CSVs**: `Duration(us)`, `Wait Time(us)`, `Start Time(us)`, `Task Duration(us)`, and all `step_trace_time.csv` fields.
- **Milliseconds** appear in: `analysis.db` (`transit_time`, `elapse_time`, `wait_time`, `synchronization_time`, `idle_time`, `StepTraceTime.*`), and in Chrome-trace UI conventions (`Start`/`Wall Duration` are described as **(ms)** in `chrome://tracing`).
- **Nanoseconds** appear in DB tables: the msprof DB reference states plainly *"Time: local Unix time, in **nanoseconds (ns)**"* — `TASK.startNs/endNs`, `COMMUNICATION_OP.startNs/endNs`, `CANN_API.startNs/endNs`, `MSTX_EVENTS.startNs/endNs`, `STEP_TIME.startNs/endNs`, `STEP_TIME` in `ascend_pytorch_profiler_{Rank_ID}.db`, `OP_MEMORY.allocationTime/releaseTime` (ns, per the guide's `escape`/memory section).
- **Memory unit flips**: `memory_record` is **MB** in Class A and **KB** in Class B; `npu_module_mem` likewise MB (A) vs KB (B). `operator_memory` is **KB** in the guide's Class A table (`Size(KB)`) but `OP_MEMORY` in the DB stores **bytes** (`size ... in Byte`). `memory_record.csv` Class A also carries a `Total Active (MB)` column not present in Class B.
- **Bandwidth**: `(GB/s)` in `op_summary` PMU columns and in `analysis.db` bandwidth fields; **bytes/s** in DB tables (`NIC.bandwidth`, `HCCS.txThroughput`, `ROCE.bandwidth`).
- **`Duration(us)` vs `Duration(ms)`**: both exist in the ecosystem, in different files — always read the unit out of the header, never assume.

### 8.4 Time base
- **Relative, per-capture — not absolute.** Trace `ts` values are offsets from the profiling session origin. The msprof DB is the exception: `SESSION_TIME_INFO.startTimeNs/endTimeNs` are **Unix time (ns)**, and `TASK.startNs` etc. are documented as *local Unix time* — so **DB timestamps and JSON `ts` are on different bases** and must not be mixed.
- **Cross-rank drift is real and expected.** From a practitioner's multi-card analysis: *"各卡 trace 起点不一致 ... 并不是真实的执行错位，需要手动把几个 moe 块拉齐"* — trace start points differ per card and are not a real execution misalignment; they must be manually aligned (MindStudio Insight's *Set Base Slice* + L/R keys). The recommended alignment anchor is an **independent collective** (e.g. `ReduceScatter`) aligned on its **right edge**, because a fused `dispatch`/`combine` op's true sync point is the inner `alltoall`'s right end, not the fused op's edges. Also: *"由于 dispatch 算子内部包含 alltoall 操作，而 alltoall 一定是全卡同步的"* ([Ascend-Inference-wiki §7.2](https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md)). **A parser that assumes a common t0 across ranks will produce wrong cross-rank deltas.**
- The msprof doc also warns of a **0–1 ms delay** between actual AI Core frequency change time and the software-monitored time, which makes recorded op durations around a frequency change inconsistent with reality.
- Class B `step_trace_*.json` mixes `ns` (in the JSON timeline) and `µs` (in the CSV) — the doc explicitly says `FP Start` etc. are **ns** in JSON and **µs** in CSV.

### 8.5 Semantic traps
- **`Duration` in `kernel_details.csv` includes scheduling wait time.** From a practitioner: *"`Duration` | 算子执行时间，**其中包含算子的调度等待时间**，这就是我们要优化、要削减的值"*. And `Task Duration(us)` is officially *"including the time spent scheduling the task to the accelerator, execution time on the accelerator, and the completion response time"*. So `Duration != pure compute time`. For pure compute use `aicore_time(us)` / `aiv_time(us)`, with the caveat that they are theoretical and wrong under frequency changes.
- **`Start Time` is not sorted.** Real files are not in chronological order: *"按 `Start Time` 排序——采集时记录顺序不一定是从前到后的，先排序保证时间轴正确"* — **always sort by `Start Time` before reconstructing a timeline.** This is one of the highest-impact parser bugs.
- **`N/A` in numeric columns** is normal and documented for: `Task ID`/`Stream ID` of `communication`-type ops, `OP Type`/`Task Type`/`Block Num`/`Input *`/`Output *` at `task_time=l0`, `Mix Block Num`/`HF32 Eligible`/`OP State` at `l0`, `FP Start`/`BP End` in offline inference, `size=0` for `notify` tasks, and per-product unsupported PMU fields.
- **`Block Num` / `Block Dim` is documented as inaccurate**: *"`Block Dim` | 算子使用的核数... 注意该值有时统计得并不准"*.
- **`Task ID`/`Stream ID` are `N/A` for communication ops** because a communication op is a *sequence* of tasks each with its own IDs — so any parser that groups by `(Task ID, Stream ID)` will collapse communication ops incorrectly.
- **A communication op is not one task.** Similarly, the `Communication` timeline track displays *only level-0 data* for MC² fusion.
- **`Input Shapes` for scalars** is `; ; ; ;` — non-empty but meaningless; don't treat as a real shape.
- **Empty vs negative operator memory.** `operator_memory.csv` legitimately contains negative and empty values when allocation/deallocation falls outside the profiling window — documented at length. Do not "clean" them.
- **`op_summary.csv` operator counts can change** for a reason unrelated to your model: on Atlas A2/A3, `MatMul` converts to a MIX operator under specific criteria (inner axis > 1000, theoretical MAC > 50 µs, inner axis not 516B-aligned), so the `MatMul` count *drops* and `Task Type` changes `AI_Core` → `MIX_AIC`.
- **`Cube`/`Vector` PMU data is only present at `task_time=l1` + `aic_mode=task-based`**; at `l0` the columns exist but read `N/A`. `SAMPLE_PMU_*` requires `sample-based`.
- **`msprof_*.json` "stores data within iterations. Data outside iterations is not displayed."** So the JSON is not a superset of the CSVs — a parser cannot use it to recover pre-iteration events.
- **`kernel_details.csv` Step Id may be absent/null** (warmup with async ops; or the CSV simply lacks the column if `schedule` wasn't used). Official guidance: *"If a StepID null value appears in kernel_details.csv, users can view the step information of that operator through the trace_view.json file, or re-collect Profiling data."*
- **`Device_id` in `api_statistic.csv` can be the string `host`**, not an integer.
- **`String`-typed numerics**: trace `ts` is a JSON **string** with a fractional part (`"1707359574357536.879"`); `analysis.db` `step` and `CommAnalyzer*` `step` are **TEXT** (`step12`); `src_rank`/`dst_rank` in `CommAnalyzerMatrix` are **TEXT** while the corresponding DB fields `srcRank`/`dstRank` are INTEGER.
- **`rank` vs `device_id`.** In Class A the rank is in the directory name; in Class B you must read `device_{id}/` or `RANK_DEVICE_MAP`. `RANK_DEVICE_MAP` documents `rankId` as *"fixed at –1"* in the PyTorch DB export and `deviceId = -1` when not collected — so **do not trust `RANK_DEVICE_MAP` to relate rank to device.** Also note Class B `Dst Rank = 4294967295` means a local on-chip (non-network) operation on non-Ascend-950 products.
- **Same logical table, two spellings across DBs.** `msprof_{timestamp}.db` and `ascend_pytorch_profiler_{Rank_ID}.db` share most tables; `analysis.db` holds the `CommAnalyzer*`/`StepTraceTime` tables. `META_DATA`/`SCHEMA_VERSION*` exist for versioning — **read `SCHEMA_VERSION` before applying any column assumptions.**
- **PMU tables are product-gated**: `TASK_PMU_INFO` is only collected on Atlas 200I/500 A2, Atlas A2 training, and Atlas A2 inference products. Presence of the file does not imply presence of data.
- **`data_simplification=True` deletes `mindstudio_profiler_output/` and `mindstudio_profiler_log/`.** If your pipeline expects `mindstudio_profiler_output/*.csv` after a vllm-ascend run, check this flag first — with simplification on you only get `PROF_*/` raw data and must re-parse.

---

## 9. Recommended parser strategy (summary of the above)

1. **Classify the artifact** by top-level directory: `*_ascend_pt` / `*_ascend_ms` ⇒ Class A (`ASCEND_PROFILER_OUTPUT/` is the target); `PROF_*` ⇒ Class B (`mindstudio_profiler_output/`, plus the `PROF_*/msprof_*.db`). Handle the legacy `device_<id>/summary/` and `device_<id>/timeline/` fallbacks.
2. **Enumerate one directory per rank** for Class A; never merge ranks by directory contents.
3. **Read headers, don't index positions.** Class B column sets vary by `--task-time`, `--aic-mode`, `--aic-metrics`, `--hccl`, `--ascendcl`, `--sys-io`, `--sys-interconnection`, and by product.
4. **Normalise with an alias table** covering `us`/`µs`, `Device_id`/`Device ID`/`deviceId`, `Block Num`/`Block Dim`, `Accelerator Core`/`Accelerator`, `MIX_AIC`/`MIXED_AIC`, `OP Type`/`Op Name`/`opName`.
5. **Treat `N/A`, `YES`/`NO`, `dynamic`/`static`, `host`, and empty cells as first-class values** in every column.
6. **Sort `kernel_details.csv` by `Start Time(us)` before any temporal reasoning.**
7. **Get the phase from the operator, not the artifact.** Ask the user, or infer from shape/periodicity fingerprints.
8. **Never mix time bases**: JSON `ts` = relative µs (string, fractional); Class A CSV = µs; Class B CSV = µs; DB = ns / Unix time.
9. **Assume multi-GB `trace_view.json` and stream it.**
10. **Assert against your own fixture.** The column *sets* here are documented; the literal header bytes must be verified on a real `ASCEND_PROFILER_OUTPUT/` from your CANN version.

---

## 10. Explicit "unconfirmed" register

| Item | Status |
|---|---|
| `mstx.json` file | ❌ **Does not exist.** mstx/msproftx data ⇒ `msprof_tx_*.json` / `msprof_tx_*.csv` (Class B) and the `MSTX_EVENTS` DB table. |
| `aicore_freq.csv` | ❌ **Does not exist.** AI Core frequency exists as the `AICORE_FREQ` DB table and the `AI Core Freq` timeline lane only. |
| `aic_metrics.csv` / standalone pipe-utilization CSV | ❌ Does not exist. Pipe ratios are appended **columns** in `op_summary_*.csv` / `kernel_details.csv`; separate files are `ai_core_utilization_*.csv` and `ai_vector_core_utilization_*.csv`. |
| `op_summary.csv` inside `ASCEND_PROFILER_OUTPUT/` | ❌ Not listed by the official guide, and `IntegrateParser.CSV_FILENAME_MAP` has no `OP_SUMMARY` entry. Class B only. |
| `communication_statistic.csv` inside `ASCEND_PROFILER_OUTPUT/` | ❌ Class B only (`communication_statistic_*.csv`). Class A uses `communication.json` / `communication_matrix.json` / `analysis.db`. |
| **`{"traceEvents": [...]}` wrapper** | ❌ **Wrong for Ascend.** The root is a **bare JSON array** (§4.1). Any parser requiring `traceEvents` will fail outright. |
| `displayTimeUnit`, `baseTime`, `deviceId`, `distributedInfo`, `systemTraceEvents`, `vizViewer`, separate metadata JSON | ❌ Do not exist in Ascend trace output (consequence of the bare-array root). Sidecar metadata is `profiler_metadata.json` / `profiler_info.json`. |
| `ph:"i"` and `ph:"B"/"E"` in Ascend traces | No evidence. (`B`/`E` *do* appear in Class B `step_trace_*.json` — a different file.) |
| `bp` values other than `"e"` | Unconfirmed; only `"e"` observed. |
| `pid` literal values `"Host"` / `"Device"` | ❌ Do not exist. Use `process_name` metadata `args.name`. |
| `args.FLOPs` | ❌ Wrong spelling — torch_npu writes **`flops`** (lowercase). |
| `args.task type`, `args.device id`, `args.stream id`, `args.connection_id` (as trace args) | ❌ `Task Type`, `Device Id`, `Stream Id` are the real spellings (used by mstt). `connection_id` is documented as a **panel field**; literal trace-JSON spelling not code-proven. |
| `args.step`, `args.Actual Time`, `args.aicore time`/`aicore_time`, `args.count`, `args.bandwidth`, `args.transport_type`, `args.group name`/`group_name`, `args.rank`, `args.op_type`, `args.Thread id`, `args.Output Shapes`, `args.Output type` | ❌ Not Ascend trace `args`. They are CSV columns (`aicore_time`), DB columns (`count`, `bandwidth`, `transportType`, `groupName`, `rankId`, `taskType`), or slice names (`ProfilerStep#<n>`). |
| Ascend `ph:"M"` metadata names beyond the five confirmed | ❌ No evidence for `Ascend Profile level`, `Device Index`, `Device ID`, `soc_version`, `step`. |
| Exact per-event `args` for **Overlap Analysis / AI Core Freq / Communication** lane slices | Unconfirmed (slice *names* `Computing`, `Communication(Not Overlapped)`, `Free` are confirmed; full device-side arg dicts are not). |
| `MEMCPY_H2D` / `MEMCPY_D2H` literal strings | Not found; the documented enum uses `host to device` / `device to host` (plus `... ex` variants), as integer ids in the DB. |
| `Task Type` values `MIX_AIV`, `FFTS_PLUS`, `MEMCPY`, `SDMA`, `PCIE_DMA`, `SYSTEM`, `Memset`, `Event`, `Barrier`, `RTS` | Not confirmed. |
| HCCL op names `AllReduce`, `AlltoAll`, `Broadcast`, `Send`, `Recv`, `hccl_allreduce_`, `hcom_all_gather_`, `Notify Wait`, `Notify Record` | Not confirmed as literal emitted strings. |
| `aclrtMemcpy` / `aclrtMemcpyAsync` rows in `api_statistic.csv` | Not confirmed (plausible; `Runtime` level exists). |
| `communication.json` / `communication_matrix.json` internal JSON key layout | Partially confirmed — the key *strings* `Bandwidth(GB/s)`, `Transit Time(ms)`, `Transit Size(MB)`, `Transport Type`, `Op Name`, `Total Op Info`, `Communication Time Info`, `Communication Bandwidth Info`, `Start Timestamp(us)`, `Wait Time(ms)`, `Synchronization Time(ms)`, `Size Distribution`, `Large Packet Ratio`, `Elapse Time(ms)`, `Idle Time(ms)` are confirmed as belonging to these files; the full nested object structure is not. Use the `analysis.db` `CommAnalyzer*` schemas as the field model. |
| Exact literal CSV header lines (byte-for-byte) for every file | Not published. Official docs provide screenshots only. Class A `operator_details.csv` and `step_trace_time.csv` headers **are** source-verbatim; verify the rest against a real capture. |
| `ascend_pytorch_profiler_0.db` `_0` suffix rule | Doc-observed only; the source constant is `ascend_pytorch_profiler.db`. Glob `ascend_pytorch_profiler*.db`. |
| `logs/` dir contents; `host/info.json`, `host_start.log`, `start_info` field layout | Confirmed to *exist* (source), contents not fully enumerated. `start_info` is Python-dict-literal text with `freq`, `start_cnt`, `start_monotonic`, `syscnt_enable`. |
| `summary/` and `timeline/` contents | Deleted by default; only referenced as removal targets in torch_npu source. |
| vLLM-Ascend `/start_profile` body / `profile_prefix` HTTP param | ❌ Does not exist in v0.6.6 / v0.8.5 / v0.9.1 / main. `profile_prefix` is Python-only. |
| Exact upstream vLLM release that removed `VLLM_TORCH_PROFILER_DIR` | Doc says "January 19, 2026, vLLM mainline"; the specific PR/tag was not pinned. |
| `VLLM_TORCH_PROFILER_WITH_MODULES`, `VLLM_TORCH_PROFILER_WITH_SHAPES` | ❌ No evidence they ever existed. Real name is `VLLM_TORCH_PROFILER_RECORD_SHAPES`. |
| Official hiascend JS-rendered pages (`atlasprofiling_16_0035.html`, `atlasprofiling_16_0072.html`, `atlasprofiling_16_1149.html`, …) | Could not be extracted — they return only a nav shell to non-browser clients. Equivalent content was obtained from the `Ascend/msprof` and `mindstudio-docs` GitHub markdown sources behind them. |
