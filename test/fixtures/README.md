# Test fixtures

## `ascend-trace_view.sample.json` (committed)

A **truncated prefix** of a real Ascend profiling trace:

* Source: [`Ascend/mstt`](https://github.com/Ascend/mstt) —
  `profiler/msprof_analyze/test/ut/advisor/advisor_backend/timeline_advice/trace_view.json`
* License: Apache License 2.0 (see the repository `LICENSE` third-party notices)
* Size: ~100 KB of the original ~635 KB
* Modification: cut off mid-event, with the tool wrapper that produced the local
  copy stripped. Nothing else was changed.

It is the single most valuable fixture in this suite because it is real, and it
pins down three parser behaviours that documentation alone gets wrong:

1. the file is a **bare JSON array** — there is no `{"traceEvents": [...]}` wrapper;
2. `ts` is a **decimal string** (`"1704161511420306.491"`) while `dur` is a number;
3. the **closing bracket is missing**, exactly as an interrupted export leaves it.

`test/pipeline.test.mjs` asserts all three, including that the truncation is
disclosed in the parse warnings.

## `decode-comm-bound/`, `prefill-compute-bound/`, `host-schedule-bound/` (generated)

Three synthetic vLLM-Ascend capture sets, one per bottleneck class the analyzer
must distinguish. They are **not committed**: `node test/make-fixture.mjs`
regenerates them deterministically (seeded PRNG), and `node test/all.test.mjs`
creates them on first run if they are missing.

Each directory mirrors a `torch_npu` export:

```
trace_view.json            bare array, ts as decimal strings, device + host events
kernel_details.csv         Device_id,Name,Type,Accelerator Core,Start Time(us),...,mac_ratio,mte2_ratio
operator_details.csv       Host/Device Self+Total durations
op_statistic.csv           per operator type aggregates
step_trace_time.csv        Computing / Communication(Not Overlapped) / Overlapped / Free + Stage
profiler_info_0.json       capture metadata
communication.json         HCCL summaries
```

| Scenario | Shape | Expected verdict |
| --- | --- | --- |
| `decode-comm-bound` | 24 steps, a small-message AllReduce per layer that cannot overlap, high device busy, low MAC | 跨卡通信瓶颈 (score ≥ 40, outranks host) |
| `prefill-compute-bound` | 6 long chunked-prefill steps dominated by MatMul/attention, MAC 0.72 | NPU 计算瓶颈, subtype `compute-bound` |
| `host-schedule-bound` | 20 eager decode steps, ~1500 host ops per step, device idle gaps, low MAC | Host 调度瓶颈 (primary), copy as secondary |

The generator lives in `test/make-fixture.mjs`; its header documents every knob.
Regenerate with:

```sh
node test/make-fixture.mjs            # writes into test/fixtures/
node test/make-fixture.mjs D:\tmp\fx  # or elsewhere, e.g. to try the page on your own
```

The generated sets are also handy as **demo data** for the analyzer page: ingest
one by path (the page's "按路径分析" box) instead of uploading.
