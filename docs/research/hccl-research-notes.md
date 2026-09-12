# HCCL / MEMCPY in Ascend profiling — raw findings notes

## constant.py (mstt) — https://raw.giteeusercontent.com/ascend/mstt/raw/master/profiler/msprof_analyze/prof_common/constant.py
- COMM_JSON = "communication.json"; COMM_MATRIX_JSON = "communication_matrix.json"
- CLUSTER_COMM_JSON = "cluster_communication.json"; CLUSTER_COMMUNICATION_MATRIX_JSON = "cluster_communication_matrix.json"; COMMUNICATION_GROUP_JSON = "communication_group.json"
- P2P="p2p"; COLLECTIVE="collective"; TOTAL="total"; STEP_ID="step_id"; RANK_ID="rank_id"; GROUP_NAME="group_name"
- COMM_OP_TYPE="comm_op_type"; COMM_OP_NAME="comm_op_name"; COMM_OP_INFO="comm_op_info"
- TOTAL_OP_INFO="Total Op Info"; COMMUNICATION_TIME_INFO="Communication Time Info"; START_TIMESTAMP="Start Timestamp(us)"; COMMUNICATION_BANDWIDTH_INFO="Communication Bandwidth Info"
- HCOM_SEND="hcom_send"; HCOM_RECEIVE="hcom_receive"
- SYNCHRONIZATION_TIME_RATIO="Synchronization Time Ratio"; SYNCHRONIZATION_TIME_MS="Synchronization Time(ms)"
- WAIT_TIME_RATIO="Wait Time Ratio"; TRANSIT_TIME_MS="Transit Time(ms)"; TRANSIT_SIZE_MB="Transit Size(MB)"
- SIZE_DISTRIBUTION="Size Distribution"; WAIT_TIME_MS="Wait Time(ms)"; OP_NAME="Op Name"; BANDWIDTH_GB_S="Bandwidth(GB/s)"
- ELAPSE_TIME_MS="Elapse Time(ms)"; IDLE_TIME_MS="Idle Time(ms)"; LARGE_PACKET_RATIO="Large Packet Ratio"; TYPE="type"
- TRANSPORT_TYPE="Transport Type"; DATA_TYPE="data_type"
- NOTIFY_RECORD="Notify_Record"; NOTIFY_WAIT="Notify_Wait"   <-- underscore
- Bars: NPU_BAR="Ascend Hardware"; COMM_BAR="Communication"; OVERLAP_BAR="Overlap Analysis"
- Overlap events: COMPUTING_EVENT="Computing"; FREE_EVENT="Free"; UNCOVERED_COMMUNICATION_EVENT="Communication(Not Overlapped)"
- MC2_TIME="mc2"; MC2_COMPUTING="mc2_p"; MC2_COMMUNICATION="mc2_m"; MC2_NUMBER="mc2_num"
- Advisor: TASK_TYPE="Task Type"; AI_CORE="AI_CORE"; AI_CPU="AI_CPU"; MIX_AIC="MIX_AIC"
- KERNEL_DETAILS_CSV="kernel_details.csv"; STEP_TIME_CSV="step_trace_time.csv"; PT_PROF_SUFFIX="ascend_pt"
- HCCL_EVENT="hccl_event"; OVERLAP_ANALYSIS_EVENT="overlap_event"; KERNEL_EVENT="kernel_event"; FWD_BWD_FLOW="fwd_to_bwd"; TORCH_TO_NPU_FLOW="torch_to_device"
- DB tables: COMMUNICATION_OP; COMMUNICATION_TASK_INFO; COMMUNICATION_SCHEDULE_TASK_INFO; CommAnalyzerBandwidth; CommAnalyzerTime; CommAnalyzerMatrix; CommunicationGroup; ClusterCommAnalyzerMatrix
- ProfilerTableConstant: OP_ID="opId"; OP_NAME="opName"; START_NS="startNS"; END_NS="endNS"; CONNECTION_ID="connectionId"; GROUP_NAME="groupName"; RELAY="relay"; RETRY="retry"; DATA_TYPE="dataType"; ALG_TYPE="algType"; COUNT="count"; OP_TYPE="opType"; WAIT_NS="waitNS"
- INVALID_RANK_NUM = 4294967295

## str_constant.py (msprof msparser) — https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d.../analysis/common_func/ms_constant/str_constant.py
- TRANSIT_TYPE = ["RDMA","HCCS","PCIE","SDMA","SIO"]  (5 keys of "Communication Bandwidth Info")
- TOTAL = "Total HCCL Operators"
- ON_CHIP="ON_CHIP"; HCCS="HCCS"; PCIE="PCIE"; SIO="SIO"; HCCS_SW="HCCS_SW"; STANDARD_ROCE="STANDARD_ROCE"; RDMA="RDMA"; SDMA="SDMA"; LOCAL="LOCAL"
- NOTIFY_WAIT = "Notify_Wait"
- RDMA_PAYLOAD_PREPARE="RDMA_PAYLOAD_PREPARE"; RDMA_SEND_PAYLOAD="RDMA_SEND_PAYLOAD"; RDMA_PAYLOAD_ACK="RDMA_PAYLOAD_ACK"
- REDUCE_TBE="Reduce TBE"; RDMA_SEND="RDMASend"
- SDMA_TRANSIT_ITEMS = ["Memcpy", "Reduce_Inline"]   <-- literal "Memcpy" op name inside HCCL op
- COMMUNICATION_TIME_INFO="Communication Time Info"; COMMUNICATION_BANDWIDTH_INFO="Communication Bandwidth Info"
- AYNC_MEMCPY = "MemcpyAsync"
- HOST_TO_DEVICE="HostToDevice"; MSTX="MsTx"; ASYNC_NPU="async_npu"; ASYNC_ACL_NPU="async_acl_npu"
- LEVEL_MAP = {"acl":"AscendCL","runtime":"Runtime","model":"Model","node":"Node"}
- AICPU_KERNEL="AicpuKernel"; AIV_KERNEL="AivKernel"
- OpAnalysisType: START_TIME='Start Timestamp(us)'; ELAPSE_TIME="Elapse Time(ms)"; TRANSIT_TIME="Transit Time(ms)"; WAIT_TIME="Wait Time(ms)"; SYNCHRONIZATION_TIME="Synchronization Time(ms)"; IDLE_TIME='Idle Time(ms)'; WAIT_TIME_RATIO="Wait Time Ratio"; SYNCHRONIZATION_TIME_RATIO="Synchronization Time Ratio"
- OpBandWidthType: TRANSIT_SIZE_MB="Transit Size(MB)"; TRANSIT_TIME_MS="Transit Time(ms)"; BANDWIDTH_GB_S="Bandwidth(GB/s)"; BANDWIDTH_UTILIZATION="Bandwidth(Utilization)"; LARGE_PACKET_RATIO="Large Packet Ratio"; SIZE_DISTRIBUTION="Size Distribution"
- CommunicationMatrixInfo: SRC_RANK="Src Rank"; DST_RANK="Dst Rank"; TRANSPORT_TYPE="Transport Type"; TRANSIT_SIZE_MB="Transit Size(MB)"; TRANSIT_TIME_MS="Transit Time(ms)"; BANDWIDTH_GB_S="Bandwidth(GB/s)"; BANDWIDTH_UTILIZATION="Bandwidth(Utilization)"; LARGE_PACKET_RATIO="Large Packet Ratio"
- TransportType(IntEnum): HCCS=0; PCIE=1; RDMA=2; LOCAL=3; SIO=4
- TASK_TYPE_MAPPING numeric -> text (raw device task types), includes:
  "0" kernel AI core task; "1" kernel AI cpu task; "2" event record task; "3" stream wait event task; "4" fusion issue task;
  "5" memory copy task; "6" maintenance task; "7" create stream task; "8" kernel data dump task; "9" event notify task;
  "10" pctrace enable task; "11" create L2 addr task; "12" model maintaince task; "13" model execute task;
  "14" notify wait task; "15" notify record task; "16" HCCL rdma cpy task; "17" L2 SDMA memory copy task;
  "18" stream switch task; "19" stream active task; "20" label set task; "21" label switch task; "22" label goto task;
  "23" profiler trace task; "24" event reset task; "25" HCCL rdma db cpy task; "26" profiler trace task;
  "50" stars common task; "51" ffts task; "52" ffts plus task; "64" profiling enable task; "65" profiling disable task;
  "66" AI vector task; "67" add model end graph task; "68" AICPU schedule task; ...; "116" task update task
- DVPP_ENGINE_TYPE: 0 VDEC,1 JPEGD,2 PNGD,3 JPEGE,4 VPC,5 VENC,6 SCD
- Msvp data types: AIC="aic"; AIV="aiv"; HWTS="hwts"; DVPP_DATA="dvpp"; NIC_DATA="nic"; ROCE_DATA="roce"; AICPU="aicpu"; CTRL_CPU="ctrlcpu"

## communication_parser.py (msprof) — https://raw.githubusercontent.com/kali20gakki/msprof/2e1b851d.../analysis/msparser/cluster/communication_parser.py
- op_info[hccl_name][rank_id]["Communication Time Info"] = {...}; ["Communication Bandwidth Info"] = {...}
- op_info["total"] (StrConstant.TOTAL = "Total HCCL Operators") aggregated over all ops+ranks
- start time uses `is_master == 1` events, trans_into_local_time
- "Communication Bandwidth Info" keyed by 5 transport types HCCS/PCIE/SIO/SDMA/RDMA; SDMA = sum of PCIE+HCCS+SIO
- bandwidth key selection: link_type HCCS_SW -> "HCCS"; PCIE/HCCS/SIO -> itself; else -> "SDMA"
- event fields used: hccl_name, transport_type, link_type, is_master, plane_id, timestamp, duration, size, ldma/rdma_type, op_name, local_rank, remote_rank, name
- notify wait duration -> Synchronization Time(ms) (first wait) and Wait Time(ms)
- Idle Time(ms) = Elapse Time(ms) - Transit Time(ms) - Wait Time(ms)
- rdma_type literal: 'RDMA_SEND_PAYLOAD'
- event.name literals "Memcpy", "Reduce_Inline"

## meta_parser.py HcclAnalysisTool
- is_send_or_recv_op: 'send' in op_name.lower() or 'receive' in op_name.lower()
- StandardBandWidth by chip; MessageSizeThreshold per transport
- convert_to_enum: HCCS or HCCS_SW -> 0; PCIE -> 1; RDMA -> 2; LOCAL -> 3; SIO -> 4; else -1
- is_valid_link: local_rank != 0xffffffff (not None) and remote_rank is not None
- SizeDistribution: {size: [count, total_time_ms]}

## msprof Profile Data File Reference (EN) — https://raw.githubusercontent.com/mindstudio-docs/master/master/msprof/docs/en/user_guide/profile_data_file_references.md
- Directory: PROF_XXX/{host/data, device_{id}/data, msprof_{ts}.db, mindstudio_profiler_output/*}
- Communication + Overlap Analysis tracks only in multi-rank/multi-node/cluster scenarios
- Flow cats: async_npu, MsTx, async_task_queue, HostToDevice, fwd_bwd
- Communication track fields (Table 1):
  - Group * *Communication* (communication group name, as reported)
  - Plane ID
  - large-op info: rank_size, connection_id, data_type, alg_type (MESH/RING/NB/HD/NHR/PIPELINE/PAIRWISE/STAR), count, relay(yes/no), retry(yes/no)
  - small-op info: notify id (invalid => 18446744073709551615), duration estimated(us), stream id, task id, task type, src rank, dst rank (4294967295 = local on-chip on non-950), transport type (LOCAL/SDMA/RDMA/UB/RoCE), size(Byte), data type, link type (HCCS,PCIE,RoCE,UBoE,SIO,HCCS_SW,STANDARD_ROCE,UB,ON_CHIP), bandwidth(GB/s)
- Overlap Analysis fields: Communication, Communication(Not Overlapped), Computing, Free, Start, Wall Duration
- communication_statistic_*.csv fields: Device_id, OP Type, Count, Total Time(us), Min Time(us), Avg Time(us), Max Time(us), Ratio(%)
- op_summary Task Type: "Valid values include: `AI_CORE`, `AI_VECTOR_CORE`, `AI_CPU, CCU`, and `DPU`"
- op_summary: communication task type ops -> Task ID and Stream ID are N/A; OP State N/A for communication
- api_statistic track = CANN track; layers AscendCL/Runtime/Node/Model/Communication; Mode = ACL_OP/ACL_MODEL/ACL_RTS
- MC2: comm-stream op name = fused name + "Aicpu" e.g. AllgatherMatmulAicpu; comm track shows only level-0 data
- SIO fields: dat_rx/dat_tx/req_rx/req_tx/rsp_rx/rsp_tx/snp_rx/snp_tx
- op_summary note: MatMul MIX conversion changes Task Type from AI_Core to MIX_AIC

## Ascend-Inference-wiki profiling analysis — https://raw.githubusercontent.com/xuchi-0808/Ascend-Inference-wiki/master/docs/explanations/ascend-profiling-analysis.md
- ASCEND_PROFILER_OUTPUT contains kernel_details.csv, trace_view.json, op_statistic.csv, api_statistic.csv, operator_details.csv, step_trace_time.csv, communication.json, communication_matrix.json, analyse.done
- kernel_details.csv columns listed: Device_id, Model ID, Task ID, Stream ID, Name, Type, Op State, Accelerator, Start Time, Duration, Wait Time, Block Dim, Input Shapes, Input Data Types, Input Format, Output Shapes/..., Context ID
- Accelerator values: AI_VECTOR_CORE (pure VECTOR), AI_CORE (CUBE), MIXED_AIC (CUBE+VECTOR)
- op_statistic.csv columns: Device_id, OP Type, Core Type, Count, Total Time, Min Time, Avg Time, Max Time, Ratio(%)
- timeline lanes: Python, CANN (Runtime/RTS), Ascend Hardware, AI Core Freq, Communication, Overlap Analysis
- communication op name example: "HcclAllGather..." ; also AllGather, ReduceScatter, EVENT_WAIT
- op name examples: MoeDistributeDispatchV2, dequant_swiglu_quant, MatMulV2, dynamic_quant, HcPreInv..., HCPost_...

## MindStudio Insight msinsight design doc (Communication)
- URL: https://raw.githubusercontent.com/mindstudio-docs/master/master/msinsight/docs/zh/development_guide/design/Communication.md
- Interfaces: communication/matrix/bandwidthInfo, communication/duration/iterations, communication/matrix/group, communication/matrix/sortOpNames, communication/duration/operatorNames, communication/operatorLists, communication/duration/list, communication/operatorDetails, communication/distribution, communication/bandwidth
- code entries: modules/cluster/src/utils/RequestUtils.ts; server/src/modules/defs/ProtocolDefs.h; server/src/modules/communication/CommunicationPlugin.h
