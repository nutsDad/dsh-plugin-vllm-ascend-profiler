/**
 * Markdown report generation.
 *
 * The report is the analysis chain written out: bottleneck location, the
 * quantified evidence behind it, the vLLM-Ascend root causes, the prioritised
 * action plan, and the expected benefit — plus the profiling provenance and
 * every caveat the dataset carries, because a performance report without its
 * measurement caveats is a liability.
 *
 * @module dsh-plugin-vllm-ascend-profiler/report/markdown
 */

import { CATEGORIES, subtypeLabel } from '../model/classify.js';

/**
 * Render the full Markdown report.
 *
 * @param {object} input - report input.
 * @param {object} input.dataset - dataset from `buildDataset`.
 * @param {object} input.analysis - result of `analyzeDataset`.
 * @param {Record<string, string>} [input.charts] - chart snapshots as PNG data
 *   URLs, embedded so the exported Markdown is self-contained.
 * @param {object} [input.options] - `includeSections` toggles.
 * @returns {string} Markdown document.
 */
export function renderMarkdownReport({ dataset, analysis, charts = {}, options = {} }) {
  const sections = [
    header(dataset, analysis),
    sectionProvenance(dataset),
    sectionBottleneck(analysis),
    sectionEvidence(analysis),
    sectionRootCause(analysis),
    sectionActions(analysis),
    sectionBenefit(analysis),
    sectionCharts(charts),
    sectionTopOperators(dataset),
    sectionPhases(dataset),
    sectionCategories(dataset),
    sectionUtilization(dataset),
    sectionCaveats(dataset, analysis),
  ];
  void options;
  return sections.filter((part) => part !== '').join('\n\n');
}

function header(dataset, analysis) {
  const window = dataset.meta.window;
  return [
    '# vLLM-Ascend Profiling 性能分析报告',
    '',
    `- 生成时间：${new Date(analysis.generatedAt).toLocaleString('zh-CN')}`,
    `- 采集窗口：${((window.end - window.start) / 1000).toFixed(2)}ms（${String(dataset.meta.counts.deviceEvents)} 个设备事件 / ${String(dataset.meta.counts.hostEvents)} 个 Host 事件）`,
    `- 识别产物：${dataset.meta.validation?.profileType === 'vllm-ascend' ? 'vLLM-Ascend' : 'Ascend NPU'} profiling（证据权重 ${String(dataset.meta.validation?.weight ?? 0)}）`,
    `- 主导瓶颈：**${analysis.bottleneck.label}**（得分 ${analysis.bottleneck.score.toFixed(0)}/100，作用范围：${analysis.bottleneck.scopeLabel ?? analysis.bottleneck.scope ?? '全量'}）`,
    '',
    `> ${analysis.bottleneck.summary}`,
  ].join('\n');
}

function sectionProvenance(dataset) {
  const lines = ['## 0. 数据来源与解析说明', '', '| 文件 | 类型 | 大小 | 行/事件数 | 识别证据 |', '| --- | --- | --- | --- | --- |'];
  for (const file of dataset.meta.files) {
    lines.push(`| ${file.name} | ${file.kind} | ${formatBytes(file.bytes)} | ${file.events ?? file.rows ?? '-'} | ${file.markers.slice(0, 2).join('；') || '-'} |`);
  }
  lines.push('');
  if (dataset.meta.sampling?.applied === true) {
    const sampling = dataset.meta.sampling;
    lines.push(`- **采样解析已启用**：策略「${sampling.strategy}」，等距步长 ${String(sampling.stride)}，扫描 ${String(sampling.seen)} 个事件、保留 ${String(sampling.kept)} 个。泳道图为抽样结果；累计耗时优先取自 CANN 统计表（${dataset.meta.totalsSource}）。`);
  } else {
    lines.push(`- 时间线为全量解析；累计耗时来源：${dataset.meta.totalsSource}。`);
  }
  lines.push(`- 阶段划分来源：${dataset.phases.sourceLabel}（置信度 ${dataset.phases.confidence}）——${dataset.phases.reason ?? dataset.phases.confidenceReason}`);
  if (dataset.meta.encodingNotes.length > 0) lines.push(`- 编码提示：${dataset.meta.encodingNotes.join('；')}`);
  return lines.join('\n');
}

function sectionBottleneck(analysis) {
  const lines = ['## ① 瓶颈类型定位', '', analysis.steps.locate.summary, ''];
  lines.push('| 作用范围 | Host 调度 | NPU 计算 | 跨卡通信 | 数据拷贝 | 结论 |', '| --- | --- | --- | --- | --- | --- |');
  for (const scope of analysis.steps.locate.perScope) {
    const score = (id) => {
      const candidate = scope.candidates.find((entry) => entry.id === id);
      return candidate === undefined ? '-' : `${candidate.score.toFixed(0)}`;
    };
    lines.push(`| ${scope.scopeLabel} | ${score('host')} | ${score('compute')} | ${score('comm')} | ${score('copy')} | ${scope.verdict === 'balanced' ? '未达门限' : scope.primaryCandidate.label} |`);
  }
  lines.push('', '> 得分 = 门限化严重度加权（0–100）。≥40 视为达到瓶颈门限；未达门限的候选同样列出，便于判断"接近瓶颈"的程度。');
  lines.push('');
  for (const scope of analysis.steps.locate.perScope) {
    lines.push(`### ${scope.scopeLabel}`, '');
    lines.push(`- 主导瓶颈：**${scope.primaryCandidate.label}**（得分 ${scope.primaryCandidate.score.toFixed(0)}/100）`);
    lines.push(`- 判定摘要：${scope.primaryCandidate.summary}`);
    if ((scope.primaryCandidate.missing ?? []).length > 0) {
      lines.push(`- 未达门限项：${scope.primaryCandidate.missing.join('；')}`);
    }
    lines.push('');
  }
  return lines.join('\n');
}

function sectionEvidence(analysis) {
  const lines = ['## ② 量化证据', '', '### 全量窗口', '', '| 分组 | 指标 | 数值 | 说明 |', '| --- | --- | --- | --- |'];
  for (const row of analysis.steps.evidence.overall) {
    lines.push(`| ${row.group} | ${row.metric} | ${formatValue(row.value, row.unit)} | ${row.note} |`);
  }
  for (const [phaseId, rows] of Object.entries(analysis.steps.evidence.phases)) {
    const label = analysis.phaseIndicators[phaseId]?.label ?? phaseId;
    lines.push('', `### 阶段：${label}`, '', '| 分组 | 指标 | 数值 | 说明 |', '| --- | --- | --- | --- |');
    for (const row of rows) lines.push(`| ${row.group} | ${row.metric} | ${formatValue(row.value, row.unit)} | ${row.note} |`);
  }
  const evidenceDetail = analysis.steps.locate.perScope;
  lines.push('', '### 门限比对明细', '', '| 作用范围 | 指标 | 实测 | 门限判定 | 数据来源 |', '| --- | --- | --- | --- | --- |');
  for (const scope of evidenceDetail) {
    for (const candidate of scope.candidates) {
      for (const row of candidate.evidence) {
        lines.push(`| ${scope.scopeLabel} | ${candidate.short} · ${row.metric} | ${formatValue(row.value, row.unit === '%' ? '%' : row.unit)} | ${row.comparison} | ${row.source} |`);
      }
    }
  }
  return lines.join('\n');
}

function sectionRootCause(analysis) {
  const lines = ['## ③ 根因推断', ''];
  if (analysis.steps.cause.items.length === 0) {
    lines.push('未定位到达到门限的瓶颈，因此不输出根因推断（避免无据结论）。请参考第 ⑤ 节的"补齐采集"建议。');
    return lines.join('\n');
  }
  lines.push('| # | 根因假设 | 适用瓶颈 | 适用阶段 |', '| --- | --- | --- | --- |');
  analysis.steps.cause.items.forEach((item, index) => {
    lines.push(`| ${String(index + 1)} | ${item.title} | ${item.bottleneck} | ${item.phase} |`);
  });
  lines.push('');
  analysis.steps.cause.items.forEach((item, index) => {
    lines.push(`### ③.${String(index + 1)} ${item.title}`, '');
    lines.push('**作用机理**', '', item.mechanism, '');
    lines.push('**vLLM-Ascend 侧的实现原因**', '', item.vllmBehaviour, '');
    if (item.triggers.length > 0) {
      lines.push('**触发该推断的数据**', '', '| 指标 | 数值 | 说明 |', '| --- | --- | --- |');
      for (const trigger of item.triggers) {
        lines.push(`| ${trigger.metric} | ${formatValue(trigger.value, trigger.unit)} | ${trigger.note} |`);
      }
      lines.push('');
    }
    lines.push('**现场确认方法**', '');
    for (const check of item.checks) lines.push(`- ${check}`);
    lines.push('');
  });
  return lines.join('\n');
}

function sectionActions(analysis) {
  const lines = ['## ④ 可落地优化方案', ''];
  const items = analysis.steps.actions.items;
  if (items.length === 0) {
    lines.push('没有匹配到可执行的优化项：请先按"补齐采集"建议完善 profiling 产物，或调整阶段标注。');
    return lines.join('\n');
  }
  lines.push(`共 ${String(items.length)} 项，按优先级与实测严重度排序（高 ${String(analysis.steps.actions.grouped.高.length)} / 中 ${String(analysis.steps.actions.grouped.中.length)} / 低 ${String(analysis.steps.actions.grouped.低.length)}）。`, '');
  items.forEach((item, index) => {
    lines.push(`### ④.${String(index + 1)} 【${item.priority}】${item.title}`, '');
    lines.push(`- 适用阶段：${(item.phases ?? [item.phase]).map(phaseLabel).join('、')}${item.confirmInVersion ? '（含需按版本确认的开关）' : ''}`);
    lines.push(`- 优先级得分：${item.priorityScore.toFixed(1)}`);
    lines.push(`- 依据：${item.rationale}`);
    lines.push('');
    lines.push('**执行动作**', '');
    for (const action of item.actions) lines.push(`- \`${action.type}\` ${action.text}`);
    lines.push('');
    if (item.expectedGain !== undefined) {
      lines.push('**预期收益**', '');
      lines.push(`- 指标：${item.expectedGain.metric}`);
      lines.push(`- 估算：${item.expectedGain.estimatePct.toFixed(1)}%（区间 ${item.expectedGain.rangePct?.[0]?.toFixed(1) ?? '?'}% – ${item.expectedGain.rangePct?.[1]?.toFixed(1) ?? '?'}%）`);
      lines.push(`- 推算依据：${item.expectedGain.basis}`);
      lines.push(`- 置信度：${confidenceLabel(item.expectedGain.confidence)}；前提假设：${item.expectedGain.assumption}`);
      lines.push('');
    }
    lines.push(`- 验证方法：${item.verification}`);
    lines.push(`- 风险：${item.risk}`);
    lines.push('');
  });
  return lines.join('\n');
}

function sectionBenefit(analysis) {
  const lines = ['## ⑤ 预期收益汇总', ''];
  const benefit = analysis.steps.benefit;
  if (benefit.items.length === 0) {
    lines.push(benefit.note);
    return lines.join('\n');
  }
  lines.push('| 优化项 | 优先级 | 阶段 | 目标指标 | 估算收益 | 区间 | 置信度 |', '| --- | --- | --- | --- | --- | --- | --- |');
  for (const item of benefit.items) {
    lines.push(`| ${item.title} | ${item.priority} | ${item.phase.map(phaseLabel).join('/')} | ${item.metric} | ${item.estimatePct.toFixed(1)}% | ${item.rangePct?.[0]?.toFixed(1) ?? '?'}%–${item.rangePct?.[1]?.toFixed(1) ?? '?'}% | ${confidenceLabel(item.confidence)} |`);
  }
  lines.push('');
  lines.push(`- 保守合计：**${benefit.combined.conservativePct.toFixed(1)}%**；乐观合计：**${benefit.combined.optimisticPct.toFixed(1)}%**（${String(benefit.combined.itemCount)} 项）`);
  for (const [phase, value] of Object.entries(benefit.combined.byPhase)) {
    lines.push(`- ${phaseLabel(phase)} 维度合计：${value.toFixed(1)}%`);
  }
  lines.push('', `> ${benefit.combined.note}`, '', `> ${benefit.note}`);
  return lines.join('\n');
}

/**
 * Embed the chart snapshots the browser captured.
 *
 * Data URLs keep the exported file self-contained (no external assets to lose),
 * at the cost of a larger file — which is why the snapshots are only present
 * when the user exported the PDF from a page that had rendered the charts.
 */
function sectionCharts(charts) {
  const entries = Object.entries(charts).filter(([, value]) => typeof value === 'string' && value.startsWith('data:image/'));
  if (entries.length === 0) {
    return '## 图表\n\n> 本次导出的 Markdown 未包含图表快照：请从分析页面点击「导出 PDF」以捕获泳道图与占比图表，'
      + '或在页面中查看对应图表（模块一/模块二的实时交互视图无法用静态 Markdown 表达）。';
  }
  const lines = ['## 图表', ''];
  for (const [name, dataUrl] of entries) {
    lines.push(`### ${name}`, '', `![${name}](${dataUrl})`, '');
  }
  lines.push('> 图表为导出时的快照：泳道图为完整采集窗口（非页面当前缩放视图），占比图表为页面当前统计口径下的结果。');
  return lines.join('\n');
}

function sectionTopOperators(dataset) {
  const lines = ['## 附 A. 算子耗时排行（累计总耗时）', '', '| # | 算子 | 类别 | 设备 | 调用次数 | 累计耗时(ms) | 单次均值(µs) | p95(µs) | 占算子总耗时 | 总计来源 | cross-check 偏差 |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |'];
  for (const row of dataset.ranking.byTotal) {
    lines.push([
      String(row.rank),
      escapeCell(row.name),
      CATEGORIES[row.category]?.label ?? row.category,
      row.group,
      String(row.count),
      (row.totalUs / 1000).toFixed(3),
      row.avgUs?.toFixed(1) ?? '-',
      row.p95Us?.toFixed(1) ?? '-',
      `${row.shareOfOpsPct.toFixed(2)}%`,
      row.totalsSource,
      row.crossCheckPct === undefined ? '-' : `${row.crossCheckPct.toFixed(1)}%`,
    ].join(' | ').replace(/^/, '| ').replace(/$/, ' |'));
  }
  lines.push('', `算子总耗时（分母）：${(dataset.ranking.totalOperatorUs / 1000).toFixed(3)}ms。占比 = 该算子累计耗时 ÷ 算子总耗时；"cross-check 偏差"为 trace 聚合与 CANN 统计表之间的相对差异。`);
  return lines.join('\n');
}

function sectionPhases(dataset) {
  const lines = ['## 附 B. Prefill / Decode 阶段指标', ''];
  if ((dataset.phases.phases ?? []).length === 0) {
    lines.push('未能划分阶段。');
    return lines.join('\n');
  }
  lines.push(`阶段来源：${dataset.phases.sourceLabel}；置信度 ${dataset.phases.confidence}；方法 ${dataset.phases.classificationMethod}。`, '');
  lines.push('| 阶段 | 步数 | 平均步长(ms) | NPU 忙碌率 | Host 忙占比 | 通信占设备 | 通信未掩盖 | 拷贝占墙钟 | Host 独占/步(µs) |', '| --- | --- | --- | --- | --- | --- | --- | --- | --- |');
  for (const phase of dataset.phases.phases) {
    lines.push(`| ${phase.label} | ${String(phase.stepCount)} | ${(phase.avgStepUs / 1000).toFixed(3)} | ${phase.npuUtilPct.toFixed(1)}% | ${phase.hostUtilPct.toFixed(1)}% | ${phase.commPct.toFixed(1)}% | ${phase.commNotOverlappedPct.toFixed(1)}% | ${phase.copyPct.toFixed(1)}% | ${(phase.hostBusyUs / Math.max(1, phase.stepCount)).toFixed(0)} |`);
  }
  return lines.join('\n');
}

function sectionCategories(dataset) {
  const lines = ['## 附 C. 算子大类耗时占比', '', '| 大类 | 累计耗时(ms) | 占比 | 调用次数 | 设备侧(ms) | Host 侧(ms) |', '| --- | --- | --- | --- | --- | --- |'];
  for (const item of dataset.categories.items) {
    lines.push(`| ${item.label} | ${(item.totalUs / 1000).toFixed(3)} | ${item.sharePct.toFixed(2)}% | ${String(item.count)} | ${(item.deviceUs / 1000).toFixed(3)} | ${(item.hostUs / 1000).toFixed(3)} |`);
  }
  lines.push('', `大类合计：${(dataset.categories.totalUs / 1000).toFixed(3)}ms（来源：${dataset.categories.totalsSource}）。通信/计算重叠部分的说明见附 D。`);
  lines.push('', '### 算子亚类分布（设备侧前 10）', '', '| 算子 | 亚类 | 累计耗时(ms) | 占比 |', '| --- | --- | --- | --- |');
  for (const row of dataset.ranking.byTotal.slice(0, 10)) {
    lines.push(`| ${escapeCell(row.name)} | ${subtypeLabel(row.category, row.subtype)} | ${(row.totalUs / 1000).toFixed(3)} | ${row.shareOfOpsPct.toFixed(2)}% |`);
  }
  return lines.join('\n');
}

function sectionUtilization(dataset) {
  const lines = ['## 附 D. 算力/流水利用率与重叠情况', ''];
  const utilization = dataset.utilization;
  lines.push(`- 设备忙碌时间：${(dataset.overlap.deviceBusyUs / 1000).toFixed(3)}ms（${dataset.overlap.deviceBusyPct.toFixed(1)}% 墙钟）`);
  lines.push(`- Host 忙碌时间：${(dataset.overlap.hostBusyUs / 1000).toFixed(3)}ms（${dataset.overlap.hostBusyPct.toFixed(1)}% 墙钟）`);
  lines.push(`- 通信并集：${(dataset.overlap.commUnionUs / 1000).toFixed(3)}ms，与计算重叠 ${(dataset.overlap.commOverlapUs / 1000).toFixed(3)}ms（${dataset.overlap.commOverlapPct.toFixed(1)}%），未掩盖 ${(dataset.overlap.commExposedUs / 1000).toFixed(3)}ms`);
  lines.push('');
  if (utilization.available) {
    lines.push('| 指标 | 加权值 | 权重耗时(ms) |', '| --- | --- | --- |');
    for (const metric of utilization.metrics) {
      lines.push(`| ${metric.label} | ${metric.valuePct.toFixed(1)}% | ${(metric.weightUs / 1000).toFixed(3)} |`);
    }
  } else {
    lines.push(`利用率数据不可用：${utilization.note}`);
  }
  if (utilization.flops !== undefined) {
    lines.push('', `FLOPs 换算：累计 ${(utilization.flops.totalFlops / 1e12).toFixed(2)} TFLOP、实测 ${utilization.flops.achievedTflops ?? '-'} TFLOPS${utilization.flops.utilizationPct === undefined ? '' : `、峰值利用率 ${utilization.flops.utilizationPct.toFixed(1)}%`}。${utilization.flops.note}`);
  }
  if (dataset.gaps.length > 0) {
    lines.push('', '### 设备空闲段 Top 10', '', '| 开始(ms) | 时长(µs) | 占墙钟 | 前序算子 | 后续算子 | Host 同期算子数 |', '| --- | --- | --- | --- | --- | --- |');
    for (const gap of dataset.gaps.slice(0, 10)) {
      lines.push(`| ${(gap.startUs / 1000).toFixed(3)} | ${gap.lengthUs.toFixed(1)} | ${gap.sharePct.toFixed(2)}% | ${escapeCell(gap.before ?? '-')} | ${escapeCell(gap.after ?? '-')} | ${String(gap.hostOperatorCount)} |`);
    }
  }
  return lines.join('\n');
}

function sectionCaveats(dataset, analysis) {
  const lines = ['## 附 E. 结论边界与数据质量', ''];
  const warnings = [...dataset.meta.warnings, ...analysis.warnings];
  if (warnings.length === 0) lines.push('- 本次解析未产生额外告警。');
  for (const warning of warnings) lines.push(`- ${warning}`);
  lines.push('', '### 指标口径', '');
  lines.push('- **区间并集**：同一设备/主机侧并行执行的算子会被合并计算，因此"设备忙碌时间"不会因多流并行而重复累加。');
  lines.push('- **通信未掩盖**：通信区间并集减去与计算区间重叠的部分，代表真正进入关键路径的通信时间。');
  lines.push('- **算子耗时合计**：各算子累计耗时之和，多流并行时会大于墙钟时间；占比分母即该合计值。');
  lines.push('- **Host 独占**：Host 忙碌但设备空闲的时间，是"设备在等主机"的下界估计。');
  lines.push('- **CSV 与 trace 的坐标**：CSV 的 `Start Time` 为设备绝对时间，trace 的 `ts` 为相对时间，二者不做混轴；累计耗时按单位统一后合并，并给出 cross-check 偏差。');
  return lines.join('\n');
}

function phaseLabel(phase) {
  return { prefill: 'Prefill', decode: 'Decode', both: '通用', overall: '全量窗口', unknown: '未划分' }[phase] ?? phase;
}

function confidenceLabel(confidence) {
  return { high: '高（由数据推导）', medium: '中（数据推导 + 合理假设）', low: '低（经验区间，需实测）' }[confidence] ?? confidence ?? '未知';
}

function formatValue(value, unit) {
  if (value === undefined || value === null) return 'N/A';
  if (typeof value !== 'number') return String(value);
  const text = Math.abs(value) >= 1000 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return `${text}${unit ?? ''}`;
}

function escapeCell(text) {
  return String(text).replaceAll('|', '\\|');
}

/**
 * Format a byte count.
 * @param {number} bytes - byte count.
 * @returns {string} human-readable size.
 */
export function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return 'N/A';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)}${units[unit]}`;
}
