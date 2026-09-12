/**
 * Printable HTML report ("export as PDF").
 *
 * The plugin deliberately does not bundle a PDF library: the browser's own
 * "print to PDF" produces a better result than a hand-rolled rasteriser, keeps
 * the plugin dependency-free, and lets the user control page size and margins.
 * This module renders a self-contained, print-optimised HTML document (inline
 * CSS, no external resources) that opens the print dialog on load.
 *
 * @module dsh-plugin-vllm-ascend-profiler/report/print
 */

import { CATEGORIES, subtypeLabel } from '../model/classify.js';

/**
 * Render the printable HTML report.
 *
 * @param {object} input - report input.
 * @param {object} input.dataset - dataset.
 * @param {object} input.analysis - analysis result.
 * @param {object} [input.charts] - chart images as data URLs, keyed by chart name.
 * @param {boolean} [input.autoPrint] - whether to open the print dialog on load.
 * @returns {string} a complete HTML document.
 */
export function renderPrintableReport({ dataset, analysis, charts = {}, autoPrint = true }) {
  const body = [
    header(dataset, analysis),
    provenance(dataset),
    locate(analysis),
    evidence(analysis),
    causes(analysis),
    actions(analysis),
    benefit(analysis),
    chartsSection(charts),
    operators(dataset),
    caveats(dataset, analysis),
  ].join('\n');
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>vLLM-Ascend Profiling 分析报告</title>
<style>${PRINT_CSS}</style>
</head>
<body>
<main>
${body}
</main>
<footer>由 DeepSeek Harness 插件 vllm-ascend-profiler-analyzer 生成 · ${escapeHtml(new Date().toLocaleString('zh-CN'))}</footer>
${autoPrint ? '<script>addEventListener("load",()=>{setTimeout(()=>print(),300)})</script>' : ''}
</body>
</html>`;
}

const PRINT_CSS = `
:root { --ink:#111; --muted:#555; --line:#c9ced6; --accent:#1d4ed8; }
* { box-sizing: border-box; }
body { margin:0; color:var(--ink); font:12px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif; background:#fff; }
main { max-width: 980px; margin: 0 auto; padding: 24px 28px 8px; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 22px 0 8px; padding-bottom: 4px; border-bottom: 1.5px solid var(--line); break-after: avoid; }
h3 { font-size: 13px; margin: 14px 0 6px; break-after: avoid; }
h4 { font-size: 12px; margin: 10px 0 4px; }
p, li { orphans: 3; widows: 3; }
table { width:100%; border-collapse: collapse; margin: 8px 0 12px; font-size: 11px; break-inside: avoid; }
th, td { border: 0.5px solid var(--line); padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #f2f4f8; font-weight: 600; }
code, .mono { font-family: "Cascadia Mono", Consolas, monospace; font-size: 11px; }
blockquote { margin: 8px 0; padding: 8px 12px; border-left: 3px solid var(--accent); background:#f6f8ff; }
.summary-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin: 12px 0; }
.summary-card { border:0.5px solid var(--line); border-radius:6px; padding:8px 10px; break-inside: avoid; }
.summary-card .k { color:var(--muted); font-size:10px; }
.summary-card .v { font-size:15px; font-weight:600; }
.tag { display:inline-block; padding:1px 6px; border-radius:8px; background:#eef1f7; font-size:10px; margin-right:4px; }
.pri-高 { background:#fde8e8; color:#a11; }
.pri-中 { background:#fef3d7; color:#8a5a00; }
.pri-低 { background:#e8f4ea; color:#1d6b2f; }
.pri { display:inline-block; padding:1px 6px; border-radius:8px; font-size:10px; font-weight:600; }
.charts { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.charts figure { margin:0; break-inside: avoid; }
.charts img { width:100%; border:0.5px solid var(--line); border-radius:6px; }
figcaption { color:var(--muted); font-size:10px; margin-top:4px; }
footer { color:var(--muted); font-size:10px; text-align:center; padding: 10px 0 20px; }
.note { color: var(--muted); font-size: 11px; }
.warn { color:#a11; }
@media print {
  main { max-width:none; padding: 0 6mm; }
  h2 { break-before: auto; }
  .page-break { break-before: page; }
  footer { position: fixed; bottom: 0; width: 100%; }
}
`;

function header(dataset, analysis) {
  const headline = analysis.bottleneck;
  return `<h1>vLLM-Ascend Profiling 性能分析报告</h1>
<p class="note">采集窗口 ${((dataset.meta.window.end - dataset.meta.window.start) / 1000).toFixed(2)}ms ·
设备事件 ${String(dataset.meta.counts.deviceEvents)} · Host 事件 ${String(dataset.meta.counts.hostEvents)} · 算子 ${String(dataset.meta.counts.operators)} · rank 数 ${String(dataset.meta.counts.ranks)}</p>
<div class="summary-grid">
  <div class="summary-card"><div class="k">主导瓶颈</div><div class="v">${escapeHtml(headline.short ?? headline.label)}</div><div class="note">得分 ${headline.score.toFixed(0)}/100</div></div>
  <div class="summary-card"><div class="k">作用范围</div><div class="v">${escapeHtml(headline.scopeLabel ?? phaseLabel(headline.scope ?? 'overall'))}</div><div class="note">${escapeHtml(headline.subtype ?? '综合判定')}</div></div>
  <div class="summary-card"><div class="k">NPU 忙碌率</div><div class="v">${analysis.indicators.deviceBusyPct.toFixed(1)}%</div><div class="note">Host 忙 ${analysis.indicators.hostBusyPct.toFixed(1)}%</div></div>
  <div class="summary-card"><div class="k">保守收益合计</div><div class="v">${analysis.steps.benefit.combined.conservativePct.toFixed(1)}%</div><div class="note">乐观 ${analysis.steps.benefit.combined.optimisticPct.toFixed(1)}%</div></div>
</div>
<blockquote>${escapeHtml(headline.summary ?? '')}</blockquote>`;
}

function provenance(dataset) {
  const rows = dataset.meta.files.map((file) => `<tr><td class="mono">${escapeHtml(file.name)}</td><td>${escapeHtml(file.kind)}</td><td>${String(file.events ?? file.rows ?? '-')}</td></tr>`).join('');
  const sampling = dataset.meta.sampling?.applied === true
    ? `<p class="warn">trace 已采样解析：${escapeHtml(dataset.meta.sampling.strategy)}（步长 ${String(dataset.meta.sampling.stride)}，扫描 ${String(dataset.meta.sampling.seen)} → 保留 ${String(dataset.meta.sampling.kept)}）。泳道图为抽样结果，累计耗时来源：${escapeHtml(dataset.meta.totalsSource)}。</p>`
    : `<p class="note">时间线为全量解析，累计耗时来源：${escapeHtml(dataset.meta.totalsSource)}。</p>`;
  return `<h2>0. 数据来源与解析说明</h2>
<table><thead><tr><th>文件</th><th>类型</th><th>行/事件数</th></tr></thead><tbody>${rows}</tbody></table>
${sampling}
<p class="note">阶段划分：${escapeHtml(dataset.phases.sourceLabel)}（置信度 ${escapeHtml(dataset.phases.confidence)}）</p>`;
}

function locate(analysis) {
  const rows = analysis.steps.locate.perScope.map((scope) => {
    const score = (id) => scope.candidates.find((candidate) => candidate.id === id)?.score.toFixed(0) ?? '-';
    return `<tr><td>${escapeHtml(scope.scopeLabel)}</td><td>${score('host')}</td><td>${score('compute')}</td><td>${score('comm')}</td><td>${score('copy')}</td><td>${scope.verdict === 'balanced' ? '未达门限' : escapeHtml(scope.primaryCandidate.label)}</td></tr>`;
  }).join('');
  const details = analysis.steps.locate.perScope.map((scope) => `<h3>${escapeHtml(scope.scopeLabel)}</h3>
<p><strong>${escapeHtml(scope.primaryCandidate.label)}</strong> — ${escapeHtml(scope.primaryCandidate.summary)}</p>
${(scope.primaryCandidate.missing ?? []).length > 0 ? `<p class="note">未达门限：${escapeHtml(scope.primaryCandidate.missing.join('；'))}</p>` : ''}`).join('');
  return `<h2>① 瓶颈类型定位</h2>
<p>${escapeHtml(analysis.steps.locate.summary)}</p>
<table><thead><tr><th>作用范围</th><th>Host 调度</th><th>NPU 计算</th><th>跨卡通信</th><th>数据拷贝</th><th>结论</th></tr></thead><tbody>${rows}</tbody></table>
${details}`;
}

function evidence(analysis) {
  const table = (rows) => `<table><thead><tr><th>分组</th><th>指标</th><th>数值</th><th>说明</th></tr></thead><tbody>${rows.map((row) => `<tr><td>${escapeHtml(row.group)}</td><td>${escapeHtml(row.metric)}</td><td>${escapeHtml(formatValue(row.value, row.unit))}</td><td>${escapeHtml(row.note)}</td></tr>`).join('')}</tbody></table>`;
  const phases = Object.entries(analysis.steps.evidence.phases)
    .map(([id, rows]) => `<h3>阶段：${escapeHtml(analysis.phaseIndicators[id]?.label ?? id)}</h3>${table(rows)}`)
    .join('');
  const gates = analysis.steps.locate.perScope.flatMap((scope) => scope.candidates.flatMap((candidate) => candidate.evidence
    .map((row) => `<tr><td>${escapeHtml(scope.scopeLabel)}</td><td>${escapeHtml(candidate.short)} · ${escapeHtml(row.metric)}</td><td>${escapeHtml(formatValue(row.value, row.unit))}</td><td>${escapeHtml(row.comparison)}</td><td>${escapeHtml(row.source)}</td></tr>`))).join('');
  return `<h2 class="page-break">② 量化证据</h2>
<h3>全量窗口</h3>${table(analysis.steps.evidence.overall)}
${phases}
<h3>门限比对明细</h3>
<table><thead><tr><th>作用范围</th><th>指标</th><th>实测</th><th>门限判定</th><th>数据来源</th></tr></thead><tbody>${gates}</tbody></table>`;
}

function causes(analysis) {
  if (analysis.steps.cause.items.length === 0) {
    return '<h2>③ 根因推断</h2><p>未定位到达到门限的瓶颈，因此不输出根因推断。</p>';
  }
  return `<h2 class="page-break">③ 根因推断</h2>${analysis.steps.cause.items.map((item, index) => `<h3>③.${String(index + 1)} ${escapeHtml(item.title)}</h3>
<p><span class="tag">${escapeHtml(item.bottleneck)}</span><span class="tag">${escapeHtml(phaseLabel(item.phase))}</span></p>
<h4>作用机理</h4><p>${escapeHtml(item.mechanism)}</p>
<h4>vLLM-Ascend 侧的实现原因</h4><p>${escapeHtml(item.vllmBehaviour)}</p>
<h4>触发该推断的数据</h4>
<table><thead><tr><th>指标</th><th>数值</th><th>说明</th></tr></thead><tbody>${item.triggers.map((trigger) => `<tr><td>${escapeHtml(trigger.metric)}</td><td>${escapeHtml(formatValue(trigger.value, trigger.unit))}</td><td>${escapeHtml(trigger.note)}</td></tr>`).join('')}</tbody></table>
<h4>现场确认方法</h4><ul>${item.checks.map((check) => `<li>${escapeHtml(check)}</li>`).join('')}</ul>`).join('')}`;
}

function actions(analysis) {
  const items = analysis.steps.actions.items;
  if (items.length === 0) return '<h2>④ 可落地优化方案</h2><p>无匹配的优化项。</p>';
  return `<h2 class="page-break">④ 可落地优化方案</h2>${items.map((item, index) => `<h3>④.${String(index + 1)} <span class="pri pri-${item.priority}">${item.priority}</span> ${escapeHtml(item.title)}</h3>
<p class="note">适用阶段：${(item.phases ?? [item.phase]).map((phase) => escapeHtml(phaseLabel(phase))).join('、')} · 优先级得分 ${item.priorityScore.toFixed(1)}${item.confirmInVersion ? ' · 含需按版本确认的开关' : ''}</p>
<p>${escapeHtml(item.rationale)}</p>
<ul>${item.actions.map((action) => `<li><code>${escapeHtml(action.type)}</code> ${escapeHtml(action.text)}</li>`).join('')}</ul>
${item.expectedGain === undefined ? '' : `<p><strong>预期收益</strong>：${escapeHtml(item.expectedGain.metric)} ${item.expectedGain.estimatePct.toFixed(1)}%（区间 ${item.expectedGain.rangePct?.[0]?.toFixed(1) ?? '?'}%–${item.expectedGain.rangePct?.[1]?.toFixed(1) ?? '?'}%，置信度 ${escapeHtml(confidenceLabel(item.expectedGain.confidence))}）<br><span class="note">推算依据：${escapeHtml(item.expectedGain.basis)}；前提：${escapeHtml(item.expectedGain.assumption)}</span></p>`}
<p class="note">验证：${escapeHtml(item.verification)} · 风险：${escapeHtml(item.risk)}</p>`).join('')}`;
}

function benefit(analysis) {
  const benefitBlock = analysis.steps.benefit;
  const rows = benefitBlock.items.map((item) => `<tr><td>${escapeHtml(item.title)}</td><td>${item.priority}</td><td>${item.phase.map((phase) => escapeHtml(phaseLabel(phase))).join('/')}</td><td>${escapeHtml(item.metric)}</td><td>${item.estimatePct.toFixed(1)}%</td><td>${item.rangePct?.[0]?.toFixed(1) ?? '?'}%–${item.rangePct?.[1]?.toFixed(1) ?? '?'}%</td><td>${escapeHtml(confidenceLabel(item.confidence))}</td></tr>`).join('');
  return `<h2>⑤ 预期收益汇总</h2>
<table><thead><tr><th>优化项</th><th>优先级</th><th>阶段</th><th>目标指标</th><th>估算</th><th>区间</th><th>置信度</th></tr></thead><tbody>${rows}</tbody></table>
<p>保守合计 <strong>${benefitBlock.combined.conservativePct.toFixed(1)}%</strong>，乐观合计 <strong>${benefitBlock.combined.optimisticPct.toFixed(1)}%</strong>。</p>
<p class="note">${escapeHtml(benefitBlock.combined.note)} ${escapeHtml(benefitBlock.note)}</p>`;
}

function chartsSection(charts) {
  const entries = Object.entries(charts).filter(([, value]) => typeof value === 'string' && value.startsWith('data:image'));
  if (entries.length === 0) return '';
  return `<h2 class="page-break">图表快照</h2><div class="charts">${entries.map(([name, data]) => `<figure><img src="${data}" alt="${escapeHtml(name)}"><figcaption>${escapeHtml(name)}</figcaption></figure>`).join('')}</div>`;
}

function operators(dataset) {
  const rows = dataset.ranking.byTotal.map((row) => `<tr><td>${String(row.rank)}</td><td class="mono">${escapeHtml(row.name)}</td><td>${escapeHtml(CATEGORIES[row.category]?.label ?? row.category)}</td><td>${escapeHtml(subtypeLabel(row.category, row.subtype))}</td><td>${row.group}</td><td>${String(row.count)}</td><td>${(row.totalUs / 1000).toFixed(3)}</td><td>${row.avgUs?.toFixed(1) ?? '-'}</td><td>${row.shareOfOpsPct.toFixed(2)}%</td><td>${escapeHtml(row.totalsSource)}</td></tr>`).join('');
  const categories = dataset.categories.items.map((item) => `<tr><td>${escapeHtml(item.label)}</td><td>${(item.totalUs / 1000).toFixed(3)}</td><td>${item.sharePct.toFixed(2)}%</td><td>${String(item.count)}</td><td>${(item.deviceUs / 1000).toFixed(3)}</td><td>${(item.hostUs / 1000).toFixed(3)}</td></tr>`).join('');
  const gaps = dataset.gaps.slice(0, 10).map((gap) => `<tr><td>${(gap.startUs / 1000).toFixed(3)}</td><td>${gap.lengthUs.toFixed(1)}</td><td>${gap.sharePct.toFixed(2)}%</td><td class="mono">${escapeHtml(gap.before ?? '-')}</td><td class="mono">${escapeHtml(gap.after ?? '-')}</td><td>${String(gap.hostOperatorCount)}</td></tr>`).join('');
  return `<h2 class="page-break">附 A. 算子耗时排行</h2>
<table><thead><tr><th>#</th><th>算子</th><th>类别</th><th>亚类</th><th>设备</th><th>次数</th><th>累计(ms)</th><th>均值(µs)</th><th>占比</th><th>来源</th></tr></thead><tbody>${rows}</tbody></table>
<h2>附 B. 算子大类占比</h2>
<table><thead><tr><th>大类</th><th>累计(ms)</th><th>占比</th><th>次数</th><th>设备(ms)</th><th>Host(ms)</th></tr></thead><tbody>${categories}</tbody></table>
<h2>附 C. 设备空闲段 Top 10</h2>
<table><thead><tr><th>开始(ms)</th><th>时长(µs)</th><th>占墙钟</th><th>前序算子</th><th>后续算子</th><th>Host 同期算子数</th></tr></thead><tbody>${gaps}</tbody></table>`;
}

function caveats(dataset, analysis) {
  const warnings = [...dataset.meta.warnings, ...analysis.warnings];
  return `<h2>附 D. 结论边界与数据质量</h2>
<ul>${warnings.map((warning) => `<li>${escapeHtml(warning)}</li>`).join('') || '<li>本次解析未产生额外告警。</li>'}</ul>
<h3>指标口径</h3>
<ul>
<li><strong>区间并集</strong>：并行执行的算子会被合并，忙碌时间不会因多流并行重复累加。</li>
<li><strong>通信未掩盖</strong>：通信区间并集减去与计算重叠的部分。</li>
<li><strong>算子耗时合计</strong>：各算子累计耗时之和，多流并行时会大于墙钟时间。</li>
<li><strong>CSV 与 trace 坐标</strong>：CSV 的绝对设备时间与 trace 的相对时间不做混轴。</li>
</ul>`;
}

function phaseLabel(phase) {
  return { prefill: 'Prefill', decode: 'Decode', both: '通用', overall: '全量窗口', unknown: '未划分' }[phase] ?? phase;
}

function confidenceLabel(confidence) {
  return { high: '高（数据推导）', medium: '中（数据+假设）', low: '低（经验区间）' }[confidence] ?? confidence ?? '未知';
}

function formatValue(value, unit) {
  if (value === undefined || value === null) return 'N/A';
  if (typeof value !== 'number') return String(value);
  const text = Math.abs(value) >= 1000 ? value.toFixed(0) : Math.abs(value) >= 10 ? value.toFixed(1) : value.toFixed(2);
  return `${text}${unit ?? ''}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}
