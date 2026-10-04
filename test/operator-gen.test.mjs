import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';

import { parseProfileSet } from '../lib/parse/index.js';
import { buildDataset } from '../lib/model/dataset.js';
import { analyzeDataset } from '../lib/analysis/index.js';
import { buildHandoff } from '../lib/handoff.js';
import { OPERATOR_DIR, planOperator, renderOperatorProject, writeOperatorProject } from '../lib/operator-gen.js';

const here = dirname(fileURLToPath(import.meta.url));

/** Build the handoff a 执行 press would build, for one scenario + advice. */
async function handoffFor(scenario, adviceId) {
  const directory = join(here, 'fixtures', scenario);
  const inputs = readdirSync(directory).map((name) => ({ name, buffer: readFileSync(join(directory, name)) }));
  const parse = await parseProfileSet({ inputs });
  assert.equal(parse.ok, true, JSON.stringify(parse.errors));
  const dataset = buildDataset(parse);
  const analysis = analyzeDataset(dataset);
  return buildHandoff({ adviceId, dataset, analysis, datasetId: scenario, label: scenario, roots: [] });
}

test('执行 generates an operator project for the advice it came from', async () => {
  const task = await handoffFor('prefill-compute-bound', 'compute.quantize');
  const project = renderOperatorProject(task, { generatedAt: '2026-01-02T03:04:05.000Z' });

  assert.equal(project.ok, true);
  assert.equal(project.kind, 'quantize');
  assert.equal(project.op, 'dyn_quant_dequant');
  // The four source artefacts plus the design, README and registration guide.
  const paths = project.files.map((file) => file.path).sort();
  assert.equal(paths.length, 6);
  assert.ok(paths.includes('csrc/ops/dyn_quant_dequant/op_kernel/dyn_quant_dequant.cpp'));
  assert.ok(paths.includes('csrc/ops/dyn_quant_dequant/op_host/dyn_quant_dequant.cpp'));
  assert.ok(paths.includes('csrc/ops/dyn_quant_dequant/design.md'));
  assert.ok(paths.includes('csrc/ops/dyn_quant_dequant/test/dyn_quant_dequant-test-cases.md'));
  assert.ok(paths.includes('csrc/ops/dyn_quant_dequant/register-patch.md'));

  // The optimisation is stated, and the acceptance bar comes from the profiling task.
  assert.match(project.summary.optimization, /量化与反量化融合/);
  assert.equal(project.summary.acceptance[0].key, 'computeUs');
  assert.equal(project.summary.acceptance[0].thresholdPct, 15);
  assert.deepEqual(project.summary.skills.slice(0, 2), ['ascendc', 'ascend-opplugin']);
  assert.match(project.summary.operators.map((operator) => operator.name).join(','), /MatMulV2/);
});

test('the generated kernel follows the AscendC rules the skill enforces', async () => {
  for (const [scenario, adviceId] of [
    ['prefill-compute-bound', 'compute.quantize'],
    ['host-schedule-bound', 'host.reduce-dispatch'],
    ['host-schedule-bound', 'host.enable-graph-mode'],
  ]) {
    const task = await handoffFor(scenario, adviceId);
    const project = renderOperatorProject(task);
    const kernel = project.files.find((file) => file.path.includes('op_kernel')).content;
    const host = project.files.find((file) => file.path.includes('op_host')).content;
    assert.match(kernel, /constexpr int32_t BUFFER_NUM = 2;/, `${adviceId}: double buffering`);
    assert.match(kernel, /DataCopyPad\(/, `${adviceId}: GM<->UB through DataCopyPad`);
    assert.ok(!/AscendC::DataCopy\(/.test(kernel), `${adviceId}: never plain DataCopy`);
    assert.ok(!/std::(sqrt|exp|abs|min|max)\(/.test(kernel), `${adviceId}: no std:: math in a kernel`);
    assert.match(kernel, /CAST_NONE|CAST_RINT/, `${adviceId}: explicit casts`);
    assert.match(host, /GetCoreNumAiv\(\)/, `${adviceId}: cores from the platform API`);
    assert.match(host, /GetCoreMemSize/, `${adviceId}: UB size from the platform API`);
    assert.match(host, /TORCH_CHECK\(/, `${adviceId}: input validation`);
    assert.match(host, /EXEC_KERNEL_CMD\(/, `${adviceId}: launch`);
    // Design + test cases carry the provenance and the acceptance table.
    const design = project.files.find((file) => file.path.endsWith('design.md')).content;
    const cases = project.files.find((file) => file.path.includes('-test-cases.md')).content;
    assert.match(design, new RegExp(adviceId));
    assert.match(cases, /SUPPORTED_DTYPES/);
    assert.match(cases, /BOUNDARY_VALUES/);
    assert.match(cases, /30/);
  }
});

test('an elementwise advice gets the fused-chain template', async () => {
  const task = await handoffFor('host-schedule-bound', 'host.reduce-dispatch');
  const fused = planOperator({ adviceId: 'compute.fuse-small-ops', focus: 'kernel' });
  assert.equal(fused.kind, 'elementwise');
  const project = renderOperatorProject({ ...task, adviceId: 'compute.fuse-small-ops' });
  assert.equal(project.kind, 'elementwise');
  const kernel = project.files.find((file) => file.path.includes('op_kernel')).content;
  assert.match(kernel, /AscendC::Exp\(/, 'silu needs Exp');
  assert.match(project.summary.optimization, /三次下发融合成一次/);
});

test('the project is written into the workspace and the files are real', async () => {
  const task = await handoffFor('prefill-compute-bound', 'compute.quantize');
  const project = renderOperatorProject(task);
  const workspace = mkdtempSync(join(tmpdir(), 'vap-opgen-'));
  const written = writeOperatorProject(project, { workspace });

  assert.equal(written.ok, true);
  assert.equal(written.dir, join(workspace, OPERATOR_DIR, project.op));
  assert.equal(written.files.length, 6);
  for (const file of written.files) assert.ok(existsSync(file), `${file} must exist`);
  const kernel = readFileSync(join(written.dir, `csrc/ops/${project.op}/op_kernel/${project.op}.cpp`), 'utf8');
  assert.ok(kernel.length > 2000, 'the kernel is a real source file, not a stub');

  // A failed task renders an error instead of throwing.
  const failed = renderOperatorProject({ ok: false, error: '未知的优化项' });
  assert.equal(failed.ok, false);
  assert.match(failed.error, /未知的优化项/);
  assert.equal(writeOperatorProject(failed, { workspace }).ok, false);

  rmSync(workspace, { recursive: true, force: true });
});
