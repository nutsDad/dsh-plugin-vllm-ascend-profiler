import test from 'node:test';
import assert from 'node:assert/strict';

import {
  compareNumeric,
  reference,
  selfCheck,
  simulateElementwise,
  simulateQuantize,
  simulateRowNorm,
  toFp16,
} from '../lib/operator-sim.js';

/**
 * The fp16 conversion is the foundation of every other check: if it rounds like a
 * real half, the tolerances below mean what they say.
 */
test('fp16 rounding matches IEEE-754 binary16', () => {
  const cases = [
    [1, 1],
    [1.0009765625, 1.0009765625], // exactly representable
    [0.1, 0.0999755859375], // nearest half
    [65504, 65504], // largest finite half
    [70000, Infinity], // overflow
    [1e-8, 0], // underflow to zero
    [-2.5, -2.5],
    [3.14159, 3.140625],
  ];
  for (const [input, expected] of cases) {
    const got = toFp16(input);
    if (expected === Infinity) assert.equal(got, Infinity, `${String(input)} must overflow`);
    else assert.ok(Math.abs(got - expected) < 1e-12, `toFp16(${String(input)}) = ${String(got)}, expected ${String(expected)}`);
  }
});

/**
 * The point of the simulation: the arithmetic and data flow of the generated kernel
 * (fp32 accumulation, fused gamma, up-cast) reproduce the documented formula.
 */
test('the fused row-norm simulation matches the reference formula', () => {
  const rows = 64;
  const cols = 128;
  const x = Array.from({ length: rows * cols }, (_, index) => Math.sin(index * 0.37) * 3);
  const gamma = Array.from({ length: cols }, (_, index) => 1 + (index % 7) * 0.05);
  const simulated = simulateRowNorm({ rows, cols, dtype: 'float16', x, gamma, eps: 1e-6 });
  const error = compareNumeric(simulated.output, reference.rowNorm({ rows, cols, x, gamma, eps: 1e-6 }));

  assert.equal(simulated.output.length, rows * cols);
  assert.equal(simulated.scales.length, rows);
  assert.ok(error.within(4.9e-4 * 8), `maxRel ${String(error.maxRel)} must stay inside 8 x fp16 eps`);
  // eps protects an all-zero row instead of dividing by zero.
  const zeros = simulateRowNorm({ rows: 1, cols: 4, dtype: 'float32', x: [0, 0, 0, 0], gamma: [1, 1, 1, 1], eps: 1e-6 });
  assert.deepEqual(zeros.output, [0, 0, 0, 0]);
  assert.ok(Number.isFinite(zeros.scales[0]));
});

test('padding the reduction axis does not change the result', () => {
  const rows = 5;
  const cols = 17;
  const x = Array.from({ length: rows * cols }, (_, index) => ((index % 11) - 5) * 0.7);
  const gamma = Array.from({ length: cols }, () => 1.25);
  const plain = simulateRowNorm({ rows, cols, dtype: 'float32', x, gamma });
  const padded = simulateRowNorm({ rows, cols, dtype: 'float32', x, gamma, padTo: 32 });
  assert.equal(padded.padded, true);
  assert.equal(padded.colsAlign, 32);
  const error = compareNumeric(padded.output, plain.output);
  assert.equal(error.maxAbs, 0, 'the padded path must produce identical values');
});

/** The quantise path is lossy by construction — the check is the step bound. */
test('the dynamic quantise round trip stays inside half a step', () => {
  const rows = 16;
  const cols = 96;
  const x = Array.from({ length: rows * cols }, (_, index) => Math.cos(index * 0.11) * (1 + (index % 13)));
  const simulated = simulateQuantize({ rows, cols, dtype: 'float16', x });
  const expected = reference.quantize({ rows, cols, x: x.map((value) => toFp16(value)) });

  // Same stored input on both sides: the arithmetic must agree tightly.
  const error = compareNumeric(simulated.output, expected.output);
  assert.ok(error.within(4.9e-4 * 2), `quantise arithmetic maxRel ${String(error.maxRel)}`);
  // Scale is exactly max|x|/127 for every row (fp32 rounding of 1/127 allowed).
  for (let row = 0; row < rows; row += 1) {
    const slice = x.slice(row * cols, (row + 1) * cols).map((value) => Math.abs(toFp16(value)));
    const expectedScale = Math.max(...slice) / 127;
    assert.ok(
      Math.abs(simulated.scales[row] - expectedScale) / expectedScale < 1e-6,
      `row ${String(row)} scale ${String(simulated.scales[row])} vs ${String(expectedScale)}`,
    );
  }
  // And the round trip respects the int8 step bound.
  const halfStep = Math.max(...simulated.scales) / 2;
  const roundTrip = compareNumeric(simulated.output, x);
  assert.ok(roundTrip.maxAbs <= halfStep + 0.01, `round trip ${String(roundTrip.maxAbs)} vs half step ${String(halfStep)}`);
  // int8 range is respected.
  assert.ok(simulated.quantized.every((value) => value >= -128 && value <= 127));
});

test('the elementwise chain simulates silu(a*b)', () => {
  const a = [0, 1, -1, 2.5, 8];
  const b = [1, 1, 1, -2, 1];
  const simulated = simulateElementwise({ dtype: 'float32', a, b });
  const error = compareNumeric(simulated.output, reference.elementwise({ a, b }));
  assert.ok(error.within(1e-6), `maxRel ${String(error.maxRel)}`);
  // silu(0) = 0, silu(-5) = -5/(1+e^5) ~ -0.0335, silu(8) ~ 8.
  assert.equal(simulated.output[0], 0);
  assert.ok(Math.abs(simulated.output[3] - -0.0335) < 1e-3, `silu(-5) = ${String(simulated.output[3])}`);
  assert.ok(Math.abs(simulated.output[4] - 8 / (1 + Math.exp(-8))) < 1e-5, 'silu saturates towards the input for large positives');
});

/** The self-check wrapper is what the page/report consumes. */
test('self-check reports pass/fail per operator class and dtype', () => {
  for (const kind of ['row-norm', 'quantize', 'elementwise']) {
    for (const dtype of ['float16', 'bfloat16', 'float32']) {
      const report = selfCheck({ kind, dtype, rows: 24, cols: 40, seed: 11 });
      assert.equal(report.passed, true, `${kind}/${dtype}: ${JSON.stringify(report.error)}`);
      assert.ok(report.error.maxRel <= report.tolerance);
      assert.ok(report.notes.length >= 2, 'the report must explain what was checked');
      assert.equal(report.kind, kind);
    }
  }
  // Non-aligned and degenerate shapes must pass too.
  for (const [rows, cols] of [[997, 17], [1, 1], [3, 4096]]) {
    const report = selfCheck({ kind: 'row-norm', dtype: 'float16', rows, cols, seed: 2 });
    assert.equal(report.passed, true, `${String(rows)}x${String(cols)}`);
  }
  // Determinism: the same seed must give the same numbers.
  assert.deepEqual(selfCheck({ kind: 'quantize', seed: 4 }).error, selfCheck({ kind: 'quantize', seed: 4 }).error);
});
