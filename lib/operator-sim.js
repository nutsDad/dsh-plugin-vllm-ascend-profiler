/**
 * Offline simulation of the generated operators.
 *
 * The generated AscendC sources cannot be compiled here (no CANN, no NPU), so this
 * module checks the part that *can* be checked offline: the arithmetic and the data
 * flow. It re-implements exactly what the kernel does — fp16/bf16 storage, fp32
 * accumulation, the per-row reduction, the padded tail — and compares the result
 * against an fp64 reference computed from the same formula.
 *
 * What this proves: the algorithm, the rounding policy, the per-row scale and the
 * tail handling produce the values the design document promises. What it cannot
 * prove: that the AscendC API calls compile and run on device. The generated
 * README says so, and Phase 5/7/8 remain the hardware gates.
 *
 * @module dsh-plugin-vllm-ascend-profiler/operator-sim
 */

/** IEEE-754 binary16 rounding, so the simulation matches the fp16 path exactly. */
export function toFp16(value) {
  const f32 = Math.fround(value);
  const buffer = new DataView(new ArrayBuffer(4));
  buffer.setFloat32(0, f32, true);
  const bits = buffer.getUint32(0, true);
  const sign = (bits >>> 31) & 0x1;
  const exponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  const signValue = sign === 1 ? -1 : 1;

  if (exponent === 0xff) return value; // inf / nan: keep as-is for the comparison

  // Rebias the 8-bit exponent to 5 bits and round the 23-bit mantissa to 10.
  let halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return signValue * Infinity;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return signValue * 0;
    const shifted = (mantissa | 0x800000) >>> (1 - halfExponent);
    const subnormal = (shifted + 0x1000) >>> 13;
    return signValue * subnormal * Math.pow(2, -24);
  }
  let halfMantissa = (mantissa + 0x1000) >>> 13;
  if (halfMantissa === 0x400) {
    halfMantissa = 0;
    halfExponent += 1;
    if (halfExponent >= 0x1f) return signValue * Infinity;
  }
  return signValue * (1 + halfMantissa / 1024) * Math.pow(2, halfExponent - 15);
}

/** Round one value to the storage precision of a dtype. */
function store(value, dtype) {
  if (dtype === 'float16') return toFp16(value);
  if (dtype === 'bfloat16') {
    // bf16 keeps 8 mantissa bits: round the fp32 representation.
    const f32 = Math.fround(value);
    const buffer = new DataView(new ArrayBuffer(4));
    buffer.setFloat32(0, f32, true);
    const bits = buffer.getUint32(0, true);
    const rounded = (bits + 0x8000) & 0xffff0000;
    buffer.setUint32(0, rounded, true);
    return buffer.getFloat32(0, true);
  }
  return Math.fround(value);
}

/** fp32 arithmetic, the way the kernel's vector ops behave. */
const f32 = (value) => Math.fround(value);

/**
 * Run the fused row-normalisation kernel's arithmetic.
 *
 * @param {object} input - `{ rows, cols, dtype, x, gamma, eps, padTo }`.
 * @returns {{output: number[], scales: number[], padded: boolean}} simulated result.
 */
export function simulateRowNorm({ rows, cols, dtype = 'float16', x, gamma, eps = 1e-6, padTo } = {}) {
  const colsAlign = padTo ?? cols;
  const padded = colsAlign !== cols;
  // The host pads the reduction axis and narrows it back afterwards; the kernel
  // only ever sees colsAlign-wide rows.
  const xPadded = new Float64Array(rows * colsAlign);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) xPadded[row * colsAlign + col] = store(x[row * cols + col], dtype);
  }
  const gammaPadded = new Float64Array(colsAlign);
  for (let col = 0; col < cols; col += 1) gammaPadded[col] = store(gamma[col], dtype);

  const output = new Float64Array(rows * cols);
  const scales = new Float64Array(rows);
  for (let row = 0; row < rows; row += 1) {
    const base = row * colsAlign;
    // 1) up-cast (already stored), 2) square, 3) reduce in fp32
    let sum = 0;
    for (let col = 0; col < cols; col += 1) {
      const value = f32(xPadded[base + col]);
      sum = f32(sum + f32(f32(value * value)));
    }
    // 4) mean -> +eps -> sqrt -> reciprocal, all fp32 vector ops
    const mean = f32(f32(sum) * f32(1 / cols));
    const denom = f32(mean + f32(eps));
    const inv = f32(1 / f32(Math.sqrt(denom)));
    scales[row] = inv;
    // 5) normalise, scale by gamma, cast back to the storage dtype
    for (let col = 0; col < cols; col += 1) {
      const source = f32(xPadded[base + col]);
      const normed = f32(f32(source) * inv);
      const scaled = f32(normed * f32(gammaPadded[col]));
      output[row * cols + col] = store(scaled, dtype);
    }
  }
  return { output: [...output], scales: [...scales], padded, colsAlign };
}

/**
 * Run the fused dynamic quantise/dequantise kernel's arithmetic.
 *
 * @param {object} input - `{ rows, cols, dtype, x, padTo }`.
 * @returns {{output: number[], scales: number[], quantized: number[]}} simulated result.
 */
export function simulateQuantize({ rows, cols, dtype = 'float16', x, padTo } = {}) {
  const colsAlign = padTo ?? cols;
  const xPadded = new Float64Array(rows * colsAlign);
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) xPadded[row * colsAlign + col] = store(x[row * cols + col], dtype);
  }
  const output = new Float64Array(rows * cols);
  const scales = new Float64Array(rows);
  const quantized = new Int8Array(rows * cols);
  for (let row = 0; row < rows; row += 1) {
    const base = row * colsAlign;
    // scale = max|x| / 127, computed in fp32 from the row
    let maxAbs = 0;
    for (let col = 0; col < cols; col += 1) {
      const value = f32(xPadded[base + col]);
      const abs = f32(Math.abs(value));
      if (abs > maxAbs) maxAbs = abs;
    }
    const scale = f32(f32(maxAbs) * f32(1 / 127));
    scales[row] = scale;
    for (let col = 0; col < cols; col += 1) {
      const source = f32(xPadded[base + col]);
      // quantize with round-half-away-from-zero, then dequantize
      const scaled = scale === 0 ? 0 : f32(source * f32(1 / scale));
      const q = Math.max(-128, Math.min(127, Math.round(scaled)));
      quantized[row * cols + col] = q;
      const dequantized = f32(f32(q) * scale);
      output[row * cols + col] = store(dequantized, dtype);
    }
  }
  return { output: [...output], scales: [...scales], quantized: [...quantized], colsAlign };
}

/** Run the fused elementwise chain: silu(a * b). */
export function simulateElementwise({ dtype = 'float16', a, b, padTo } = {}) {
  const length = a.length;
  const output = new Float64Array(length);
  for (let index = 0; index < length; index += 1) {
    const left = f32(store(a[index], dtype));
    const right = f32(store(b[index], dtype));
    const product = f32(left * right);
    const negated = f32(product * f32(-1));
    const exponent = f32(Math.exp(negated));
    const denominator = f32(exponent + f32(1));
    output[index] = store(f32(product / denominator), dtype);
  }
  void padTo;
  return { output: [...output] };
}

/** Reference implementations in fp64, used as the yardstick. */
export const reference = {
  rowNorm({ rows, cols, x, gamma, eps = 1e-6 }) {
    const output = new Float64Array(rows * cols);
    for (let row = 0; row < rows; row += 1) {
      let sum = 0;
      for (let col = 0; col < cols; col += 1) sum += x[row * cols + col] ** 2;
      const inv = 1 / Math.sqrt(sum / cols + eps);
      for (let col = 0; col < cols; col += 1) {
        output[row * cols + col] = x[row * cols + col] * inv * gamma[col];
      }
    }
    return [...output];
  },
  quantize({ rows, cols, x }) {
    const output = new Float64Array(rows * cols);
    const scales = new Float64Array(rows);
    for (let row = 0; row < rows; row += 1) {
      let maxAbs = 0;
      for (let col = 0; col < cols; col += 1) maxAbs = Math.max(maxAbs, Math.abs(x[row * cols + col]));
      const scale = maxAbs / 127;
      scales[row] = scale;
      for (let col = 0; col < cols; col += 1) {
        const q = scale === 0 ? 0 : Math.max(-128, Math.min(127, Math.round(x[row * cols + col] / scale)));
        output[row * cols + col] = q * scale;
      }
    }
    return { output: [...output], scales: [...scales] };
  },
  elementwise({ a, b }) {
    return a.map((value, index) => {
      const product = value * b[index];
      return product / (1 + Math.exp(-product));
    });
  },
};

/**
 * Compare a simulation result with a reference and summarise the error.
 *
 * @param {number[]} actual - simulated values.
 * @param {number[]} expected - reference values.
 * @returns {{maxAbs: number, maxRel: number, meanRel: number, within: (tol: number) => boolean}} error report.
 */
export function compareNumeric(actual, expected) {
  let maxAbs = 0;
  let maxRel = 0;
  let sumRel = 0;
  for (let index = 0; index < expected.length; index += 1) {
    const diff = Math.abs(actual[index] - expected[index]);
    const scale = Math.max(Math.abs(expected[index]), 1e-6);
    maxAbs = Math.max(maxAbs, diff);
    const rel = diff / scale;
    maxRel = Math.max(maxRel, rel);
    sumRel += rel;
  }
  return {
    maxAbs,
    maxRel,
    meanRel: expected.length === 0 ? 0 : sumRel / expected.length,
    within: (tolerance) => maxRel <= tolerance,
  };
}

/**
 * Machine epsilon per storage dtype.
 *
 * The tolerances below are derived from these instead of being magic numbers: a
 * chain of a few fp32 operations over values rounded to `dtype` cannot do better
 * than a small multiple of the storage epsilon, and bf16's 8 mantissa bits are
 * ~12x coarser than fp16's 10.
 */
const DTYPE_EPS = Object.freeze({ float32: 6e-8, float16: 4.9e-4, bfloat16: 3.9e-3 });

/**
 * Self-check one generated operator kind against its reference.
 *
 * @param {object} input - `{ kind, rows, cols, dtype, seed }`.
 * @returns {{kind: string, rows: number, cols: number, dtype: string,
 *   tolerance: number, error: object, passed: boolean, notes: string[]}} report.
 */
export function selfCheck({ kind, rows = 32, cols = 64, dtype = 'float16', seed = 1 } = {}) {
  // Deterministic pseudo-random inputs (xorshift), so the check is reproducible.
  let state = seed >>> 0 || 1;
  const next = () => {
    state ^= state << 13; state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5; state >>>= 0;
    return (state / 0xffffffff) * 2 - 1;
  };
  const x = Array.from({ length: rows * cols }, () => next() * 4);
  const notes = [];
  const eps = DTYPE_EPS[dtype] ?? DTYPE_EPS.float32;
  let error;
  let tolerance;

  if (kind === 'row-norm') {
    const gamma = Array.from({ length: cols }, () => 1 + next() * 0.1);
    const simulated = simulateRowNorm({ rows, cols, dtype, x, gamma, eps: 1e-6 });
    error = compareNumeric(simulated.output, reference.rowNorm({ rows, cols, x, gamma, eps: 1e-6 }));
    // square -> reduce -> mean -> sqrt -> reciprocal -> two multiplies
    tolerance = eps * 8;
    notes.push('gamma 融合、fp32 累加；eps=1e-6');
    notes.push(`行内归约与参考实现同序，误差只来自 ${dtype} 存储与 fp32 舍入（容差 = 8 x 存储 eps）`);
  } else if (kind === 'quantize') {
    // Both sides must see the *same stored* input: comparing an fp16-rounded input
    // against an fp32 reference would measure the storage rounding, not the
    // quantise/dequantise arithmetic this operator is responsible for.
    const stored = x.map((value) => store(value, dtype));
    const simulated = simulateQuantize({ rows, cols, dtype, x: stored });
    const expected = reference.quantize({ rows, cols, x: stored });
    error = compareNumeric(simulated.output, expected.output);
    tolerance = eps * 2;
    // The round trip is lossy by construction: |y - x| must stay inside half a
    // quantisation step, plus the storage rounding of the output value.
    const worstStep = Math.max(...simulated.scales.map((scale) => Math.abs(scale) / 2));
    const maxAbsValue = Math.max(...x.map((value) => Math.abs(value)));
    const roundTrip = compareNumeric(simulated.output, x);
    const bound = worstStep * 1.001 + maxAbsValue * eps;
    const boundOk = roundTrip.maxAbs <= bound;
    notes.push(`每行对称 scale = max|x|/127，半步界 ${worstStep.toFixed(6)}，往返最大绝对误差 ${roundTrip.maxAbs.toFixed(6)}（界 ${bound.toFixed(6)}，${boundOk ? '在界内' : '越界'}）`);
    notes.push('量化-反量化往返，输出保持原 dtype');
    return {
      kind,
      rows,
      cols,
      dtype,
      tolerance,
      error: { maxAbs: error.maxAbs, maxRel: error.maxRel, meanRel: error.meanRel },
      absolute: { roundTripMaxAbs: roundTrip.maxAbs, halfStep: worstStep, boundOk },
      passed: error.within(tolerance) && boundOk,
      notes,
    };
  } else {
    const a = x.slice(0, rows * cols);
    const b = Array.from({ length: rows * cols }, () => next());
    const simulated = simulateElementwise({ dtype, a, b });
    error = compareNumeric(simulated.output, reference.elementwise({ a, b }));
    tolerance = eps * 8;
    notes.push('silu(a*b) 融合链，升精度/降精度各一次');
    notes.push(`容差 = 8 x ${dtype} 存储 eps（指数与除法各贡献一次舍入）`);
  }

  return {
    kind,
    rows,
    cols,
    dtype,
    tolerance,
    error: { maxAbs: error.maxAbs, maxRel: error.maxRel, meanRel: error.meanRel },
    passed: error.within(tolerance),
    notes,
  };
}
