/**
 * Gradient descent trajectory runner.
 * Pure math — no rendering.
 */

import { gradientAt, lossAt, nearestKnownMinimum } from "./surfaces.js";
import { createOptimizerState, getOptimizer } from "./optimizers.js";

/** True if every value in the descent point is a finite number. */
export function isFinitePoint(x, y, loss, grad) {
  return (
    Number.isFinite(x) &&
    Number.isFinite(y) &&
    Number.isFinite(loss) &&
    Number.isFinite(grad[0]) &&
    Number.isFinite(grad[1])
  );
}

/**
 * Run gradient descent for a given number of steps.
 * Stops early (cleanly) if the optimizer diverges to a non-finite value,
 * so the visual layer never receives NaN/Infinity positions.
 * @returns {Array<{step, x, y, z, loss, grad, gradNorm, diverged?}>}
 */
/** Clip a gradient so its L2 norm never exceeds maxNorm (gradient clipping). */
export function clipGradient(grad, maxNorm) {
  if (!Number.isFinite(maxNorm) || maxNorm <= 0) return grad;
  const norm = Math.hypot(grad[0], grad[1]);
  if (norm <= maxNorm) return grad;
  const scale = maxNorm / norm;
  return [grad[0] * scale, grad[1] * scale];
}

export function runDescent(surface, optimizerId, start, steps, options = {}) {
  const lr = options.lr ?? getOptimizer(optimizerId).defaultLr;
  const clipNorm = options.clipNorm ?? Infinity;
  const state = createOptimizerState(optimizerId, options);
  const optimizer = getOptimizer(optimizerId);

  let x = start.x;
  let y = start.y;
  let optState = state;

  const trajectory = [];

  for (let step = 0; step <= steps; step++) {
    const grad = gradientAt(surface, x, y);
    const loss = lossAt(surface, x, y);

    // Divergence guard: never emit a non-finite point. Mark the last good
    // point as diverged so the UI can show it stopped exploding.
    if (!isFinitePoint(x, y, loss, grad)) {
      if (trajectory.length) trajectory[trajectory.length - 1].diverged = true;
      break;
    }

    const gradNorm = Math.hypot(grad[0], grad[1]);

    trajectory.push({
      step,
      x,
      y,
      z: loss,
      loss,
      grad: [...grad],
      gradNorm,
    });

    if (step === steps) break;

    // Gradient clipping keeps steps bounded on steep surfaces (e.g. Rosenbrock)
    // so the optimizer stays numerically stable. The true gradient is still
    // reported above; only the *step* uses the clipped value.
    const stepGrad = clipGradient(grad, clipNorm);
    const result = optimizer.step(x, y, stepGrad, lr, optState);
    x = result.x;
    y = result.y;
    optState = result.state;
  }

  return trajectory;
}

export function runDescentUntilConverged(
  surface,
  optimizerId,
  start,
  options = {}
) {
  const maxSteps = options.maxSteps ?? 2000;
  const tolerance = options.tolerance ?? 1e-4;
  const lr = options.lr ?? getOptimizer(optimizerId).defaultLr;
  const state = createOptimizerState(optimizerId, options);
  const optimizer = getOptimizer(optimizerId);

  let x = start.x;
  let y = start.y;
  let optState = state;
  const trajectory = [];

  for (let step = 0; step <= maxSteps; step++) {
    const grad = gradientAt(surface, x, y);
    const loss = lossAt(surface, x, y);
    const gradNorm = Math.hypot(grad[0], grad[1]);

    trajectory.push({ step, x, y, z: loss, loss, grad: [...grad], gradNorm });

    if (gradNorm < tolerance) break;
    if (step === maxSteps) break;

    const result = optimizer.step(x, y, grad, lr, optState);
    x = result.x;
    y = result.y;
    optState = result.state;
  }

  return trajectory;
}

export function summarizeDescent(trajectory, surface, tolerance = 0.5) {
  const first = trajectory[0];
  const last = trajectory[trajectory.length - 1];
  const nearest = nearestKnownMinimum(surface, last.x, last.y, tolerance);

  return {
    steps: trajectory.length - 1,
    startLoss: first.loss,
    endLoss: last.loss,
    endX: last.x,
    endY: last.y,
    endGradNorm: last.gradNorm,
    reachedKnownMinimum: nearest !== null,
    nearestMinimum: nearest,
    lossDecreased: last.loss < first.loss,
  };
}
