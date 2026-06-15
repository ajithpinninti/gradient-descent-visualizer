/**
 * Gradient descent trajectory runner.
 * Pure math — no rendering.
 */

import { gradientAt, lossAt, nearestKnownMinimum } from "./surfaces.js";
import { createOptimizerState, getOptimizer } from "./optimizers.js";

/**
 * Run gradient descent for a given number of steps.
 * @returns {Array<{step, x, y, z, loss, grad, gradNorm}>}
 */
export function runDescent(surface, optimizerId, start, steps, options = {}) {
  const lr = options.lr ?? getOptimizer(optimizerId).defaultLr;
  const state = createOptimizerState(optimizerId, options);
  const optimizer = getOptimizer(optimizerId);

  let x = start.x;
  let y = start.y;
  let optState = state;

  const trajectory = [];

  for (let step = 0; step <= steps; step++) {
    const grad = gradientAt(surface, x, y);
    const loss = lossAt(surface, x, y);
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

    const result = optimizer.step(x, y, grad, lr, optState);
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
