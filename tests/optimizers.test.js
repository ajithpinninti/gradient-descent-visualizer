import { SURFACES } from "../src/surfaces.js";
import { OPTIMIZERS, getOptimizer, createOptimizerState } from "../src/optimizers.js";
import { runDescent, runDescentUntilConverged, summarizeDescent } from "../src/descent.js";
import { assert, assertClose, runSuite } from "./harness.js";

await runSuite("optimizers", {
  "SGD step on bowl moves toward origin": () => {
    const opt = OPTIMIZERS.sgd;
    const state = opt.createState();
    const result = opt.step(2, 2, [4, 4], 0.1, state);
    assert(result.x < 2 && result.y < 2, "should move toward origin");
    assertClose(result.x, 1.6, 1e-9);
    assertClose(result.y, 1.6, 1e-9);
  },

  "Momentum accumulates velocity": () => {
    const opt = OPTIMIZERS.momentum;
    let state = opt.createState(0.9);
    const r1 = opt.step(2, 2, [4, 4], 0.1, state);
    state = r1.state;
    assert(Math.abs(r1.state.vx) > 0, "velocity should be non-zero");
    const r2 = opt.step(r1.x, r1.y, [4, 4], 0.1, state);
    assert(Math.abs(r2.x - r1.x) > Math.abs(r1.x - 2), "second step should be larger");
  },

  "RMSProp step reduces coordinates on bowl": () => {
    const opt = OPTIMIZERS.rmsprop;
    const state = opt.createState();
    const result = opt.step(2, 2, [4, 4], 0.1, state);
    assert(result.x < 2 && result.y < 2);
  },

  "Adam step reduces coordinates on bowl": () => {
    const opt = OPTIMIZERS.adam;
    const state = opt.createState();
    const result = opt.step(2, 2, [4, 4], 0.1, state);
    assert(result.x < 2 && result.y < 2);
  },

  "getOptimizer throws on unknown": () => {
    let threw = false;
    try { getOptimizer("fake"); } catch { threw = true; }
    assert(threw);
  },

  "createOptimizerState works for all optimizers": () => {
    for (const id of Object.keys(OPTIMIZERS)) {
      const state = createOptimizerState(id);
      assert(state !== undefined);
    }
  },
});

await runSuite("optimizers convergence", {
  "SGD converges on bowl to origin": () => {
    const traj = runDescentUntilConverged(SURFACES.bowl, "sgd", { x: 2.5, y: 2.5 }, { lr: 0.1 });
    const summary = summarizeDescent(traj, SURFACES.bowl, 0.3);
    assert(summary.reachedKnownMinimum, "should reach origin");
    assert(summary.lossDecreased);
    assertClose(summary.endLoss, 0, 0.01);
  },

  "Momentum converges on bowl": () => {
    const traj = runDescentUntilConverged(SURFACES.bowl, "momentum", { x: 2.5, y: 2.5 }, { lr: 0.05 });
    const summary = summarizeDescent(traj, SURFACES.bowl, 0.3);
    assert(summary.reachedKnownMinimum);
    assertClose(summary.endLoss, 0, 0.01);
  },

  "Adam converges on bowl": () => {
    const traj = runDescentUntilConverged(SURFACES.bowl, "adam", { x: 2.5, y: 2.5 }, { lr: 0.1 });
    const summary = summarizeDescent(traj, SURFACES.bowl, 0.3);
    assert(summary.reachedKnownMinimum);
    assertClose(summary.endLoss, 0, 0.05);
  },

  "RMSProp converges on bowl": () => {
    const traj = runDescentUntilConverged(SURFACES.bowl, "rmsprop", { x: 2.5, y: 2.5 }, { lr: 0.05 });
    const summary = summarizeDescent(traj, SURFACES.bowl, 0.3);
    assert(summary.reachedKnownMinimum);
    assertClose(summary.endLoss, 0, 0.05);
  },

  "SGD on himmelblau reaches one of four minima": () => {
    const traj = runDescentUntilConverged(
      SURFACES.himmelblau,
      "sgd",
      { x: 0, y: 0 },
      { lr: 0.01, maxSteps: 5000, tolerance: 0.01 }
    );
    const summary = summarizeDescent(traj, SURFACES.himmelblau, 0.5);
    assert(summary.reachedKnownMinimum, "should reach a known minimum");
    assert(summary.lossDecreased);
    assertClose(summary.endLoss, 0, 0.01);
  },

  "Adam on rosenbrock reaches (1,1)": () => {
    const traj = runDescentUntilConverged(
      SURFACES.rosenbrock,
      "adam",
      { x: -1.5, y: 2.5 },
      { lr: 0.05, maxSteps: 5000, tolerance: 0.01 }
    );
    const summary = summarizeDescent(traj, SURFACES.rosenbrock, 0.5);
    assert(summary.reachedKnownMinimum);
    assertClose(summary.endLoss, 0, 0.05);
  },

  "Momentum beats SGD step count on rosenbrock (typically)": () => {
    const start = { x: -1.5, y: 2.5 };
    const sgdTraj = runDescentUntilConverged(SURFACES.rosenbrock, "sgd", start, { lr: 0.001, maxSteps: 10000, tolerance: 0.05 });
    const momTraj = runDescentUntilConverged(SURFACES.rosenbrock, "momentum", start, { lr: 0.001, maxSteps: 10000, tolerance: 0.05 });
    // Momentum should reach lower loss in same steps OR fewer steps to same tolerance
    const sgdEnd = sgdTraj[sgdTraj.length - 1].loss;
    const momEnd = momTraj[momTraj.length - 1].loss;
    assert(momEnd <= sgdEnd + 0.1, "momentum should not be worse than SGD on rosenbrock");
  },
});

console.log("\nAll optimizer tests passed.");
