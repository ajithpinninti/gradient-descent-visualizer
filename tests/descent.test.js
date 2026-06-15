import { SURFACES } from "../src/surfaces.js";
import { runDescent, runDescentUntilConverged, summarizeDescent } from "../src/descent.js";
import { assert, assertClose, runSuite } from "./harness.js";

await runSuite("descent trajectory", {
  "runDescent returns correct length": () => {
    const traj = runDescent(SURFACES.bowl, "sgd", { x: 2, y: 2 }, 10);
    assert(traj.length === 11, "steps+1 points");
  },

  "z equals f(x,y) at every point": () => {
    const surface = SURFACES.bowl;
    const traj = runDescent(surface, "sgd", { x: 2, y: 2 }, 5, { lr: 0.1 });
    for (const pt of traj) {
      assertClose(pt.z, surface.f(pt.x, pt.y), 1e-9);
      assertClose(pt.loss, pt.z, 1e-9);
    }
  },

  "loss decreases on convex bowl with SGD": () => {
    const traj = runDescent(SURFACES.bowl, "sgd", { x: 2.5, y: 2.5 }, 20, { lr: 0.1 });
    const first = traj[0].loss;
    const last = traj[traj.length - 1].loss;
    assert(last < first, "loss should decrease");
  },

  "grad array has 2 elements at each step": () => {
    const traj = runDescent(SURFACES.bowl, "sgd", { x: 1, y: 1 }, 3);
    for (const pt of traj) {
      assert(pt.grad.length === 2);
      assertClose(pt.gradNorm, Math.hypot(pt.grad[0], pt.grad[1]), 1e-9);
    }
  },

  "runDescentUntilConverged stops when grad norm small": () => {
    const traj = runDescentUntilConverged(
      SURFACES.bowl,
      "sgd",
      { x: 1, y: 1 },
      { lr: 0.1, tolerance: 0.01, maxSteps: 500 }
    );
    const last = traj[traj.length - 1];
    assert(last.gradNorm < 0.01, "should converge");
    assert(traj.length < 500, "should stop early");
  },

  "summarizeDescent reports correct fields": () => {
    const traj = runDescent(SURFACES.bowl, "sgd", { x: 2, y: 2 }, 10, { lr: 0.1 });
    const summary = summarizeDescent(traj, SURFACES.bowl);
    assert(summary.steps === 10);
    assert(typeof summary.startLoss === "number");
    assert(typeof summary.endLoss === "number");
    assert(summary.lossDecreased);
  },

  "wavy bowl: SGD from default start reaches origin": () => {
    const start = SURFACES.wavy.defaultStart;
    const traj = runDescentUntilConverged(SURFACES.wavy, "sgd", start, { lr: 0.05, maxSteps: 3000, tolerance: 0.01 });
    const summary = summarizeDescent(traj, SURFACES.wavy, 0.3);
    assert(summary.reachedKnownMinimum);
    assertClose(summary.endLoss, 0, 0.05);
  },
});

console.log("\nAll descent tests passed.");
