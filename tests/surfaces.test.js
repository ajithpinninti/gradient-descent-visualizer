import {
  SURFACES,
  getSurface,
  numericalGrad,
  gradientNorm,
  nearestKnownMinimum,
} from "../src/surfaces.js";
import { assert, assertClose, assertArrayClose, runSuite } from "./harness.js";

const TOL = 1e-4;
const GRAD_TOL = 1e-3;

function randomIn(min, max) {
  return min + Math.random() * (max - min);
}

function verifyAnalyticGrad(surface, n = 20) {
  const { xMin, xMax, yMin, yMax, f, grad } = surface;
  for (let i = 0; i < n; i++) {
    const x = randomIn(xMin, xMax);
    const y = randomIn(yMin, yMax);
    const analytic = grad(x, y);
    const numeric = numericalGrad(f, x, y);
    assertArrayClose(analytic, numeric, GRAD_TOL, `${surface.id}@${x.toFixed(2)},${y.toFixed(2)}`);
  }
}

function verifyKnownMinima(surface) {
  for (const m of surface.knownMinima) {
    const g = surface.grad(m.x, m.y);
    assertArrayClose(g, [0, 0], 0.05, `${surface.id} min at (${m.x}, ${m.y}) grad`);
    // Should be lower than a point offset from minimum
    const fMin = surface.f(m.x, m.y);
    const fOffset = surface.f(m.x + 0.3, m.y + 0.3);
    assert(fMin <= fOffset, `${surface.id}: minimum should be lower than nearby point`);
  }
}

await runSuite("surfaces", {
  "getSurface returns bowl": () => {
    const s = getSurface("bowl");
    assert(s.id === "bowl");
  },

  "getSurface throws on unknown": () => {
    let threw = false;
    try {
      getSurface("nonexistent");
    } catch {
      threw = true;
    }
    assert(threw);
  },

  "bowl f(0,0) === 0": () => {
    assertClose(SURFACES.bowl.f(0, 0), 0, TOL);
  },

  "bowl grad(1,2) === [2,4]": () => {
    assertArrayClose(SURFACES.bowl.grad(1, 2), [2, 4], TOL);
  },

  "saddle grad(0,0) === [0,0] but not a minimum": () => {
    assertArrayClose(SURFACES.saddle.grad(0, 0), [0, 0], TOL);
    // Along x: f increases; along y: f decreases
    assert(SURFACES.saddle.f(0.1, 0) > SURFACES.saddle.f(0, 0));
    assert(SURFACES.saddle.f(0, 0.1) < SURFACES.saddle.f(0, 0));
  },

  "himmelblau has 4 known minima with zero gradient": () => {
    verifyKnownMinima(SURFACES.himmelblau);
    assert(SURFACES.himmelblau.knownMinima.length === 4);
  },

  "rosenbrock minimum at (1,1)": () => {
    assertClose(SURFACES.rosenbrock.f(1, 1), 0, TOL);
    assertArrayClose(SURFACES.rosenbrock.grad(1, 1), [0, 0], 0.01);
  },

  "wavy bowl: origin is global minimum on grid": () => {
    const s = SURFACES.wavy;
    const fOrigin = s.f(0, 0);
    assertClose(fOrigin, 0, TOL);
    const steps = 40;
    for (let i = 0; i <= steps; i++) {
      for (let j = 0; j <= steps; j++) {
        const x = -3 + (6 * i) / steps;
        const y = -3 + (6 * j) / steps;
        assert(s.f(x, y) >= fOrigin - 1e-9, `f(${x.toFixed(2)},${y.toFixed(2)}) should be >= f(0,0)`);
      }
    }
  },

  "wavy bowl: has higher-loss ripples (local structure) away from origin": () => {
    const s = SURFACES.wavy;
    assert(s.f(1.5, 1.5) > s.f(0, 0), "ripple hills exist away from origin");
  },

  "analytic grad matches numerical grad (all surfaces)": () => {
    for (const surface of Object.values(SURFACES)) {
      verifyAnalyticGrad(surface, 15);
    }
  },

  "known minima have zero gradient (where defined)": () => {
    for (const surface of Object.values(SURFACES)) {
      if (surface.knownMinima.length) {
        verifyKnownMinima(surface);
      }
    }
  },

  "nearestKnownMinimum finds bowl origin": () => {
    const m = nearestKnownMinimum(SURFACES.bowl, 0.1, -0.1, 0.5);
    assert(m !== null);
    assertClose(m.x, 0, 0.01);
    assertClose(m.y, 0, 0.01);
  },

  "gradientNorm is non-negative": () => {
    assert(gradientNorm(SURFACES.bowl, 1, 1) >= 0);
    assertClose(gradientNorm(SURFACES.bowl, 0, 0), 0, TOL);
  },
});

console.log("\nAll surface tests passed.");
