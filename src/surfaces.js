/**
 * Loss surfaces f(x, y) with analytic gradients.
 * Each surface teaches one gradient-descent concept.
 */

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

/** Numerical gradient via central finite differences (for test verification). */
export function numericalGrad(f, x, y, h = 1e-5) {
  const dfdx = (f(x + h, y) - f(x - h, y)) / (2 * h);
  const dfdy = (f(x, y + h) - f(x, y - h)) / (2 * h);
  return [dfdx, dfdy];
}

export const SURFACES = {
  bowl: {
    id: "bowl",
    name: "Bowl (convex)",
    description: "One global minimum at (0, 0). The simplest gradient descent case.",
    domain: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
    f(x, y) {
      return x * x + y * y;
    },
    grad(x, y) {
      return [2 * x, 2 * y];
    },
    knownMinima: [{ x: 0, y: 0 }],
    defaultStart: { x: 2.5, y: 2.5 },
  },

  saddle: {
    id: "saddle",
    name: "Saddle point",
    description: "Gradient is zero at the origin, but it is a saddle — not a minimum.",
    domain: { xMin: -2, xMax: 2, yMin: -2, yMax: 2 },
    f(x, y) {
      return x * x - y * y;
    },
    grad(x, y) {
      return [2 * x, -2 * y];
    },
    knownMinima: [],
    defaultStart: { x: 1.5, y: 1.5 },
  },

  himmelblau: {
    id: "himmelblau",
    name: "Himmelblau (4 minima)",
    description: "Four equal global minima — where you start determines where you land.",
    domain: { xMin: -5, xMax: 5, yMin: -5, yMax: 5 },
    f(x, y) {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return a * a + b * b;
    },
    grad(x, y) {
      const a = x * x + y - 11;
      const b = x + y * y - 7;
      return [4 * x * a + 2 * b, 2 * a + 4 * y * b];
    },
    knownMinima: [
      { x: 3.0, y: 2.0 },
      { x: -2.805118, y: 3.131312 },
      { x: -3.779310, y: -3.283186 },
      { x: 3.584428, y: -1.848126 },
    ],
    defaultStart: { x: 0, y: 0 },
  },

  rosenbrock: {
    id: "rosenbrock",
    name: "Rosenbrock (banana valley)",
    description: "A narrow curved valley — plain SGD is slow; Momentum and Adam help.",
    domain: { xMin: -2, xMax: 2, yMin: -1, yMax: 3 },
    f(x, y) {
      const a = 1 - x;
      const b = y - x * x;
      return a * a + 100 * b * b;
    },
    grad(x, y) {
      const b = y - x * x;
      return [-2 * (1 - x) - 400 * x * b, 200 * b];
    },
    knownMinima: [{ x: 1, y: 1 }],
    defaultStart: { x: -1.5, y: 2.5 },
  },

  wavy: {
    id: "wavy",
    name: "Wavy bowl (local minima)",
    description: "Ripples create local minima; the deepest valley is still at (0, 0).",
    domain: { xMin: -3, xMax: 3, yMin: -3, yMax: 3 },
    f(x, y) {
      // Always >= 0: bowl + non-negative ripple term. Min at origin.
      const ripple = 0.15 * (1 - Math.cos(2 * x)) * (1 - Math.cos(2 * y));
      return x * x + y * y + ripple;
    },
    grad(x, y) {
      const dRippleDx = 0.15 * 2 * Math.sin(2 * x) * (1 - Math.cos(2 * y));
      const dRippleDy = 0.15 * 2 * (1 - Math.cos(2 * x)) * Math.sin(2 * y);
      return [2 * x + dRippleDx, 2 * y + dRippleDy];
    },
    knownMinima: [{ x: 0, y: 0 }],
    defaultStart: { x: 2.5, y: -2.0 },
  },
};

export function getSurface(id) {
  const surface = SURFACES[id];
  if (!surface) {
    throw new Error(`Unknown surface: ${id}`);
  }
  return surface;
}

export function lossAt(surface, x, y) {
  return surface.f(x, y);
}

export function gradientAt(surface, x, y) {
  return surface.grad(x, y);
}

export function gradientNorm(surface, x, y) {
  const [gx, gy] = surface.grad(x, y);
  return Math.hypot(gx, gy);
}

/** Sample z values on a grid for mesh rendering (used later in visual layer). */
export function sampleGrid(surface, resolution = 40) {
  const { xMin, xMax, yMin, yMax } = surface.domain;
  const xs = [];
  const ys = [];
  const zs = [];

  for (let i = 0; i <= resolution; i++) {
    const x = xMin + (i / resolution) * (xMax - xMin);
    for (let j = 0; j <= resolution; j++) {
      const y = yMin + (j / resolution) * (yMax - yMin);
      xs.push(x);
      ys.push(y);
      zs.push(surface.f(x, y));
    }
  }

  return { xs, ys, zs, resolution };
}

export function nearestKnownMinimum(surface, x, y, tolerance = 0.5) {
  if (!surface.knownMinima.length) return null;

  let best = null;
  let bestDist = Infinity;

  for (const m of surface.knownMinima) {
    const d = Math.hypot(x - m.x, y - m.y);
    if (d < bestDist) {
      bestDist = d;
      best = m;
    }
  }

  return bestDist <= tolerance ? best : null;
}

export { clamp };
