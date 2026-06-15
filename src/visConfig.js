/**
 * Visualization configs per surface — tuned via tests/_bench so every
 * optimizer stays numerically stable (finite) and behaves pedagogically.
 *
 * - clipNorm: gradient clipping cap (keeps steep surfaces stable)
 * - steps: how many descent steps to precompute for the animation
 * - defaultLr: learning rate used in single-optimizer mode
 * - raceLr: per-optimizer learning rate used in "race" mode (SGD needs a
 *   smaller rate than Adam, so one shared rate would make some balls diverge)
 */
export const VIS_CONFIG = {
  bowl: {
    clipNorm: Infinity,
    steps: 120,
    defaultLr: 0.06,
    raceLr: { sgd: 0.06, momentum: 0.04, rmsprop: 0.1, adam: 0.1 },
  },
  saddle: {
    clipNorm: 1.5,
    steps: 60,
    defaultLr: 0.02,
    raceLr: { sgd: 0.02, momentum: 0.012, rmsprop: 0.04, adam: 0.04 },
  },
  himmelblau: {
    clipNorm: 6,
    steps: 160,
    defaultLr: 0.02,
    raceLr: { sgd: 0.02, momentum: 0.02, rmsprop: 0.1, adam: 0.1 },
  },
  rosenbrock: {
    clipNorm: 4,
    steps: 400,
    defaultLr: 0.01,
    raceLr: { sgd: 0.02, momentum: 0.01, rmsprop: 0.05, adam: 0.012 },
  },
  wavy: {
    clipNorm: 6,
    steps: 140,
    defaultLr: 0.05,
    raceLr: { sgd: 0.05, momentum: 0.04, rmsprop: 0.08, adam: 0.08 },
  },
};

export function getVisConfig(id) {
  return VIS_CONFIG[id] ?? { clipNorm: Infinity, steps: 150, defaultLr: 0.05, raceLr: {} };
}
