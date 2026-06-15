/**
 * Optimizers — pure update rules, no ML libraries.
 * Each returns { x, y, state } after one step.
 */

export const OPTIMIZERS = {
  sgd: {
    id: "sgd",
    name: "SGD",
    description: "Plain gradient descent: step opposite to the gradient.",
    defaultLr: 0.1,
    createState() {
      return {};
    },
    step(x, y, grad, lr, state) {
      const [gx, gy] = grad;
      return {
        x: x - lr * gx,
        y: y - lr * gy,
        state,
      };
    },
  },

  momentum: {
    id: "momentum",
    name: "Momentum",
    description: "Accumulates velocity — rolls through flat regions and overshoots less on curves.",
    defaultLr: 0.05,
    defaultMomentum: 0.9,
    createState(momentum = 0.9) {
      return { vx: 0, vy: 0, momentum };
    },
    step(x, y, grad, lr, state) {
      const [gx, gy] = grad;
      const { momentum } = state;
      const vx = momentum * state.vx - lr * gx;
      const vy = momentum * state.vy - lr * gy;
      return {
        x: x + vx,
        y: y + vy,
        state: { ...state, vx, vy },
      };
    },
  },

  rmsprop: {
    id: "rmsprop",
    name: "RMSProp",
    description: "Adapts learning rate per dimension using running average of squared gradients.",
    defaultLr: 0.05,
    defaultBeta: 0.9,
    createState(beta = 0.9, eps = 1e-8) {
      return { sx: 0, sy: 0, beta, eps };
    },
    step(x, y, grad, lr, state) {
      const [gx, gy] = grad;
      const { beta, eps } = state;
      const sx = beta * state.sx + (1 - beta) * gx * gx;
      const sy = beta * state.sy + (1 - beta) * gy * gy;
      return {
        x: x - (lr * gx) / Math.sqrt(sx + eps),
        y: y - (lr * gy) / Math.sqrt(sy + eps),
        state: { ...state, sx, sy },
      };
    },
  },

  adam: {
    id: "adam",
    name: "Adam",
    description: "Combines momentum + adaptive rates — default optimizer in most deep learning.",
    defaultLr: 0.1,
    defaultBeta1: 0.9,
    defaultBeta2: 0.999,
    createState(beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
      return { mx: 0, my: 0, vx: 0, vy: 0, t: 0, beta1, beta2, eps };
    },
    step(x, y, grad, lr, state) {
      const [gx, gy] = grad;
      const { beta1, beta2, eps } = state;
      const t = state.t + 1;

      const mx = beta1 * state.mx + (1 - beta1) * gx;
      const my = beta1 * state.my + (1 - beta1) * gy;
      const vx = beta2 * state.vx + (1 - beta2) * gx * gx;
      const vy = beta2 * state.vy + (1 - beta2) * gy * gy;

      const mxHat = mx / (1 - Math.pow(beta1, t));
      const myHat = my / (1 - Math.pow(beta1, t));
      const vxHat = vx / (1 - Math.pow(beta2, t));
      const vyHat = vy / (1 - Math.pow(beta2, t));

      return {
        x: x - (lr * mxHat) / (Math.sqrt(vxHat) + eps),
        y: y - (lr * myHat) / (Math.sqrt(vyHat) + eps),
        state: { ...state, mx, my, vx, vy, t },
      };
    },
  },
};

export function getOptimizer(id) {
  const opt = OPTIMIZERS[id];
  if (!opt) throw new Error(`Unknown optimizer: ${id}`);
  return opt;
}

export function createOptimizerState(id, options = {}) {
  const opt = getOptimizer(id);
  switch (id) {
    case "momentum":
      return opt.createState(options.momentum ?? opt.defaultMomentum);
    case "rmsprop":
      return opt.createState(options.beta ?? opt.defaultBeta);
    case "adam":
      return opt.createState(
        options.beta1 ?? opt.defaultBeta1,
        options.beta2 ?? opt.defaultBeta2
      );
    default:
      return opt.createState();
  }
}
