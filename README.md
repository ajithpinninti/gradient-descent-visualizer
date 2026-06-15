# Gradient Descent Visualizer

> **Watch a ball roll down a 3D loss surface and actually understand gradient descent — built from scratch in JavaScript, no ML libraries.**

[![Live Demo](#)](#) · [How it works](#how-it-works) · [The math](#the-math) · [Run locally](#run-locally)

---

## What you'll learn in 60 seconds

If you've ever wondered *"how does a neural network find the best weights?"* — this is the answer, made visible:

1. **A loss function** is a landscape. High points = bad predictions. Low valleys = good predictions.
2. **The gradient** points uphill (steepest increase). To minimize loss, we walk **opposite** to the gradient.
3. **Gradient descent** repeats: compute gradient → take a small step downhill → repeat until you reach a minimum.
4. **Local vs global minima**: some landscapes have many valleys. Where you start matters — you might get stuck in a *local* minimum instead of the *global* best.

This project lets you **see** all of that in 3D.

---

## Demo

> **Status:** All math verified ✓ (33 tests) · 3D visualization next — open `index.html` once added.

<!-- Replace with GIF once visual layer is ready -->
<!-- ![Gradient descent on a bowl surface](assets/demo-bowl.gif) -->

**Try it:** open `index.html` in your browser (after the visual layer is added).

---

## How it works

This repo is built in **small, verified chunks**:

| Chunk | What | Status |
|-------|------|--------|
| 0 | Repo scaffold + test harness | ✅ |
| 1 | Loss surfaces + analytic gradients (math) | ✅ |
| 2 | Optimizers: SGD, Momentum, RMSProp, Adam | ✅ |
| 3 | Trajectory generator (path downhill) | ✅ |
| 4 | Three.js 3D surface + animated marker | 🔄 |
| 5 | Controls, loss chart, README GIFs | ⏳ |

Every math module has **automated tests** before any pixels are drawn.

---

## The math

### Gradient

For a function `f(x, y)`, the gradient is:

```
∇f = [ ∂f/∂x , ∂f/∂y ]
```

It tells you which direction increases `f` fastest. **Gradient descent** updates:

```
(x, y) ← (x, y) − learning_rate × ∇f(x, y)
```

### Surfaces included

| Surface | Formula | Teaches |
|---------|---------|---------|
| **Bowl** | `x² + y²` | One global minimum — the simplest case |
| **Saddle** | `x² − y²` | Gradient is zero but it's not a minimum |
| **Himmelblau** | `(x²+y−11)² + (x+y²−7)²` | Four equal global minima — start point matters |
| **Rosenbrock** | `(1−x)² + 100(y−x²)²` | Narrow curved valley — why Momentum/Adam help |
| **Wavy bowl** | `x² + y² + 0.15(1−cos2x)(1−cos2y)` | Rippled landscape — global min at origin, hills in between |

### Optimizers (from scratch)

- **SGD** — plain gradient step
- **Momentum** — accumulates velocity, rolls through flat regions
- **RMSProp** — adapts step size per dimension
- **Adam** — combines momentum + adaptive rates (default in most deep learning)

---

## Run locally

**Requirements:** [Node.js](https://nodejs.org/) 18+ (for math tests only — the demo is plain HTML/JS)

```bash
# Clone (once pushed to GitHub)
git clone https://github.com/ajithpinninti/gradient-descent-visualizer.git
cd gradient-descent-visualizer

# Run math tests
npm test

# Open demo (after visual layer is added)
# Just open index.html in Chrome/Firefox, or:
npx serve .
```

---

## Project structure

```
gradient-descent-visualizer/
├── src/
│   ├── surfaces.js      # f(x,y) and ∇f — verified against finite differences
│   ├── optimizers.js    # SGD, Momentum, RMSProp, Adam
│   ├── descent.js       # runDescent() → trajectory array
│   ├── scene3d.js       # Three.js mesh (visual layer)
│   └── main.js          # UI wiring
├── tests/
│   ├── surfaces.test.js
│   ├── optimizers.test.js
│   └── descent.test.js
└── index.html
```

---

## Why this exists

Machine learning tutorials often show gradient descent as a formula. This project makes it **tangible** — a ball rolling down a surface you can rotate, zoom, and experiment with.

Built while working on **[DistilBook](https://distilbook.com)** — turn any topic into clear whiteboard-style explainers. This README is written so you can paste it into DistilBook to auto-generate a walkthrough video of how gradient descent works.

---

## License

MIT — use freely, share widely.

---

## Author

**[ajithpinninti](https://github.com/ajithpinninti)**

If this helped you understand gradient descent, ⭐ the repo — it helps others find it on r/learnmachinelearning.
