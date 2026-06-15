# Gradient Descent Visualizer

> **Watch a ball roll down a 3D loss surface and actually understand gradient descent — built from scratch in JavaScript, no ML libraries.**

[![Live Demo](#)](#) · [How it works](#how-it-works) · [The math](#the-math) · [Run locally](#run-locally)

📖 **Full write-up:** [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) — the complete, fact-checked explanation of the project, the math, how it was built, and measured observations. 🎬 **Video script:** [`docs/VIDEO_SCRIPT.md`](docs/VIDEO_SCRIPT.md) — scene-by-scene narration ready for DistilBook.

---

## What you'll learn in 60 seconds

If you've ever wondered *"how does a neural network find the best weights?"* — this is the answer, made visible:

1. **A loss function** is a landscape. High points = bad predictions. Low valleys = good predictions.
2. **The gradient** points uphill (steepest increase). To minimize loss, we walk **opposite** to the gradient.
3. **Gradient descent** repeats: compute gradient → take a small step downhill → repeat until you reach a minimum.
4. **Local vs global minima**: some landscapes have many valleys. Where you start matters — you might get stuck in a *local* minimum instead of the *global* best.

This project lets you **see** all of that in 3D — drag to orbit, scroll to zoom, press Play, and watch the ball roll.

---

## Demo

> **Status:** All math verified ✓ (33 tests) · 3D demo ready — see [Run locally](#run-locally).

<!-- Add GIF here after recording from the demo -->
<!-- ![Optimizer race on Rosenbrock surface](assets/demo-race.gif) -->

**Default view:** a ball starts on a smooth bowl and rolls to the bottom using gradient descent. Switch surfaces to see saddles, multiple minima, and narrow valleys.

**Optimizer race:** enable "Optimizer race" to watch SGD, Momentum, RMSProp, and Adam descend the same surface simultaneously — great for seeing why Adam is the default in deep learning.

---

## How it works (for DistilBook / video generation)

This section is written so you can paste the README into **DistilBook** to auto-generate a whiteboard explainer video.

### Scene 1 — The loss landscape

Imagine a 3D terrain. Height = error (loss). The neural network's job is to find the **lowest point** on this terrain — that's the best set of weights.

We visualize five classic surfaces:

- **Bowl** — one simple valley. Always works.
- **Saddle** — flat in the middle but curves up and down in different directions. Shows why "gradient = 0" doesn't always mean "found the answer."
- **Himmelblau** — four identical bottom points. Same algorithm, different starting points → different answers.
- **Rosenbrock** — a long curved banana-shaped valley. Plain gradient descent crawls; smarter optimizers fly.
- **Wavy bowl** — ripples on the way down. Shows why learning rate and momentum matter.

### Scene 2 — The gradient

At any point on the surface, the **gradient** ∇f tells you the steepest uphill direction. Gradient descent flips that:

```
new position = old position − learning_rate × gradient
```

In the demo, the ball always moves **opposite** to the gradient arrow — downhill.

### Scene 3 — One step, repeated

Each frame of the animation is one step:

1. Compute loss at current (x, y)
2. Compute gradient at current (x, y)
3. Optimizer decides how big a step to take
4. Move the ball
5. Repeat

The **loss chart** at the bottom shows loss dropping over steps — that's the whole training loop in miniature.

### Scene 4 — Optimizers compared

| Optimizer | Idea | When it shines |
|-----------|------|----------------|
| **SGD** | Plain step downhill | Simple surfaces (bowl) |
| **Momentum** | Build up speed, roll through flats | Rosenbrock, long valleys |
| **RMSProp** | Smaller steps where gradient is large | Noisy or uneven terrain |
| **Adam** | Momentum + adaptive rates | Default choice in deep learning |

Toggle **Optimizer race** to see all four balls on the same surface.

### Scene 5 — Connection to neural networks

When you train a neural network, there are millions of weights — so the "landscape" has millions of dimensions. We can't draw that, but the **idea is identical**: compute loss, compute gradient (via backpropagation), take a step, repeat.

This 2D demo is the same algorithm, just on a surface you can see.

---

## How it works

This repo is built in **small, verified chunks**:

| Chunk | What | Status |
|-------|------|--------|
| 0 | Repo scaffold + test harness | ✅ |
| 1 | Loss surfaces + analytic gradients (math) | ✅ |
| 2 | Optimizers: SGD, Momentum, RMSProp, Adam | ✅ |
| 3 | Trajectory generator (path downhill) | ✅ |
| 4 | Three.js 3D surface + animated marker | ✅ |
| 5 | Controls, loss chart, README GIFs | ✅ |

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

**Requirements:** [Node.js](https://nodejs.org/) 18+ (for math tests). The demo needs a local server (ES modules).

```bash
git clone https://github.com/ajithpinninti/gradient-descent-visualizer.git
cd gradient-descent-visualizer

# Run all math tests (33 tests — must pass before trusting the visuals)
npm test

# Start local server and open demo
npx serve .
# Then open http://localhost:3000 in Chrome or Firefox
```

**Quick test checklist (open in browser):**

1. Default **Bowl** surface loads — colored 3D terrain visible
2. Click **▶ Play** — blue ball rolls toward center, loss chart drops
3. Switch to **Himmelblau** — ball finds one of four minima
4. Enable **Optimizer race** — four colored balls descend together
5. Switch to **Rosenbrock** — Adam/Momentum beat plain SGD visually

---

## Project structure

```
gradient-descent-visualizer/
├── src/
│   ├── surfaces.js      # f(x,y) and ∇f — verified against finite differences
│   ├── optimizers.js    # SGD, Momentum, RMSProp, Adam
│   ├── descent.js       # runDescent() + gradient clipping + divergence guard
│   ├── visConfig.js     # tuned per-surface learning rates / clip values
│   ├── scene3d.js       # Three.js mesh + bloom glow (visual layer)
│   ├── lossChart.js     # 2D synced loss curve
│   └── main.js          # UI wiring + smooth animation
├── tests/
│   ├── surfaces.test.js
│   ├── optimizers.test.js
│   └── descent.test.js
└── index.html
```

### Numerical stability (a real ML detail)

Steep surfaces like **Rosenbrock** have huge gradients near the start. With one shared learning rate, plain SGD would "explode" to infinity and the ball would vanish. The demo handles this exactly like real deep-learning code:

- **Gradient clipping** — the step size is capped so it never blows up (the *true* gradient is still shown in the stats; only the *step* is clipped).
- **Per-optimizer learning rates** in race mode — SGD needs a smaller rate than Adam, so each optimizer gets a rate that keeps it stable. These are tuned and locked in `visConfig.js`.
- **Divergence guard** — if any run still goes non-finite, it stops cleanly instead of drawing `NaN`.

This is why, on Rosenbrock, you'll see **SGD stall in the valley while Momentum and Adam reach the bottom** — that's the real behavior, not a bug.

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
