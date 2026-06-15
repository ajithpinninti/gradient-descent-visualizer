import { SURFACES } from "./surfaces.js";
import { OPTIMIZERS } from "./optimizers.js";
import { runDescent } from "./descent.js";
import { getVisConfig } from "./visConfig.js";
import { Scene3D, BALL_COLORS } from "./scene3d.js";
import { LossChart } from "./lossChart.js";

const CAPTIONS = {
  bowl: "Convex bowl: one global minimum. Gradient descent always finds the bottom.",
  saddle: "Saddle point: the gradient is zero at the center, but it's not a minimum — the ball rolls off instead of settling.",
  himmelblau: "Four equal minima: where you start determines which valley you fall into.",
  rosenbrock: "Banana valley: plain SGD crawls and stalls — watch Momentum and Adam reach the bottom.",
  wavy: "Rippled landscape: small hills on the way down to the global minimum at the center.",
};

const OPTIMIZER_IDS = Object.keys(OPTIMIZERS);
const SURFACE_IDS = Object.keys(SURFACES);

function hexColor(num) {
  return `#${num.toString(16).padStart(6, "0")}`;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

class App {
  constructor() {
    this.canvas3d = document.getElementById("canvas-3d");
    this.canvasLoss = document.getElementById("canvas-loss");
    this.scene = new Scene3D(this.canvas3d);
    this.chart = new LossChart(this.canvasLoss);

    this.surfaceSelect = document.getElementById("surface-select");
    this.optimizerSelect = document.getElementById("optimizer-select");
    this.lrSlider = document.getElementById("lr-slider");
    this.lrValue = document.getElementById("lr-value");
    this.speedSlider = document.getElementById("speed-slider");
    this.btnPlay = document.getElementById("btn-play");
    this.btnReset = document.getElementById("btn-reset");
    this.showTrail = document.getElementById("show-trail");
    this.showRace = document.getElementById("show-race");
    this.surfaceDesc = document.getElementById("surface-desc");
    this.caption = document.getElementById("caption");
    this.legend = document.getElementById("legend");

    this.statStep = document.getElementById("stat-step");
    this.statLoss = document.getElementById("stat-loss");
    this.statGrad = document.getElementById("stat-grad");
    this.statPos = document.getElementById("stat-pos");

    this.trajectories = [];
    this.optIds = [];
    this.maxLen = 1;
    this.playhead = 0; // continuous step position for smooth motion
    this.playing = false;
    this.lastTime = 0;

    this._populateSelects();
    this._bindEvents();
    this._applySurfaceDefaults();
    this.reset();
    this._loop();
  }

  _populateSelects() {
    for (const id of SURFACE_IDS) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = SURFACES[id].name;
      this.surfaceSelect.appendChild(opt);
    }
    for (const id of OPTIMIZER_IDS) {
      const opt = document.createElement("option");
      opt.value = id;
      opt.textContent = OPTIMIZERS[id].name;
      this.optimizerSelect.appendChild(opt);
    }
    this.optimizerSelect.value = "adam";
  }

  _bindEvents() {
    this.surfaceSelect.addEventListener("change", () => {
      this._applySurfaceDefaults();
      this.reset();
    });
    this.optimizerSelect.addEventListener("change", () => this.reset());
    this.lrSlider.addEventListener("input", () => {
      this.lrValue.textContent = Number(this.lrSlider.value).toFixed(3);
      this.reset();
    });
    this.showTrail.addEventListener("change", () => this.reset());
    this.showRace.addEventListener("change", () => {
      this.lrSlider.disabled = this.showRace.checked;
      this.reset();
    });
    this.btnPlay.addEventListener("click", () => this.togglePlay());
    this.btnReset.addEventListener("click", () => {
      this.playhead = 0;
      this.pause();
      this._render();
    });

    window.addEventListener("resize", () => {
      this.scene.resize();
      this.chart.resize();
      this.chart.draw();
    });
  }

  get surface() {
    return SURFACES[this.surfaceSelect.value];
  }

  _applySurfaceDefaults() {
    const cfg = getVisConfig(this.surfaceSelect.value);
    this.lrSlider.value = String(cfg.defaultLr);
    this.lrValue.textContent = cfg.defaultLr.toFixed(3);
  }

  reset() {
    this.pause();
    this.playhead = 0;

    const surface = this.surface;
    const cfg = getVisConfig(surface.id);
    const race = this.showRace.checked;
    this.optIds = race ? OPTIMIZER_IDS : [this.optimizerSelect.value];

    this.surfaceDesc.textContent = surface.description;
    this.caption.innerHTML = `<strong>${surface.name}:</strong> ${CAPTIONS[surface.id] ?? surface.description}`;

    this.scene.setSurface(surface);
    this.scene.setupMarkers(this.optIds, this.showTrail.checked);
    this._updateLegend();

    this.trajectories = this.optIds.map((id) => {
      const lr = race ? cfg.raceLr[id] ?? cfg.defaultLr : Number(this.lrSlider.value);
      return runDescent(surface, id, surface.defaultStart, cfg.steps, {
        lr,
        clipNorm: cfg.clipNorm,
      });
    });

    this.maxLen = Math.max(...this.trajectories.map((t) => t.length));

    this._render();
    this.btnPlay.textContent = "▶ Play";
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  play() {
    if (this.playhead >= this.maxLen - 1) this.playhead = 0;
    this.playing = true;
    this.btnPlay.textContent = "⏸ Pause";
    this.lastTime = performance.now();
  }

  pause() {
    this.playing = false;
    this.btnPlay.textContent = "▶ Play";
  }

  _loop() {
    const now = performance.now();
    if (this.playing) {
      const dt = (now - this.lastTime) / 1000;
      const stepsPerSec = Number(this.speedSlider.value);
      this.playhead += dt * stepsPerSec;

      if (this.playhead >= this.maxLen - 1) {
        this.playhead = this.maxLen - 1;
        this.pause();
      }
      this._render();
    }
    this.lastTime = now;
    this.scene.render();
    requestAnimationFrame(() => this._loop());
  }

  /** Interpolated point on a trajectory at the (float) playhead. */
  _pointAt(traj, playhead) {
    const maxIdx = traj.length - 1;
    if (playhead >= maxIdx) return traj[maxIdx];
    const i = Math.floor(playhead);
    const frac = playhead - i;
    const a = traj[i];
    const b = traj[i + 1];
    const x = lerp(a.x, b.x, frac);
    const y = lerp(a.y, b.y, frac);
    const z = this.surface.f(x, y); // hug the surface for accuracy
    return { x, y, z, loss: z, gradNorm: lerp(a.gradNorm, b.gradNorm, frac), step: a.step };
  }

  _render() {
    for (let i = 0; i < this.trajectories.length; i++) {
      const traj = this.trajectories[i];
      const head = this._pointAt(traj, this.playhead);
      const upto = Math.min(Math.floor(this.playhead) + 1, traj.length);
      const trailPoints = traj.slice(0, upto).map((p) => ({ x: p.x, y: p.y, z: p.z }));
      trailPoints.push({ x: head.x, y: head.y, z: head.z });
      this.scene.updateMarker(i, head.x, head.y, head.z, trailPoints);
    }
    this._updateStats();
    this._updateChart();
  }

  _updateLegend() {
    this.legend.innerHTML = "";
    for (const id of this.optIds) {
      const item = document.createElement("div");
      item.className = "legend-item";

      const dot = document.createElement("span");
      dot.className = "legend-dot";
      const color = hexColor(BALL_COLORS[id] ?? 0xffffff);
      dot.style.background = color;
      dot.style.color = color;

      const label = document.createElement("span");
      label.textContent = OPTIMIZERS[id]?.name ?? id;

      item.appendChild(dot);
      item.appendChild(label);
      this.legend.appendChild(item);
    }
  }

  _updateStats() {
    const traj = this.trajectories[0];
    if (!traj) return;
    const pt = this._pointAt(traj, this.playhead);
    this.statStep.textContent = String(Math.round(this.playhead));
    this.statLoss.textContent = pt.loss.toFixed(4);
    this.statGrad.textContent = pt.gradNorm.toFixed(4);
    this.statPos.textContent = `(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`;
  }

  _updateChart() {
    const upto = Math.floor(this.playhead) + 1;
    const series = this.trajectories.map((traj, i) => {
      const id = this.optIds[i];
      return {
        id,
        color: hexColor(BALL_COLORS[id] ?? 0xffffff),
        data: traj.slice(0, Math.min(upto, traj.length)).map((p) => ({ step: p.step, loss: p.loss })),
      };
    });
    this.chart.setSeries(series);
  }
}

new App();
