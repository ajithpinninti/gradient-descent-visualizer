import { SURFACES } from "./surfaces.js";
import { OPTIMIZERS } from "./optimizers.js";
import { runDescent } from "./descent.js";
import { Scene3D, BALL_COLORS } from "./scene3d.js";
import { LossChart } from "./lossChart.js";

const CAPTIONS = {
  bowl: "Convex bowl: one global minimum. Gradient descent always finds the bottom.",
  saddle: "Saddle point: gradient is zero at the center, but it's not a minimum — the surface goes up in some directions.",
  himmelblau: "Four equal minima: where you start determines which valley you fall into.",
  rosenbrock: "Banana valley: narrow curved path. Plain SGD is slow here — try Momentum or Adam.",
  wavy: "Rippled landscape: small hills between you and the global minimum at the center.",
};

const OPTIMIZER_IDS = Object.keys(OPTIMIZERS);
const SURFACE_IDS = Object.keys(SURFACES);

function hexColor(num) {
  return `#${num.toString(16).padStart(6, "0")}`;
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

    this.statStep = document.getElementById("stat-step");
    this.statLoss = document.getElementById("stat-loss");
    this.statGrad = document.getElementById("stat-grad");
    this.statPos = document.getElementById("stat-pos");

    this.trajectories = [];
    this.frameIndices = [];
    this.playing = false;
    this.rafId = null;
    this.lastFrameTime = 0;

    this._populateSelects();
    this._bindEvents();
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
  }

  _bindEvents() {
    this.surfaceSelect.addEventListener("change", () => this.reset());
    this.optimizerSelect.addEventListener("change", () => this.reset());
    this.lrSlider.addEventListener("input", () => {
      this.lrValue.textContent = Number(this.lrSlider.value).toFixed(2);
      this.reset();
    });
    this.speedSlider.addEventListener("input", () => {});
    this.showTrail.addEventListener("change", () => this.reset());
    this.showRace.addEventListener("change", () => this.reset());
    this.btnPlay.addEventListener("click", () => this.togglePlay());
    this.btnReset.addEventListener("click", () => this.reset());

    window.addEventListener("resize", () => {
      this.scene.resize();
      this.chart.resize();
      this.chart.draw();
    });
  }

  get surface() {
    return SURFACES[this.surfaceSelect.value];
  }

  get lr() {
    return Number(this.lrSlider.value);
  }

  reset() {
    this.pause();
    const surface = this.surface;
    const race = this.showRace.checked;
    const optimizerIds = race ? OPTIMIZER_IDS : [this.optimizerSelect.value];

    this.surfaceDesc.textContent = surface.description;
    this.caption.innerHTML = `<strong>${surface.name}:</strong> ${CAPTIONS[surface.id] ?? surface.description}`;

    this.scene.setSurface(surface);
    this.scene.setupMarkers(optimizerIds, this.showTrail.checked);

    const steps = surface.id === "rosenbrock" ? 400 : 200;
    this.trajectories = optimizerIds.map((id) =>
      runDescent(surface, id, surface.defaultStart, steps, { lr: this.lr })
    );
    this.frameIndices = optimizerIds.map(() => 0);

    this._updateStats(0);
    this._updateChart();
    this._renderFrame(0);
    this.btnPlay.textContent = "▶ Play";
  }

  togglePlay() {
    if (this.playing) this.pause();
    else this.play();
  }

  play() {
    this.playing = true;
    this.btnPlay.textContent = "⏸ Pause";
    this.lastFrameTime = performance.now();
  }

  pause() {
    this.playing = false;
    this.btnPlay.textContent = "▶ Play";
  }

  _loop() {
    const now = performance.now();
    const speed = Number(this.speedSlider.value);

    if (this.playing && now - this.lastFrameTime > 1000 / speed) {
      this.lastFrameTime = now;
      let allDone = true;

      for (let i = 0; i < this.trajectories.length; i++) {
        if (this.frameIndices[i] < this.trajectories[i].length - 1) {
          this.frameIndices[i]++;
          allDone = false;
        }
      }

      const leadIdx = this.frameIndices[0];
      this._renderFrame(leadIdx);
      this._updateStats(leadIdx);
      this._updateChart();

      if (allDone) this.pause();
    }

    this.scene.render();
    this.rafId = requestAnimationFrame(() => this._loop());
  }

  _renderFrame(stepIdx) {
    for (let i = 0; i < this.trajectories.length; i++) {
      const traj = this.trajectories[i];
      const idx = Math.min(this.frameIndices[i], traj.length - 1);
      const trailPoints = traj.slice(0, idx + 1);
      const pt = traj[idx];
      this.scene.updateMarker(i, pt.x, pt.y, pt.z, trailPoints);
    }
  }

  _updateStats(stepIdx) {
    const pt = this.trajectories[0]?.[stepIdx];
    if (!pt) return;
    this.statStep.textContent = String(pt.step);
    this.statLoss.textContent = pt.loss.toFixed(4);
    this.statGrad.textContent = pt.gradNorm.toFixed(4);
    this.statPos.textContent = `(${pt.x.toFixed(2)}, ${pt.y.toFixed(2)})`;
  }

  _updateChart() {
    const series = this.trajectories.map((traj, i) => {
      const id = this.showRace.checked ? OPTIMIZER_IDS[i] : this.optimizerSelect.value;
      const idx = this.frameIndices[i];
      return {
        id,
        color: hexColor(BALL_COLORS[id] ?? 0xffffff),
        data: traj.slice(0, idx + 1).map((p) => ({ step: p.step, loss: p.loss })),
      };
    });
    this.chart.setSeries(series);
  }
}

new App();
