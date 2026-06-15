export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.series = [];
    this.resize();
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio, 2);
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    this.canvas.width = w * dpr;
    this.canvas.height = h * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = w;
    this.height = h;
  }

  setSeries(seriesList) {
    // seriesList: [{ id, color, data: [{step, loss}] }]
    this.series = seriesList;
    this.draw();
  }

  draw() {
    const { ctx, width, height } = this;
    ctx.clearRect(0, 0, width, height);

    if (!this.series.length || !this.series[0].data.length) return;

    const pad = { l: 40, r: 12, t: 8, b: 20 };
    const plotW = width - pad.l - pad.r;
    const plotH = height - pad.t - pad.b;

    let maxStep = 0;
    let maxLoss = 0;
    let minLoss = Infinity;

    for (const s of this.series) {
      for (const pt of s.data) {
        maxStep = Math.max(maxStep, pt.step);
        maxLoss = Math.max(maxLoss, pt.loss);
        minLoss = Math.min(minLoss, pt.loss);
      }
    }

    if (maxLoss === minLoss) maxLoss = minLoss + 1;
    const lossRange = maxLoss - minLoss;

    // Grid
    ctx.strokeStyle = "#2a2a3a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    ctx.lineTo(pad.l, pad.t + plotH);
    ctx.lineTo(pad.l + plotW, pad.t + plotH);
    ctx.stroke();

    // Labels
    ctx.fillStyle = "#8888a0";
    ctx.font = "10px Segoe UI, sans-serif";
    ctx.fillText("loss", 4, pad.t + 10);
    ctx.fillText(maxStep.toString(), pad.l + plotW - 10, height - 4);
    ctx.fillText("0", pad.l, height - 4);

    for (const s of this.series) {
      if (s.data.length < 2) continue;
      ctx.strokeStyle = s.color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < s.data.length; i++) {
        const pt = s.data[i];
        const x = pad.l + (pt.step / Math.max(maxStep, 1)) * plotW;
        const y = pad.t + plotH - ((pt.loss - minLoss) / lossRange) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }
}
