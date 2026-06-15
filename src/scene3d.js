import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { sampleGrid } from "./surfaces.js";

const BALL_COLORS = {
  sgd: 0x6c8cff,
  momentum: 0xff6c8c,
  rmsprop: 0x4ade80,
  adam: 0xfbbf24,
};

export class Scene3D {
  constructor(canvas) {
    this.canvas = canvas;
    this.surfaceMesh = null;
    this.trailLines = [];
    this.balls = [];
    this.trails = [];

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0a0a0f);
    this.scene.fog = new THREE.Fog(0x0a0a0f, 20, 50);

    const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 200);
    this.camera.position.set(8, 7, 10);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxPolarAngle = Math.PI / 2.05;

    const ambient = new THREE.AmbientLight(0xffffff, 0.4);
    const dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(5, 10, 7);
    this.scene.add(ambient, dir);

    const grid = new THREE.GridHelper(14, 14, 0x2a2a3a, 0x1a1a2a);
    grid.position.y = -0.01;
    this.scene.add(grid);

    this.zScale = 1;
    this.origin = { x: 0, y: 0 };
  }

  _worldPos(x, y, z) {
    return new THREE.Vector3(
      x - this.origin.x,
      z * this.zScale,
      y - this.origin.y
    );
  }

  setSurface(surface) {
    if (this.surfaceMesh) {
      this.scene.remove(this.surfaceMesh);
      this.surfaceMesh.geometry.dispose();
      this.surfaceMesh.material.dispose();
    }

    const { xs, ys, zs, resolution } = sampleGrid(surface, 50);
    const { xMin, xMax, yMin, yMax } = surface.domain;

    this.origin = {
      x: (xMin + xMax) / 2,
      y: (yMin + yMax) / 2,
    };

    let zMin = Infinity;
    let zMax = -Infinity;
    for (const z of zs) {
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
    }
    const zRange = Math.max(zMax - zMin, 0.01);
    this.zScale = 4 / zRange;

    const geometry = new THREE.PlaneGeometry(
      xMax - xMin,
      yMax - yMin,
      resolution,
      resolution
    );

    const positions = geometry.attributes.position;
    const colors = new Float32Array(positions.count * 3);

    for (let i = 0; i <= resolution; i++) {
      for (let j = 0; j <= resolution; j++) {
        const idx = i * (resolution + 1) + j;
        const x = xs[idx];
        const y = ys[idx];
        const z = zs[idx];

        positions.setXYZ(
          idx,
          x - this.origin.x,
          z * this.zScale,
          y - this.origin.y
        );

        const t = (z - zMin) / zRange;
        const color = new THREE.Color().setHSL(0.62 - t * 0.35, 0.7, 0.35 + t * 0.25);
        colors[idx * 3] = color.r;
        colors[idx * 3 + 1] = color.g;
        colors[idx * 3 + 2] = color.b;
      }
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.15,
      roughness: 0.65,
      side: THREE.DoubleSide,
      wireframe: false,
    });

    this.surfaceMesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.surfaceMesh);

    this._fitCamera(surface);
  }

  _fitCamera(surface) {
    const { xMin, xMax, yMin, yMax } = surface.domain;
    const span = Math.max(xMax - xMin, yMax - yMin);
    this.camera.position.set(span * 0.9, span * 0.7, span * 1.1);
    this.controls.target.set(0, 1.5, 0);
    this.controls.update();
  }

  clearMarkers() {
    for (const ball of this.balls) {
      this.scene.remove(ball);
      ball.geometry.dispose();
      ball.material.dispose();
    }
    for (const trail of this.trails) {
      this.scene.remove(trail);
      trail.geometry.dispose();
      trail.material.dispose();
    }
    this.balls = [];
    this.trails = [];
  }

  setupMarkers(optimizerIds, showTrail) {
    this.clearMarkers();

    for (const id of optimizerIds) {
      const color = BALL_COLORS[id] ?? 0xffffff;
      const ball = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 24, 24),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.35,
          metalness: 0.3,
          roughness: 0.4,
        })
      );
      ball.position.set(0, 2, 0);
      this.scene.add(ball);
      this.balls.push(ball);

      if (showTrail) {
        const trailGeo = new THREE.BufferGeometry();
        const trailMat = new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: 0.75,
        });
        const trail = new THREE.Line(trailGeo, trailMat);
        this.scene.add(trail);
        this.trails.push(trail);
      } else {
        this.trails.push(null);
      }
    }
  }

  updateMarker(index, x, y, z, trailPoints) {
    const ball = this.balls[index];
    if (!ball) return;

    const pos = this._worldPos(x, y, z);
    ball.position.copy(pos);

    const trail = this.trails[index];
    if (trail && trailPoints.length > 1) {
      const verts = trailPoints.map((p) => this._worldPos(p.x, p.y, p.z));
      trail.geometry.setFromPoints(verts);
    }
  }

  resize() {
    const w = this.canvas.clientWidth;
    const h = this.canvas.clientHeight;
    if (w === 0 || h === 0) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

export { BALL_COLORS };
