import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
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
    this.wireMesh = null;
    this.balls = [];
    this.trails = [];
    this.targetMarkers = [];
    this.clock = new THREE.Clock();

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05060a);
    this.scene.fog = new THREE.Fog(0x05060a, 18, 46);

    const aspect = canvas.clientWidth / Math.max(canvas.clientHeight, 1);
    this.camera = new THREE.PerspectiveCamera(50, aspect, 0.1, 200);
    this.camera.position.set(8, 7, 10);

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(canvas.clientWidth, canvas.clientHeight, false);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.maxPolarAngle = Math.PI / 2.05;
    this.controls.minDistance = 4;
    this.controls.maxDistance = 40;

    const ambient = new THREE.AmbientLight(0x404a6a, 0.6);
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(6, 12, 8);
    const rim = new THREE.DirectionalLight(0x6c8cff, 0.5);
    rim.position.set(-8, 4, -6);
    this.scene.add(ambient, key, rim);

    const grid = new THREE.GridHelper(16, 16, 0x223, 0x111a2a);
    grid.position.y = -0.02;
    this.scene.add(grid);

    // Post-processing: subtle bloom so the balls / trails / targets glow.
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(
      new THREE.Vector2(canvas.clientWidth, canvas.clientHeight),
      0.7, // strength
      0.5, // radius
      0.75 // threshold (only bright emissive things bloom)
    );
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.composer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.composer.setSize(canvas.clientWidth, canvas.clientHeight);

    this.zScale = 1;
    this.origin = { x: 0, y: 0 };
  }

  _worldPos(x, y, z) {
    return new THREE.Vector3(x - this.origin.x, z * this.zScale, y - this.origin.y);
  }

  setSurface(surface) {
    this._disposeSurface();

    const { xs, ys, zs, resolution } = sampleGrid(surface, 60);
    const { xMin, xMax, yMin, yMax } = surface.domain;

    this.origin = { x: (xMin + xMax) / 2, y: (yMin + yMax) / 2 };
    this.surface = surface;

    let zMin = Infinity;
    let zMax = -Infinity;
    for (const z of zs) {
      zMin = Math.min(zMin, z);
      zMax = Math.max(zMax, z);
    }
    const zRange = Math.max(zMax - zMin, 0.01);
    this.zScale = 4 / zRange;
    this.zMin = zMin;
    this.zRange = zRange;

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

        positions.setXYZ(idx, x - this.origin.x, z * this.zScale, y - this.origin.y);

        const t = (z - zMin) / zRange;
        // deep indigo valleys -> warm magenta peaks
        const color = new THREE.Color().setHSL(0.66 - t * 0.5, 0.65, 0.18 + t * 0.32);
        colors[idx * 3] = color.r;
        colors[idx * 3 + 1] = color.g;
        colors[idx * 3 + 2] = color.b;
      }
    }

    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      metalness: 0.1,
      roughness: 0.7,
      side: THREE.DoubleSide,
      flatShading: false,
    });
    this.surfaceMesh = new THREE.Mesh(geometry, material);
    this.scene.add(this.surfaceMesh);

    // Faint wireframe overlay for the "neat grid" look.
    const wireMat = new THREE.MeshBasicMaterial({
      color: 0x9fb4ff,
      wireframe: true,
      transparent: true,
      opacity: 0.06,
    });
    this.wireMesh = new THREE.Mesh(geometry.clone(), wireMat);
    this.scene.add(this.wireMesh);

    this._buildTargetMarkers(surface);
    this._fitCamera(surface);
  }

  _buildTargetMarkers(surface) {
    for (const m of surface.knownMinima ?? []) {
      const z = surface.f(m.x, m.y);
      const group = new THREE.Group();

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.32, 0.04, 12, 40),
        new THREE.MeshStandardMaterial({
          color: 0xffffff,
          emissive: 0xffe08a,
          emissiveIntensity: 2.2,
        })
      );
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      group.position.copy(this._worldPos(m.x, m.y, z));
      group.position.y += 0.02;
      this.scene.add(group);
      this.targetMarkers.push(group);
    }
  }

  _disposeSurface() {
    for (const obj of [this.surfaceMesh, this.wireMesh]) {
      if (obj) {
        this.scene.remove(obj);
        obj.geometry.dispose();
        obj.material.dispose();
      }
    }
    this.surfaceMesh = null;
    this.wireMesh = null;

    for (const g of this.targetMarkers) {
      this.scene.remove(g);
      g.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose();
      });
    }
    this.targetMarkers = [];
  }

  _fitCamera(surface) {
    const { xMin, xMax, yMin, yMax } = surface.domain;
    const span = Math.max(xMax - xMin, yMax - yMin);
    this.camera.position.set(span * 0.85, span * 0.7, span * 1.05);
    this.controls.target.set(0, 1.4, 0);
    this.controls.update();
  }

  clearMarkers() {
    for (const ball of this.balls) {
      this.scene.remove(ball);
      ball.geometry.dispose();
      ball.material.dispose();
    }
    for (const trail of this.trails) {
      if (!trail) continue;
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
        new THREE.SphereGeometry(0.16, 28, 28),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 1.9,
          metalness: 0.2,
          roughness: 0.35,
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
          opacity: 0.9,
          blending: THREE.AdditiveBlending,
        });
        const trail = new THREE.Line(trailGeo, trailMat);
        trail.frustumCulled = false;
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
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) return;

    ball.position.copy(this._worldPos(x, y, z));

    const trail = this.trails[index];
    if (trail && trailPoints && trailPoints.length > 1) {
      const verts = trailPoints
        .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y) && Number.isFinite(p.z))
        .map((p) => this._worldPos(p.x, p.y, p.z));
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
    this.composer.setSize(w, h);
  }

  render() {
    const t = this.clock.getElapsedTime();
    const pulse = 1 + 0.18 * Math.sin(t * 3);
    for (const g of this.targetMarkers) g.scale.setScalar(pulse);

    this.controls.update();
    this.composer.render();
  }
}

export { BALL_COLORS };
