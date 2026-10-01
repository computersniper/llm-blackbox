// Shared spatial trace viewer. Uses the inference page's renderer and camera.
import { Engine, THREE, label } from '../stage/engine.js';
import { esc } from '../ui.js';

const COLORS = { layer: 0x6b9bff, tower: 0x6b9bff, matrix: 0x5ef0d4, token: 0xffb65c, head: 0xb39bff, weight: 0xff6b93, box: 0x5ef0d4, bar: 0x6b9bff };

export class TraceStage {
  constructor(host, { onPick } = {}) {
    this.host = host;
    this.onPick = onPick;
    this.objects = new Map();
    this.flows = [];
    this.progress = 0;
    this.generation = 0;
    this.spec = null;
    this.mount = document.createElement('div');
    this.mount.className = 'trace-canvas';
    host.prepend(this.mount);
    try {
      this.engine = new Engine(this.mount, {
        onPick: (info) => { if (info) this.onPick?.(info.id); },
        onFrame: (dt, time) => this.animate(dt, time),
        onFreeChange: (free) => { if (this.reset) this.reset.hidden = !free; },
      });
    } catch (error) {
      this.mount.remove();
      console.warn('3D 舞台不可用，保留数据视图。', error.message);
      return;
    }
    host.classList.add('has-lab3d');
    this.group = new THREE.Group();
    this.engine.scene.add(this.group);
    this.chrome = document.createElement('div');
    this.chrome.className = 'trace-chrome';
    this.chrome.innerHTML = '<div class="trace-heading"><small>MODEL TRACE · 3D</small><strong></strong></div><div class="trace-actions"><button type="button" class="trace-data-button" aria-expanded="false" aria-controls="view">查看步骤数据</button><button type="button" class="trace-follow" hidden>回到跟随视角</button></div><p>拖动旋转 · 滚轮推进 · 点击物体选择步骤 · 双击聚焦</p>';
    host.append(this.chrome);
    this.reset = this.chrome.querySelector('.trace-follow');
    this.reset.onclick = () => { this.engine.exitFree(); this.fit(); };
    const toggle = this.chrome.querySelector('.trace-data-button');
    toggle.onclick = () => {
      const open = host.classList.toggle('trace-data-open');
      toggle.setAttribute('aria-expanded', String(open));
      toggle.textContent = open ? '收起步骤数据' : '查看步骤数据';
      const debug = host.querySelector('.dbg');
      if (this.host.clientWidth < 700 && debug) {
        if (open) { this.restoreDebug = !debug.classList.contains('folded'); if (this.restoreDebug) host.querySelector('#btnDbgFold')?.click(); }
        else if (this.restoreDebug && debug.classList.contains('folded')) { host.querySelector('#btnDbgFold')?.click(); this.restoreDebug = false; }
      }
    };
    this.observer = new MutationObserver(() => this.setActive(!document.body.classList.contains('mode-pick')));
    this.observer.observe(document.body, { attributes: true, attributeFilter: ['class'] });
    this.resizeObserver = new ResizeObserver(() => this.fit());
    this.resizeObserver.observe(host);
    const debug = host.querySelector('.dbg');
    if (debug) this.resizeObserver.observe(debug);
    this.setActive(!document.body.classList.contains('mode-pick'));
  }

  setActive(active) { if (this.engine) this.engine.active = !!active; }

  update(spec) {
    if (!this.engine) return;
    const signature = JSON.stringify([spec.depth, spec.title, spec.layout, spec.nodes, spec.edges]);
    const rebuild = signature !== this.signature;
    const depthChanged = spec.depth !== this.spec?.depth;
    this.spec = spec;
    if (this.host.clientWidth < 700 && !this.mobileFolded) {
      const debug = this.host.querySelector('.dbg');
      if (debug && !debug.classList.contains('folded')) this.host.querySelector('#btnDbgFold')?.click();
      this.mobileFolded = true;
    }
    this.chrome.querySelector('strong').textContent = spec.title || '模型运行记录';
    this.chrome.querySelector('p').textContent = (spec.nodes || []).some((node) => node.values?.length > 64)
      ? '拖动旋转 · 点击物体选择步骤 · 数值网格为实测数组抽样'
      : '拖动旋转 · 滚轮推进 · 点击物体选择步骤 · 双击聚焦';
    if (rebuild) {
      const oldPositions = new Map([...this.objects].map(([id, object]) => [id, object.root.position.clone()]));
      this.clear();
      const nodes = spec.nodes || [];
      const tower = spec.layout === 'tower' || (nodes.length > 1 && nodes.every((n) => n.kind === 'layer'));
      const cols = this.host.clientWidth < 700 ? Math.min(3, nodes.length) : spec.layout === 'grid' ? Math.ceil(Math.sqrt(nodes.length)) : nodes.length <= 8 ? nodes.length : Math.min(4, nodes.length);
      nodes.forEach((node, i) => {
        const pos = node.position || (tower ? [0, i * 1.25, 0] : [(i % cols - (cols - 1) / 2) * 5.2, 1.3, (Math.floor(i / cols)) * 4.2]);
        this.addNode({ ...node, labelVisible: node.labelVisible ?? (nodes.length <= 6 || i % Math.ceil(nodes.length / 5) === 0) }, pos);
      });
      const edges = spec.edges || nodes.slice(1).map((n, i) => [nodes[i].id, n.id]);
      for (const [from, to] of edges) this.addEdge(from, to);
      if (nodes.length) {
        const bounds = new THREE.Box3().setFromObject(this.group);
        const size = bounds.getSize(new THREE.Vector3());
        const center = bounds.getCenter(new THREE.Vector3());
        this.center = center;
        this.size = size;
        const floor = new THREE.Mesh(new THREE.BoxGeometry(Math.max(8, size.x + 4), .08, Math.max(7, size.z + 4)), new THREE.MeshStandardMaterial({ color: 0x0a1528, roughness: .5, metalness: .4 }));
        floor.position.set(center.x, bounds.min.y - .4, center.z);
        floor.userData.noFocus = true;
        this.group.add(floor);
        const grid = new THREE.GridHelper(Math.max(size.x + 4, size.z + 4, 8), 16, 0x20394e, 0x112338);
        grid.position.copy(floor.position); grid.position.y += .05;
        this.group.add(grid);
      }
      for (const [id, object] of this.objects) {
        object.target = object.root.position.clone();
        object.root.position.copy(oldPositions.get(id) || object.target.clone().add(new THREE.Vector3(0, -1.1, 0)));
      }
      this.signature = signature;
      if (depthChanged) this.engine.exitFree();
      if (!this.engine.free) this.fit();
    }
    this.active = spec.active;
    for (const [id, object] of this.objects) {
      const selected = id === this.active;
      object.mesh.material.emissiveIntensity = selected ? .65 : .08;
      object.label.element.classList.toggle('on', selected);
      object.label.element.classList.toggle('hide', !selected && !object.labelVisible);
      const detail = object.label.element.querySelector('small');
      if (detail) detail.hidden = !selected;
      object.outline.material.color.setHex(selected ? 0xffb65c : object.color);
      object.outline.material.opacity = selected ? .9 : .32;
    }
    this.setActive(!document.body.classList.contains('mode-pick'));
  }

  addNode(node, position) {
    const color = node.color ?? COLORS[node.kind] ?? COLORS.box;
    const root = new THREE.Group();
    root.position.set(...position);
    this.group.add(root);
    const layer = node.kind === 'layer';
    const tower = node.kind === 'tower';
    const matrix = node.kind === 'matrix' || !!node.imageUrl;
    const barHeight = .12 + 4 * Math.max(0, Math.min(1, Number(node.value) || 0));
    const geometry = layer ? new THREE.BoxGeometry(5.6, .22, 3.2) : tower ? new THREE.BoxGeometry(3.6, 3.6, 2.8) : matrix ? new THREE.BoxGeometry(3.6, 2.7, .2) : node.kind === 'head' ? new THREE.IcosahedronGeometry(1, 1) : node.kind === 'bar' ? new THREE.BoxGeometry(.8, barHeight, .8) : node.kind === 'token' ? new THREE.BoxGeometry(1.9, 1, 1.3) : new THREE.BoxGeometry(2.9, 1.65, 2);
    const material = new THREE.MeshStandardMaterial({ color: 0x142c43, emissive: color, emissiveIntensity: .08, metalness: .35, roughness: .34, transparent: true, opacity: layer ? .68 : .88 });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.pick = { id: node.id, click: true };
    root.add(mesh);
    if (tower) {
      material.opacity = .14;
      const count = Math.min(64, Math.max(1, Math.round(Number(node.layers) || 1)));
      for (let i = 0; i < count; i++) {
        const plate = new THREE.Mesh(new THREE.BoxGeometry(3.45, .025, 2.6), new THREE.MeshStandardMaterial({ color, transparent: true, opacity: .3, metalness: .35, roughness: .4 }));
        plate.position.y = count === 1 ? 0 : -1.6 + 3.2 * i / (count - 1);
        root.add(plate);
      }
    }
    if (node.kind === 'bar') mesh.position.y = barHeight / 2;
    this.engine.pickables.push(mesh);
    const outline = new THREE.LineSegments(new THREE.EdgesGeometry(geometry), new THREE.LineBasicMaterial({ color, transparent: true, opacity: .32 }));
    root.add(outline);
    outline.position.copy(mesh.position);
    const tag = label(`<b>${esc(node.label || node.id)}</b>${node.detail ? `<small>${esc(String(node.detail))}</small>` : ''}`, 'lbl trace-label');
    tag.position.set(0, layer ? .48 : tower ? 2.2 : node.kind === 'bar' ? barHeight + .3 : 1.8, 0);
    root.add(tag);
    if (node.imageUrl) {
      const aspect = Number(node.imageAspect) > 0 ? Number(node.imageAspect) : 3.45 / 2.55;
      const width = Math.min(3.45, 2.55 * aspect), height = Math.min(2.55, 3.45 / aspect);
      const plane = new THREE.Mesh(new THREE.PlaneGeometry(width, height), new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide }));
      plane.position.z = .12;
      root.add(plane);
      const generation = this.generation;
      new THREE.TextureLoader().load(node.imageUrl, (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        if (generation === this.generation) { plane.material.map = texture; plane.material.needsUpdate = true; }
        else texture.dispose();
      });
    } else if (node.values?.length) this.addValues(root, node.values, layer, matrix);
    if (node.kind === 'head') {
      const ring = new THREE.Mesh(new THREE.TorusGeometry(1.4, .025, 8, 48), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: .6 }));
      ring.rotation.x = Math.PI / 2; root.add(ring);
    }
    this.objects.set(node.id, { root, mesh, label: tag, outline, color, labelVisible: node.labelVisible });
  }

  addValues(root, raw, layer, matrixPanel) {
    // Read real values only; evenly sample large vectors to bound rendering cost.
    const count = Math.min(64, raw.length);
    const values = Array.from({ length: count }, (_, i) => Number(raw[Math.floor(i * raw.length / count)]) || 0);
    const max = Math.max(...values.map(Math.abs), 1e-9);
    const cols = Math.ceil(Math.sqrt(count));
    const cell = layer ? .22 : .27;
    const cells = new THREE.InstancedMesh(new THREE.BoxGeometry(cell * .78, cell * .78, .08), new THREE.MeshBasicMaterial(), count);
    const matrix = new THREE.Matrix4();
    const teal = new THREE.Color(0x5ef0d4), rose = new THREE.Color(0xff6b93), dark = new THREE.Color(0x112338);
    values.forEach((value, i) => {
      if (layer) matrix.makeRotationX(-Math.PI / 2).setPosition((i % cols - (cols - 1) / 2) * cell, .14, (Math.floor(i / cols) - (cols - 1) / 2) * cell);
      else matrix.makeTranslation((i % cols - (cols - 1) / 2) * cell, (Math.floor(i / cols) - (cols - 1) / 2) * cell, matrixPanel ? .14 : 1.03);
      cells.setMatrixAt(i, matrix);
      cells.setColorAt(i, dark.clone().lerp(value < 0 ? rose : teal, Math.sqrt(Math.abs(value) / max)));
    });
    root.add(cells);
  }

  addEdge(from, to) {
    const a = this.objects.get(from), b = this.objects.get(to);
    if (!a || !b) return;
    const start = a.root.position.clone(), end = b.root.position.clone();
    const middle = start.clone().lerp(end, .5); middle.y += .65;
    const curve = new THREE.QuadraticBezierCurve3(start, middle, end);
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(32)), new THREE.LineBasicMaterial({ color: 0x5ef0d4, transparent: true, opacity: .24 }));
    this.group.add(line);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(.075, 8, 8), new THREE.MeshBasicMaterial({ color: 0x5ef0d4 }));
    this.group.add(dot);
    this.flows.push({ from, to, curve, dot, line });
  }

  fit(follow = true) {
    if (!this.engine || !this.center) return;
    const small = this.host.clientWidth < 700;
    const debug = this.host.querySelector('.dbg');
    const bottom = small ? (debug?.getBoundingClientRect().height || 32) + 136 : 110;
    this.engine.setInsets(small ? 0 : (debug?.getBoundingClientRect().width || 0) + 28, bottom);
    if (this.engine.free || !follow) return;
    const angle = new THREE.Vector3(small ? .24 : .7, small ? .62 : .48, 1).normalize();
    const right = new THREE.Vector3(1, 0, -angle.x / angle.z).normalize();
    const up = new THREE.Vector3().crossVectors(angle, right).normalize();
    const projected = (axis) => Math.abs(axis.x) * this.size.x + Math.abs(axis.y) * this.size.y + Math.abs(axis.z) * this.size.z;
    const distance = this.engine.fitDistance(projected(right) + 3, projected(up) + 2, 1.12);
    const direction = angle.multiplyScalar(distance);
    this.engine.setView(this.center.clone().add(direction), this.center, { snap: !this.hasView });
    this.hasView = true;
  }

  frame(progress) { this.progress = Number.isFinite(progress) ? progress : 0; }

  animate(dt, time) {
    for (const flow of this.flows) {
      const selected = flow.from === this.active || flow.to === this.active;
      flow.dot.visible = selected;
      flow.line.material.opacity = selected ? .58 : .16;
      if (selected) flow.dot.position.copy(flow.curve.getPoint(Math.max(0, Math.min(1, this.progress))));
    }
    for (const [id, object] of this.objects) {
      if (object.target) object.root.position.lerp(object.target, 1 - Math.exp(-dt * 5));
      if (id === this.active) object.mesh.material.emissiveIntensity = .45 + Math.sin(time * 2) * .12;
    }
  }

  clear() {
    this.generation++;
    this.group.traverse((object) => {
      object.geometry?.dispose();
      if (object.material) {
        for (const material of Array.isArray(object.material) ? object.material : [object.material]) { material.map?.dispose(); material.dispose(); }
      }
      if (object.isCSS2DObject) object.element.remove();
    });
    this.group.clear(); this.objects.clear(); this.flows = []; this.engine.pickables = [];
  }
}
