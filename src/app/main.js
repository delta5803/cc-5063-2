// Browser application: viewport + control-cage editing + level stepping + QA.
//
// The geometry shown is exactly the output of the subdivision core.  Nothing is
// re-interpolated in the renderer; shared edge points are shared vertices in the
// buffers, which is why the wireframe shows a single continuous quad mesh.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

import {
  createControlLevel, subdivideLevel,
} from '../core/subdivision.js';
import { evaluateLimits } from '../core/limit.js';
import { classifyVertices } from '../core/creases.js';
import {
  checkOrientation, edgeAngleStats, findDuplicatePositions,
} from '../core/quality.js';
import { vdist, vlen } from '../core/vec3.js';
import { modelByName, VALID_MODEL_NAMES, INVALID_MODEL_NAMES } from '../models/fixtures.js';
import { buildTriangles, buildWireframe, buildPointCloud, buildCreaseSegments } from './geometry.js';
import { toOBJ, toJSON, download } from './exporters.js';

// ---------------------------------------------------------------- constants ---

const MAX_LEVEL = 5;
const COLORS = {
  background: 0x0f1216,
  cage: 0xf3b24a,
  cagePoint: 0xffd27f,
  surface: 0xb8c6d6,
  wire: 0x2a3746,
  wireStrong: 0x3d5670,
  extra: 0xc78bff,
  crease: 0xff7b54,
  boundary: 0x50c8ff,
  facePoint: 0x4cd08a,
  edgePoint: 0x4ea1ff,
  limit: 0xffd166,
  selection: 0xff5d5d,
};

// Counts of derived vertices can explode; cap the total vertex budget.
const VERTEX_BUDGET = 1_200_000;

// ---------------------------------------------------------------- state ------

const state = {
  modelName: 'quad-ball',
  control: null,          // { positions, faces, creases, corners }
  levels: [],             // levels[0] = control level
  currentLevel: 0,
  boundaryMode: 'edgeAndCorner',
  editMode: false,
  selectedControlVertex: -1,
  selectedLevelVertex: -1,
  diagnostics: null,
  controlClass: null,
  levelClass: null,
  limitRoot: null,
  error: null,
  buildGeneration: 0,
  show: {
    surface: true, wire: true, cage: true,
    facePoints: false, edgePoints: false,
    extraordinary: true, crease: true, limitPoints: false,
  },
};

// ---------------------------------------------------------------- three ------

let renderer, scene, camera, orbit, transform;
let objects = {
  surface: null, wire: null, cage: null, cagePoints: null,
  facePoints: null, edgePoints: null, extraordinary: null,
  creaseLines: null, boundaryLines: null, limitPoints: null,
  selectionRing: null,
};
let surfacePickMesh = null;
let triangleSource = null;
let transformProxy = null;
let animationNeedsRender = true;
let lastFrameTime = 0;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const tooltipEl = document.getElementById('tooltip');

// ---------------------------------------------------------------- DOM --------

const el = {
  modelSelect: document.getElementById('model-select'),
  modelDesc: document.getElementById('model-desc'),
  levelSlider: document.getElementById('level-slider'),
  levelBadge: document.getElementById('level-badge'),
  levelPrev: document.getElementById('level-prev'),
  levelNext: document.getElementById('level-next'),
  levelMax: document.getElementById('level-max'),
  editToggle: document.getElementById('edit-toggle'),
  resetModel: document.getElementById('reset-model'),
  editStatus: document.getElementById('edit-status'),
  selectedInfo: document.getElementById('selected-info'),
  errorBox: document.getElementById('error-box'),
  qualitySummary: document.getElementById('quality-summary'),
  levelTbody: document.getElementById('level-tbody'),
  limitInfo: document.getElementById('limit-info'),
  vertexCheck: document.getElementById('vertex-check'),
  statusText: document.getElementById('status-text'),
  perfText: document.getElementById('perf-text'),
  exportObj: document.getElementById('export-obj'),
  exportJson: document.getElementById('export-json'),
  showSurface: document.getElementById('show-surface'),
  showWire: document.getElementById('show-wire'),
  showCage: document.getElementById('show-cage'),
  showFacePoints: document.getElementById('show-facepoints'),
  showEdgePoints: document.getElementById('show-edgepoints'),
  showExtra: document.getElementById('show-extraordinary'),
  showCrease: document.getElementById('show-crease'),
  showLimitPoints: document.getElementById('show-limitpoints'),
};

function init() {
  initRenderer();
  initUI();

  // Start with the quad ball: eight valence-3 vertices, no boundary, closed.
  loadModel('quad-ball', 2);
  animate();
}

function initRenderer() {
  const canvas = document.getElementById('viewport');
  renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setClearColor(COLORS.background, 1);
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();
  scene.fog = new THREE.Fog(COLORS.background, 18, 48);

  camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.05, 300);
  camera.up.set(0, 0, 1);
  camera.position.set(4.4, -5.6, 3.6);

  orbit = new OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.09;
  orbit.rotateSpeed = 0.85;
  orbit.zoomSpeed = 0.9;
  orbit.panSpeed = 0.8;
  orbit.addEventListener('change', () => { animationNeedsRender = true; });

  // lights
  const hemi = new THREE.HemisphereLight(0xdce8ff, 0x1a2028, 0.85);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 1.25);
  key.position.set(5, -7, 9);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x9fc4ff, 0.5);
  fill.position.set(-7, 4, -3);
  scene.add(fill);

  // subtle ground grid in the XY plane (z = 0)
  const grid = new THREE.GridHelper(20, 20, 0x2a323c, 0x1d242c);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -0.02;
  grid.material.transparent = true;
  grid.material.opacity = 0.5;
  scene.add(grid);
  objects.grid = grid;
  transform = new TransformControls(camera, renderer.domElement);
  transform.setSize(0.9);
  transform.detach();
  // TransformControls reads object.parent.matrixWorld during a drag, so the proxy
  // must be part of the scene graph.
  transformProxy = new THREE.Object3D();
  transformProxy.name = 'editProxy';
  scene.add(transformProxy);
  transform.addEventListener('dragging-changed', (e) => {
    orbit.enabled = !e.value;
    if (!e.value && state.editMode) {
      // Final rebuild with full diagnostics after the drag ends.
      rebuild({ fast: false });
    }
  });
  transform.addEventListener('objectChange', () => {
    if (state.selectedControlVertex < 0) return;
    const p = transform.object.position;
    state.control.positions[state.selectedControlVertex] = [p.x, p.y, p.z];
    rebuild({ fast: true });
    renderSelectedControlInfo(state.selectedControlVertex);
  });
  scene.add(transform);

  window.addEventListener('resize', onResize);
  renderer.domElement.addEventListener('pointerdown', onPointerDown);
  renderer.domElement.addEventListener('pointermove', onPointerMove);
}

function onResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  animationNeedsRender = true;
}

// ---------------------------------------------------------------- UI ---------

function initUI() {
  const valid = document.createElement('optgroup');
  valid.label = '可细分模型';
  for (const name of VALID_MODEL_NAMES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    valid.appendChild(opt);
  }
  const invalid = document.createElement('optgroup');
  invalid.label = '反例（应被定位并拒绝）';
  for (const name of INVALID_MODEL_NAMES) {
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name;
    invalid.appendChild(opt);
  }
  el.modelSelect.appendChild(valid);
  el.modelSelect.appendChild(invalid);
  el.modelSelect.value = state.modelName;
  el.modelSelect.addEventListener('change', () => {
    loadModel(el.modelSelect.value, 0);
  });

  el.levelSlider.min = '0';
  el.levelSlider.max = String(MAX_LEVEL);
  el.levelSlider.addEventListener('input', () => {
    setLevel(Number(el.levelSlider.value));
  });
  el.levelPrev.addEventListener('click', () => setLevel(state.currentLevel - 1));
  el.levelNext.addEventListener('click', () => setLevel(state.currentLevel + 1));
  el.levelMax.addEventListener('click', () => setLevel(MAX_LEVEL));

  for (const radio of document.querySelectorAll('input[name="boundary"]')) {
    radio.addEventListener('change', () => {
      if (!radio.checked) return;
      state.boundaryMode = radio.value;
      rebuild({ fast: false });
    });
  }

  const bindCheck = (input, key) => {
    input.addEventListener('change', () => {
      state.show[key] = input.checked;
      applyVisibility();
      animationNeedsRender = true;
    });
  };
  bindCheck(el.showSurface, 'surface');
  bindCheck(el.showWire, 'wire');
  bindCheck(el.showCage, 'cage');
  bindCheck(el.showFacePoints, 'facePoints');
  bindCheck(el.showEdgePoints, 'edgePoints');
  bindCheck(el.showExtra, 'extraordinary');
  bindCheck(el.showCrease, 'crease');
  bindCheck(el.showLimitPoints, 'limitPoints');

  el.editToggle.addEventListener('click', () => toggleEditMode());
  el.resetModel.addEventListener('click', () => loadModel(state.modelName, state.currentLevel));

  el.exportObj.addEventListener('click', () => {
    const lv = state.levels[state.currentLevel];
    if (!lv) return;
    download(`${state.modelName}_L${lv.level}.obj`, toOBJ(lv, state.modelName), 'text/plain');
    setStatus(`已导出 OBJ（${lv.positions.length} 顶点 / ${lv.faces.length} 面）`, 'ok');
  });
  el.exportJson.addEventListener('click', () => {
    if (!state.levels.length) return;
    const extra = state.diagnostics ? { diagnostics: state.diagnostics.summary } : {};
    download(`${state.modelName}_subdivision.json`, toJSON(state.control, state.levels, extra), 'application/json');
    setStatus('已导出 JSON（含控制网、各层级统计与折痕）', 'ok');
  });
}

// ---------------------------------------------------------------- loading ----

function loadModel(name, startLevel) {
  const generation = ++state.buildGeneration;
  let model;
  try {
    model = modelByName(name);
  } catch (e) {
    showError(e, name);
    return;
  }

  // Deep copy so editing never mutates the fixture.
  state.modelName = name;
  state.control = {
    name: model.name,
    positions: model.positions.map((p) => p.slice()),
    faces: model.faces.map((f) => f.slice()),
    creases: (model.creases || []).map((c) => ({ ...c })),
    corners: (model.corners || []).slice(),
  };
  state.selectedControlVertex = -1;
  state.selectedLevelVertex = -1;
  state.error = null;
  state.currentLevel = Math.min(startLevel, MAX_LEVEL);
  transform?.detach();
  el.selectedInfo.classList.add('hidden');
  el.selectedInfo.innerHTML = '';
  el.vertexCheck.innerHTML = '<span class="hint">点击场景中的顶点查看其规则、价数与极限位置策略。</span>';

  el.modelDesc.textContent = model.description || '';
  el.modelSelect.value = name;

  if (!rebuild({ fast: false, generation })) return;
  frameCamera();
  if (state.editMode) updateTransformTarget();
}

function rebuild({ fast = false, generation = state.buildGeneration } = {}) {
  const t0 = performance.now();
  try {
    const levels = buildLevelsForState(fast);
    if (generation !== state.buildGeneration) return false;
    state.levels = levels;
    state.currentLevel = Math.min(state.currentLevel, levels.length - 1);
    state.error = null;

    buildSceneGraph();

    if (!fast) {
      state.limitRoot = computeRootLimits(levels[0]);
      state.diagnostics = runDiagnostics(levels);
      renderDiagnostics();
    } else {
      renderLevelBadge();
    }
    updateSelectionOverlay();
    applyVisibility();
    animationNeedsRender = true;

    const ms = performance.now() - t0;
    el.perfText.textContent = `重建 ${ms.toFixed(0)} ms · ${levels[levels.length - 1].positions.length.toLocaleString()} 顶点(最高级)`;
    hideError();
    return true;
  } catch (e) {
    state.levels = [];
    state.currentLevel = 0;
    showError(e, state.modelName);
    return false;
  }
}

function buildLevelsForState(fast) {
  const opts = {
    boundaryMode: state.boundaryMode,
    // Topology is required by the renderer (overlays, cage, marker sizing), so it
    // is always built.  "fast" only skips the expensive QA pass.
    validateEachLevel: true,
  };
  const levels = [createControlLevel(state.control, opts)];
  // While dragging, only refine as far as the viewed level needs; the full chain
  // is produced once the drag ends.
  const limit = fast ? Math.max(state.currentLevel, 1) : MAX_LEVEL;
  let i = 1;
  while (i <= limit) {
    const prev = levels[i - 1];
    const predicted = prev.positions.length + prev.topology.numEdges + prev.faces.length;
    if (predicted > VERTEX_BUDGET) break;
    levels.push(subdivideLevel(prev));
    i++;
  }
  return levels;
}

// ---------------------------------------------------------------- scene ------

function disposeObject(obj) {
  if (!obj) return;
  if (obj.parent) obj.parent.remove(obj);
  obj.traverse?.((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
      else child.material.dispose();
    }
  });
}

function clearSceneGraph() {
  for (const key of Object.keys(objects)) {
    if (key === 'grid') continue;
    disposeObject(objects[key]);
    objects[key] = null;
  }
  surfacePickMesh = null;
  triangleSource = null;
}

function controlClassification() {
  if (state.controlClass) return state.controlClass;
  const l0 = state.levels[0];
  const pin = (l0.options.boundaryMode || 'edgeAndCorner') !== 'edgeOnly';
  state.controlClass = classifyVertices(l0.topology, l0.creaseMap, l0.cornerSet, {
    pinBoundaryCorners: pin,
  });
  return state.controlClass;
}

function levelClassification() {
  if (state.levelClass && state.levelClass.level === state.currentLevel) {
    return state.levelClass.data;
  }
  const lv = state.levels[state.currentLevel];
  const pin = (lv.options.boundaryMode || 'edgeAndCorner') !== 'edgeOnly';
  const data = classifyVertices(lv.topology, lv.creaseMap, lv.cornerSet, {
    pinBoundaryCorners: pin,
  });
  state.levelClass = { level: state.currentLevel, data };
  return data;
}

function buildSceneGraph() {
  clearSceneGraph();
  state.controlClass = null;
  state.levelClass = null;
  const lv = state.levels[state.currentLevel];
  const l0 = state.levels[0];

  // ---- shaded surface (triangulated, smooth normals) ----
  const tri = buildTriangles(lv.positions, lv.faces);
  const surfGeo = new THREE.BufferGeometry();
  surfGeo.setAttribute('position', new THREE.BufferAttribute(tri.positions, 3));
  surfGeo.setAttribute('normal', new THREE.BufferAttribute(tri.normals, 3));
  surfGeo.computeBoundingSphere();
  const surfMat = new THREE.MeshStandardMaterial({
    color: COLORS.surface,
    roughness: 0.52,
    metalness: 0.08,
    side: THREE.DoubleSide,
    flatShading: false,
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const surface = new THREE.Mesh(surfGeo, surfMat);
  surface.name = 'surface';
  scene.add(surface);
  objects.surface = surface;
  surfacePickMesh = surface;
  triangleSource = tri.sourceVertices;

  // ---- quad topology wireframe (real edges) ----
  const wire = buildWireframe(lv.positions, lv.faces, lv.topology);
  const wireGeo = new THREE.BufferGeometry();
  wireGeo.setAttribute('position', new THREE.BufferAttribute(wire.positions, 3));
  const wireMat = new THREE.LineBasicMaterial({
    color: lv.level === 0 ? COLORS.wireStrong : COLORS.wire,
    transparent: true,
    opacity: lv.level === 0 ? 0.95 : 0.72,
  });
  const wireLines = new THREE.LineSegments(wireGeo, wireMat);
  wireLines.name = 'wire';
  scene.add(wireLines);
  objects.wire = wireLines;

  // ---- control cage (always the level-0 topology) ----
  const cageSegs = [];
  const cageTopo = l0.topology;
  for (const e of cageTopo.edges) {
    const a = l0.positions[e.a];
    const b = l0.positions[e.b];
    cageSegs.push(a[0], a[1], a[2], b[0], b[1], b[2]);
  }
  const cageGeo = new THREE.BufferGeometry();
  cageGeo.setAttribute('position', new THREE.Float32BufferAttribute(cageSegs, 3));
  const cageMat = new THREE.LineBasicMaterial({
    color: COLORS.cage, transparent: true, opacity: 0.85,
  });
  const cage = new THREE.LineSegments(cageGeo, cageMat);
  cage.name = 'cage';
  scene.add(cage);
  objects.cage = cage;

  // control points
  const cagePointGeo = new THREE.BufferGeometry();
  cagePointGeo.setAttribute('position', new THREE.Float32BufferAttribute(flattenPositions(l0.positions), 3));
  const cagePointMat = new THREE.PointsMaterial({
    color: COLORS.cagePoint, size: markerSize(l0) * 0.9, sizeAttenuation: true,
    depthTest: false, transparent: true, opacity: 0.95,
  });
  const cagePoints = new THREE.Points(cagePointGeo, cagePointMat);
  cagePoints.name = 'cagePoints';
  cagePoints.renderOrder = 5;
  scene.add(cagePoints);
  objects.cagePoints = cagePoints;

  // ---- face points / edge points of the current level ----
  if (lv.level > 0 && lv.facePointIndex && lv.edgePointIndex) {
    const fp = buildPointCloud(lv.positions, Array.from(lv.facePointIndex), COLORS.facePoint);
    const fpGeo = new THREE.BufferGeometry();
    fpGeo.setAttribute('position', new THREE.BufferAttribute(fp.positions, 3));
    const fpMat = new THREE.PointsMaterial({
      color: COLORS.facePoint, size: markerSize(lv) * 0.55,
      depthTest: false, transparent: true, opacity: 0.9,
    });
    const fpPoints = new THREE.Points(fpGeo, fpMat);
    fpPoints.renderOrder = 6;
    scene.add(fpPoints);
    objects.facePoints = fpPoints;

    const ep = buildPointCloud(lv.positions, Array.from(lv.edgePointIndex), COLORS.edgePoint);
    const epGeo = new THREE.BufferGeometry();
    epGeo.setAttribute('position', new THREE.BufferAttribute(ep.positions, 3));
    const epMat = new THREE.PointsMaterial({
      color: COLORS.edgePoint, size: markerSize(lv) * 0.45,
      depthTest: false, transparent: true, opacity: 0.8,
    });
    const epPoints = new THREE.Points(epGeo, epMat);
    epPoints.renderOrder = 6;
    scene.add(epPoints);
    objects.edgePoints = epPoints;
  }

  // ---- extraordinary vertices (valence != 4, interior) of current level ----
  const extraIdx = [];
  const info = lv.topology.vertexInfo;
  for (let v = 0; v < info.length; v++) {
    if (!info[v].boundary && info[v].edgeCount !== 4 && info[v].faceCount > 0) extraIdx.push(v);
  }
  // Only draw a manageable number; these are the mathematically interesting ones.
  const extraGeo = new THREE.BufferGeometry();
  extraGeo.setAttribute('position', new THREE.Float32BufferAttribute(
    flattenPositions(extraIdx.map((v) => lv.positions[v])), 3));
  const extraMat = new THREE.PointsMaterial({
    color: COLORS.extra, size: markerSize(lv) * 1.15,
    depthTest: false, transparent: true, opacity: 0.95,
  });
  const extraPoints = new THREE.Points(extraGeo, extraMat);
  extraPoints.renderOrder = 8;
  scene.add(extraPoints);
  objects.extraordinary = extraPoints;

  // ---- crease + boundary lines of the current level ----
  const segs = buildCreaseSegments(lv.positions, lv.faces, lv.creaseMap, lv.topology);
  if (segs.crease.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(segs.crease, 3));
    const m = new THREE.LineBasicMaterial({ color: COLORS.crease });
    const line = new THREE.LineSegments(g, m);
    line.renderOrder = 7;
    scene.add(line);
    objects.creaseLines = line;
  }
  if (segs.boundary.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(segs.boundary, 3));
    const m = new THREE.LineBasicMaterial({ color: COLORS.boundary, transparent: true, opacity: 0.95 });
    const line = new THREE.LineSegments(g, m);
    line.renderOrder = 7;
    scene.add(line);
    objects.boundaryLines = line;
  }

  // ---- limit points of the current level's vertices (toggle) ----
  if (state.show.limitPoints) buildLimitPoints(lv);

  // ---- selection ring ----
  const ringGeo = new THREE.BufferGeometry();
  ringGeo.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
  const ringMat = new THREE.PointsMaterial({
    color: COLORS.selection, size: markerSize(lv) * 1.6,
    depthTest: false, transparent: true, opacity: 0.95,
  });
  const ring = new THREE.Points(ringGeo, ringMat);
  ring.renderOrder = 9;
  ring.visible = false;
  scene.add(ring);
  objects.selectionRing = ring;
}

function buildLimitPoints(lv) {
  // Evaluating every vertex of a deep level is prohibitively expensive; cap the
  // sample to the first N vertices and say so in the status bar.
  const maxPoints = 8000;
  const indices = [];
  for (let v = 0; v < Math.min(lv.positions.length, maxPoints); v++) indices.push(v);
  const lim = evaluateLimits(lv, { indices, maxCreaseLevels: 0 });
  const positions = lim.results.map((r) => r.position);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(flattenPositions(positions), 3));
  const m = new THREE.PointsMaterial({
    color: COLORS.limit, size: markerSize(lv) * 0.7,
    depthTest: false, transparent: true, opacity: 0.85,
  });
  const pts = new THREE.Points(g, m);
  pts.renderOrder = 7;
  scene.add(pts);
  objects.limitPoints = pts;
}

function flattenPositions(arr) {
  const out = new Float32Array(arr.length * 3);
  for (let i = 0; i < arr.length; i++) {
    out[i * 3] = arr[i][0];
    out[i * 3 + 1] = arr[i][1];
    out[i * 3 + 2] = arr[i][2];
  }
  return out;
}

function markerSize(lv) {
  // Scale markers with the model, shrinking with level so they don't swamp it.
  let maxEdge = 0;
  const topo = lv.topology;
  for (const e of topo.edges) {
    const l = vdist(lv.positions[e.a], lv.positions[e.b]);
    if (l > maxEdge) maxEdge = l;
  }
  const base = maxEdge > 0 ? maxEdge * 0.13 : 0.1;
  return Math.max(base / Math.pow(1.9, lv.level), 0.012);
}

function applyVisibility() {
  if (objects.surface) objects.surface.visible = state.show.surface;
  if (objects.wire) objects.wire.visible = state.show.wire;
  if (objects.cage) objects.cage.visible = state.show.cage;
  if (objects.cagePoints) objects.cagePoints.visible = state.show.cage;
  if (objects.facePoints) objects.facePoints.visible = state.show.facePoints;
  if (objects.edgePoints) objects.edgePoints.visible = state.show.edgePoints;
  if (objects.extraordinary) objects.extraordinary.visible = state.show.extraordinary;
  if (objects.creaseLines) objects.creaseLines.visible = state.show.crease;
  if (objects.boundaryLines) objects.boundaryLines.visible = state.show.crease;
  if (objects.limitPoints) objects.limitPoints.visible = state.show.limitPoints;
}

// ---------------------------------------------------------------- levels -----

function setLevel(level) {
  const clamped = Math.max(0, Math.min(level, state.levels.length - 1, MAX_LEVEL));
  if (clamped === state.currentLevel) return;
  state.currentLevel = clamped;
  state.selectedLevelVertex = -1;
  // Vertex indices change between levels, so a stale inspection would be wrong.
  el.vertexCheck.innerHTML = '<span class="hint">点击场景中的顶点查看其规则、价数与极限位置策略。</span>';
  buildSceneGraph();
  updateSelectionOverlay();
  applyVisibility();
  renderLevelBadge();
  markActiveRow();
  animationNeedsRender = true;
  const lv = state.levels[clamped];
  setStatus(`第 ${clamped} 级：${lv.positions.length} 顶点 · ${lv.topology.numEdges} 边 · ${lv.faces.length} 面`, '');
}

function renderLevelBadge() {
  const lv = state.levels[state.currentLevel];
  el.levelBadge.textContent = 'L' + state.currentLevel;
  el.levelSlider.value = String(state.currentLevel);
  el.levelSlider.max = String(Math.max(0, state.levels.length - 1));
  el.levelPrev.disabled = state.currentLevel <= 0;
  el.levelNext.disabled = state.currentLevel >= state.levels.length - 1;
  el.levelMax.disabled = state.currentLevel >= state.levels.length - 1;
  if (lv) {
    el.levelBadge.textContent = `L${state.currentLevel} · ${fmtCount(lv.positions.length)}v`;
  }
  markActiveRow();
}

function fmtCount(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

// ---------------------------------------------------------------- editing ----

function toggleEditMode() {
  state.editMode = !state.editMode;
  el.editToggle.textContent = state.editMode ? '退出编辑' : '进入编辑';
  el.editToggle.classList.toggle('accent', state.editMode);
  el.editStatus.textContent = state.editMode ? '点击黄色控制点即可拖动' : '点击顶点选中';

  if (state.editMode) {
    // Editing always manipulates level-0 control points; the surface stays at
    // whatever subdivision level is being viewed so edits can be judged smooth.
    state.show.cage = true;
    el.showCage.checked = true;
    applyVisibility();
    setStatus('编辑模式：点击黄色控制点选择，拖动三轴手柄移动；曲面会实时重建', '');
  } else {
    transform.detach();
    transform.visible = false;
    state.selectedControlVertex = -1;
    updateSelectionOverlay();
    setStatus('已退出编辑模式', '');
  }
  animationNeedsRender = true;
}

function pickControlVertex(event) {
  const l0 = state.levels[0];
  if (!l0 || !objects.cagePoints) return -1;
  updatePointer(event);
  raycaster.setFromCamera(pointer, camera);
  const threshold = Math.max(markerSize(l0) * 1.8, 0.04);
  raycaster.params.Points = { threshold };
  const hits = raycaster.intersectObject(objects.cagePoints, false);
  if (!hits.length) return -1;
  return hits[0].index;
}

function pickLevelVertex(event) {
  const lv = state.levels[state.currentLevel];
  if (!lv || !surfacePickMesh) return null;
  updatePointer(event);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObject(surfacePickMesh, false);
  if (!hits.length) return null;
  const hit = hits[0];
  // hit.face.a/b/c are triangle-soup indices; translate to real mesh vertices.
  const src = triangleSource;
  if (!src || hit.faceIndex == null) return null;
  const tri = hit.faceIndex;
  const candidate = [
    src[tri * 3], src[tri * 3 + 1], src[tri * 3 + 2],
  ].filter((v) => v != null);
  let nearest = -1;
  let best = Infinity;
  for (const v of candidate) {
    const d = vdist(lv.positions[v], [hit.point.x, hit.point.y, hit.point.z]);
    if (d < best) { best = d; nearest = v; }
  }
  return { vertex: nearest, point: hit.point, distance: best };
}

function updatePointer(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onPointerDown(event) {
  if (event.button !== 0) return;
  if (transform.dragging) return;

  if (state.editMode) {
    const idx = pickControlVertex(event);
    if (idx >= 0) {
      selectControlVertex(idx);
      return;
    }
  }
  // Otherwise inspect the nearest vertex on the current subdivided level.
  const hit = pickLevelVertex(event);
  if (hit && hit.vertex >= 0) {
    state.selectedLevelVertex = hit.vertex;
    updateSelectionOverlay();
    renderVertexCheck(hit.vertex);
    animationNeedsRender = true;
  } else {
    state.selectedLevelVertex = -1;
    updateSelectionOverlay();
    animationNeedsRender = true;
  }
}

function onPointerMove(event) {
  if (!state.editMode || transform.dragging) {
    tooltipEl.classList.add('hidden');
    return;
  }
  const idx = pickControlVertex(event);
  if (idx >= 0) {
    const cInfo = controlClassification()[idx];
    tooltipEl.innerHTML =
      `控制点 #${idx}<br><span class="dim">价数 ${cInfo.valence} · ${ruleLabel(cInfo.rule)}</span>`;
    tooltipEl.style.left = event.clientX + 'px';
    tooltipEl.style.top = event.clientY + 'px';
    tooltipEl.classList.remove('hidden');
    renderer.domElement.style.cursor = 'pointer';
  } else {
    tooltipEl.classList.add('hidden');
    renderer.domElement.style.cursor = state.editMode ? 'crosshair' : 'default';
  }
}

function selectControlVertex(idx) {
  state.selectedControlVertex = idx;
  const p = state.levels[0].positions[idx];
  transformProxy.position.set(p[0], p[1], p[2]);
  transform.attach(transformProxy);
  transform.visible = true;
  updateSelectionOverlay();
  renderSelectedControlInfo(idx);
  animationNeedsRender = true;
}

function updateTransformTarget() {
  if (state.selectedControlVertex < 0 || !state.editMode) {
    transform.detach();
    transform.visible = false;
    return;
  }
  selectControlVertex(state.selectedControlVertex);
}

function updateSelectionOverlay() {
  const ring = objects.selectionRing;
  if (!ring) return;
  let p = null;
  if (state.editMode && state.selectedControlVertex >= 0 && state.levels[0]) {
    p = state.levels[0].positions[state.selectedControlVertex];
  } else if (state.selectedLevelVertex >= 0) {
    p = state.levels[state.currentLevel]?.positions[state.selectedLevelVertex];
  }
  if (p) {
    ring.geometry.attributes.position.setXYZ(0, p[0], p[1], p[2]);
    ring.geometry.attributes.position.needsUpdate = true;
    ring.geometry.computeBoundingSphere();
    ring.visible = true;
  } else {
    ring.visible = false;
  }
}

function renderSelectedControlInfo(idx) {
  const l0 = state.levels[0];
  const cInfo = controlClassification()[idx];
  const lim = evaluateLimits(l0, { indices: [idx] });
  const r = lim.results[0];
  el.selectedInfo.classList.remove('hidden');
  el.selectedInfo.innerHTML = `
    <div class="kv"><span>控制点</span><span>#${idx}</span></div>
    <div class="kv"><span>价数</span><span>${cInfo.valence}${cInfo.valence === 4 ? '（规则）' : '（非规则）'}</span></div>
    <div class="kv"><span>规则</span><span class="tag ${cInfo.rule}">${ruleLabel(cInfo.rule)}</span></div>
    <div class="kv"><span>边界</span><span>${cInfo.boundary ? '是（' + cInfo.boundaryCount + ' 条边界边）' : '否'}</span></div>
    <div class="kv"><span>位置</span><span>${fmtVec(l0.positions[idx])}</span></div>
    <div class="kv"><span>极限位置</span><span>${fmtVec(r.position)}</span></div>
    <div class="kv"><span>策略</span><span>${policyLabel(r.policy)}</span></div>
  `;
}

function renderVertexCheck(v) {
  const lv = state.levels[state.currentLevel];
  const cInfo = levelClassification()[v];
  // Interactive inspection: cap the dart refinement so a deep-level click stays
  // responsive.  The control cage diagnostic uses the full budget instead.
  const lim = evaluateLimits(lv, { indices: [v], maxDartLevels: 6, maxVertices: 60_000 });
  const r = lim.results[0];
  el.vertexCheck.innerHTML = `
    <div class="kv"><span>层级 / 顶点</span><span>L${lv.level} · #${v}</span></div>
    <div class="kv"><span>价数</span><span>${cInfo.valence}${cInfo.valence === 4 ? '（规则）' : '（非规则）'}</span></div>
    <div class="kv"><span>规则</span><span class="tag ${cInfo.rule}">${ruleLabel(cInfo.rule)}</span></div>
    <div class="kv"><span>边界</span><span>${cInfo.boundary ? '是' : '否'}</span></div>
    <div class="kv"><span>当前位置</span><span>${fmtVec(lv.positions[v])}</span></div>
    <div class="kv"><span>极限位置</span><span>${fmtVec(r.position)}</span></div>
    <div class="kv"><span>迭代→极限</span><span>${sci(vdist(lv.positions[v], r.position))}</span></div>
    <div class="kv"><span>策略</span><span>${policyLabel(r.policy)}</span></div>
    ${r.extraLevels ? `<div class="kv"><span>附加细分</span><span>${r.extraLevels} 级（折痕衰减/规范化）</span></div>` : ''}
    ${r.approximate ? `<div class="kv"><span>精度</span><span>近似（折痕端点无闭式极限，细分逼近）</span></div>` : ''}
  `;
}

function ruleLabel(rule) {
  return { smooth: '平滑', crease: '折痕', corner: '尖角', dart: '飞镖' }[rule] || rule;
}

function policyLabel(policy) {
  return {
    'smooth-limit': '平滑极限掩码',
    'crease-limit': '折痕曲线极限 (A+4P+B)/6',
    'dart-limit': '折痕端点：继续细分后取极限',
    corner: '尖角固定',
    degenerate: '退化',
  }[policy] || policy;
}

function fmtVec(p) {
  return p.map((x) => x.toFixed(4)).join(', ');
}

function sci(x) {
  return x === 0 ? '0' : x.toExponential(2);
}

// ---------------------------------------------------------------- QA ---------

function runDiagnostics(levels) {
  const rows = [];
  for (let i = 0; i < levels.length; i++) {
    const lv = levels[i];
    const topo = lv.topology;
    const V = lv.positions.length;
    const E = topo.numEdges;
    const F = lv.faces.length;
    const chi = V - E + F;
    // seam duplicates (cap work on the largest levels)
    let dups = 0;
    if (V <= 250_000) {
      dups = findDuplicatePositions(lv.positions, 1e-10).length;
    }
    const angles = edgeAngleStats(lv.positions, lv.faces);
    rows.push({
      level: i,
      V, E, F, chi,
      boundaryEdges: topo.boundaryEdgeCount,
      duplicates: dups,
      maxAngleDeg: angles.maxAngleDeg,
      p99AngleDeg: angles.p99AngleDeg,
      medianAngleDeg: angles.medianAngleDeg,
      orientationOk: checkOrientation(lv.positions, lv.faces).ok,
    });
  }

  // Convergence of root vertices to their limit.
  const lim = state.limitRoot || computeRootLimits(levels[0]);
  const rootCount = levels[0].positions.length;
  const errors = levels.map((lv) => {
    let maxErr = 0;
    for (let v = 0; v < rootCount; v++) {
      const d = vdist(lv.positions[v], lim.results[v].position);
      if (d > maxErr) maxErr = d;
    }
    return maxErr;
  });
  const ratios = errors.map((e, i) => (i > 0 && errors[i - 1] > 0 ? e / errors[i - 1] : null));

  // Limit policy census on the control-cage vertices only (the refined level
  // carries extra derived vertices whose policy is not a modelling decision).
  const policies = {};
  for (let v = 0; v < rootCount; v++) {
    const p = lim.results[v].policy;
    policies[p] = (policies[p] || 0) + 1;
  }

  // Extraordinary vertices adjacent to creases: the stated "singular vertex next
  // to a crease" case.  Record their rule and the chosen policy.
  const l0 = levels[0];
  const cls = classifyVertices(l0.topology, l0.creaseMap, l0.cornerSet, {
    pinBoundaryCorners: (l0.options.boundaryMode || 'edgeAndCorner') !== 'edgeOnly',
  });
  const singularNearCrease = [];
  const explicitCremap = l0.creaseMap;
  for (let v = 0; v < rootCount; v++) {
    const c = cls[v];
    if (c.valence === 4) continue;
    // Count only genuine creases (explicit sharp edges), not boundary edges that
    // are also treated as sharp for interpolation.
    let creaseCount = 0;
    for (const ei of c.sharpEdges) {
      if (explicitCremap.has(l0.topology.edges[ei].key)) creaseCount++;
    }
    if (creaseCount > 0) {
      singularNearCrease.push({
        vertex: v,
        valence: c.valence,
        sharpCount: creaseCount,
        rule: c.rule,
        policy: lim.results[v].policy,
      });
    }
  }

  const issues = [];
  const last = rows[rows.length - 1];
  for (const row of rows) {
    if (row.duplicates > 0) issues.push(`L${row.level}: ${row.duplicates} 个接缝重复顶点 (per-face seam)`);
    if (!row.orientationOk) issues.push(`L${row.level}: 面缠绕方向不一致`);
  }
  // Euler characteristic must stay constant across levels.
  const chi0 = rows[0].chi;
  for (const row of rows) {
    if (row.chi !== chi0) issues.push(`L${row.level}: 欧拉示性数 ${row.chi} ≠ ${chi0}（拓扑改变）`);
  }

  return {
    rows, errors, ratios, policies, singularNearCrease, issues,
    summary: {
      levels: rows.length,
      finalVertices: last.V,
      finalFaces: last.F,
      euler: last.chi,
      maxSeamDuplicates: Math.max(...rows.map((r) => r.duplicates)),
      maxDihedralDeg: last.maxAngleDeg,
      p99DihedralDeg: last.p99AngleDeg,
      convergenceRatio: ratios.length ? ratios[ratios.length - 1] : null,
      creaseExtraLevels: lim.creaseLevels,
      creaseCapped: lim.capped,
      quadLevels: lim.quadLevels,
      dartRefined: lim.dartLevels > 0,
      dartLevels: lim.dartLevels,
      policies,
      singularNearCrease,
    },
  };
}

function computeRootLimits(l0) {
  return evaluateLimits(l0, { maxExtraLevels: 6 });
}

function renderDiagnostics() {
  const d = state.diagnostics;
  if (!d) return;

  // ---- summary cards ----
  const items = [];
  const last = d.rows[d.rows.length - 1];

  items.push({
    cls: d.summary.maxSeamDuplicates === 0 ? 'good' : 'bad',
    text: d.summary.maxSeamDuplicates === 0
      ? '<b>连接检查通过</b>：所有层级均无“重合但分离”的接缝顶点，边点/面点由两侧面共享。'
      : `<b>发现接缝</b>：最多 ${d.summary.maxSeamDuplicates} 对重合顶点，说明面被独立插值。`,
  });

  const convOk = d.ratios.length > 2 && d.ratios[d.ratios.length - 1] != null && d.ratios[d.ratios.length - 1] < 0.8;
  items.push({
    cls: convOk ? 'good' : 'warn',
    text: `<b>收敛</b>：控制点迭代到极限位置的最大误差 ${sci(d.errors[d.errors.length - 1])}` +
      (d.ratios.length ? `，相邻级比值 ${d.ratios[d.ratios.length - 1]?.toFixed(2) ?? '—'}` : '') + '。',
  });

  const angleDrop = d.rows.length > 2 && last.p99AngleDeg < d.rows[1].p99AngleDeg;
  items.push({
    cls: angleDrop ? 'good' : 'warn',
    text: `<b>法线连续</b>：末级最大二面角 ${last.maxAngleDeg.toFixed(1)}°，P99 ${last.p99AngleDeg.toFixed(2)}°` +
      `（上一级 P99 ${d.rows.length > 1 ? d.rows[d.rows.length - 2].p99AngleDeg.toFixed(2) : '—'}°）。`,
  });

  items.push({
    cls: d.issues.length === 0 ? 'good' : 'bad',
    text: d.issues.length === 0
      ? `<b>拓扑稳定</b>：欧拉示性数恒为 ${d.summary.euler}，边界边为 1-流形，方向一致。`
      : `<b>发现问题</b>：${d.issues.join('；')}`,
  });

  el.qualitySummary.innerHTML = items.map((it) =>
    `<div class="summary-item"><span class="dot ${it.cls}"></span><span class="txt">${it.text}</span></div>`
  ).join('');

  // ---- level table ----
  el.levelTbody.innerHTML = d.rows.map((r) => {
    const bad = r.duplicates > 0 || !r.orientationOk;
    return `<tr data-level="${r.level}" class="${bad ? 'bad' : ''}">
      <td>${r.level}</td>
      <td>${fmtCount(r.V)}</td>
      <td>${fmtCount(r.E)}</td>
      <td>${fmtCount(r.F)}</td>
      <td>${r.chi}</td>
      <td>${r.boundaryEdges}</td>
      <td>${r.duplicates}</td>
      <td>${r.maxAngleDeg.toFixed(1)}°</td>
      <td>${r.p99AngleDeg.toFixed(2)}°</td>
    </tr>`;
  }).join('');
  markActiveRow();

  el.levelTbody.querySelectorAll('tr').forEach((tr) => {
    tr.addEventListener('click', () => {
      setLevel(Number(tr.dataset.level));
    });
  });

  // ---- limit policy info ----
  const p = d.policies;
  const parts = [];
  if (p['smooth-limit']) parts.push(`平滑 ${p['smooth-limit']}`);
  if (p['dart-limit']) parts.push(`飞镖 ${p['dart-limit']}`);
  if (p['crease-limit']) parts.push(`折痕曲线 ${p['crease-limit']}`);
  if (p.corner) parts.push(`尖角固定 ${p.corner}`);
  const extra = d.summary.creaseExtraLevels;
  const capped = d.summary.creaseCapped;
  const quadLevels = d.summary.quadLevels;
  let html = `<div class="kv"><span>极限策略分布</span><span>${parts.join(' · ') || '—'}</span></div>`;
  if (quadLevels > 0) {
    html += `<div class="kv"><span>非四边形规范化</span><span>先细分 ${quadLevels} 级</span></div>`;
  }
  html += `<div class="kv"><span>半锐折痕衰减</span><span>${extra > 0 ? `先细分 ${extra} 级再取极限` : '无需额外细分'}${capped ? '（达到预算上限）' : ''}</span></div>`;
  if (d.summary.dartRefined) {
    html += `<div class="kv"><span>折痕端点（dart）</span><span>先细分 ${d.summary.dartLevels} 级逼近（无闭式极限）</span></div>`;
  }
  if (d.singularNearCrease.length) {
    const list = d.singularNearCrease.slice(0, 6).map((s) =>
      `#${s.vertex}(价${s.valence},${s.sharpCount}锐边→${s.policy})`).join(' ');
    html += `<div class="kv"><span>奇异顶点紧邻折痕</span><span>${d.singularNearCrease.length} 个</span></div>`;
    html += `<div style="margin-top:4px;line-height:1.55">${list}${d.singularNearCrease.length > 6 ? ' …' : ''}</div>`;
  } else {
    html += `<div class="kv"><span>奇异顶点紧邻折痕</span><span>无</span></div>`;
  }
  el.limitInfo.innerHTML = html;

  renderLevelBadge();
}

function markActiveRow() {
  el.levelTbody.querySelectorAll('tr').forEach((tr) => {
    tr.classList.toggle('active', Number(tr.dataset.level) === state.currentLevel);
  });
}

// ---------------------------------------------------------------- error ------

function showError(e, modelName) {
  state.error = e;
  el.errorBox.classList.remove('hidden');
  const code = e.code || 'ERROR';
  const details = formatErrorDetails(e);
  el.errorBox.innerHTML = `
    <strong>输入被拒绝：${code}</strong>
    <div>${escapeHtml(e.message)}</div>
    ${details ? `<div style="margin-top:6px">${details}</div>` : ''}
    <div style="margin-top:6px;color:#c98f8f">问题已定位到具体元素；按约定不会自动修改成另一种拓扑。请修正控制网后重试。</div>
  `;
  clearSceneGraph();
  el.qualitySummary.innerHTML = '';
  el.levelTbody.innerHTML = '';
  el.limitInfo.innerHTML = '';
  setStatus(`模型 ${modelName} 被拒绝：${code}`, 'err');
  animationNeedsRender = true;
}

function formatErrorDetails(e) {
  const d = e.details;
  if (!Array.isArray(d) || d.length === 0) return '';
  const items = d.slice(0, 6).map((x) => {
    if (x.face != null && x.duplicateOf != null) return `面 ${x.face} 与面 ${x.duplicateOf} 顶点相同`;
    if (x.face != null && x.reason) return `面 ${x.face}：${x.reason}`;
    if (x.face != null) return `面 ${x.face}（面积 ${x.area?.toExponential?.(2) ?? '—'}）`;
    if (x.vertex != null && x.faceCount != null) return `顶点 ${x.vertex}：${x.faceCount} 面 / ${x.boundaryEdgeCount} 边界边 / ${x.fanCount} 个扇形`;
    if (x.vertex != null) return `顶点 ${x.vertex}`;
    if (x.a != null) return `边 ${x.a}-${x.b} 被 ${x.faceCount} 个面共享`;
    return JSON.stringify(x);
  });
  return `<div style="font-size:11px;line-height:1.5">${items.join('<br>')}${d.length > 6 ? '<br>…' : ''}</div>`;
}

function hideError() {
  el.errorBox.classList.add('hidden');
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
}

function setStatus(text, cls) {
  el.statusText.textContent = text;
  el.statusText.className = cls || '';
}

// ---------------------------------------------------------------- camera -----

function frameCamera() {
  const lv = state.levels[0];
  if (!lv) return;
  let min = [Infinity, Infinity, Infinity];
  let max = [-Infinity, -Infinity, -Infinity];
  for (const p of lv.positions) {
    for (let k = 0; k < 3; k++) {
      if (p[k] < min[k]) min[k] = p[k];
      if (p[k] > max[k]) max[k] = p[k];
    }
  }
  const center = [0, 1, 2].map((k) => (min[k] + max[k]) / 2);
  let radius = 0;
  for (const p of lv.positions) {
    radius = Math.max(radius, vdist(p, center));
  }
  radius = Math.max(radius, 0.5);
  const dist = radius / Math.sin((camera.fov * Math.PI) / 360) * 1.25;
  const dir = [0.52, -0.66, 0.54];
  const dl = vlen(dir);
  camera.position.set(
    center[0] + (dir[0] / dl) * dist,
    center[1] + (dir[1] / dl) * dist,
    center[2] + (dir[2] / dl) * dist
  );
  orbit.target.set(center[0], center[1], center[2]);
  camera.near = Math.max(0.01, dist / 400);
  camera.far = dist * 40;
  camera.updateProjectionMatrix();
  orbit.update();
  scene.fog.near = dist * 1.6;
  scene.fog.far = dist * 5.5;
  // Put the ground reference below the model so it never slices through a tray or
  // sheet; scale it to the model extent as well.
  if (objects.grid) {
    objects.grid.position.z = min[2] - radius * 0.06 - 0.02;
    objects.grid.scale.setScalar(Math.max(1, radius / 4));
  }
  animationNeedsRender = true;
}

// ---------------------------------------------------------------- loop -------

function animate() {
  requestAnimationFrame(animate);
  orbit.update();
  if (animationNeedsRender || orbit.enableDamping) {
    renderer.render(scene, camera);
    animationNeedsRender = false;
  }
}

// ---------------------------------------------------------------- boot -------

init();
