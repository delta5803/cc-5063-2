// Automatic failure localisation: shrink a failing control net to a small
// neighbourhood that still reproduces the seam error.
//
// Given a control mesh, a fault, and a set of "seed" vertices (e.g. the traced
// chain endpoints), this finds the smallest face neighbourhood whose subdivision
// still exhibits the error.  The result is a standalone mesh that can be reloaded
// without the original scene -- that is the point: a defect report has to be
// replayable from the saved mesh alone.
//
// Algorithm:
//   1. Start from the whole cage.
//   2. BFS face-rings outward from the seed faces, building the induced submesh.
//   3. Keep the smallest ring whose faulted build still shows the metric.
//   4. Verify the saved mesh in isolation (build from its arrays, re-run the
//      metric) so a stale reference cannot make the reduced repro pass.

import { buildTopology } from '../src/core/topology.js';
import { createControlLevel, subdivideLevel, FAULT } from '../src/core/subdivision.js';
import { seamMetrics, findSeamPairs } from '../src/core/seams.js';
import { vdist } from '../src/core/vec3.js';

/**
 * Build the induced submesh over a face set, remapping vertices.
 * Creases and crease-chains are restricted to the surviving edges; a chain whose
 * edges are split is dropped rather than silently truncated.
 */
export function inducedSubmesh(mesh, faceSet) {
  const faces = [...faceSet].sort((a, b) => a - b).map((f) => mesh.faces[f].slice());
  const used = new Map();
  const positions = [];
  for (const f of faces) {
    for (let i = 0; i < f.length; i++) {
      const v = f[i];
      if (!used.has(v)) {
        used.set(v, positions.length);
        positions.push(mesh.positions[v].slice());
      }
      f[i] = used.get(v);
    }
  }
  // Actual edge set of the induced faces; a crease can only survive if the edge
  // still exists in the reduced mesh, otherwise createControlLevel would (rightly)
  // reject the chain.
  const edgeSet = new Set();
  for (const f of faces) {
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = f[(i + 1) % f.length];
      edgeSet.add(a < b ? `${a}:${b}` : `${b}:${a}`);
    }
  }
  // `edgeSet` holds remapped ids, so remap before testing membership.
  const keepEdge = (a, b) => {
    if (!used.has(a) || !used.has(b)) return false;
    const ra = used.get(a);
    const rb = used.get(b);
    return edgeSet.has(ra < rb ? `${ra}:${rb}` : `${rb}:${ra}`);
  };
  const creases = (mesh.creases || [])
    .filter((c) => keepEdge(c.a, c.b))
    .map((c) => ({ a: used.get(c.a), b: used.get(c.b), sharpness: c.sharpness }));
  const creaseChains = [];
  for (const chain of (mesh.creaseChains || [])) {
    if (chain.edges.every(([a, b]) => keepEdge(a, b))) {
      creaseChains.push({
        id: chain.id,
        edges: chain.edges.map(([a, b]) => [used.get(a), used.get(b)]),
      });
    }
  }
  const corners = (mesh.corners || []).filter((v) => used.has(v)).map((v) => used.get(v));
  return {
    name: mesh.name ? `${mesh.name}-repro` : 'reduced-repro',
    description: `reduced repro of ${mesh.name || 'mesh'} (${faces.length} face(s))`,
    positions,
    faces,
    creases,
    corners,
    creaseChains,
  };
}

/** Faces within `rings` face-adjacency hops of any seed face. */
export function faceRingsAround(mesh, seedFaceIds, rings) {
  const topo = buildTopology(mesh.positions, mesh.faces, { allowBoundary: true });
  const out = new Set(seedFaceIds);
  let frontier = new Set(seedFaceIds);
  for (let r = 0; r < rings; r++) {
    const next = new Set();
    for (const e of topo.edges) {
      if (e.faces.length !== 2) continue;
      const [f0, f1] = e.faces;
      if (frontier.has(f0) && !out.has(f1)) next.add(f1);
      if (frontier.has(f1) && !out.has(f0)) next.add(f0);
    }
    for (const f of next) out.add(f);
    frontier = next;
    if (!frontier.size) break;
  }
  return out;
}

/** Faces incident to any of the given vertices. */
export function facesTouching(mesh, vertexIds) {
  const want = new Set(vertexIds);
  const out = new Set();
  for (let f = 0; f < mesh.faces.length; f++) {
    for (const v of mesh.faces[f]) {
      if (want.has(v)) { out.add(f); break; }
    }
  }
  return out;
}

/** Faces containing any edge of any crease chain. */
export function facesOfChains(mesh) {
  const out = new Set();
  const keys = new Set();
  for (const chain of (mesh.creaseChains || [])) {
    for (const [a, b] of chain.edges) keys.add(a < b ? `${a}:${b}` : `${b}:${a}`);
  }
  for (let f = 0; f < mesh.faces.length; f++) {
    const face = mesh.faces[f];
    for (let i = 0; i < face.length; i++) {
      const a = face[i];
      const b = face[(i + 1) % face.length];
      if (keys.has(a < b ? `${a}:${b}` : `${b}:${a}`)) { out.add(f); break; }
    }
  }
  return out;
}

function buildWithFault(mesh, level, fault, faultLevel) {
  const clean = [];
  const test = [];
  let c = createControlLevel(mesh, {});
  let t = createControlLevel(mesh, {});
  clean.push(c);
  test.push(t);
  for (let i = 0; i < level; i++) {
    c = subdivideLevel(c);
    t = subdivideLevel(t, i === faultLevel ? { fault, faultLevel } : {});
    clean.push(c);
    test.push(t);
  }
  return { clean: c, test: t };
}

/**
 * A metric callback returns { ok, score }, where `ok` says the failure is still
 * present on the reduced mesh and `score` lets the search prefer stronger repros.
 */
function defaultMetric(mesh, level, fault, faultLevel) {
  const { clean, test } = buildWithFault(mesh, level, fault, faultLevel);
  const seeds = [];
  for (let v = 0; v < mesh.positions.length; v++) seeds.push(v);
  const m = seamMetrics(clean, test, seeds, { maxDartLevels: 0 });
  const score = Math.max(m.positionDelta.max, m.limitPositionDelta.max);
  return { ok: m.positionDelta.max > 1e-9 || m.limitPositionDelta.max > 1e-9, score, metrics: m };
}

/**
 * Shrink `mesh` while `fault` still produces a seam error.
 *
 * @param {object} mesh      full control net
 * @param {object} options
 * @param {string} options.fault       FAULT.* to inject
 * @param {number} options.faultLevel  subdivision level at which to inject
 * @param {number} options.level       level at which to measure
 * @param {number[]} [options.seedVertices] vertices to centre the search on
 * @param {Function} [options.metric]  custom {ok,score} probe
 * @returns {{mesh, rings, verified, metric, dropped}}
 */
export function minimiseFailingMesh(mesh, options = {}) {
  const fault = options.fault || FAULT.WRONG_INHERIT;
  const faultLevel = options.faultLevel != null ? options.faultLevel : 0;
  const level = options.level != null ? options.level : 1;
  const metric = options.metric || ((m) => defaultMetric(m, level, fault, faultLevel));

  const seedFaces = options.seedVertices
    ? facesTouching(mesh, options.seedVertices)
    : facesOfChains(mesh);

  const topo = buildTopology(mesh.positions, mesh.faces, { allowBoundary: true });
  // Highest ring that still yields a manifold mesh; grow until it covers all.
  const maxRings = 12;
  let best = null;
  let firstWorking = null;
  const attempts = [];
  for (let r = 0; r <= maxRings; r++) {
    const set = faceRingsAround(mesh, seedFaces, r);
    let sub;
    try {
      sub = inducedSubmesh(mesh, set);
    } catch (e) {
      attempts.push({ rings: r, faces: set.size, error: e.message });
      continue;
    }
    if (sub.faces.length === 0) continue;
    let res;
    try {
      res = metric(sub);
    } catch (e) {
      attempts.push({ rings: r, faces: sub.faces.length, error: e.message });
      continue;
    }
    attempts.push({ rings: r, faces: sub.faces.length, ok: res.ok, score: res.score });
    if (res.ok) {
      if (!firstWorking) firstWorking = { rings: r, mesh: sub, metric: res };
      best = { rings: r, mesh: sub, metric: res };
      break;
    }
  }
  if (!best) {
    // No reduction preserved the error.  Fail loudly rather than returning the
    // original mesh and claiming success.
    const err = new Error(
      `Could not reduce a mesh that still shows the ${fault} seam error ` +
      `(tried rings 0..${maxRings})`
    );
    err.code = 'REDUCTION_FAILED';
    err.details = attempts;
    throw err;
  }

  const verified = verifyStandalone(best.mesh, fault, faultLevel, level, metric);
  return {
    mesh: best.mesh,
    rings: best.rings,
    originalFaces: mesh.faces.length,
    originalVertices: mesh.positions.length,
    reducedFaces: best.mesh.faces.length,
    reducedVertices: best.mesh.positions.length,
    manifoldOk: best.mesh.faces.length > 0,
    verified,
    metric: best.metric,
    attempts,
    droppedFaces: mesh.faces.length - best.mesh.faces.length,
    // Kept for callers that want to log sizes; topology of the original.
    originalEdges: topo.numEdges,
  };
}

/**
 * Reload the reduced mesh from serialised arrays (no references to the original
 * scene) and re-run the metric.  Guards against a reduction that only "works"
 * because it shares mutable state with the parent.
 */
export function verifyStandalone(mesh, fault, faultLevel, level, metric) {
  // structuredClone keeps Infinity sharpness; JSON would turn it into null and
  // silently drop the creases, making the "standalone" check meaningless.
  const copy = structuredClone({
    name: mesh.name, positions: mesh.positions, faces: mesh.faces,
    creases: mesh.creases, corners: mesh.corners, creaseChains: mesh.creaseChains,
  });
  const res = metric(copy);
  return {
    ok: res.ok,
    score: res.score,
    faces: copy.faces.length,
    vertices: copy.positions.length,
  };
}

/**
 * Serialise a repro mesh to JSON with Infinity encoded explicitly, so the file
 * can be saved, committed, and reloaded without the original scene.
 */
export function serializeRepro(mesh, meta = {}) {
  return JSON.stringify({
    format: 'cc-seam-repro-v1',
    ...meta,
    mesh: {
      ...mesh,
      creases: (mesh.creases || []).map((c) => ({
        a: c.a,
        b: c.b,
        sharpness: Number.isFinite(c.sharpness) ? c.sharpness : 'inf',
      })),
    },
  }, null, 2) + '\n';
}

export function parseRepro(text) {
  const data = JSON.parse(text);
  if (data.format !== 'cc-seam-repro-v1') {
    throw new Error(`Not a seam repro file: format ${data.format}`);
  }
  const m = data.mesh;
  return {
    mesh: {
      ...m,
      creases: (m.creases || []).map((c) => ({
        a: c.a,
        b: c.b,
        sharpness: c.sharpness === 'inf' ? Infinity : c.sharpness,
      })),
    },
    meta: { ...data, mesh: undefined },
  };
}

/**
 * Check the saved repro still reproduces when evaluated through the public entry
 * points only (createControlLevel / subdivideLevel / seamMetrics).
 */
export function replayReducedMesh(saved, fault, faultLevel, level) {
  const { clean, test } = buildWithFault(saved, level, fault, faultLevel);
  const seeds = [];
  for (let v = 0; v < saved.positions.length; v++) seeds.push(v);
  const metrics = seamMetrics(clean, test, seeds, { maxDartLevels: 0 });
  const seams = findSeamPairs(test, 1e-9).map((p) => vdist(test.positions[p.i], test.positions[p.j]));
  return {
    reproduced: metrics.positionDelta.max > 1e-9 || metrics.limitPositionDelta.max > 1e-9,
    metrics,
    seamPairs: seams.length,
  };
}
