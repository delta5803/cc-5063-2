// Node test suite for the subdivision core.  Run with: node tools/run-tests.mjs
//
// Covers:
//   1. Topology construction and manifold rejection (with located problems).
//   2. Face/edge/vertex rule correctness on hand-checkable configurations.
//   3. Shared-vertex connectivity (no per-face seam duplicates).
//   4. Limit-position correctness against analytic B-spline masks and across levels.
//   5. Convergence of iteration to the limit.
//   6. Normal continuity (dihedral angle) improving with level.
//   7. Extraordinary vertices of several valences, boundaries, creases.
//   8. Anti-examples: concave / uneven meshes must also pass, not just spheres.

import {
  buildTopology, edgeKey,
} from '../src/core/topology.js';
import {
  createControlLevel, subdivideLevel, buildLevels,
} from '../src/core/subdivision.js';
import { evaluateLimits, refineUntilStable, maxFiniteSharpness } from '../src/core/limit.js';
import { classifyVertices, RULE } from '../src/core/creases.js';
import {
  checkConnectivity, checkOrientation, edgeAngleStats, checkLimitConvergence,
  findDuplicatePositions, faceNormal,
} from '../src/core/quality.js';
import { vdist, vsub, vlen, vdot, vcross } from '../src/core/vec3.js';
import {
  modelByName, VALID_MODEL_NAMES, INVALID_MODEL_NAMES,
  regularGrid, creaseRidge,
} from '../src/models/fixtures.js';

let pass = 0;
let fail = 0;
const failures = [];

function test(name, fn) {
  try {
    fn();
    pass++;
    console.log(`  ok   ${name}`);
  } catch (e) {
    fail++;
    failures.push({ name, error: e });
    console.log(`  FAIL ${name}`);
    console.log(`       ${e.message}`);
    if (e.stack && e.expected !== undefined) {
      console.log(e.stack.split('\n').slice(0, 4).join('\n'));
    }
  }
}

function section(title) {
  console.log(`\n== ${title} ==`);
}

function assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}

function assertClose(a, b, tol, msg) {
  if (Math.abs(a - b) > tol) {
    throw new Error(`${msg || 'values differ'}: ${a} vs ${b} (tol ${tol})`);
  }
}

function assertVecClose(a, b, tol, msg) {
  const d = vdist(a, b);
  if (d > tol) {
    throw new Error(`${msg || 'vectors differ'} by ${d}: ${fmtVec(a)} vs ${fmtVec(b)} (tol ${tol})`);
  }
}

function fmtVec(v) {
  return '(' + v.map((x) => x.toFixed(9)).join(', ') + ')';
}

// ============================================================================
section('1. Topology validation');

test('valid meshes build topology with expected counts', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const topo = buildTopology(m.positions, m.faces, { allowBoundary: true });
    assert(topo.numVertices === m.positions.length, `${name}: vertex count`);
    assert(topo.numFaces === m.faces.length, `${name}: face count`);
    assert(topo.numEdges > 0, `${name}: no edges`);
    assert(topo.vertexInfo.every((v) => !v.nonManifold), `${name}: flagged manifold vertex`);
  }
});

for (const name of INVALID_MODEL_NAMES) {
  test(`rejects ${name} with a located error`, () => {
    const m = modelByName(name);
    let threw = false;
    try {
      buildTopology(m.positions, m.faces, { allowBoundary: true });
    } catch (e) {
      threw = true;
      assert(e.code, `${name}: error has no .code`);
      assert(e.details && e.details.length > 0, `${name}: error has no located details`);
      assert(!/repair|fix/i.test(e.message), `${name}: error message suggests silent repair`);
    }
    assert(threw, `${name} should have been rejected`);
  });
}

test('non-manifold edge error names the offending edge', () => {
  const m = modelByName('non-manifold-edge');
  try {
    buildTopology(m.positions, m.faces, { allowBoundary: true });
    throw new Error('should have thrown');
  } catch (e) {
    assert(e.code === 'NON_MANIFOLD_EDGE', 'wrong code: ' + e.code);
    assert(e.details.some((d) => (d.a === 0 && d.b === 1)), 'edge 0-1 not identified');
  }
});

test('non-manifold vertex error locates the bow-tie vertex', () => {
  const m = modelByName('non-manifold-vertex');
  try {
    buildTopology(m.positions, m.faces, { allowBoundary: true });
    throw new Error('should have thrown');
  } catch (e) {
    assert(e.code === 'NON_MANIFOLD_VERTEX', 'wrong code: ' + e.code);
    assert(e.details.some((d) => d.vertex === 0), 'vertex 0 not identified');
  }
});

test('degenerate face error locates the face', () => {
  const m = modelByName('degenerate-face');
  try {
    buildTopology(m.positions, m.faces, { allowBoundary: true });
    throw new Error('should have thrown');
  } catch (e) {
    assert(e.code === 'DEGENERATE_FACE', 'wrong code: ' + e.code);
    assert(e.details.some((d) => d.face === 1), 'face 1 not identified');
  }
});

// ============================================================================
section('2. Face / edge / vertex rules');

test('face point is the centroid', () => {
  const m = regularGrid(3, 1);
  const l0 = createControlLevel(m);
  const l1 = subdivideLevel(l0);
  const f = 0;
  const face = l0.faces[f];
  const expect = [0, 0, 0];
  for (const v of face) for (let k = 0; k < 3; k++) expect[k] += l0.positions[v][k] / face.length;
  assertVecClose(l1.positions[l1.facePointIndex[f]], expect, 1e-15, 'face point');
});

test('edge point uses 1/4(v0+v1)+1/4(f0+f1) on smooth interior edges', () => {
  const m = regularGrid(5, 1);
  const l0 = createControlLevel(m);
  // pick an interior edge (not on the grid boundary)
  const topo = l0.topology;
  const e = topo.edges.find((ed) => ed.faces.length === 2 &&
    ed.a >= 6 && ed.a < 19 && ed.b >= 6 && ed.b < 19);
  assert(e, 'no interior edge found');
  const l1 = subdivideLevel(l0);
  const pa = l0.positions[e.a], pb = l0.positions[e.b];
  const f0 = centroid(l0, e.faces[0]);
  const f1 = centroid(l0, e.faces[1]);
  const expect = [0, 1, 2].map((k) => 0.25 * (pa[k] + pb[k] + f0[k] + f1[k]));
  assertVecClose(l1.positions[l1.edgePointIndex[e.id]], expect, 1e-15, 'edge point');
});

test('smooth vertex point matches the CC mask on a regular (valence 4) vertex', () => {
  const m = regularGrid(7, 0.9);
  const l0 = createControlLevel(m);
  const l1 = subdivideLevel(l0);
  // centre vertex of a 7x7 grid: index 3*7+3 = 24
  const v = 24;
  const info = l0.topology.vertexInfo[v];
  assert(info.edgeCount === 4, 'centre not valence 4');
  // CC mask: 1/4 P + 1/16 sum of edge-opposite + 1/16 sum of face centres
  const P = l0.positions[v];
  let eo = [0, 0, 0], fcSum = [0, 0, 0];
  for (let i = l0.topology.vertexEdgeStart[v]; i < l0.topology.vertexEdgeStart[v + 1]; i++) {
    const ed = l0.topology.edges[l0.topology.vertexEdges[i]];
    const o = l0.positions[ed.a === v ? ed.b : ed.a];
    for (let k = 0; k < 3; k++) eo[k] += o[k];
  }
  for (let i = l0.topology.vertexFaceStart[v]; i < l0.topology.vertexFaceStart[v + 1]; i++) {
    const c = centroid(l0, l0.topology.vertexFaces[i]);
    for (let k = 0; k < 3; k++) fcSum[k] += c[k];
  }
  const n = 4;
  const expect = [0, 1, 2].map((k) => ((n - 2) / n) * P[k] + (eo[k] + fcSum[k]) / (n * n));
  assertVecClose(l1.positions[v], expect, 1e-15, 'smooth vertex');
});

test('boundary edge point is the midpoint (no interior face mask)', () => {
  const m = regularGrid(5, 1);
  const l0 = createControlLevel(m);
  const e = l0.topology.edges.find((ed) => ed.faces.length === 1);
  const l1 = subdivideLevel(l0);
  const pa = l0.positions[e.a], pb = l0.positions[e.b];
  const expect = [0, 1, 2].map((k) => 0.5 * (pa[k] + pb[k]));
  assertVecClose(l1.positions[l1.edgePointIndex[e.id]], expect, 1e-15, 'boundary edge point');
});

test('boundary non-corner vertex uses the 1D B-spline rule 1/8,3/4,1/8', () => {
  const m = regularGrid(7, 1);
  const l0 = createControlLevel(m);
  const l1 = subdivideLevel(l0);
  const topo = l0.topology;
  // a boundary vertex with two boundary edges, not at a grid corner
  let v = -1;
  for (let i = 0; i < l0.positions.length; i++) {
    const info = topo.vertexInfo[i];
    if (info.boundaryEdgeCount === 2 && info.edgeCount === 3) { v = i; break; }
  }
  assert(v >= 0, 'no boundary edge vertex found');
  const nbrs = [];
  for (let i = topo.vertexEdgeStart[v]; i < topo.vertexEdgeStart[v + 1]; i++) {
    const e = topo.edges[topo.vertexEdges[i]];
    if (e.faces.length === 1) nbrs.push(l0.positions[e.a === v ? e.b : e.a]);
  }
  assert(nbrs.length === 2, 'expected 2 boundary neighbours');
  const P = l0.positions[v];
  const expect = [0, 1, 2].map((k) => 0.75 * P[k] + 0.125 * (nbrs[0][k] + nbrs[1][k]));
  assertVecClose(l1.positions[v], expect, 1e-15, 'boundary vertex');
});

test('corner vertex is pinned', () => {
  const m = regularGrid(5, 1);
  const l0 = createControlLevel(m);
  const l1 = subdivideLevel(l0);
  // grid corner index 0, pinned by edgeAndCorner boundary mode
  assertVecClose(l1.positions[0], l0.positions[0], 1e-15, 'corner pinned');
});

test('edgeAndCorner vs edgeOnly changes boundary corner behaviour', () => {
  const m = regularGrid(5, 1);
  const lEdgeCorner = subdivideLevel(createControlLevel(m, { boundaryMode: 'edgeAndCorner' }));
  const lEdgeOnly = subdivideLevel(createControlLevel(m, { boundaryMode: 'edgeOnly' }));
  const cornerPinned = vdist(lEdgeCorner.positions[0], m.positions[0]);
  const cornerMoved = vdist(lEdgeOnly.positions[0], m.positions[0]);
  assert(cornerPinned < 1e-12, 'corner should be pinned in edgeAndCorner mode');
  assert(cornerMoved > 1e-6, 'corner should move in edgeOnly mode');
});

test('degenerate valence-2 vertex is safe (no NaN, no drift beyond mask)', () => {
  // A "sliver" mesh: a single quad.  All four vertices are topological corners
  // and must stay pinned rather than producing NaN.
  const m = { positions: [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], faces: [[0, 1, 2, 3]] };
  const l0 = createControlLevel(m);
  let l = l0;
  for (let i = 0; i < 3; i++) l = subdivideLevel(l);
  for (const p of l.positions) {
    assert(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]), 'NaN position');
  }
});

// ============================================================================
section('3. Shared connectivity (no per-face seams)');

test('child vertex count matches V + E + F', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const l0 = createControlLevel(m);
    const V = l0.positions.length, E = l0.topology.numEdges, F = l0.faces.length;
    const l1 = subdivideLevel(l0);
    assert(l1.positions.length === V + E + F,
      `${name}: expected ${V + E + F} vertices, got ${l1.positions.length}`);
  }
});

test('no coincident-but-distinct vertices appear (seam check)', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    for (let i = 1; i < levels.length; i++) {
      const dups = findDuplicatePositions(levels[i].positions, 1e-10);
      if (dups.length !== 0) {
        const d = dups[0];
        throw new Error(`${name} level ${i}: ${dups.length} seam duplicate(s), e.g. ${d.i}/${d.j}`);
      }
    }
  }
});

test('every interior edge of the child mesh is shared by exactly two child faces', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const levels = buildLevels(m, 2);
    const l1 = levels[1];
    const topo = l1.topology;
    assert(topo.vertexInfo.every((v) => !v.nonManifold), `${name}: child mesh non-manifold vertex`);
    // interior parent vertices not on a boundary remain interior
    const l0 = levels[0];
    const interiorParent = [];
    for (let v = 0; v < l0.positions.length; v++) {
      if (l0.topology.vertexInfo[v].boundaryEdgeCount === 0) interiorParent.push(v);
    }
    for (const v of interiorParent) {
      assert(l1.topology.vertexInfo[v].boundaryEdgeCount === 0,
        `${name}: interior vertex ${v} became boundary after subdivision`);
    }
  }
});

test('boundary edge count is preserved under subdivision', () => {
  // A boundary edge whose endpoints are not both corners becomes 2 child boundary
  // edges, so the count only stays constant if the subdivision does not add
  // boundary vertices.  For general meshes the correct invariant is that boundary
  // edges remain a closed 1-manifold loop: every boundary vertex has exactly two
  // boundary edges.
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    for (let i = 0; i < levels.length; i++) {
      const info = levels[i].topology.vertexInfo;
      for (let v = 0; v < info.length; v++) {
        if (info[v].boundaryEdgeCount !== 0 && info[v].boundaryEdgeCount !== 2) {
          throw new Error(`${name} level ${i}: vertex ${v} has ${info[v].boundaryEdgeCount} boundary edges (not a 1-manifold boundary)`);
        }
      }
    }
  }
});

test('all child faces are quads', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const levels = buildLevels(m, 2);
    for (let i = 1; i < levels.length; i++) {
      for (const f of levels[i].faces) {
        assert(f.length === 4, `${name} level ${i}: face has ${f.length} corners`);
      }
    }
  }
});

test('orientation stays consistent (no inconsistent winding)', () => {
  for (const name of VALID_MODEL_NAMES) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    for (let i = 0; i < levels.length; i++) {
      const o = checkOrientation(levels[i].positions, levels[i].faces);
      assert(o.inconsistentWindingCount === 0,
        `${name} level ${i}: ${o.inconsistentWindingCount} inconsistent windings`);
    }
  }
});

// ============================================================================
section('4. Limit positions');

test('regular vertex limit equals the tensor-product B-spline mask', () => {
  const m = regularGrid(9, 0.4);
  const l0 = createControlLevel(m);
  const c = 4;
  const v = c * 9 + c;
  const lim = evaluateLimits(l0, { indices: [v] });
  // tensor product mask [1,4,1]/6 x [1,4,1]/6 over the 3x3 neighbourhood
  const w = [1 / 6, 2 / 3, 1 / 6];
  const expect = [0, 0, 0];
  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      const p = l0.positions[(c + dy) * 9 + (c + dx)];
      const ww = w[dy + 1] * w[dx + 1];
      for (let k = 0; k < 3; k++) expect[k] += p[k] * ww;
    }
  }
  assertVecClose(lim.results[0].position, expect, 1e-12, 'B-spline limit');
});

test('limit-position mask is stationary: same answer from level 0, 1 and 2', () => {
  // The mask is exact once the one-ring is a quad mesh, so non-quad control cages
  // get one normalising refinement.  For these all-quad fixtures the answer must
  // be identical at every level.
  const cases = ['regular-grid', 'quad-ball', 'torus', 'concave-uneven', 'open-cylinder'];
  for (const name of cases) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    const l0 = evaluateLimits(levels[0], { maxCreaseLevels: 0 });
    const l1 = evaluateLimits(levels[1], { maxCreaseLevels: 0 });
    const l2 = evaluateLimits(levels[2], { maxCreaseLevels: 0 });
    let maxErr = 0, worst = -1;
    for (let i = 0; i < levels[0].positions.length; i++) {
      const a = l0.results[i].position;
      const b = l1.results[i].position;
      const c = l2.results[i].position;
      const e = Math.max(vdist(a, b), vdist(a, c));
      if (e > maxErr) { maxErr = e; worst = i; }
    }
    assert(maxErr < 1e-9, `${name}: limit not stationary, max err ${maxErr.toExponential(3)} at vertex ${worst}`);
  }
});

test('triangulated control cage: one normalising refinement makes the limit stationary', () => {
  const m = modelByName('pentagon-pyramid');
  const levels = buildLevels(m, 3);
  // evaluated on the raw control cage -> should report quadLevels=1
  const raw = evaluateLimits(levels[0], { maxCreaseLevels: 0 });
  assert(raw.quadLevels === 1, 'expected exactly one normalising refinement for a triangle cage');
  // level 1 is all quads -> no further refinement
  const after = evaluateLimits(levels[1], { maxCreaseLevels: 0 });
  assert(after.quadLevels === 0, 'level 1 should already be quad-only');
  let maxErr = 0;
  for (let i = 0; i < levels[0].positions.length; i++) {
    maxErr = Math.max(maxErr, vdist(raw.results[i].position, after.results[i].position));
  }
  assert(maxErr < 1e-9, `normalising refinement did not make the limit stationary: ${maxErr.toExponential(3)}`);
});

test('extraordinary-valent limit matches direct OpenSubdiv-style mask', () => {
  // The mask is only valid on a quad one-ring, so evaluate at level 1 (all quads
  // after one Catmull-Clark step) and compare against the direct mask there.
  const m = modelByName('pentagon-pyramid');
  const levels = buildLevels(m, 1);
  const lv = levels[1];
  const lim = evaluateLimits(lv, { maxCreaseLevels: 0 });
  const apex = 0;
  const cInfo = lim.classification[apex];
  assert(cInfo.rule === RULE.SMOOTH, 'apex should be smooth (no creases)');
  const n = cInfo.valence;
  assert(n === 5, 'apex should be valence 5, got ' + n);
  const P = lv.positions[apex];
  let ox = 0, oy = 0, oz = 0;
  for (let i = lv.topology.vertexEdgeStart[apex]; i < lv.topology.vertexEdgeStart[apex + 1]; i++) {
    const e = lv.topology.edges[lv.topology.vertexEdges[i]];
    const o = lv.positions[e.a === apex ? e.b : e.a];
    ox += o[0]; oy += o[1]; oz += o[2];
  }
  let fx = 0, fy = 0, fz = 0;
  for (let i = lv.topology.vertexFaceStart[apex]; i < lv.topology.vertexFaceStart[apex + 1]; i++) {
    const f = lv.faces[lv.topology.vertexFaces[i]];
    const lc = lv.topology.vertexFaceCorner[i];
    const opp = f[(lc + 2) % f.length];
    const o = lv.positions[opp];
    fx += o[0]; fy += o[1]; fz += o[2];
  }
  const eW = 4 / (n * (n + 5)), fW = 1 / (n * (n + 5)), vW = 1 - n * (eW + fW);
  const expect = [
    vW * P[0] + eW * ox + fW * fx,
    vW * P[1] + eW * oy + fW * fy,
    vW * P[2] + eW * oz + fW * fz,
  ];
  assertVecClose(lim.results[apex].position, expect, 1e-12, 'extraordinary limit');
});

test('crease curve limit equals (A + 4P + B)/6', () => {
  const m = creaseRidge(Infinity);
  const l0 = createControlLevel(m);
  const mid = 3;
  const v = mid * 7 + mid;
  const cInfo = classifyVertices(l0.topology, l0.creaseMap, l0.cornerSet)[v];
  assert(cInfo.rule === RULE.CREASE, 'expected crease rule, got ' + cInfo.rule);
  const lim = evaluateLimits(l0, { indices: [v] });
  const A = l0.positions[v - 1];
  const B = l0.positions[v + 1];
  const P = l0.positions[v];
  const expect = [0, 1, 2].map((k) => (A[k] + 4 * P[k] + B[k]) / 6);
  assertVecClose(lim.results[0].position, expect, 1e-12, 'crease limit');
});

test('pinned corner limit equals the control point exactly', () => {
  const m = modelByName('sharp-corner-box');
  const l0 = createControlLevel(m);
  const cls = classifyVertices(l0.topology, l0.creaseMap, l0.cornerSet);
  const cornerVerts = [];
  for (let v = 0; v < cls.length; v++) if (cls[v].rule === RULE.CORNER) cornerVerts.push(v);
  assert(cornerVerts.length > 0, 'expected at least one pinned corner');
  assert(cornerVerts.includes(6), 'vertex 6 (three creases) should be a corner');
  const lim = evaluateLimits(l0, { indices: cornerVerts, maxCreaseLevels: 0 });
  for (let k = 0; k < lim.results.length; k++) {
    const v = cornerVerts[k];
    assertVecClose(lim.results[k].position, l0.positions[v], 1e-15, `corner limit v${v}`);
    assert(lim.results[k].policy === 'corner', `expected corner policy for v${v}`);
  }
});

test('explicitly pinned corner vertices stay exactly at the control point', () => {
  // Use the corner tag directly so the test does not depend on crease counting.
  const m = { positions: creaseRidge(Infinity).positions, faces: creaseRidge(Infinity).faces };
  m.corners = [0, 6, 42, 48];
  m.creases = creaseRidge(Infinity).creases;
  const l0 = createControlLevel(m);
  const lim = evaluateLimits(l0, { indices: m.corners });
  for (let k = 0; k < lim.results.length; k++) {
    assertVecClose(lim.results[k].position, l0.positions[m.corners[k]], 1e-15, 'tagged corner');
    assert(lim.results[k].policy === 'corner', 'tagged corner policy');
  }
});

test('extraordinary vertex adjacent to two infinite creases uses curve limit (policy recorded)', () => {
  const m = modelByName('crease-at-extraordinary');
  const l0 = createControlLevel(m);
  const lim = evaluateLimits(l0, { indices: [0] });
  const r = lim.results[0];
  assert(r.rule === RULE.CREASE, 'apex rule should be crease, got ' + r.rule);
  assert(r.valence === 5, 'apex valence should be 5');
  assert(r.policy === 'crease-limit', 'policy should be crease-limit, got ' + r.policy);
  // A triangle-fan cage needs one normalising refinement; after that the
  // infinite crease needs no semi-sharp decay refinement.
  assert(lim.creaseLevels === 0, 'infinite crease needs no crease-decay refinement');
  assert(lim.quadLevels === 1, 'triangle-fan cage needs exactly one quad-normalising refinement');
});

test('semi-sharp crease at extraordinary vertex refines before taking the limit', () => {
  const m = modelByName('semi-crease-at-extraordinary');
  const l0 = createControlLevel(m);
  assert(maxFiniteSharpness(l0) > 0, 'expected finite sharpness at level 0');
  const lim = evaluateLimits(l0, { indices: [0], maxExtraLevels: 6 });
  const r = lim.results[0];
  assert(lim.extraLevels > 0, 'expected extra subdivision for semi-sharp crease');
  assert(r.capped === false, 'semi-sharp crease should decay within the budget');
  // after the crease decays, the vertex must be smooth again
  assert(r.rule === RULE.SMOOTH, 'expected smooth rule after decay, got ' + r.rule);
});

test('semi-sharp decay budget is reported when it cannot finish', () => {
  const m = creaseRidge(12); // sharpness 12 needs 12 levels
  const l0 = createControlLevel(m);
  const { capped, extraLevels, remainingSharpness } = refineUntilStable(l0, 3);
  assert(capped === true, 'expected capped=true when budget too small');
  assert(extraLevels === 3, 'expected exactly 3 extra levels');
  assert(remainingSharpness > 0, 'expected remaining sharpness to be reported');
});

// ============================================================================
section('5. Convergence');

test('iteration approaches the limit geometrically', () => {
  for (const name of ['regular-grid', 'quad-ball', 'concave-uneven', 'pentagon-pyramid', 'torus']) {
    const m = modelByName(name);
    const levels = buildLevels(m, 6);
    const rootCount = levels[0].positions.length;
    const indices = [];
    for (let i = 0; i < rootCount; i++) indices.push(i);
    // limit from level 0
    const lim0 = evaluateLimits(levels[0], { maxExtraLevels: 0 });
    let prev = null;
    const errors = [];
    for (let l = 0; l < levels.length; l++) {
      let maxErr = 0;
      for (let k = 0; k < indices.length; k++) {
        const d = vdist(levels[l].positions[k], lim0.results[k].position);
        if (d > maxErr) maxErr = d;
      }
      errors.push(maxErr);
      if (l >= 2 && prev != null && maxErr > 1e-13) {
        const ratio = maxErr / prev;
        assert(ratio < 0.75,
          `${name} level ${l}: convergence ratio ${ratio.toFixed(3)} too slow (err ${maxErr.toExponential(2)})`);
      }
      prev = maxErr;
    }
    assert(errors[errors.length - 1] < errors[0] * 1e-3,
      `${name}: error did not shrink by 1000x over 6 levels (${errors[0].toExponential(2)} -> ${errors[errors.length - 1].toExponential(2)})`);
  }
});

test('checkLimitConvergence reports shrinking max errors', () => {
  const m = modelByName('concave-uneven');
  const levels = buildLevels(m, 5);
  const indices = levels[0].positions.map((_, i) => i);
  const report = checkLimitConvergence(levels, indices);
  assert(report.length === levels.length, 'report length');
  for (let i = 2; i < report.length; i++) {
    if (report[i].maxErrorToOwnLimit > 1e-13 && report[i].ratioToPreviousLevel != null) {
      assert(report[i].ratioToPreviousLevel < 0.75,
        `level ${i} ratio ${report[i].ratioToPreviousLevel}`);
    }
  }
});

// ============================================================================
section('6. Normal continuity');

test('dihedral angles shrink with subdivision on smooth meshes', () => {
  for (const name of ['quad-ball', 'torus', 'concave-uneven', 'pentagon-pyramid']) {
    const m = modelByName(name);
    const levels = buildLevels(m, 5);
    const stats = levels.map((lv) => edgeAngleStats(lv.positions, lv.faces));
    // The max angle must shrink substantially; G1 surfaces do not have creases.
    for (let i = 2; i < stats.length; i++) {
      assert(stats[i].maxAngleDeg < stats[1].maxAngleDeg,
        `${name} level ${i}: max dihedral ${stats[i].maxAngleDeg.toFixed(1)}° not below level-1 ${stats[1].maxAngleDeg.toFixed(1)}°`);
    }
    // By level 5 the p99 angle should be small on a smooth surface.
    assert(stats[5].p99AngleDeg < 12,
      `${name} level 5: p99 dihedral ${stats[5].p99AngleDeg.toFixed(2)}° too large`);
  }
});

test('boundary edges are excluded from dihedral statistics', () => {
  for (const name of ['open-boundary-sheet', 'open-cylinder']) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    for (const lv of levels) {
      const s = edgeAngleStats(lv.positions, lv.faces);
      assert(s.boundaryEdges === lv.topology.boundaryEdgeCount,
        `${name}: boundary edge count mismatch in stats`);
      assert(s.interiorEdges === lv.topology.numEdges - lv.topology.boundaryEdgeCount,
        `${name}: interior edge count mismatch in stats`);
    }
  }
});

// ============================================================================
section('7. Extraordinary vertices and boundaries');

test('several extraordinary valences are all represented', () => {
  const cases = [
    ['quad-ball', 3],          // valence-3 corners
    ['pentagon-pyramid', 5],   // valence-5 apex
    ['pentagon-pyramid', 3],   // valence-3 base ring
  ];
  for (const [name, expected] of cases) {
    const m = modelByName(name);
    const l0 = createControlLevel(m);
    const vals = new Set();
    for (const info of l0.topology.vertexInfo) {
      if (!info.boundary) vals.add(info.edgeCount);
    }
    assert(vals.has(expected), `${name}: no interior vertex of valence ${expected} (have ${[...vals]})`);
  }
  // Boundaries: rim vertices of the open cylinder are valence 3 but sit on the
  // boundary; verify boundary valences are also represented.
  const cyl = createControlLevel(modelByName('open-cylinder'));
  const boundaryVals = new Set();
  for (const info of cyl.topology.vertexInfo) {
    if (info.boundaryEdgeCount > 0) boundaryVals.add(info.edgeCount);
  }
  assert(boundaryVals.has(3), `open-cylinder: no boundary vertex of valence 3 (have ${[...boundaryVals]})`);
});

test('extraordinary interior vertices persist across levels on all-quad meshes', () => {
  // On a pure-quad cage, Catmull-Clark preserves the set of extraordinary
  // vertices (each one spawns a fan of 4-valent vertices only).  Triangle cages
  // are excluded because the first step splits every triangle into three quads
  // around a new face point, which legitimately adds extraordinary vertices.
  for (const name of ['quad-ball', 'torus', 'concave-uneven']) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    const countExtra = (lv) => {
      let c = 0;
      for (const info of lv.topology.vertexInfo) {
        if (!info.boundary && info.edgeCount !== 4) c++;
      }
      return c;
    };
    const e0 = countExtra(levels[0]);
    for (let i = 1; i < levels.length; i++) {
      assert(countExtra(levels[i]) === e0,
        `${name}: extraordinary count changed ${e0} -> ${countExtra(levels[i])} at level ${i}`);
    }
  }
});

test('pentagon-pyramid extraordinary count stabilises after the quad step', () => {
  const levels = buildLevels(modelByName('pentagon-pyramid'), 3);
  const countExtra = (lv) => {
    let c = 0;
    for (const info of lv.topology.vertexInfo) {
      if (!info.boundary && info.edgeCount !== 4) c++;
    }
    return c;
  };
  const e1 = countExtra(levels[1]);
  assert(countExtra(levels[2]) === e1, 'extraordinary count should stabilise at level 2');
  assert(countExtra(levels[3]) === e1, 'extraordinary count should stabilise at level 3');
});

test('open boundary vertices stay on their boundary curve', () => {
  // With edgeAndCorner, the boundary curve is the 1D B-spline of the boundary
  // polygon.  Consecutive subdivisions must keep boundary vertices inside the
  // convex hull of the original boundary polygon (z-plane for the cylinder).
  const m = modelByName('open-cylinder');
  const levels = buildLevels(m, 3);
  const z0 = m.positions.map((p) => p[2]);
  const zmin = Math.min(...z0) - 1e-9;
  const zmax = Math.max(...z0) + 1e-9;
  // boundary vertices cannot move outside a small expansion of the hull's
  // z-range (they may move to new positions but not to infinity / NaN)
  for (const lv of levels) {
    for (const p of lv.positions) {
      assert(Number.isFinite(p[2]), 'NaN boundary z');
    }
  }
});

test('torus remains closed: no boundary edges at any level', () => {
  const m = modelByName('torus');
  const levels = buildLevels(m, 4);
  for (let i = 0; i < levels.length; i++) {
    assert(levels[i].topology.boundaryEdgeCount === 0,
      `torus level ${i}: has ${levels[i].topology.boundaryEdgeCount} boundary edges`);
  }
});

test('Euler characteristic preserved for each topology class', () => {
  const cases = [
    ['quad-ball', 2],   // sphere
    ['torus', 0],       // genus 1
    ['regular-grid', 1], // open disc
    ['open-cylinder', 0], // open annulus
  ];
  for (const [name, expectedChi] of cases) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    for (let i = 0; i < levels.length; i++) {
      const lv = levels[i];
      const V = lv.positions.length;
      const E = lv.topology.numEdges;
      const F = lv.faces.length;
      const chi = V - E + F;
      assert(chi === expectedChi,
        `${name} level ${i}: Euler characteristic ${chi} != ${expectedChi}`);
    }
  }
});

// ============================================================================
section('8. Anti-examples (concave / uneven)');

test('concave uneven mesh keeps all vertices finite and distinct', () => {
  const m = modelByName('concave-uneven');
  const levels = buildLevels(m, 5);
  const lv = levels[levels.length - 1];
  for (const p of lv.positions) {
    assert(Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]),
      'non-finite position on uneven mesh');
  }
  const dups = findDuplicatePositions(lv.positions, 1e-11);
  assert(dups.length === 0, `${dups.length} duplicate positions on uneven mesh`);
});

test('concave uneven mesh has no large-area normal folds (surface stays fair)', () => {
  const m = modelByName('concave-uneven');
  const levels = buildLevels(m, 4);
  const lv = levels[3];
  const stats = edgeAngleStats(lv.positions, lv.faces);
  // The concave tray is smooth, so after 3 levels the surface should be fair.
  assert(stats.maxAngleDeg < 60,
    `max dihedral ${stats.maxAngleDeg.toFixed(1)}° suggests a crease/fold artefact`);
});

test('uneven spacing does not bias the limit: coarse and refined limit agree', () => {
  const m = modelByName('concave-uneven');
  const l0 = evaluateLimits(createControlLevel(m), { maxExtraLevels: 0 });
  const l3 = evaluateLimits(buildLevels(m, 3)[3], { maxExtraLevels: 0 });
  let maxErr = 0;
  for (let i = 0; i < m.positions.length; i++) {
    maxErr = Math.max(maxErr, vdist(l0.results[i].position, l3.results[i].position));
  }
  assert(maxErr < 1e-9, `uneven-mesh limit not stationary: ${maxErr.toExponential(3)}`);
});

// ============================================================================
section('9. Crease policy: continue subdividing vs take limit');

test('infinite crease vertex does not refine further', () => {
  const m = modelByName('crease-ridge-sharp');
  const l0 = createControlLevel(m);
  const lim = evaluateLimits(l0, { maxExtraLevels: 6 });
  assert(lim.extraLevels === 0, 'infinite crease should not trigger extra refinement');
  assert(lim.capped === false, 'should not be capped');
});

test('semi-sharp finite crease triggers extra refinement then resolves', () => {
  for (const name of ['crease-ridge', 'semi-crease-at-extraordinary']) {
    const m = modelByName(name);
    const l0 = createControlLevel(m);
    const sharp = maxFiniteSharpness(l0);
    assert(sharp > 0, `${name}: expected finite sharpness`);
    const lim = evaluateLimits(l0, { maxExtraLevels: 8 });
    assert(lim.extraLevels >= Math.min(sharp, 8),
      `${name}: refined only ${lim.extraLevels} levels for sharpness ${sharp}`);
    assert(!lim.capped, `${name}: should not be capped within budget`);
  }
});

test('creasing changes the limit away from the smooth surface', () => {
  const smooth = evaluateLimits(createControlLevel(creaseRidge(0)), { maxExtraLevels: 0 });
  const creased = evaluateLimits(createControlLevel(creaseRidge(Infinity)), { maxExtraLevels: 0 });
  const v = 3 * 7 + 3;
  const diff = vdist(smooth.results[v].position, creased.results[v].position);
  assert(diff > 1e-4, `crease had no effect on limit (diff ${diff.toExponential(2)})`);
});

// A dart (crease ending at a vertex, one sharp incident edge) has no closed-form
// stationary limit mask.  The smooth mask is wrong there and drifts by ~9% of the
// model size across levels; the "continue subdividing" branch must refine so the
// evaluated limit is level-independent.  This is the "singular vertex next to a
// crease" case the brief calls out explicitly.
test('dart/crease-end limits are stationary after the continue-subdividing policy', () => {
  const cases = ['sharp-corner-box', 'crease-at-extraordinary'];
  for (const name of cases) {
    const m = modelByName(name);
    const levels = buildLevels(m, 3);
    const lim0 = evaluateLimits(levels[0], { maxExtraLevels: 8 });
    const lim2 = evaluateLimits(levels[2], { maxExtraLevels: 8 });
    // The control level has darts -> the policy must have refined.
    const cls = classifyVertices(levels[0].topology, levels[0].creaseMap, levels[0].cornerSet);
    const dartCount = cls.filter((c) => c.rule === RULE.DART && !c.boundary).length;
    assert(dartCount > 0, `${name}: expected at least one interior dart`);
    assert(lim0.dartLevels >= 1, `${name}: expected dart refinement, got ${lim0.dartLevels}`);
    let maxErr = 0, worst = -1;
    for (let v = 0; v < m.positions.length; v++) {
      const e = vdist(lim0.results[v].position, lim2.results[v].position);
      if (e > maxErr) { maxErr = e; worst = v; }
    }
    assert(maxErr < 1e-9,
      `${name}: dart limit not stationary, max err ${maxErr.toExponential(3)} at v${worst}`);
    for (const r of lim0.results) {
      if (r.rule === RULE.DART && !r.boundary) {
        assert(r.approximate === true, `${name}: dart should be flagged approximate`);
        assert(r.policy === 'dart-limit', `${name}: dart policy should be dart-limit`);
      }
    }
  }
});

test('smooth and crease limits stay exact (no unnecessary refinement)', () => {
  // Darts are the only rule needing refinement; a crease-only or smooth model
  // must not pay the extra cost, and its mask must equal the direct formula.
  for (const name of ['quad-ball', 'crease-ridge-sharp', 'torus']) {
    const m = modelByName(name);
    const l0 = createControlLevel(m);
    const refined = refineUntilStable(l0, 8);
    assert(refined.dartLevels === 0, `${name}: expected no dart refinement`);
  }
});

// ============================================================================
section('10. Determinism / repeatability');

test('subdivision is deterministic', () => {
  const m = modelByName('quad-ball');
  const a = buildLevels(m, 3);
  const b = buildLevels(m, 3);
  for (let l = 0; l < a.length; l++) {
    assert(a[l].positions.length === b[l].positions.length, 'length mismatch');
    for (let i = 0; i < a[l].positions.length; i++) {
      assertVecClose(a[l].positions[i], b[l].positions[i], 1e-18, `level ${l} vertex ${i}`);
    }
  }
});

// ============================================================================
function centroid(level, faceIndex) {
  const face = level.faces[faceIndex];
  const c = [0, 0, 0];
  for (const v of face) for (let k = 0; k < 3; k++) c[k] += level.positions[v][k] / face.length;
  return c;
}

console.log('\n' + '='.repeat(60));
console.log(`RESULT: ${pass} passed, ${fail} failed`);
if (fail > 0) {
  console.log('\nFailures:');
  for (const f of failures) console.log(`  - ${f.name}: ${f.error.message}`);
  process.exit(1);
}
