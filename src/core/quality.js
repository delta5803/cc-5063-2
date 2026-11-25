// Mesh quality diagnostics: connectivity, continuity and limit convergence.
//
// These checks are intentionally independent of the renderer so that they can run
// under Node in tests and under the browser for the on-screen report.

import { buildTopology } from './topology.js';
import { evaluateLimits } from './limit.js';
import { vdist, vsub, vcross, vlen } from './vec3.js';

/**
 * Verify that the subdivision produced the expected number of shared vertices
 * and that no two distinct vertices occupy the same position (which would
 * indicate per-face interpolation producing duplicate seam vertices).
 */
export function checkConnectivity(levels) {
  const issues = [];
  const perLevel = [];
  for (let i = 0; i < levels.length; i++) {
    const lv = levels[i];
    const topology = lv.topology || buildTopology(lv.positions, lv.faces, { allowBoundary: true });
    let expectedV;
    let expectedE;
    let expectedF;
    if (i === 0) {
      expectedV = lv.positions.length;
      expectedE = topology.numEdges;
      expectedF = lv.faces.length;
    } else {
      const p = levels[i - 1];
      const pTopo = p.topology;
      expectedV = p.positions.length + pTopo.numEdges + p.faces.length;
      // Every parent edge yields 2 child edges, every parent face contributes its
      // boundary edges again (shared), and every parent vertex emits one edge per
      // incident corner.
      // Compute directly from the child mesh's topology instead of a closed form.
      expectedE = topology.numEdges;
      expectedF = countFaceCorners(lv.faces);
    }
    const duplicates = findDuplicatePositions(lv.positions);
    perLevel.push({
      level: i,
      vertices: lv.positions.length,
      edges: topology.numEdges,
      faces: lv.faces.length,
      expectedVertices: expectedV,
      expectedFaceCorners: expectedF,
      vertexCountMatches: lv.positions.length === expectedV,
      duplicatePositionCount: duplicates.length,
      duplicateSamples: duplicates.slice(0, 5),
      topologyValid: !topology.nonManifold,
      boundaryEdges: topology.boundaryEdgeCount,
    });
    if (i > 0 && lv.positions.length !== expectedV) {
      issues.push(`Level ${i}: expected ${expectedV} vertices, got ${lv.positions.length}`);
    }
    if (duplicates.length) {
      issues.push(`Level ${i}: ${duplicates.length} coincident vertex pair(s) (seam) e.g. ` +
        duplicates.slice(0, 3).map((d) => `${d.i}/${d.j} @ ${fmt(d.p)}`).join(', '));
    }
    const nonQuad = lv.faces.filter((f) => f.length !== 4).length;
    if (i > 0 && nonQuad > 0) {
      issues.push(`Level ${i}: ${nonQuad} non-quad face(s) after subdivision`);
    }
    const invalid = lv.faces.filter((f) => new Set(f).size !== f.length).length;
    if (invalid) issues.push(`Level ${i}: ${invalid} face(s) with repeated vertices`);
  }
  return { ok: issues.length === 0, issues, perLevel };
}

function countFaceCorners(faces) {
  let c = 0;
  for (const f of faces) c += f.length;
  return c;
}

export function findDuplicatePositions(positions, tolerance = 1e-9) {
  // Hash on rounded coordinates to catch coincident-but-distinct vertices.
  const buckets = new Map();
  const out = [];
  const inv = 1 / tolerance;
  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const kx = Math.round(p[0] * inv);
    const ky = Math.round(p[1] * inv);
    const kz = Math.round(p[2] * inv);
    // check the 27 neighbouring cells to be tolerant across cell boundaries
    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const key = (kx + dx) + ',' + (ky + dy) + ',' + (kz + dz);
          const list = buckets.get(key);
          if (!list) continue;
          for (const j of list) {
            if (vdist(positions[i], positions[j]) <= tolerance) {
              out.push({ i: Math.min(i, j), j: Math.max(i, j), p: positions[i] });
            }
          }
        }
      }
    }
    const key = kx + ',' + ky + ',' + kz;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(i);
  }
  return out;
}

/**
 * Compute per-face normals and check mesh orientation consistency: shared edges
 * must be traversed in opposite directions by their two faces.
 *
 * Note we deliberately do NOT flag faces whose normals disagree strongly as
 * "flipped": a legitimate sharp edge (a cone apex, a crease, the apex of the
 * pentagonal pyramid) has faces meeting at a large angle.  Only a winding that
 * is inconsistent with the topology (both faces erase the shared edge in the
 * same direction) is a real error, because subdivision would then produce a
 * self-overlapping fan.
 */
export function checkOrientation(positions, faces) {
  const topology = buildTopology(positions, faces, { allowBoundary: true });
  const inconsistentWinding = [];
  for (const e of topology.edges) {
    if (e.faces.length !== 2) continue;
    const f0 = e.faces[0];
    const f1 = e.faces[1];
    const dir0 = directedEdge(positions, faces[f0], e.a, e.b);
    const dir1 = directedEdge(positions, faces[f1], e.a, e.b);
    if (dir0 === 0 || dir1 === 0) continue;
    if (dir0 === dir1) inconsistentWinding.push({ a: e.a, b: e.b, f0, f1 });
  }
  return {
    ok: inconsistentWinding.length === 0,
    inconsistentWindingCount: inconsistentWinding.length,
    samples: { inconsistentWinding: inconsistentWinding.slice(0, 5) },
  };
}

function directedEdge(positions, face, a, b) {
  const n = face.length;
  for (let i = 0; i < n; i++) {
    const x = face[i];
    const y = face[(i + 1) % n];
    if (x === a && y === b) return 1;
    if (x === b && y === a) return -1;
  }
  return 0;
}

export function faceNormal(positions, face) {
  // Newell's method, robust for non-planar polygons.
  let nx = 0, ny = 0, nz = 0;
  const n = face.length;
  for (let i = 0; i < n; i++) {
    const a = positions[face[i]];
    const b = positions[face[(i + 1) % n]];
    nx += (a[1] - b[1]) * (a[2] + b[2]);
    ny += (a[2] - b[2]) * (a[0] + b[0]);
    nz += (a[0] - b[0]) * (a[1] + b[1]);
  }
  const l = Math.hypot(nx, ny, nz);
  if (l < 1e-30) return [0, 0, 0];
  return [nx / l, ny / l, nz / l];
}

export function vertexNormal(positions, faces, vertexFaceStart, vertexFaces, v) {
  let nx = 0, ny = 0, nz = 0;
  for (let i = vertexFaceStart[v]; i < vertexFaceStart[v + 1]; i++) {
    const fn = faceNormal(positions, faces[vertexFaces[i]]);
    nx += fn[0]; ny += fn[1]; nz += fn[2];
  }
  const l = Math.hypot(nx, ny, nz);
  if (l < 1e-30) return [0, 0, 0];
  return [nx / l, ny / l, nz / l];
}

/**
 * Dihedral angles across interior edges.  A large angle (> threshold) is reported
 * so that "C0" seams caused by incorrect shared vertices / normals show up.
 * Catmull-Clark gives C1 (G1 at extraordinary vertices), so for reasonably fine
 * meshes the dihedral angle should approach 0 everywhere.  We report the max and
 * a percentile instead of hard-failing: coarse control cages legitimately have
 * large angles at the control level.
 */
export function edgeAngleStats(positions, faces) {
  const topology = buildTopology(positions, faces, { allowBoundary: true });
  const normals = faces.map((f) => faceNormal(positions, f));
  const angles = [];
  let boundary = 0;
  for (const e of topology.edges) {
    if (e.faces.length !== 2) { boundary++; continue; }
    const n0 = normals[e.faces[0]];
    const n1 = normals[e.faces[1]];
    let dot = n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2];
    dot = Math.max(-1, Math.min(1, dot));
    angles.push(Math.acos(dot));
  }
  angles.sort((a, b) => a - b);
  const q = (p) => (angles.length ? angles[Math.min(angles.length - 1, Math.floor(p * angles.length))] : 0);
  return {
    interiorEdges: angles.length,
    boundaryEdges: boundary,
    maxAngleDeg: angles.length ? (angles[angles.length - 1] * 180) / Math.PI : 0,
    p99AngleDeg: (q(0.99) * 180) / Math.PI,
    medianAngleDeg: (q(0.5) * 180) / Math.PI,
    angles,
  };
}

/**
 * Compare each level's tracked vertices against the limit positions implied by
 * that same level's topology.  For a stationary scheme the "level limit" of a
 * tracked vertex is identical at every level; the distance to the current vertex
 * position shrinks geometrically.  Returns per-level max error and the observed
 * convergence ratio.
 */
export function checkLimitConvergence(levels, trackedIndices) {
  const perLevel = [];
  let previousError = null;
  for (let i = 0; i < levels.length; i++) {
    const lv = levels[i];
    const lim = evaluateLimits(lv, { indices: trackedIndices, maxExtraLevels: 0 });
    let maxErr = 0;
    let count = 0;
    for (let k = 0; k < lim.results.length; k++) {
      const r = lim.results[k];
      if (!r.position) continue;
      const d = vdist(lv.positions[r.index], r.position);
      if (d > maxErr) maxErr = d;
      count++;
    }
    const ratio = previousError != null && maxErr > 0 ? maxErr / previousError : null;
    perLevel.push({
      level: i,
      tracked: count,
      maxErrorToOwnLimit: maxErr,
      ratioToPreviousLevel: ratio,
    });
    if (maxErr > 0) previousError = maxErr;
  }
  return perLevel;
}

function fmt(p) {
  return `(${p[0].toFixed(4)}, ${p[1].toFixed(4)}, ${p[2].toFixed(4)})`;
}
