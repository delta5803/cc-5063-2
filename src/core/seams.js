// Seam analysis and limit-normal comparison.
//
// "Seam" here means either:
//   * a projection seam: two distinct vertices at the same position (the slit of
//     the stroke-regression fixture), or
//   * a crease seam: an interior chain where an injected inheritance fault makes
//     the two sides disagree.
//
// The two quantities the acceptance tests must keep apart are
//   * vertex position difference  -- where the surface sits, and
//   * limit normal difference     -- which way the limit surface faces.
// A position shift with unchanged normals is a translation; a normal change with
// unchanged positions is a kink.  Reporting them separately is what lets a test
// say which one a fault caused.

import { buildTopology } from './topology.js';
import { evaluateLimits } from './limit.js';
import { vdist, vdot, vlen } from './vec3.js';
import { findDuplicatePositions } from './quality.js';

/** Coincident-but-distinct vertices: the intentional or accidental seams. */
export function findSeamPairs(level, tolerance = 1e-9) {
  return findDuplicatePositions(level.positions, tolerance);
}

/** Per-vertex incidence summary for the two sides of each seam pair. */
export function describeSeams(level, pairs) {
  const topo = level.topology || buildTopology(level.positions, level.faces, { allowBoundary: true });
  return pairs.map(({ i, j, p }) => ({
    i,
    j,
    position: p,
    iValence: topo.vertexEdgeStart[i + 1] - topo.vertexEdgeStart[i],
    jValence: topo.vertexEdgeStart[j + 1] - topo.vertexEdgeStart[j],
    iBoundaryEdges: topo.vertexInfo[i].boundaryEdgeCount,
    jBoundaryEdges: topo.vertexInfo[j].boundaryEdgeCount,
    iFaces: topo.vertexInfo[i].faceCount,
    jFaces: topo.vertexInfo[j].faceCount,
  }));
}

/**
 * Limit positions for every vertex of a level, keyed by the level's own vertex
 * indices.  Vertices the evaluator does not return keep their current position.
 */
export function limitPositions(level, options = {}) {
  const lim = evaluateLimits(level, {
    maxCreaseLevels: 0,
    maxDartLevels: options.maxDartLevels != null ? options.maxDartLevels : 4,
    maxVertices: options.maxVertices,
  });
  const out = new Array(level.positions.length);
  for (let i = 0; i < out.length; i++) out[i] = level.positions[i];
  for (const r of lim.results) if (r.position) out[r.index] = r.position;
  return out;
}

/** Newell normal of a polygon whose positions are given by `positions`. */
export function polygonNormal(positions, face) {
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
  return l > 1e-30 ? [nx / l, ny / l, nz / l] : [0, 0, 0];
}

/**
 * Limit normal at a vertex: the area-binomial average of the normals of its
 * incident faces *evaluated at their limit positions*.  This is the direction the
 * delivered limit surface faces, not the coarse cage normal.
 */
export function vertexLimitNormal(level, limitPos, v) {
  const topo = level.topology || buildTopology(level.positions, level.faces, { allowBoundary: true });
  let nx = 0, ny = 0, nz = 0;
  const f0 = topo.vertexFaceStart[v];
  const f1 = topo.vertexFaceStart[v + 1];
  for (let i = f0; i < f1; i++) {
    const fn = polygonNormal(limitPos, level.faces[topo.vertexFaces[i]]);
    nx += fn[0]; ny += fn[1]; nz += fn[2];
  }
  const l = Math.hypot(nx, ny, nz);
  return l > 1e-30 ? [nx / l, ny / l, nz / l] : [0, 0, 0];
}

function angleBetween(a, b) {
  const la = vlen(a);
  const lb = vlen(b);
  if (la < 1e-30 || lb < 1e-30) return 0;
  const d = Math.max(-1, Math.min(1, vdot(a, b) / (la * lb)));
  return Math.acos(d);
}

/**
 * Compare a faulted level against its clean counterpart at a set of vertices.
 * Both levels must share topology (same control net, same subdivision count), so
 * vertex indices correspond one-to-one.
 */
export function seamMetrics(cleanLevel, testLevel, vertices, options = {}) {
  if (cleanLevel.positions.length !== testLevel.positions.length ||
      cleanLevel.faces.length !== testLevel.faces.length) {
    throw new Error('seamMetrics needs two levels with identical topology');
  }
  const cleanLim = limitPositions(cleanLevel, options);
  const testLim = limitPositions(testLevel, options);

  let posMax = 0, posSum = 0, posWorst = -1;
  let limMax = 0, limSum = 0, limWorst = -1;
  let angMax = 0, angSum = 0, angWorst = -1;
  let count = 0;
  for (const v of vertices) {
    const dp = vdist(cleanLevel.positions[v], testLevel.positions[v]);
    posSum += dp; count++;
    if (dp > posMax) { posMax = dp; posWorst = v; }

    const dl = vdist(cleanLim[v], testLim[v]);
    limSum += dl;
    if (dl > limMax) { limMax = dl; limWorst = v; }

    const ang = angleBetween(
      vertexLimitNormal(cleanLevel, cleanLim, v),
      vertexLimitNormal(testLevel, testLim, v),
    );
    angSum += ang;
    if (ang > angMax) { angMax = ang; angWorst = v; }
  }
  return {
    count,
    positionDelta: {
      max: posMax,
      mean: count ? posSum / count : 0,
      worstVertex: posWorst,
    },
    limitPositionDelta: {
      max: limMax,
      mean: count ? limSum / count : 0,
      worstVertex: limWorst,
    },
    normalDelta: {
      maxAngleDeg: (angMax * 180) / Math.PI,
      meanAngleDeg: count ? (angSum / count) * 180 / Math.PI : 0,
      worstVertex: angWorst,
    },
  };
}

/**
 * Topological integrity of a subdivided level.  Deliberately independent of the
 * geometric checks: a build can have perfect positions and still be broken here
 * (or vice versa).
 */
export function topologyEvidence(level) {
  const topo = level.topology || buildTopology(level.positions, level.faces, { allowBoundary: true });
  const nonManifoldVertices = [];
  for (let v = 0; v < topo.vertexInfo.length; v++) {
    if (topo.vertexInfo[v].nonManifold) nonManifoldVertices.push(v);
  }
  const nonQuadFaces = level.faces.filter((f) => f.length !== 4).length;
  const repeatedVertexFaces = level.faces.filter((f) => new Set(f).size !== f.length).length;
  const boundaryNotOneManifold = [];
  for (let v = 0; v < topo.vertexInfo.length; v++) {
    const b = topo.vertexInfo[v].boundaryEdgeCount;
    if (b !== 0 && b !== 2) boundaryNotOneManifold.push(v);
  }
  return {
    vertices: level.positions.length,
    edges: topo.numEdges,
    faces: level.faces.length,
    euler: level.positions.length - topo.numEdges + level.faces.length,
    boundaryEdges: topo.boundaryEdgeCount,
    nonManifoldVertices,
    nonManifoldEdgeCount: topo.edges.filter((e) => e.faces.length > 2).length,
    nonQuadFaces,
    repeatedVertexFaces,
    boundaryNotOneManifold,
    ok: nonManifoldVertices.length === 0 && nonManifoldFaces(level) === 0 &&
      nonQuadFaces === 0 && repeatedVertexFaces === 0 && boundaryNotOneManifold.length === 0,
  };
}

function nonManifoldFaces(level) {
  const topo = level.topology || buildTopology(level.positions, level.faces, { allowBoundary: true });
  return topo.edges.filter((e) => e.faces.length > 2).length;
}
