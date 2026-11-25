// Catmull-Clark subdivision with real shared topology.
//
// Each subdivision step creates exactly one child vertex per parent face (face
// point), one per parent edge (edge point) and one per parent vertex (vertex
// point).  Child quads reference those shared indices, so the two faces on either
// side of an edge use the *same* edge point.  Per-face interpolation into a local
// grid would instead produce two coincident-but-distinct edge points and a visible
// seam; that is explicitly not done here.
//
// Rules implemented (Catmull-Clark + OpenSubdiv "Catmark" creasing):
//   face point     centroid of the parent face
//   edge point     midpoint for sharp/boundary edges, else 1/4(v0+v1)+1/4(f0+f1).
//                  Semi-sharp edges (0 < s < 1) blend midpoint and smooth masks.
//   vertex point   smooth CC mask, cubic B-spline crease mask 3/4 P + 1/8(A+B),
//                  or pinned corner.
//   sharpness      finite sharpness decays by 1 per level; boundary edges are
//                  infinitely sharp.  Rule transitions between parent and child
//                  are blended the same way OpenSubdiv's Catmark scheme does.
//
// Output is quad-only (an n-gon produces n child quads); the input may be any
// manifold polygon mesh.

import { buildTopology } from './topology.js';
import {
  buildCreaseMap, classifyVertices, decaySharpness, normalizeCorners,
  transitionWeight, RULE,
} from './creases.js';

export const DEFAULT_OPTIONS = Object.freeze({
  boundaryMode: 'edgeAndCorner', // 'edgeAndCorner' | 'edgeOnly'
  validateEachLevel: true,
});

export function pinBoundaryCornersFromOptions(options) {
  return (options.boundaryMode || 'edgeAndCorner') !== 'edgeOnly';
}

function faceCentroid(positions, face) {
  let x = 0, y = 0, z = 0;
  for (let i = 0; i < face.length; i++) {
    const p = positions[face[i]];
    x += p[0]; y += p[1]; z += p[2];
  }
  const inv = 1 / face.length;
  return [x * inv, y * inv, z * inv];
}

/**
 * Create a level-0 object from a control mesh.
 */
export function createControlLevel(control, options = {}) {
  const opts = { ...DEFAULT_OPTIONS, ...options };
  const positions = control.positions.map((p) => [p[0], p[1], p[2]]);
  const faces = control.faces.map((f) => f.slice());
  const cornerSet = normalizeCorners(control.corners);
  const creaseMap = buildCreaseMap(control.creases, positions, opts);

  const topology = buildTopology(positions, faces, { allowBoundary: true });

  // Every crease must reference an actual edge of the cage; silently ignoring a
  // mistyped crease would hide modelling errors.
  for (const c of (control.creases || [])) {
    if (!(c.sharpness > 0)) continue;
    const key = c.a < c.b ? c.a + ':' + c.b : c.b + ':' + c.a;
    if (!topology.edgeIndexByKey.has(key)) {
      const err = new Error(
        `Crease references edge ${c.a}-${c.b}, which does not exist in the control mesh`
      );
      err.code = 'INVALID_CREASE';
      err.details = [{ a: c.a, b: c.b }];
      throw err;
    }
  }
  for (const v of cornerSet) {
    if (!Number.isInteger(v) || v < 0 || v >= positions.length) {
      const err = new Error(`Corner tag references invalid vertex ${v}`);
      err.code = 'INVALID_CORNER';
      err.details = [{ vertex: v }];
      throw err;
    }
  }

  return {
    level: 0,
    positions,
    faces,
    creaseMap,
    cornerSet,
    options: opts,
    topology,
    rootVertexCount: positions.length,
    edgePointIndex: null,
    facePointIndex: null,
  };
}

/**
 * Subdivide one level.
 *
 * @param {object} level  as returned by createControlLevel / subdivideLevel
 * @returns {object} the next level, including its validated topology
 */
export function subdivideLevel(level, optionsOverride = {}) {
  const opts = { ...level.options, ...optionsOverride };
  const { positions, faces, creaseMap, cornerSet } = level;
  const topology = level.topology || buildTopology(positions, faces, { allowBoundary: true });
  const pinCorners = pinBoundaryCornersFromOptions(opts);

  const V = positions.length;
  const F = faces.length;
  const E = topology.numEdges;
  const edges = topology.edges;
  const faceStart = topology.faceStart;

  const classification = classifyVertices(topology, creaseMap, cornerSet, {
    pinBoundaryCorners: pinCorners,
  });

  // Boundary corner vertices (one face, two boundary edges) are pinned for all
  // levels when the boundary mode asks for it, so they survive as explicit tags.
  const pinnedSet = new Set(cornerSet);
  if (pinCorners) {
    for (let v = 0; v < V; v++) {
      if (classification[v].isTopologicalCorner) pinnedSet.add(v);
    }
  }

  // --- face points ----------------------------------------------------------
  const faceCenters = new Array(F);
  for (let f = 0; f < F; f++) faceCenters[f] = faceCentroid(positions, faces[f]);

  // --- per-edge parent/child sharpness -------------------------------------
  const edgeParentSharp = new Float64Array(E);
  const edgeChildSharp = new Float64Array(E);
  for (let ei = 0; ei < E; ei++) {
    const e = edges[ei];
    if (e.faces.length < 2) {
      edgeParentSharp[ei] = Infinity;
      edgeChildSharp[ei] = Infinity;
    } else {
      const s = creaseMap.get(e.key) || 0;
      edgeParentSharp[ei] = s;
      edgeChildSharp[ei] = decaySharpness(s);
    }
  }

  // --- edge points ----------------------------------------------------------
  const edgePoints = new Array(E);
  for (let ei = 0; ei < E; ei++) {
    const e = edges[ei];
    const pa = positions[e.a];
    const pb = positions[e.b];
    const s = edgeParentSharp[ei];
    const mx = (pa[0] + pb[0]) * 0.5;
    const my = (pa[1] + pb[1]) * 0.5;
    const mz = (pa[2] + pb[2]) * 0.5;
    if (e.faces.length < 2 || s >= 1) {
      edgePoints[ei] = [mx, my, mz];
      continue;
    }
    const f0 = faceCenters[e.faces[0]];
    const f1 = faceCenters[e.faces[1]];
    const sx = (pa[0] + pb[0] + f0[0] + f1[0]) * 0.25;
    const sy = (pa[1] + pb[1] + f0[1] + f1[1]) * 0.25;
    const sz = (pa[2] + pb[2] + f0[2] + f1[2]) * 0.25;
    if (s > 0) {
      edgePoints[ei] = [
        mx * s + sx * (1 - s),
        my * s + sy * (1 - s),
        mz * s + sz * (1 - s),
      ];
    } else {
      edgePoints[ei] = [sx, sy, sz];
    }
  }

  // --- vertex points --------------------------------------------------------
  const vertexPoints = new Array(V);

  for (let v = 0; v < V; v++) {
    const start = topology.vertexEdgeStart[v];
    const end = topology.vertexEdgeStart[v + 1];
    const nEdges = end - start;
    const P = positions[v];

    if (nEdges === 0) {
      vertexPoints[v] = [P[0], P[1], P[2]];
      continue;
    }

    const cInfo = classification[v];
    const parentRule = cInfo.rule;

    const incidentEdgeIds = [];
    const parentSharp = [];
    const childSharp = [];
    for (let i = start; i < end; i++) {
      const ei = topology.vertexEdges[i];
      incidentEdgeIds.push(ei);
      parentSharp.push(edgeParentSharp[ei]);
      childSharp.push(edgeChildSharp[ei]);
    }

    let childSharpCount = 0;
    for (let i = 0; i < childSharp.length; i++) if (childSharp[i] > 0) childSharpCount++;
    let childRule;
    if (pinnedSet.has(v) || childSharpCount >= 3) childRule = RULE.CORNER;
    else if (childSharpCount === 2) childRule = RULE.CREASE;
    else if (childSharpCount === 1) childRule = RULE.DART;
    else childRule = RULE.SMOOTH;

    const smoothPos = () => smoothVertexPosition(v, topology, positions, faceCenters, nEdges);

    const rulePos = (rule, sharpEdgeIds) => {
      if (rule === RULE.CORNER) return [P[0], P[1], P[2]];
      if (rule === RULE.CREASE && sharpEdgeIds.length === 2) {
        const a = edgeOpposite(edges, positions, sharpEdgeIds[0], v);
        const b = edgeOpposite(edges, positions, sharpEdgeIds[1], v);
        return [
          P[0] * 0.75 + (a[0] + b[0]) * 0.125,
          P[1] * 0.75 + (a[1] + b[1]) * 0.125,
          P[2] * 0.75 + (a[2] + b[2]) * 0.125,
        ];
      }
      return smoothPos();
    };

    // Smooth and dart vertices use the smooth mask unconditionally (a dart is
    // smooth except that its single sharp edge stays sharp for edge points).
    if (parentRule === RULE.SMOOTH || parentRule === RULE.DART) {
      vertexPoints[v] = smoothPos();
      continue;
    }

    const pPos = rulePos(parentRule, cInfo.sharpEdges);
    if (childRule === parentRule) {
      vertexPoints[v] = pPos;
      continue;
    }

    const childSharpEdgeIds = [];
    for (let i = 0; i < incidentEdgeIds.length; i++) {
      if (childSharp[i] > 0) childSharpEdgeIds.push(incidentEdgeIds[i]);
    }
    const cPos = rulePos(childRule, childSharpEdgeIds);
    const w = transitionWeight(parentSharp, childSharp);
    vertexPoints[v] = [
      pPos[0] * w + cPos[0] * (1 - w),
      pPos[1] * w + cPos[1] * (1 - w),
      pPos[2] * w + cPos[2] * (1 - w),
    ];
  }

  // --- assemble child mesh --------------------------------------------------
  const newPositions = new Array(V + E + F);
  for (let v = 0; v < V; v++) newPositions[v] = vertexPoints[v];
  for (let ei = 0; ei < E; ei++) newPositions[V + ei] = edgePoints[ei];
  for (let f = 0; f < F; f++) newPositions[V + E + f] = faceCenters[f];

  const edgePointIndex = new Int32Array(E);
  for (let ei = 0; ei < E; ei++) edgePointIndex[ei] = V + ei;
  const facePointIndex = new Int32Array(F);
  for (let f = 0; f < F; f++) facePointIndex[f] = V + E + f;

  const newFaces = [];
  for (let f = 0; f < F; f++) {
    const face = faces[f];
    const n = face.length;
    const base = faceStart[f];
    const fp = facePointIndex[f];
    for (let i = 0; i < n; i++) {
      const vCur = face[i];
      const ePrev = topology.faceEdges[base + ((i - 1 + n) % n)];
      const eNext = topology.faceEdges[base + i];
      // One child quad per parent corner: vertex point, forward edge point,
      // face point, backward edge point.  Edge points are shared between the
      // two faces adjacent to the parent edge.
      newFaces.push([
        vCur,
        edgePointIndex[eNext],
        fp,
        edgePointIndex[ePrev],
      ]);
    }
  }

  // --- propagate creases to child edges ------------------------------------
  const newCreaseMap = new Map();
  for (let ei = 0; ei < E; ei++) {
    const e = edges[ei];
    if (e.faces.length < 2) continue;
    const s = edgeChildSharp[ei];
    if (!(s > 0)) continue;
    const ep = edgePointIndex[ei];
    newCreaseMap.set(edgeKeyOf(e.a, ep), s);
    newCreaseMap.set(edgeKeyOf(e.b, ep), s);
  }

  const newCornerSet = new Set();
  for (const v of cornerSet) if (v < V) newCornerSet.add(v);

  const newLevel = {
    level: level.level + 1,
    positions: newPositions,
    faces: newFaces,
    creaseMap: newCreaseMap,
    cornerSet: newCornerSet,
    options: opts,
    topology: null,
    rootVertexCount: level.rootVertexCount,
    edgePointIndex,
    facePointIndex,
  };

  if (opts.validateEachLevel !== false) {
    newLevel.topology = buildTopology(newPositions, newFaces, { allowBoundary: true });
  }
  return newLevel;
}

function edgeKeyOf(a, b) {
  return a < b ? a + ':' + b : b + ':' + a;
}

function edgeOpposite(edges, positions, ei, v) {
  const e = edges[ei];
  return positions[e.a === v ? e.b : e.a];
}

/**
 * Catmull-Clark smooth vertex mask:
 *   ((n-3) P + 2 R_avg + F_avg) / n
 * which equals ((n-2)/n) P + 1/n^2 sum(face centers) + 1/n^2 sum(opposites).
 * Only called for manifold interior (or dart) vertices, where #faces == #edges.
 */
function smoothVertexPosition(v, topology, positions, faceCenters, nEdges) {
  const P = positions[v];
  const n = nEdges;
  if (n < 3) return [P[0], P[1], P[2]];
  let fx = 0, fy = 0, fz = 0;
  const fStart = topology.vertexFaceStart[v];
  const fEnd = topology.vertexFaceStart[v + 1];
  if (fEnd - fStart !== n) {
    // Would indicate a non-manifold vertex that slipped through validation.
    throw new Error(`Vertex ${v} has ${fEnd - fStart} faces but ${n} edges; not manifold`);
  }
  for (let i = fStart; i < fEnd; i++) {
    const fc = faceCenters[topology.vertexFaces[i]];
    fx += fc[0]; fy += fc[1]; fz += fc[2];
  }
  let ox = 0, oy = 0, oz = 0;
  for (let i = topology.vertexEdgeStart[v]; i < topology.vertexEdgeStart[v + 1]; i++) {
    const e = topology.edges[topology.vertexEdges[i]];
    const o = positions[e.a === v ? e.b : e.a];
    ox += o[0]; oy += o[1]; oz += o[2];
  }
  const wP = (n - 2) / n;
  const w = 1 / (n * n);
  return [
    P[0] * wP + (fx + ox) * w,
    P[1] * wP + (fy + oy) * w,
    P[2] * wP + (fz + oz) * w,
  ];
}

/**
 * Subdivide n times; returns the levels array (index = level number).
 */
export function buildLevels(control, maxLevel, options = {}) {
  const levels = [createControlLevel(control, options)];
  for (let i = 1; i <= maxLevel; i++) {
    levels.push(subdivideLevel(levels[i - 1]));
  }
  return levels;
}
