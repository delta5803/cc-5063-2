// Limit-surface evaluation.
//
// Catmull-Clark is a stationary refinement scheme: the limit position of a vertex
// does not depend on which refinement level we evaluate the limit mask at, once
// the neighbourhood is quad-structured.  We use the OpenSubdiv "Catmark" masks:
//
//   smooth / dart vertex   vW*P + eW*sum(edge-opposite) + fW*sum(face-opposite)
//                          eW = 4 / (n(n+5)), fW = 1 / (n(n+5)), vW = 1 - n(eW+fW)
//   crease vertex          (A + 4P + B) / 6   along the two sharp edges
//   corner vertex          P                   (pinned)
//
// The face weights act on the *opposite* vertex of each incident face (the
// diagonal of a quad), not on the face centroid.  Using face centroids (an easy
// mistake) yields a different, non-stationary point; the tests compare against a
// tensor-product B-spline mask to guard against that.
//
// Two cases require refining before the mask is valid:
//
//   1. Non-quad control meshes.  The limit mask above is derived for a vertex
//      whose one-ring is a quad mesh.  On a control cage containing triangles or
//      n-gons the mask is *not* the limit.  One Catmull-Clark step makes every
//      face a quad, after which the mask is exact and stationary.  We therefore
//      perform a single normalising refinement for non-quad levels.
//
//   2. Semi-sharp creases.  A finite sharpness decays over levels, so the vertex
//      rule can change.  The limit is only reached after the sharpness has fully
//      decayed.  For vertices that still carry finite sharpness we keep
//      subdividing (within a budget) and report whether the budget was enough.
//
//   3. Darts (a crease ending at a vertex, i.e. exactly one sharp incident edge).
//      The smooth mask is exact for smooth and crease vertices (verified
//      stationary to ~1e-16), but it is NOT the limit for a dart: the terminating
//      sharp edge perturbs the eigen-structure and there is no closed-form limit
//      stencil.  Evaluating the smooth mask at the control level therefore gives
//      a point that drifts by up to ~9% of the model size as levels are compared.
//      For darts we follow the "continue subdividing" branch: refine a bounded
//      number of levels and read the mask there, where it converges to the true
//      iterated limit, and report the extra levels and approximation status.
//
// This module reports, per requested vertex, whether the limit was taken from a
// mask, from a curve, from a pinned corner, or whether extra refinement was
// required ("singular vertex next to a crease" case).

import { buildTopology } from './topology.js';
import { classifyVertices, RULE } from './creases.js';
import { subdivideLevel, pinBoundaryCornersFromOptions } from './subdivision.js';

export function maxFiniteSharpness(level) {
  let m = 0;
  for (const s of level.creaseMap.values()) {
    if (Number.isFinite(s) && s > m) m = s;
  }
  return m;
}

export function isAllQuads(faces) {
  for (const f of faces) if (f.length !== 4) return false;
  return true;
}

/**
 * Refine until the limit masks are valid:
 *   - the level is a pure quad mesh (needed at most once), and
 *   - no finite (semi-sharp) crease sharpness remains (bounded by maxExtraLevels).
 *
 * Returns the refined level together with a breakdown of why refinement was done.
 */
export function refineUntilStable(level, maxCreaseLevels = 4, maxDartLevels = 4, options = {}) {
  const onlyIndices = options.indices || null;
  const maxVertices = options.maxVertices != null ? options.maxVertices : 400_000;
  let cur = level;
  let quadLevels = 0;
  let creaseLevels = 0;
  let dartLevels = 0;

  if (!isAllQuads(cur.faces)) {
    cur = subdivideLevel(cur);
    quadLevels++;
  }

  while (creaseLevels < maxCreaseLevels && maxFiniteSharpness(cur) > 0) {
    cur = subdivideLevel(cur);
    creaseLevels++;
  }

  // Darts terminate a crease; their smooth mask is not exactly stationary, so
  // refining converges to the iterated limit.  Only refine when one of the
  // vertices actually being evaluated is a dart -- otherwise this is pure cost
  // (and refining the whole cage per hover would be unusable on deep levels).
  while (dartLevels < maxDartLevels && hasDart(cur, onlyIndices)
    && predictedSize(cur) <= maxVertices) {
    cur = subdivideLevel(cur);
    dartLevels++;
  }

  const remaining = maxFiniteSharpness(cur);
  return {
    level: cur,
    quadLevels,
    creaseLevels,
    dartLevels,
    extraLevels: quadLevels + creaseLevels + dartLevels,
    capped: remaining > 0,
    remainingSharpness: remaining,
    // Darts persist under refinement (the crease still terminates), so unless the
    // budget ran out the result is a converged approximation, not an exact mask.
    dartApproximate: dartLevels > 0 && hasDart(cur, onlyIndices),
  };
}

function predictedSize(level) {
  const topo = level.topology;
  const e = topo ? topo.numEdges : level.faces.length * 2;
  return level.positions.length + e + level.faces.length;
}

function hasDart(level, onlyIndices = null) {
  const topo = level.topology || buildTopology(level.positions, level.faces, { allowBoundary: true });
  const cls = classifyVertices(topo, level.creaseMap, level.cornerSet);
  const check = onlyIndices || null;
  if (check) {
    for (const v of check) {
      const c = cls[v];
      if (c && c.rule === RULE.DART && !c.boundary) return true;
    }
    return false;
  }
  for (let v = 0; v < cls.length; v++) {
    if (cls[v].rule === RULE.DART && !cls[v].boundary) return true;
  }
  return false;
}

/**
 * Evaluate limit positions.
 *
 * @param {object} level
 * @param {object} [options]
 * @param {number[]} [options.indices]         vertex indices to evaluate (default all)
 * @param {number} [options.maxCreaseLevels=4] refinement budget for semi-sharp decay
 * @param {number} [options.maxDartLevels=4]   refinement budget for crease-ending (dart) vertices
 */
export function evaluateLimits(level, options = {}) {
  const maxCreaseLevels = options.maxCreaseLevels != null
    ? options.maxCreaseLevels
    : (options.maxExtraLevels != null ? options.maxExtraLevels : 4);
  const maxDartLevels = options.maxDartLevels != null
    ? options.maxDartLevels
    : (options.maxExtraLevels != null ? options.maxExtraLevels : 4);
  const refine = refineUntilStable(level, maxCreaseLevels, maxDartLevels, {
    indices: options.indices || null,
    maxVertices: options.maxVertices,
  });
  const evalLevel = refine.level;

  const positions = evalLevel.positions;
  const topology = evalLevel.topology
    || buildTopology(positions, evalLevel.faces, { allowBoundary: true });
  const pinCorners = pinBoundaryCornersFromOptions(evalLevel.options || {});
  const classification = classifyVertices(topology, evalLevel.creaseMap, evalLevel.cornerSet, {
    pinBoundaryCorners: pinCorners,
  });

  const indices = options.indices || range(positions.length);
  const results = new Array(indices.length);

  const faceStarts = topology.vertexFaceStart;
  const faceIds = topology.vertexFaces;
  const faceCorner = topology.vertexFaceCorner;
  const edgeStarts = topology.vertexEdgeStart;
  const edgeIds = topology.vertexEdges;
  const edges = topology.edges;
  const faces = evalLevel.faces;

  for (let k = 0; k < indices.length; k++) {
    const v = indices[k];
    const cInfo = classification[v];
    const P = positions[v];
    const n = cInfo.valence;
    let pos;
    let policy;

    if (cInfo.rule === RULE.CORNER) {
      pos = [P[0], P[1], P[2]];
      policy = 'corner';
    } else if (cInfo.rule === RULE.CREASE && cInfo.sharpEdges.length === 2) {
      const a = edgeOpposite(edges, positions, cInfo.sharpEdges[0], v);
      const b = edgeOpposite(edges, positions, cInfo.sharpEdges[1], v);
      pos = [
        (a[0] + 4 * P[0] + b[0]) / 6,
        (a[1] + 4 * P[1] + b[1]) / 6,
        (a[2] + 4 * P[2] + b[2]) / 6,
      ];
      policy = 'crease-limit';
    } else {
      // smooth or dart (OpenSubdiv uses the smooth limit mask for a dart)
      if (n < 3) {
        pos = [P[0], P[1], P[2]];
        policy = 'degenerate';
      } else {
        let ox = 0, oy = 0, oz = 0;
        for (let i = edgeStarts[v]; i < edgeStarts[v + 1]; i++) {
          const e = edges[edgeIds[i]];
          const o = positions[e.a === v ? e.b : e.a];
          ox += o[0]; oy += o[1]; oz += o[2];
        }
        let fx = 0, fy = 0, fz = 0;
        for (let i = faceStarts[v]; i < faceStarts[v + 1]; i++) {
          const f = faces[faceIds[i]];
          const opp = f[(faceCorner[i] + 2) % f.length];
          const o = positions[opp];
          fx += o[0]; fy += o[1]; fz += o[2];
        }
        const eW = 4 / (n * (n + 5));
        const fW = 1 / (n * (n + 5));
        const vW = 1 - n * (eW + fW);
        pos = [
          vW * P[0] + eW * ox + fW * fx,
          vW * P[1] + eW * oy + fW * fy,
          vW * P[2] + eW * oz + fW * fz,
        ];
        policy = cInfo.rule === RULE.DART ? 'dart-limit' : 'smooth-limit';
      }
    }

    results[k] = {
      index: v,
      position: pos,
      rule: cInfo.rule,
      valence: n,
      boundary: cInfo.boundary,
      policy,
      quadLevels: refine.quadLevels,
      creaseLevels: refine.creaseLevels,
      dartLevels: refine.dartLevels,
      extraLevels: refine.extraLevels,
      capped: refine.capped,
      remainingSharpness: refine.remainingSharpness,
      evaluatedLevel: evalLevel.level,
      approximate: cInfo.rule === RULE.DART && !cInfo.boundary,
    };
  }

  return {
    evaluatedLevel: evalLevel.level,
    quadLevels: refine.quadLevels,
    creaseLevels: refine.creaseLevels,
    dartLevels: refine.dartLevels,
    extraLevels: refine.extraLevels,
    dartApproximate: refine.dartApproximate,
    capped: refine.capped,
    remainingSharpness: refine.remainingSharpness,
    results,
    classification,
    evalLevel,
  };
}

function edgeOpposite(edges, positions, ei, v) {
  const e = edges[ei];
  return positions[e.a === v ? e.b : e.a];
}

function range(n) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = i;
  return a;
}
