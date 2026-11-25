// Crease / boundary / corner semantics.
//
// We follow the Catmull-Clark crease scheme as implemented by OpenSubdiv's
// "Catmark" scheme:
//
//   * Every boundary edge is treated as infinitely sharp.
//   * A sharp edge carries a sharpness in [0, inf].  Finite sharpness decays by
//     one per subdivision level (semi-sharp crease); infinite sharpness persists.
//   * The vertex rule is decided from the sharpness of its incident edges:
//         sharp edges >= 3            -> corner (pinned)
//         sharp edges == 2            -> crease (1D B-spline rule along the crease)
//         sharp edges == 1            -> dart   (smooth-like, but the incident
//                                                sharp edge stays sharp)
//         sharp edges == 0            -> smooth (Catmull-Clark)
//   * A vertex marked "corner" (by the user) is pinned regardless.
//   * With boundary interpolation "edge and corner", boundary corner vertices
//     (exactly one face and two boundary edges) are additionally pinned.
//
// For rendering of the limit surface we evaluate the standard limit masks at a
// chosen level.  Semi-sharp creases are handled by tracking sharpness through
// each subdivision so that finite creases fade out over levels, which is what
// makes a level-by-level comparison meaningful.

import { edgeKey } from './topology.js';

export const RULE = Object.freeze({
  SMOOTH: 'smooth',
  DART: 'dart',
  CREASE: 'crease',
  CORNER: 'corner',
});

export const SHARPNESS_INFINITE = Infinity;

/**
 * Build a per-edge sharpness lookup from a user-supplied description.
 * edges: array of { a, b, sharpness } (either order of a,b is fine).
 */
export function buildCreaseMap(edgeCreases, positions, options = {}) {
  const map = new Map();
  if (!edgeCreases) return map;
  const tol = options.tolerance != null ? options.tolerance : 1e-7;
  for (const c of edgeCreases) {
    if (!Number.isFinite(c.sharpness) && c.sharpness !== SHARPNESS_INFINITE) continue;
    if (!(c.sharpness > 0)) continue;
    map.set(edgeKey(c.a, c.b), c.sharpness);
  }
  return map;
}

export function normalizeCorners(corners) {
  const s = new Set();
  if (corners) for (const v of corners) s.add(v);
  return s;
}

/**
 * Normalise a numbered crease-chain description.
 *
 * A chain is `{ id, edges: [[a,b], ...] }`; the order of `edges` is the numbering
 * (segment 0 first).  Chains exist so a regression fixture can say "this is
 * engraved segment #k" and a stroke tracer can prove each numbered segment
 * survived refinement exactly once.  They do not create sharpness on their own:
 * the same edges must also appear in `creases`, so there is one source of truth
 * for how sharp an edge is.
 *
 * Returns the tag lookup used by the subdivision core to carry provenance to
 * child edges.
 */
export function normalizeCreaseChains(creaseChains) {
  const chains = [];
  const tags = new Map();
  if (!creaseChains) return { chains, tags };
  for (let ci = 0; ci < creaseChains.length; ci++) {
    const c = creaseChains[ci];
    const id = c.id != null ? c.id : `chain-${ci}`;
    if (tagsHasChain(tags, id)) throw new Error(`Duplicate crease chain id "${id}"`);
    if (!Array.isArray(c.edges) || c.edges.length === 0) {
      throw new Error(`Crease chain "${id}" has no edges`);
    }
    const edges = [];
    for (let i = 0; i < c.edges.length; i++) {
      const e = c.edges[i];
      if (!Array.isArray(e) || e.length !== 2 ||
        !Number.isInteger(e[0]) || !Number.isInteger(e[1]) || e[0] === e[1]) {
        throw new Error(`Crease chain "${id}" segment ${i} is not a vertex pair`);
      }
      const key = edgeKey(e[0], e[1]);
      if (tags.has(key)) {
        throw new Error(`Crease chain "${id}" reuses edge ${key}`);
      }
      // Keep the author's order: segment i runs from a to b, and b is shared with
      // segment i+1.  Stroke tracing relies on this to stay direction-consistent
      // across levels; deriving direction from the min/max edge key instead would
      // flip alternating segments.
      tags.set(key, { chainId: id, index: i, a: e[0], b: e[1] });
      edges.push([e[0], e[1]]);
    }
    chains.push({ id, edges });
  }
  return { chains, tags };
}

function tagsHasChain(tags, id) {
  for (const t of tags.values()) if (t.chainId === id) return true;
  return false;
}

/**
 * Classify every vertex of a topology given explicit creases/corners.
 * Returns an array indexed by vertex id.
 *
 * The returned object records:
 *   rule          one of RULE.*
 *   sharpEdges    indices of incident sharp/boundary edges (in local edge order)
 *   sharpCount
 *   boundary      whether any incident edge is a boundary edge
 *   boundaryCount
 *   isTopologicalCorner
 */
export function classifyVertices(topology, creaseMap, cornerSet, options = {}) {
  const {
    edges, vertexEdges, vertexEdgeStart, numVertices, vertexInfo,
  } = topology;
  const pinBoundaryCorners = options.pinBoundaryCorners !== false;
  const out = new Array(numVertices);

  for (let v = 0; v < numVertices; v++) {
    const start = vertexEdgeStart[v];
    const end = vertexEdgeStart[v + 1];
    const sharpEdges = [];
    let boundaryCount = 0;
    for (let i = start; i < end; i++) {
      const e = edges[vertexEdges[i]];
      const isBoundary = e.faces.length < 2;
      const s = isBoundary ? SHARPNESS_INFINITE : (creaseMap.get(e.key) || 0);
      if (isBoundary) boundaryCount++;
      if (s > 0) sharpEdges.push(vertexEdges[i]);
    }
    const sharpCount = sharpEdges.length;
    const info = vertexInfo ? vertexInfo[v] : null;
    const topoCorner = !!(info && info.topologicalCorner);
    const pinned = cornerSet.has(v) ||
      (pinBoundaryCorners && topoCorner) ||
      sharpCount >= 3;

    let rule;
    if (pinned) rule = RULE.CORNER;
    else if (sharpCount === 2) rule = RULE.CREASE;
    else if (sharpCount === 1) rule = RULE.DART;
    else rule = RULE.SMOOTH;

    out[v] = {
      vertex: v,
      rule,
      sharpEdges,
      sharpCount,
      boundary: boundaryCount > 0,
      boundaryCount,
      isTopologicalCorner: topoCorner,
      valence: end - start,
      pinned,
    };
  }
  return out;
}

/**
 * Compute child sharpness for an edge at the next level.  Finite sharpness
 * decays by one per level; infinite stays infinite.
 */
export function decaySharpness(s) {
  if (!(s > 0)) return 0;
  if (!Number.isFinite(s)) return SHARPNESS_INFINITE;
  return s > 1 ? s - 1 : 0;
}

/**
 * Semi-sharp blending weight for a vertex's subdivision mask.
 *
 * When a vertex transitions from a sharp rule (crease/corner) to a smooth rule
 * between parent and child, the parent mask is blended with the smooth mask.
 * We use the same fractional weight as OpenSubdiv: the average of the parent
 * edge sharpness values that decay to zero (clamped to [0,1]).
 */
export function transitionWeight(parentSharpness, childSharpness) {
  let count = 0;
  let sum = 0;
  for (let i = 0; i < parentSharpness.length; i++) {
    const p = parentSharpness[i];
    const c = childSharpness[i];
    if (!(p > 0)) continue;
    // Infinite sharpness never decays.
    if (!Number.isFinite(p)) continue;
    if (!(c > 0) || !Number.isFinite(c)) {
      if (p <= 1) {
        sum += p;
        count++;
      }
    }
  }
  if (count === 0) return 0;
  return Math.min(1, sum / count);
}
