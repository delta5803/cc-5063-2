// Mesh topology: validates a polygon control cage and builds real vertex / edge /
// face adjacency.  Subdivision operates on this adjacency rather than treating each
// face in isolation, so shared edges receive one unambiguous child vertex.
//
// The representation is intentionally plain (arrays of indices) so that the same
// data can be serialised, tested in Node, and consumed by the viewer.

import { vdist } from './vec3.js';

export function edgeKey(a, b) {
  return a < b ? a + ':' + b : b + ':' + a;
}

/**
 * Validate and index a polygon mesh.
 *
 * @param {Array<[number,number,number]>} positions
 * @param {Array<number[]>} faces  polygons, CCW when viewed from outside
 * @param {object} [options]
 * @param {boolean} [options.allowBoundary=true]
 * @returns {MeshTopology}
 */
export function buildTopology(positions, faces, options = {}) {
  const allowBoundary = options.allowBoundary !== false;
  const nVerts = positions.length;

  if (!Array.isArray(positions) || !Array.isArray(faces)) {
    throw new Error('buildTopology expects positions[] and faces[]');
  }
  if (faces.length === 0) {
    throw new Error('Mesh has no faces');
  }

  // ---- per-face checks -----------------------------------------------------
  const usedVertex = new Uint8Array(nVerts);
  const faceStart = new Int32Array(faces.length + 1);
  let cornerCount = 0;
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    faceStart[f] = cornerCount;
    if (!Array.isArray(face) || face.length < 3) {
      const err = new Error(`Face ${f} has ${face ? face.length : 0} corners; need at least 3`);
      err.code = 'DEGENERATE_FACE';
      err.details = [{ face: f, reason: 'too few corners' }];
      throw err;
    }
    const seen = new Set();
    for (let i = 0; i < face.length; i++) {
      const v = face[i];
      if (!Number.isInteger(v) || v < 0 || v >= nVerts) {
        const err = new Error(`Face ${f} references invalid vertex index ${v}`);
        err.code = 'INVALID_INDEX';
        err.details = [{ face: f, vertex: v }];
        throw err;
      }
      if (seen.has(v)) {
        const err = new Error(`Face ${f} is degenerate: vertex ${v} is repeated`);
        err.code = 'DEGENERATE_FACE';
        err.details = [{ face: f, reason: 'repeated vertex', vertex: v }];
        throw err;
      }
      seen.add(v);
      usedVertex[v] = 1;
    }
  }

  // Rebuild corner table correctly (the loop above only validated).
  cornerCount = 0;
  for (let f = 0; f < faces.length; f++) {
    faceStart[f] = cornerCount;
    cornerCount += faces[f].length;
  }
  faceStart[faces.length] = cornerCount;

  const faceCorners = new Int32Array(cornerCount);
  {
    let k = 0;
    for (let f = 0; f < faces.length; f++) {
      const face = faces[f];
      for (let i = 0; i < face.length; i++) faceCorners[k++] = face[i];
    }
  }

  // ---- edge adjacency ------------------------------------------------------
  // key -> { a, b, faces: Int32[], corners: Int32[] }
  const edgeMap = new Map();
  const faceEdges = new Int32Array(cornerCount);       // per corner: edge id
  const faceEdgeStart = faceStart;

  for (let f = 0; f < faces.length; f++) {
    const base = faceStart[f];
    const n = faces[f].length;
    for (let i = 0; i < n; i++) {
      const a = faceCorners[base + i];
      const b = faceCorners[base + ((i + 1) % n)];
      if (a === b) {
        throw new Error(`Face ${f} has a degenerate edge ${a}->${b}`);
      }
      const key = edgeKey(a, b);
      let e = edgeMap.get(key);
      if (!e) {
        e = { key, a: Math.min(a, b), b: Math.max(a, b), faces: [], corners: [] };
        edgeMap.set(key, e);
      }
      e.faces.push(f);
      e.corners.push(base + i);
      faceEdges[base + i] = -1; // assigned after map is final
    }
  }

  // ---- duplicate faces -----------------------------------------------------
  // Two faces with the same vertex set create a "pillow" with coincident
  // geometry; it is not a useful manifold input and silently repairing it would
  // change the topology, so reject it and point at both faces.
  const faceSignatures = new Map();
  const duplicateFaces = [];
  for (let f = 0; f < faces.length; f++) {
    const sig = faces[f].slice().sort((a, b) => a - b).join(',');
    const prev = faceSignatures.get(sig);
    if (prev !== undefined) duplicateFaces.push({ face: f, duplicateOf: prev });
    else faceSignatures.set(sig, f);
  }
  if (duplicateFaces.length) {
    const err = new Error(
      `Duplicate face(s): ${duplicateFaces.length} face(s) share all vertices with another face ` +
      `(e.g. face ${duplicateFaces[0].face} duplicates face ${duplicateFaces[0].duplicateOf})`
    );
    err.code = 'DUPLICATE_FACE';
    err.details = duplicateFaces;
    throw err;
  }

  // ---- manifold and degeneracy checks -------------------------------------
  const edges = [];
  const edgeIndexByKey = new Map();
  const nonManifoldEdges = [];
  let boundaryEdges = 0;

  for (const e of edgeMap.values()) {
    if (e.faces.length > 2) {
      nonManifoldEdges.push({ key: e.key, a: e.a, b: e.b, faceCount: e.faces.length, faces: e.faces.slice() });
      continue;
    }
    if (e.faces.length === 1) {
      boundaryEdges++;
      if (!allowBoundary) {
        throw new Error(`Boundary edge ${e.a}-${e.b} found but boundaries are disallowed`);
      }
    }
    const id = edges.length;
    edgeIndexByKey.set(e.key, id);
    edges.push({ id, key: e.key, a: e.a, b: e.b, faces: e.faces.slice(), corners: e.corners.slice() });
  }

  if (nonManifoldEdges.length) {
    const summary = nonManifoldEdges
      .slice(0, 8)
      .map((e) => `${e.a}-${e.b} shared by ${e.faceCount} faces`)
      .join(', ');
    const err = new Error(
      `Non-manifold input: ${nonManifoldEdges.length} edge(s) with more than two incident faces ` +
      `(${summary}${nonManifoldEdges.length > 8 ? ', ...' : ''})`
    );
    err.code = 'NON_MANIFOLD_EDGE';
    err.details = nonManifoldEdges;
    throw err;
  }

  // Fill faceEdges now that edges have stable ids.
  for (let f = 0; f < faces.length; f++) {
    const base = faceStart[f];
    const n = faces[f].length;
    for (let i = 0; i < n; i++) {
      const a = faceCorners[base + i];
      const b = faceCorners[base + ((i + 1) % n)];
      faceEdges[base + i] = edgeIndexByKey.get(edgeKey(a, b));
    }
  }

  // ---- per-vertex incidence ------------------------------------------------
  const vertexEdgeCount = new Int32Array(nVerts);
  for (const e of edges) {
    vertexEdgeCount[e.a]++;
    vertexEdgeCount[e.b]++;
  }
  const vertexEdgeStart = new Int32Array(nVerts + 1);
  for (let v = 0; v < nVerts; v++) vertexEdgeStart[v + 1] = vertexEdgeStart[v] + vertexEdgeCount[v];
  const vertexEdges = new Int32Array(vertexEdgeStart[nVerts]);
  {
    const cursor = vertexEdgeStart.slice(0, nVerts);
    for (const e of edges) {
      vertexEdges[cursor[e.a]++] = e.id;
      vertexEdges[cursor[e.b]++] = e.id;
    }
  }

  const vertexFaceCount = new Int32Array(nVerts);
  for (let f = 0; f < faces.length; f++) {
    for (const v of faces[f]) vertexFaceCount[v]++;
  }
  const vertexFaceStart = new Int32Array(nVerts + 1);
  for (let v = 0; v < nVerts; v++) vertexFaceStart[v + 1] = vertexFaceStart[v] + vertexFaceCount[v];
  const vertexFaces = new Int32Array(vertexFaceStart[nVerts]);
  {
    const cursor = vertexFaceStart.slice(0, nVerts);
    for (let f = 0; f < faces.length; f++) {
      for (const v of faces[f]) vertexFaces[cursor[v]++] = f;
    }
  }

  // Local corner index of each vertex within each incident face (for facet
  // opposite-vertex lookup in limit masks).
  const vertexFaceCorner = new Int32Array(vertexFaceStart[nVerts]);
  {
    const cursor = vertexFaceStart.slice(0, nVerts);
    for (let f = 0; f < faces.length; f++) {
      const base = faceStart[f];
      const n = faces[f].length;
      for (let i = 0; i < n; i++) {
        const v = faceCorners[base + i];
        vertexFaceCorner[cursor[v]++] = i;
      }
    }
  }

  // ---- degeneracy: zero-area faces ----------------------------------------
  const degenerateFaces = [];
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    const area = faceArea(positions, face);
    const minEdge = minFaceEdgeLength(positions, face);
    if (!(area > 1e-12) || !(minEdge > 1e-12)) {
      degenerateFaces.push({ face: f, area, minEdge });
    }
  }
  if (degenerateFaces.length) {
    const worst = degenerateFaces[0];
    const err = new Error(
      `Degenerate face(s): ${degenerateFaces.length} face(s) with near-zero area or edge length ` +
      `(first: face ${worst.face}, area ${worst.area.toExponential(2)}, min edge ${worst.minEdge.toExponential(2)})`
    );
    err.code = 'DEGENERATE_FACE';
    err.details = degenerateFaces;
    throw err;
  }
  // ---- isolated / unused vertices -----------------------------------------
  const unusedVertices = [];
  for (let v = 0; v < nVerts; v++) {
    if (!usedVertex[v]) unusedVertices.push(v);
  }

  // ---- vertex manifoldness -------------------------------------------------
  // A manifold vertex must have a single closed (interior) or open (boundary)
  // fan.  Detect disconnected fans and bow-tie vertices.
  const vertexInfo = new Array(nVerts);
  for (let v = 0; v < nVerts; v++) {
    vertexInfo[v] = classifyVertexFan(v, {
      vertexEdges, vertexEdgeStart, vertexFaces, vertexFaceStart,
      vertexFaceCorner, edges, faces, faceStart, faceCorners,
    });
  }
  const nonManifoldVertices = vertexInfo.filter((i) => i.nonManifold);

  if (nonManifoldVertices.length) {
    const sample = nonManifoldVertices
      .slice(0, 8)
      .map((i) => `v${i.vertex} (${i.boundaryEdgeCount} boundary edges, ${i.faceCount} faces, ${i.fanCount} fans)`)
      .join(', ');
    const err = new Error(
      `Non-manifold vertex/vertices: ${nonManifoldVertices.length} vertex(es) with disconnected face fans ` +
      `(${sample}${nonManifoldVertices.length > 8 ? ', ...' : ''})`
    );
    err.code = 'NON_MANIFOLD_VERTEX';
    err.details = nonManifoldVertices;
    throw err;
  }

  // ---- curvature scale for thin-face heuristics ---------------------------
  let totalArea = 0;
  let maxEdge = 0;
  for (let f = 0; f < faces.length; f++) totalArea += faceArea(positions, faces[f]);
  for (const e of edges) maxEdge = Math.max(maxEdge, vdist(positions[e.a], positions[e.b]));

  return {
    positions,
    faces,
    numVertices: nVerts,
    numFaces: faces.length,
    numEdges: edges.length,
    cornerCount,
    faceStart,
    faceCorners,
    faceEdges,
    edges,
    edgeIndexByKey,
    vertexEdgeStart,
    vertexEdges,
    vertexFaceStart,
    vertexFaces,
    vertexFaceCorner,
    vertexInfo,
    boundaryEdgeCount: boundaryEdges,
    totalArea,
    maxEdge,
    unusedVertexCount: unusedVertices.length,
    unusedVertices,
  };
}

export function faceArea(positions, face) {
  // Fan triangulation area; robust enough for validity checks on planar-ish quads.
  if (face.length < 3) return 0;
  const a = positions[face[0]];
  let area = 0;
  for (let i = 1; i < face.length - 1; i++) {
    const b = positions[face[i]];
    const c = positions[face[i + 1]];
    const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
    const nx = aby * acz - abz * acy;
    const ny = abz * acx - abx * acz;
    const nz = abx * acy - aby * acx;
    area += 0.5 * Math.hypot(nx, ny, nz);
  }
  return area;
}

function minFaceEdgeLength(positions, face) {
  let m = Infinity;
  for (let i = 0; i < face.length; i++) {
    m = Math.min(m, vdist(positions[face[i]], positions[face[(i + 1) % face.length]]));
  }
  return m;
}

function classifyVertexFan(v, ctx) {
  const {
    vertexEdges, vertexEdgeStart, vertexFaces, vertexFaceStart,
    edges, faces, faceStart, faceCorners,
  } = ctx;

  const e0 = vertexEdgeStart[v];
  const e1 = vertexEdgeStart[v + 1];
  const f0 = vertexFaceStart[v];
  const f1 = vertexFaceStart[v + 1];
  const edgeCount = e1 - e0;
  const faceCount = f1 - f0;

  if (edgeCount === 0) {
    return { vertex: v, edgeCount, faceCount, boundaryEdgeCount: 0, fanCount: 0,
      nonManifold: false, isolated: true, boundary: false };
  }

  const boundaryEdgeCount = countBoundaryAtVertex(v, vertexEdges, vertexEdgeStart, edges);

  // Build corner-adjacency graph over incident faces: two faces touch if they
  // share an incident edge of v.
  const incidentFaces = [];
  for (let i = f0; i < f1; i++) incidentFaces.push(vertexFaces[i]);
  if (incidentFaces.length === 0) {
    return { vertex: v, edgeCount, faceCount, boundaryEdgeCount, fanCount: 0,
      nonManifold: false, isolated: false, boundary: boundaryEdgeCount > 0, dangling: true };
  }

  // Union-find over incident faces.
  const localOf = new Map();
  for (let i = 0; i < incidentFaces.length; i++) localOf.set(incidentFaces[i], i);
  const parent = new Int32Array(incidentFaces.length);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };

  for (let ei = e0; ei < e1; ei++) {
    const e = edges[vertexEdges[ei]];
    if (e.faces.length === 2) {
      const l0 = localOf.get(e.faces[0]);
      const l1 = localOf.get(e.faces[1]);
      if (l0 !== undefined && l1 !== undefined) union(l0, l1);
    }
  }
  let fanCount = 0;
  for (let i = 0; i < incidentFaces.length; i++) if (find(i) === i) fanCount++;

  // Topological corner: exactly one face and two boundary edges meeting there.
  let topologicalCorner = false;
  if (faceCount === 1 && edgeCount === 2 && boundaryEdgeCount === 2) {
    topologicalCorner = true;
  }

  // For a manifold interior vertex, #faces == #edges.  On the boundary,
  // #faces == #edges - 1.  A single fan must satisfy this.
  let nonManifold = false;
  if (fanCount > 1) nonManifold = true;
  if (!topologicalCorner) {
    if (boundaryEdgeCount === 0 && faceCount !== edgeCount) nonManifold = true;
    if (boundaryEdgeCount > 0 && faceCount !== edgeCount - 1) {
      // A vertex with a boundary can legitimately have dangling edges if the
      // input is malformed; flag it.
      nonManifold = true;
    }
  }

  return {
    vertex: v,
    edgeCount,
    faceCount,
    boundaryEdgeCount,
    fanCount,
    boundary: boundaryEdgeCount > 0,
    topologicalCorner,
    nonManifold,
  };
}

export function countBoundaryAtVertex(v, vertexEdges, vertexEdgeStart, edges) {
  let c = 0;
  for (let i = vertexEdgeStart[v]; i < vertexEdgeStart[v + 1]; i++) {
    if (edges[vertexEdges[i]].faces.length === 1) c++;
  }
  return c;
}

/**
 * Count valence and boundary status quickly (used by the crease classifier).
 */
export function vertexValence(topology, v) {
  return topology.vertexEdgeStart[v + 1] - topology.vertexEdgeStart[v];
}

/**
 * Convert flat triplet positions to an array of arrays.
 */
export function positionsToArrays(flat) {
  const out = new Array(flat.length / 3);
  for (let i = 0; i < out.length; i++) {
    out[i] = [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]];
  }
  return out;
}

export function positionsFromArrays(arrs) {
  const flat = new Float64Array(arrs.length * 3);
  for (let i = 0; i < arrs.length; i++) {
    flat[i * 3] = arrs[i][0];
    flat[i * 3 + 1] = arrs[i][1];
    flat[i * 3 + 2] = arrs[i][2];
  }
  return flat;
}
