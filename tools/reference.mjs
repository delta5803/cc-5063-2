// Independent reference implementations for acceptance tests.
//
// Nothing here imports the project's subdivision core.  These routines are a
// second, from-scratch implementation written directly from the Catmull-Clark
// definition so that a bug shared between the kernel and its own "expected"
// values cannot hide.  They are deliberately simple and slow; they are only used
// by the test harness and by the optional "reference" QA panel.
//
// Provided:
//   referenceSubdivideOnce(mesh)   one Catmull-Clark step, quad-only output
//   referenceLimit(mesh, depth)    iterate then Richardson-extrapolate a vertex
//   referenceLimitAll(mesh,depth)  same for every control vertex
//   referenceQuadGridNormals(...)  analytic normal for a regular grid patch
//   referenceBoundaryCurve(...)    1D cubic B-spline curve for an open chain

// ---------------------------------------------------------------- mesh ------

function keyOf(a, b) {
  return a < b ? `${a}:${b}` : `${b}:${a}`;
}

/**
 * One Catmull-Clark refinement step, written independently of the core.
 * Input:  { positions: [[x,y,z],...], faces: [[i,...],...] }
 * Output: same shape; every face is a quad.
 *
 * Rules (smooth interior, boundary 1D B-spline, topological corners pinned):
 *   face point   f = centroid(face)
 *   edge point   interior: (v0+v1+f0+f1)/4 ; boundary: (v0+v1)/2
 *   vertex point interior: ((n-2)P + sum(opposite) + sum(faceCentroid)) / n^2
 *                boundary: 3/4 P + 1/8 (A + B)
 *                corner:   P
 * No creases: this is the smooth surface reference.  Creased expectations are
 * derived separately from the exact masks.
 */
export function referenceSubdivideOnce(mesh) {
  const { positions, faces } = mesh;
  const V = positions.length;
  const facePoints = faces.map((f) => {
    const c = [0, 0, 0];
    for (const v of f) for (let k = 0; k < 3; k++) c[k] += positions[v][k];
    return c.map((x) => x / f.length);
  });

  const edgeMap = new Map();
  for (let fi = 0; fi < faces.length; fi++) {
    const f = faces[fi];
    for (let i = 0; i < f.length; i++) {
      const a = f[i];
      const b = f[(i + 1) % f.length];
      const key = keyOf(a, b);
      let e = edgeMap.get(key);
      if (!e) { e = { a: Math.min(a, b), b: Math.max(a, b), faces: [] }; edgeMap.set(key, e); }
      e.faces.push(fi);
    }
  }
  const edges = [...edgeMap.values()];
  const edgePoints = edges.map((e) => {
    const pa = positions[e.a];
    const pb = positions[e.b];
    if (e.faces.length < 2) {
      return [0, 1, 2].map((k) => 0.5 * (pa[k] + pb[k]));
    }
    const f0 = facePoints[e.faces[0]];
    const f1 = facePoints[e.faces[1]];
    return [0, 1, 2].map((k) => 0.25 * (pa[k] + pb[k] + f0[k] + f1[k]));
  });

  const vEdges = positions.map(() => []);
  const vFaces = positions.map(() => []);
  edges.forEach((e, ei) => { vEdges[e.a].push(ei); vEdges[e.b].push(ei); });
  faces.forEach((f, fi) => { for (const v of f) vFaces[v].push(fi); });

  const vertexPoints = positions.map((P, v) => {
    const ne = vEdges[v];
    const inc = vFaces[v];
    const boundary = ne.filter((ei) => edges[ei].faces.length < 2);
    const corner = inc.length === 1 && ne.length === 2 && boundary.length === 2;
    if (corner) return P.slice();
    if (boundary.length >= 2) {
      const nb = boundary.map((ei) => {
        const e = edges[ei];
        return positions[e.a === v ? e.b : e.a];
      });
      const A = nb[0];
      const B = nb[1];
      return [0, 1, 2].map((k) => 0.75 * P[k] + 0.125 * (A[k] + B[k]));
    }
    const n = ne.length;
    if (n < 3) return P.slice();
    let ox = 0, oy = 0, oz = 0;
    for (const ei of ne) {
      const e = edges[ei];
      const o = positions[e.a === v ? e.b : e.a];
      ox += o[0]; oy += o[1]; oz += o[2];
    }
    let fx = 0, fy = 0, fz = 0;
    for (const fi of inc) {
      const c = facePoints[fi];
      fx += c[0]; fy += c[1]; fz += c[2];
    }
    return [
      ((n - 2) / n) * P[0] + (ox + fx) / (n * n),
      ((n - 2) / n) * P[1] + (oy + fy) / (n * n),
      ((n - 2) / n) * P[2] + (oz + fz) / (n * n),
    ];
  });

  const newPositions = [...vertexPoints, ...edgePoints, ...facePoints];
  const edgeIndex = new Map();
  edges.forEach((e, ei) => edgeIndex.set(keyOf(e.a, e.b), V + ei));
  const newFaces = [];
  for (let fi = 0; fi < faces.length; fi++) {
    const f = faces[fi];
    for (let i = 0; i < f.length; i++) {
      const cur = f[i];
      const nxt = f[(i + 1) % f.length];
      const prv = f[(i - 1 + f.length) % f.length];
      newFaces.push([cur, edgeIndex.get(keyOf(cur, nxt)), V + edges.length + fi, edgeIndex.get(keyOf(prv, cur))]);
    }
  }
  return { positions: newPositions, faces: newFaces };
}

/** Indices of original vertices survive refinement: original vertex v stays at v. */
export function referenceLimit(mesh, index, depth = 8) {
  let cur = { positions: mesh.positions.map((p) => p.slice()), faces: mesh.faces.map((f) => f.slice()) };
  let last = cur.positions[index];
  const seq = [];
  for (let d = 0; d < depth; d++) {
    cur = referenceSubdivideOnce(cur);
    seq.push(cur.positions[index]);
    last = cur.positions[index];
  }
  // Richardson extrapolation on the geometric tail.  Catmull-Clark converges at
  // roughly 1/4 per level, so a single extrapolation removes most of the bias:
  //   L ~ p_k + (p_k - p_{k-1}) / (r - 1), r = |p_k-p_{k-1}| / |p_{k-1}-p_{k-2}|
  const k = seq.length;
  if (k >= 3) {
    const d2 = Math.hypot(...seq[k - 2].map((x, j) => x - seq[k - 3][j]));
    const d1 = Math.hypot(...seq[k - 1].map((x, j) => x - seq[k - 2][j]));
    if (d1 > 1e-15 && d2 > 1e-15) {
      const r = d1 / d2;
      if (r > 0 && r < 1 && Math.abs(1 - r) > 1e-6) {
        return seq[k - 1].map((x, j) => x + (x - seq[k - 2][j]) / (r - 1));
      }
    }
  }
  return last.slice();
}

export function referenceLimitAll(mesh, depth = 8) {
  let cur = { positions: mesh.positions.map((p) => p.slice()), faces: mesh.faces.map((f) => f.slice()) };
  const seq = [];
  for (let d = 0; d < depth; d++) {
    cur = referenceSubdivideOnce(cur);
    seq.push(cur.positions.slice(0, mesh.positions.length).map((p) => p.slice()));
  }
  return mesh.positions.map((_, i) => {
    const k = seq.length;
    const p1 = seq[k - 1][i];
    const p0 = seq[k - 2][i];
    if (k >= 3) {
      const p2 = seq[k - 3][i];
      const d2 = Math.hypot(...p0.map((x, j) => x - p2[j]));
      const d1 = Math.hypot(...p1.map((x, j) => x - p0[j]));
      if (d1 > 1e-15 && d2 > 1e-15) {
        const r = d1 / d2;
        if (r > 0 && r < 1 && Math.abs(1 - r) > 1e-6) {
          return p1.map((x, j) => x + (x - p0[j]) / (r - 1));
        }
      }
    }
    return p1.slice();
  });
}

// ---------------------------------------------------------------- curves ----

/**
 * 1D cubic B-spline limit curve of an open polyline with pinned endpoints.
 * This is the exact limit of the boundary / crease rule (0.125, 0.75, 0.125),
 * computed by iterating the refinement on the curve alone.
 */
export function referenceBoundaryCurve(vertices, { pinned = [], depth = 12 } = {}) {
  let pts = vertices.map((p) => p.slice());
  let pinnedSet = new Set(pinned);
  for (let d = 0; d < depth; d++) {
    const n = pts.length;
    const next = [];
    const nextPinned = [];
    for (let i = 0; i < n; i++) {
      const p = pts[i];
      if (pinnedSet.has(i)) {
        next.push(p.slice());
      } else {
        const a = pts[Math.max(0, i - 1)];
        const b = pts[Math.min(n - 1, i + 1)];
        next.push([0, 1, 2].map((k) => 0.0625 * a[k] + 0.75 * p[k] + 0.0625 * b[k]));
      }
      nextPinned.push(pinnedSet.has(i));
      if (i + 1 < n) {
        next.push([0, 1, 2].map((k) => 0.5 * (p[k] + pts[i + 1][k])));
        nextPinned.push(false);
      }
    }
    pts = next;
    pinnedSet = new Set(nextPinned.map((v, i) => (v ? i : -1)).filter((i) => i >= 0));
  }
  return pts;
}

// ---------------------------------------------------------------- normal -----

/**
 * Analytic limit normal of a regular (valence-4, no crease) grid vertex.
 *
 * For a tensor-product cubic B-spline the limit surface's tangent frame at a
 * grid vertex is spanned by the two 1D limit tangents, which for uniform spacing
 * are proportional to (p[i+1,j]-p[i-1,j]) and (p[i,j+1]-p[i,j-1]).  The unit
 * normal is their cross product.  That is an independent closed form, unlike the
 * averaged face normals the viewer uses.
 */
export function referenceRegularGridNormal(grid, i, j, stride) {
  const at = (di, dj) => grid[(i + di) + (j + dj) * stride];
  const du = [0, 1, 2].map((k) => at(1, 0)[k] - at(-1, 0)[k]);
  const dv = [0, 1, 2].map((k) => at(0, 1)[k] - at(0, -1)[k]);
  const n = [
    du[1] * dv[2] - du[2] * dv[1],
    du[2] * dv[0] - du[0] * dv[2],
    du[0] * dv[1] - du[1] * dv[0],
  ];
  const l = Math.hypot(n[0], n[1], n[2]);
  return l > 0 ? n.map((x) => x / l) : [0, 0, 1];
}
