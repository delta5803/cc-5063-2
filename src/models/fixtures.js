// Test control meshes.  Every model is a plain { name, positions, faces,
// creases, corners } object so it can be fed to the core directly.
//
// Included on purpose:
//   * a regular quad grid (the easy case, but not the only acceptance test)
//   * an open sheet with an L-shaped boundary (convex and concave boundary corners)
//   * a pentagonal pyramid with uneven radii (one valence-5 and five valence-3
//     extraordinary vertices, no boundary)
//   * a lumpy "quad ball" (cube with 2x2 faces: eight valence-3 corners)
//   * an open cylinder (boundary rims with valence-3 vertices)
//   * a deformed torus (closed genus-1 surface, all regular interior vertices)
//   * a concave tray with wildly uneven edge lengths (hard case that a sphere or
//     uniform grid would hide)
//   * creased variants (semi-sharp and infinitely sharp), including a crease
//     ending at an extraordinary vertex
//   * malformed inputs (non-manifold edge, non-manifold vertex, degenerate face)
//     that must be located and rejected, never silently repaired.

// ---------------------------------------------------------------- helpers ----

function gridPositions(nx, ny, spacing) {
  const positions = [];
  const index = [];
  for (let j = 0; j < ny; j++) {
    index[j] = [];
    for (let i = 0; i < nx; i++) {
      index[j][i] = positions.length;
      positions.push([(i - (nx - 1) / 2) * spacing, (j - (ny - 1) / 2) * spacing, 0]);
    }
  }
  return { positions, index };
}

function gridFaces(index) {
  const ny = index.length;
  const nx = index[0].length;
  const faces = [];
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      faces.push([index[j][i], index[j][i + 1], index[j + 1][i + 1], index[j + 1][i]]);
    }
  }
  return faces;
}

function edgeKey(a, b) {
  return a < b ? a + ':' + b : b + ':' + a;
}

// ---------------------------------------------------------------- models -----

export function regularGrid(n = 7, spacing = 0.8) {
  const { positions, index } = gridPositions(n, n, spacing);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const [x, y] = positions[index[j][i]];
      positions[index[j][i]][2] =
        0.16 * (x * x - y * y) + 0.08 * Math.sin(2 * x) * Math.cos(1.5 * y);
    }
  }
  return {
    name: 'regular-grid',
    description: '7x7 quad grid, all interior valences 4',
    positions,
    faces: gridFaces(index),
    creases: [],
    corners: [],
  };
}

export function openBoundarySheet() {
  // L-shaped outline: a 6x6 grid with the top-right 2x2 block removed.  This has
  // four convex boundary corners and one concave boundary corner.
  const n = 6;
  const spacing = 0.7;
  const { positions, index } = gridPositions(n, n, spacing);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const [x, y] = positions[index[j][i]];
      positions[index[j][i]][2] =
        0.1 * Math.sin(1.4 * x) * Math.cos(1.1 * y) + 0.05 * x;
    }
  }
  const faces = [];
  for (let j = 0; j < n - 1; j++) {
    for (let i = 0; i < n - 1; i++) {
      if (i >= 4 && j >= 4) continue;
      faces.push([index[j][i], index[j][i + 1], index[j + 1][i + 1], index[j + 1][i]]);
    }
  }
  return {
    name: 'open-boundary-sheet',
    description: 'L-shaped open sheet: boundary corners, concave notch',
    positions,
    faces,
    creases: [],
    corners: [],
  };
}

export function pentagonPyramid() {
  // Apex (valence 5) + five base ring vertices (valence 3) + one pentagon base.
  // Ring radii are uneven so the surface is not a symmetric "easy" case, but the
  // ring stays convex (radii within a narrow band, monotonically increasing
  // angles) so the cage does not self-fold.
  const n = 5;
  const positions = [];
  const apex = positions.length;
  positions.push([0.05, -0.04, 1.25]);
  const ring = [];
  const radii = [1.0, 0.86, 1.12, 0.93, 1.06];
  const heights = [0.1, -0.06, 0.14, 0.02, -0.05];
  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2 + 0.06 * Math.sin(i * 2.3);
    ring.push(positions.length);
    positions.push([radii[i] * Math.cos(a), radii[i] * Math.sin(a), heights[i]]);
  }
  const faces = [];
  for (let i = 0; i < n; i++) {
    faces.push([apex, ring[i], ring[(i + 1) % n]]);
  }
  faces.push([ring[4], ring[3], ring[2], ring[1], ring[0]]);
  return {
    name: 'pentagon-pyramid',
    description: 'pentagonal pyramid: apex valence 5, base ring valence 3, uneven',
    positions,
    faces,
    creases: [],
    corners: [],
  };
}

export function quadBall(res = 2) {
  // Cube whose six faces are each res x res quads, stitched by shared vertices.
  // Displaced to a lumpy rounded cube: eight valence-3 corners, 24 boundary-of-
  // face vertices with valence 4, and strongly uneven edge lengths.
  const positions = [];
  const keyMap = new Map();
  const get = (p) => {
    const key = p.map((v) => v.toFixed(6)).join(',');
    let idx = keyMap.get(key);
    if (idx === undefined) {
      idx = positions.length;
      keyMap.set(key, idx);
      positions.push(p.slice());
    }
    return idx;
  };

  const A = [-1, -1, -1], B = [1, -1, -1], C = [1, 1, -1], D = [-1, 1, -1];
  const E = [-1, -1, 1], F = [1, -1, 1], G = [1, 1, 1], H = [-1, 1, 1];

  // each face given by CCW corner loop p00 -> p10 -> p11 -> p01
  const faceDefs = [
    [A, D, C, B], // z = -1
    [E, F, G, H], // z = +1
    [A, B, F, E], // y = -1
    [D, H, G, C], // y = +1
    [A, E, H, D], // x = -1
    [B, C, G, F], // x = +1
  ];

  const faces = [];
  for (const [p00, p10, p11, p01] of faceDefs) {
    const grid = [];
    for (let v = 0; v <= res; v++) {
      // non-uniform parameterisation along each face
      const tv = v / res + 0.06 * Math.sin(Math.PI * (v / res));
      grid[v] = [];
      for (let u = 0; u <= res; u++) {
        const tu = u / res + 0.06 * Math.sin(Math.PI * (u / res));
        const x = p00[0] + (p10[0] - p00[0]) * tu + (p01[0] - p00[0]) * tv;
        const y = p00[1] + (p10[1] - p00[1]) * tu + (p01[1] - p00[1]) * tv;
        const z = p00[2] + (p10[2] - p00[2]) * tu + (p01[2] - p00[2]) * tv;
        // rounded-cube displacement: pull each point toward a superellipsoid and
        // add a low-frequency lump so the surface is not a perfect sphere
        const r = Math.pow(
          Math.pow(Math.abs(x), 4) + Math.pow(Math.abs(y), 4) + Math.pow(Math.abs(z), 4),
          0.25
        );
        const lump = 1 + 0.09 * Math.sin(2.7 * x + 0.4) * Math.cos(2.2 * y - 0.3) +
          0.06 * Math.sin(1.9 * z + 1.1);
        const sx = (x / r) * lump, sy = (y / r) * lump, sz = (z / r) * lump;
        grid[v][u] = get([sx, sy, sz]);
      }
    }
    for (let v = 0; v < res; v++) {
      for (let u = 0; u < res; u++) {
        faces.push([grid[v][u], grid[v][u + 1], grid[v + 1][u + 1], grid[v + 1][u]]);
      }
    }
  }
  return {
    name: 'quad-ball',
    description: `cube with ${res}x${res} quads per face: eight valence-3 corners`,
    positions,
    faces,
    creases: [],
    corners: [],
  };
}

export function openCylinder(m = 7) {
  // Two rings of quads with open top and bottom boundaries.  Middle ring vertices
  // have valence 4; rim vertices have valence 3 and lie on the boundary.
  const positions = [];
  const rings = [];
  const heights = [-1.0, 0.0, 1.0];
  for (let k = 0; k < 3; k++) {
    const ring = [];
    for (let i = 0; i < m; i++) {
      const a = (i / m) * Math.PI * 2 + 0.15 * Math.sin(i * 1.7);
      const r = 1.0 + (k === 1 ? 0.15 : 0) + 0.07 * Math.cos(i * 2.1 + k);
      ring.push(positions.length);
      positions.push([r * Math.cos(a), r * Math.sin(a), heights[k] + 0.1 * Math.sin(2 * a)]);
    }
    rings.push(ring);
  }
  const faces = [];
  for (let k = 0; k < 2; k++) {
    for (let i = 0; i < m; i++) {
      const i1 = (i + 1) % m;
      // outward winding
      faces.push([rings[k][i], rings[k][i1], rings[k + 1][i1], rings[k + 1][i]]);
    }
  }
  return {
    name: 'open-cylinder',
    description: 'open quad cylinder: boundary rims, valence-3 rim vertices',
    positions,
    faces,
    creases: [],
    corners: [],
  };
}

export function torus(m = 10, n = 6) {
  // Deformed torus (closed genus-1).  Interior vertices are all valence 4.
  const positions = [];
  const index = [];
  for (let i = 0; i < m; i++) {
    index[i] = [];
    const a = (i / m) * Math.PI * 2 + 0.12 * Math.sin(i * 2.7);
    const R = 2.0 + 0.18 * Math.sin(3 * a);
    for (let j = 0; j < n; j++) {
      const b = (j / n) * Math.PI * 2 + 0.1 * Math.cos(i * 1.3);
      const r = 0.62 + 0.08 * Math.cos(2 * a + b);
      const x = (R + r * Math.cos(b)) * Math.cos(a);
      const y = (R + r * Math.cos(b)) * Math.sin(a);
      const z = r * Math.sin(b) * 1.15 + 0.06 * Math.sin(3 * a);
      index[i][j] = positions.length;
      positions.push([x, y, z]);
    }
  }
  const faces = [];
  for (let i = 0; i < m; i++) {
    const i1 = (i + 1) % m;
    for (let j = 0; j < n; j++) {
      const j1 = (j + 1) % n;
      faces.push([index[i][j], index[i1][j], index[i1][j1], index[i][j1]]);
    }
  }
  return {
    name: 'torus',
    description: 'deformed closed torus (genus 1), all regular interior valences',
    positions,
    faces,
    creases: [],
    corners: [],
  };
}

export function concaveUneven() {
  // Concave "tray": rectangle pressed down in the middle, with wildly uneven
  // spacing (dense near the dip, sparse at the rim).  Uniform grids and spheres
  // hide seams; uneven concave geometry exposes them.
  const xs = [-3.2, -2.7, -1.1, -0.35, 0.1, 0.5, 1.4, 2.9, 3.3];
  const ys = [-2.8, -2.4, -0.9, -0.2, 0.3, 1.2, 2.5, 3.0];
  const positions = [];
  const index = [];
  for (let j = 0; j < ys.length; j++) {
    index[j] = [];
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      const y = ys[j];
      const r2 = x * x + y * y;
      const z = -0.9 * Math.exp(-0.55 * r2) + 0.06 * x * y + 0.04 * Math.sin(1.3 * x);
      index[j][i] = positions.length;
      positions.push([x, y, z]);
    }
  }
  return {
    name: 'concave-uneven',
    description: 'concave tray with strongly uneven edge lengths (anti-example)',
    positions,
    faces: gridFaces(index),
    creases: [],
    corners: [],
  };
}

export function creasedCube() {
  // Box with all 12 edges semi-sharp (sharpness 2).  After two levels the
  // sharpness has decayed and the limit is the smooth box; the cube corners are
  // extraordinary vertices adjacent to creases, which exercises the policy.
  const positions = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ];
  const faces = [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
  ];
  const edges = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  return {
    name: 'creased-cube',
    description: 'cube, all 12 edges semi-sharp (sharpness 2)',
    positions,
    faces,
    creases: edges.map(([a, b]) => ({ a, b, sharpness: 2 })),
    corners: [],
  };
}

export function sharpCornerBox() {
  // Box with the three edges meeting at vertex 6 (the [1,1,1] corner) infinitely
  // sharp.  Vertex 6 then has 3 sharp edges -> corner rule; its neighbours 2, 5, 7
  // each have 1 sharp edge -> dart rule; the remaining vertices are smooth.  This
  // packs all four vertex rules into one fixture.
  const positions = [
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
  ];
  const faces = [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 4, 7, 3], [1, 2, 6, 5],
  ];
  const creases = [
    { a: 6, b: 2, sharpness: Infinity },
    { a: 6, b: 5, sharpness: Infinity },
    { a: 6, b: 7, sharpness: Infinity },
  ];
  return {
    name: 'sharp-corner-box',
    description: 'cube with three infinite creases meeting at corner 6 (corner + darts)',
    positions,
    faces,
    creases,
    corners: [],
  };
}

export function creaseRidge(semiSharp = 3) {
  // A bumpy 7x7 grid with a crease chain along the middle row.  The chain's two
  // endpoints terminate at ordinary interior vertices (dart rule); the interior
  // of the chain uses the crease rule.  With finite sharpness the crease fades
  // over levels, which forces the limit evaluator to keep subdividing first.
  const n = 7;
  const spacing = 0.75;
  const { positions, index } = gridPositions(n, n, spacing);
  const mid = Math.floor(n / 2);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const [x, y] = positions[index[j][i]];
      positions[index[j][i]][2] =
        0.55 * Math.exp(-0.7 * (x * x * 0.6 + y * y * 1.7)) +
        0.05 * Math.sin(1.7 * x) * Math.cos(1.1 * y);
    }
  }
  const creases = [];
  for (let i = 0; i < n - 1; i++) {
    creases.push({ a: index[mid][i], b: index[mid][i + 1], sharpness: semiSharp });
  }
  return {
    name: semiSharp === Infinity ? 'crease-ridge-sharp' : 'crease-ridge',
    description: semiSharp === Infinity
      ? 'grid bump, infinitely sharp crease chain ending at darts'
      : `grid bump, crease chain with sharpness ${semiSharp} ending at darts`,
    positions,
    faces: gridFaces(index),
    creases,
    corners: [],
  };
}

export function creaseAtExtraordinary(semiSharp = Infinity) {
  // Pentagonal pyramid where two apex edges are creased.  The apex has valence 5
  // and exactly two sharp edges: the crease rule applies to an extraordinary
  // vertex directly.  With finite sharpness the crease decays, so the limit
  // evaluator must refine further before taking the mask.
  const base = pentagonPyramid();
  const apex = 0;
  const ring = [1, 2, 3, 4, 5];
  const creases = [
    { a: apex, b: ring[0], sharpness: semiSharp },
    { a: apex, b: ring[1], sharpness: semiSharp },
  ];
  return {
    name: semiSharp === Infinity ? 'crease-at-extraordinary' : 'semi-crease-at-extraordinary',
    description: semiSharp === Infinity
      ? 'valence-5 apex with two infinite creases (curve limit at extraordinary vertex)'
      : `valence-5 apex with semi-sharp creases (sharpness ${semiSharp})`,
    positions: base.positions,
    faces: base.faces,
    creases,
    corners: [],
  };
}

// ---- invalid fixtures -------------------------------------------------------

export function nonManifoldEdgeMesh() {
  // Three quads sharing edge 0-1.
  const positions = [
    [0, 0, 0], [1, 0, 0],
    [0, 1, 0], [1, 1, 0],
    [0, -1, 0], [1, -1, 0],
    [0, 0, 1], [1, 0, 1],
  ];
  const faces = [
    [0, 1, 3, 2],
    [0, 1, 5, 4],
    [0, 1, 7, 6],
  ];
  return { name: 'non-manifold-edge', positions, faces, creases: [], corners: [] };
}

export function nonManifoldVertexMesh() {
  // Two quads touching at a single shared vertex (bow-tie), which is a
  // disconnected face fan at that vertex.
  const positions = [
    [0, 0, 0],
    [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0],
    [1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1],
  ];
  const faces = [
    [0, 1, 2, 3],
    [0, 4, 8, 5],
  ];
  return { name: 'non-manifold-vertex', positions, faces, creases: [], corners: [] };
}

export function degenerateFaceMesh() {
  const positions = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0],
  ];
  const faces = [
    [0, 1, 3, 2],
    [1, 1, 5, 4], // repeated vertex
  ];
  return { name: 'degenerate-face', positions, faces, creases: [], corners: [] };
}

export function zeroAreaFaceMesh() {
  const positions = [
    [0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [0, 1, 0], [1, 1, 0],
  ];
  const faces = [
    [0, 1, 2, 3], // collinear quad
    [0, 1, 5, 4],
  ];
  return { name: 'zero-area-face', positions, faces, creases: [], corners: [] };
}

export function duplicateFaceMesh() {
  const positions = [
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  ];
  const faces = [
    [0, 1, 2, 3],
    [0, 1, 2, 3], // exact duplicate
  ];
  return { name: 'duplicate-face', positions, faces, creases: [], corners: [] };
}

// ---------------------------------------------------------------- registry ---

export const MODEL_BUILDERS = {
  'regular-grid': () => regularGrid(),
  'open-boundary-sheet': () => openBoundarySheet(),
  'pentagon-pyramid': () => pentagonPyramid(),
  'quad-ball': () => quadBall(2),
  'open-cylinder': () => openCylinder(7),
  torus: () => torus(10, 6),
  'concave-uneven': () => concaveUneven(),
  'creased-cube': () => creasedCube(),
  'sharp-corner-box': () => sharpCornerBox(),
  'crease-ridge': () => creaseRidge(3),
  'crease-ridge-sharp': () => creaseRidge(Infinity),
  'crease-at-extraordinary': () => creaseAtExtraordinary(Infinity),
  'semi-crease-at-extraordinary': () => creaseAtExtraordinary(2),
  'non-manifold-edge': () => nonManifoldEdgeMesh(),
  'non-manifold-vertex': () => nonManifoldVertexMesh(),
  'degenerate-face': () => degenerateFaceMesh(),
  'zero-area-face': () => zeroAreaFaceMesh(),
  'duplicate-face': () => duplicateFaceMesh(),
};

export const VALID_MODEL_NAMES = [
  'regular-grid', 'open-boundary-sheet', 'pentagon-pyramid', 'quad-ball',
  'open-cylinder', 'torus', 'concave-uneven', 'creased-cube',
  'sharp-corner-box', 'crease-ridge', 'crease-ridge-sharp',
  'crease-at-extraordinary', 'semi-crease-at-extraordinary',
];

export const INVALID_MODEL_NAMES = [
  'non-manifold-edge', 'non-manifold-vertex', 'degenerate-face',
  'zero-area-face', 'duplicate-face',
];

export function modelByName(name) {
  const build = MODEL_BUILDERS[name];
  if (!build) throw new Error('Unknown model: ' + name);
  return build();
}
