// Convert core meshes into BufferGeometry-friendly arrays.
//
// The viewer wants:
//   * a triangulated surface for shading (smooth normals),
//   * a wireframe of the real quad topology (so the shared-edge structure is
//     visible and can be verified by eye),
//   * optional per-vertex overlays (face points, edge points, extraordinary
//     vertices, crease edges).
//
// Nothing here re-interpolates faces: it consumes the exact positions produced by
// the subdivision core, so the mesh on screen *is* the delivered mesh.

import { buildTopology } from '../core/topology.js';

/**
 * Triangulate the quad/polygon faces for rendering.  We use a fan from the face's
 * first vertex; child faces are convex quads in practice, and this avoids adding
 * any new vertices.
 */
export function buildTriangles(positions, faces) {
  const verts = [];
  const normals = [];
  const faceIds = [];
  // sourceVertices[3*t + k] is the mesh vertex index behind triangle t's corner k.
  // A raycast hit reports triangle-soup indices; this maps back to the real mesh.
  const sourceVertices = [];

  // smooth vertex normals from face normals (area weighted)
  const acc = new Float64Array(positions.length * 3);
  const faceNormals = new Array(faces.length);
  for (let f = 0; f < faces.length; f++) {
    const fn = newellNormal(positions, faces[f]);
    faceNormals[f] = fn;
    for (let i = 0; i < faces[f].length; i++) {
      const v = faces[f][i];
      acc[v * 3] += fn[0];
      acc[v * 3 + 1] += fn[1];
      acc[v * 3 + 2] += fn[2];
    }
  }
  const invNorm = new Float64Array(positions.length);
  for (let v = 0; v < positions.length; v++) {
    const l = Math.hypot(acc[v * 3], acc[v * 3 + 1], acc[v * 3 + 2]);
    invNorm[v] = l > 1e-30 ? 1 / l : 0;
  }

  for (let f = 0; f < faces.length; f++) {
    const face = faces[f];
    for (let i = 1; i < face.length - 1; i++) {
      const tri = [face[0], face[i], face[i + 1]];
      for (const v of tri) {
        verts.push(positions[v][0], positions[v][1], positions[v][2]);
        normals.push(acc[v * 3] * invNorm[v], acc[v * 3 + 1] * invNorm[v], acc[v * 3 + 2] * invNorm[v]);
        sourceVertices.push(v);
      }
      faceIds.push(f);
    }
  }
  return {
    positions: new Float32Array(verts),
    normals: new Float32Array(normals),
    faceIds: new Uint32Array(faceIds),
    sourceVertices: new Uint32Array(sourceVertices),
    faceNormals,
  };
}

function newellNormal(positions, face) {
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

/**
 * Line segments for the wireframe of the actual topology (deduplicated edges).
 */
export function buildWireframe(positions, faces, topology = null) {
  const topo = topology || buildTopology(positions, faces, { allowBoundary: true });
  const out = new Float32Array(topo.numEdges * 6);
  let k = 0;
  for (const e of topo.edges) {
    const a = positions[e.a];
    const b = positions[e.b];
    out[k++] = a[0]; out[k++] = a[1]; out[k++] = a[2];
    out[k++] = b[0]; out[k++] = b[1]; out[k++] = b[2];
  }
  return { positions: out, edgeCount: topo.numEdges, topology: topo };
}

/** Points for a subset of vertices, e.g. face points or edge points. */
export function buildPointCloud(positions, indices, color) {
  const out = new Float32Array(indices.length * 3);
  for (let i = 0; i < indices.length; i++) {
    const p = positions[indices[i]];
    out[i * 3] = p[0]; out[i * 3 + 1] = p[1]; out[i * 3 + 2] = p[2];
  }
  return { positions: out, count: indices.length, color };
}

/** Line segments for crease edges (to show sharpness), from a crease map. */
export function buildCreaseSegments(positions, faces, creaseMap, topology = null) {
  const topo = topology || buildTopology(positions, faces, { allowBoundary: true });
  const segs = [];
  const boundary = [];
  for (const e of topo.edges) {
    const a = positions[e.a];
    const b = positions[e.b];
    if (e.faces.length === 1) {
      boundary.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    } else if (creaseMap.has(e.key)) {
      segs.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
  }
  return {
    crease: new Float32Array(segs),
    boundary: new Float32Array(boundary),
    creaseCount: segs.length / 6,
    boundaryCount: boundary.length / 6,
  };
}
