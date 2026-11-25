// Stroke tracing for numbered crease chains.
//
// A regression fixture may carry `creaseChains: [{ id, edges: [[a,b], ...] }]`
// where the array order is the engraving's numbering.  The subdivision core
// carries that numbering to every refined level (see creaseChainSegments), and
// this module turns it back into an ordered polyline so a stroke can be driven
// from the delivered mesh.
//
// Two failures have to be told apart, and they have different signatures:
//
//   * "read the wrong parent edge"  -> consecutive numbered segments no longer
//     share an endpoint, so the traced polyline has breaks ("断笔").  Geometry is
//     untouched, so every traced vertex is exactly where it should be.
//   * "wrong interpolation weight"  -> the polyline is still connected, but the
//     traced vertices drift off the engraved line.  Connectivity is untouched.
//
// So the combination (connected?, per-vertex distance to the clean engraving)
// isolates either fault without needing a fragile independent curve fit:
//
//   clean          connected = true,  vertex deviation ~ 0
//   wrong-parent   connected = false, vertex deviation ~ 0
//   wrong-weight   connected = true,  vertex deviation > 0
//
// The engraved line itself is produced by the same subdivision kernel on an
// un-faulted copy of the control net; the fault is the *only* difference, which
// is what makes the comparison meaningful.

import { vdist } from './vec3.js';
import { FAULT, createControlLevel, subdivideLevel } from './subdivision.js';

/**
 * Order the child halves of one numbered parent segment along the segment.
 *
 * @param {Array<{key:string,a:number,b:number}>} halves
 */
function orderHalves(halves) {
  if (!halves || halves.length === 0) {
    return { points: [], ordered: [], complete: false };
  }
  const endpointCount = new Map();
  for (const h of halves) {
    endpointCount.set(h.a, (endpointCount.get(h.a) || 0) + 1);
    endpointCount.set(h.b, (endpointCount.get(h.b) || 0) + 1);
  }
  const ends = [...endpointCount.entries()].filter(([, c]) => c === 1).map(([v]) => v);
  if (ends.length !== 2) return { points: [], ordered: halves.slice(), complete: false };

  const byVertex = new Map();
  for (const h of halves) {
    if (!byVertex.has(h.a)) byVertex.set(h.a, []);
    byVertex.get(h.a).push(h);
    if (!byVertex.has(h.b)) byVertex.set(h.b, []);
    byVertex.get(h.b).push(h);
  }
  const used = new Set();
  const ordered = [];
  let cur = ends[0];
  const points = [cur];
  while (ordered.length < halves.length) {
    const cand = (byVertex.get(cur) || []).find((h) => !used.has(h));
    if (!cand) break;
    used.add(cand);
    ordered.push(cand);
    cur = cand.a === cur ? cand.b : cand.a;
    points.push(cur);
  }
  return {
    points,
    ordered,
    complete: ordered.length === halves.length && points.length === halves.length + 1,
  };
}

/**
 * Trace all numbered segments of a chain at a level.
 *
 * `joints[k]` records whether segment k ends where segment k+1 begins.  A shared
 * topology index is the strongest continuity; coincident-but-distinct vertices
 * (the projection seam at the far end of the regression chain) also count as
 * continuous geometrically but are flagged separately.
 */
export function traceCreaseChain(level, chainId, options = {}) {
  const tol = options.tolerance != null ? options.tolerance : 1e-9;
  const directory = level.creaseChainSegments;
  if (!directory || !directory.has(chainId)) {
    const err = new Error(`Level ${level.level} has no traced segments for chain "${chainId}"`);
    err.code = 'CHAIN_NOT_TRACED';
    throw err;
  }
  const buckets = directory.get(chainId);
  const segments = [];
  for (let i = 0; i < buckets.length; i++) {
    const orderedResult = orderHalves(buckets[i]);
    segments.push({
      index: i,
      path: orderedResult.points,
      complete: orderedResult.complete,
      points: orderedResult.points.map((v) => level.positions[v]),
    });
  }

  const joints = [];
  const breaks = [];
  for (let i = 0; i + 1 < segments.length; i++) {
    const a = segments[i];
    const b = segments[i + 1];
    const endV = a.path.length ? a.path[a.path.length - 1] : -1;
    const startV = b.path.length ? b.path[0] : -1;
    const endP = a.points.length ? a.points[a.points.length - 1] : null;
    const startP = b.points.length ? b.points[0] : null;
    const gap = endP && startP ? vdist(endP, startP) : Infinity;
    const sharedVertex = endV >= 0 && endV === startV;
    const coincident = !sharedVertex && gap <= tol;
    joints.push({ from: i, to: i + 1, endVertex: endV, startVertex: startV, gap, sharedVertex, coincident });
    if (!sharedVertex && !coincident) breaks.push({ from: i, to: i + 1, gap });
  }

  const polyline = [];
  for (const seg of segments) {
    for (const v of seg.path) {
      if (polyline.length && polyline[polyline.length - 1] === v) continue;
      polyline.push(v);
    }
  }

  return {
    chainId,
    level: level.level,
    segments,
    joints,
    breaks,
    connected: breaks.length === 0,
    complete: segments.every((s) => s.complete),
    polyline,
    points: polyline.map((v) => level.positions[v]),
    vertexSet: collectVertices(segments),
  };
}

function collectVertices(segments) {
  const set = new Set();
  for (const s of segments) for (const v of s.path) set.add(v);
  return [...set].sort((a, b) => a - b);
}

/**
 * Build an un-faulted ("engraved") copy of the same control net at the same
 * level.  Because the topology is identical, vertex indices correspond exactly;
 * only positions can differ between this and a faulted build.
 */
export function buildEngravedLevel(control, level, options = {}) {
  const base = withoutFaultOptions(options);
  let cur = createControlLevel(control, base);
  for (let i = 0; i < level; i++) cur = subdivideLevel(cur);
  return cur;
}

/** Build a level with a controlled fault injected at `faultLevel`. */
export function buildFaultedLevel(control, level, fault, faultLevel = 0, options = {}) {
  const base = withoutFaultOptions(options);
  let cur = createControlLevel(control, base);
  for (let i = 0; i < level; i++) {
    cur = subdivideLevel(cur, i === faultLevel ? { fault, faultLevel } : {});
  }
  return cur;
}

function withoutFaultOptions(options) {
  const out = { ...options };
  delete out.fault;
  delete out.faultLevel;
  delete out.driftTolerance;
  return out;
}

/**
 * Per-vertex position distance between two traces over the same control net.
 * Compares by vertex index, not polyline order, so a reordering fault does not
 * masquerade as a position drift.
 */
export function vertexDeviation(reference, refLevel, trace, testLevel) {
  const ref = new Set(reference.vertexSet);
  let max = 0;
  let sum = 0;
  let worst = -1;
  let count = 0;
  for (const v of trace.vertexSet) {
    if (!ref.has(v)) continue;
    const d = vdist(refLevel.positions[v], testLevel.positions[v]);
    sum += d;
    count++;
    if (d > max) { max = d; worst = v; }
  }
  return { count, max, mean: count ? sum / count : 0, worstVertex: worst };
}

/** Convenience wrapper: classify a build against a clean engraving. */
export function diagnoseStroke(control, level, chainId, options = {}) {
  const fault = options.fault || FAULT.NONE;
  const faultLevel = options.faultLevel != null ? options.faultLevel : 0;
  const clean = buildEngravedLevel(control, level, options);
  const testLevel = fault === FAULT.NONE
    ? clean
    : buildFaultedLevel(control, level, fault, faultLevel, options);

  const trace = traceCreaseChain(testLevel, chainId);
  const reference = traceCreaseChain(clean, chainId);
  const deviation = vertexDeviation(reference, clean, trace, testLevel);

  const signature = !trace.connected
    ? 'broken-pen'
    : deviation.max > (options.driftTolerance != null ? options.driftTolerance : 1e-6)
      ? 'off-engraving'
      : 'ok';

  return {
    chainId,
    level,
    fault,
    faultLevel,
    connected: trace.connected,
    complete: trace.complete,
    breaks: trace.breaks,
    drift: deviation,
    signature,
    trace,
    reference,
    testLevel,
    cleanLevel: clean,
  };
}

/**
 * Convert a trace into flat line-segment pairs ready for a LineSegments overlay.
 * Breaks are included as gaps (no synthetic segment is drawn across them), which
 * is what makes a broken pen visible instead of silently bridged.
 */
export function traceToSegments(trace) {
  const segs = [];
  for (const seg of trace.segments) {
    for (let i = 0; i + 1 < seg.points.length; i++) {
      const a = seg.points[i];
      const b = seg.points[i + 1];
      segs.push(a[0], a[1], a[2], b[0], b[1], b[2]);
    }
  }
  return new Float32Array(segs);
}
