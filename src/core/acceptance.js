// Explicit acceptance checks for subdivision and limit-surface behaviour.
//
// These are the pass/fail predicates the test harness and the browser QA panel
// both call.  They exist so "the face count went up" can never stand in for a
// real result: every check names the property it guards, returns the evidence it
// looked at, and fails on a controlled bad input.
//
// Categories:
//   subdivisionProperties(levels)   topology / connectivity invariants
//   limitAccuracy(level, reference) position and normal error vs a reference
//   seamAcceptance(clean, test)     position vs normal, kept separate
//   strokeAcceptance(diagnosis)     connected vs on-engraving, kept separate

import { buildTopology } from './topology.js';
import { evaluateLimits } from './limit.js';
import { vdist, vdot } from './vec3.js';
import { findDuplicatePositions } from './quality.js';
import { topologyEvidence, vertexLimitNormal } from './seams.js';

/**
 * Structural invariants every level of a Catmull-Clark refinement must satisfy.
 * Returns { ok, checks: [{name, ok, evidence}] } so a failing check is named.
 */
export function subdivisionProperties(levels, options = {}) {
  const checks = [];
  const tol = options.duplicateTolerance != null ? options.duplicateTolerance : 1e-9;
  const add = (name, ok, evidence) => checks.push({ name, ok: !!ok, evidence });

  // 1. Topology is manifold, quad-only, and boundary is 1-manifold.
  //    Quad-only is a property of *subdivision output*, not of the control cage:
  //    a hand-authored L0 may legitimately contain triangles (cap fans).  The
  //    cage still has to be manifold with a 1-manifold boundary.
  const quadFrom = options.quadOnlyFromLevel != null ? options.quadOnlyFromLevel : 1;
  for (const lv of levels) {
    const ev = topologyEvidence(lv);
    const quadApplies = lv.level >= quadFrom;
    const structuralOk = ev.nonManifoldVertices.length === 0 &&
      ev.nonManifoldEdgeCount === 0 && ev.repeatedVertexFaces === 0 &&
      ev.boundaryNotOneManifold.length === 0;
    const ok = structuralOk && (!quadApplies || ev.nonQuadFaces === 0);
    add(`L${lv.level}:manifold-quad-boundary`, ok, { ...ev, quadApplies });
  }

  // 2. Euler characteristic constant across levels, and equal to the expected
  //    value for the model's topology class when one is supplied.
  const chi0 = options.expectedEuler != null ? options.expectedEuler : topologyEvidence(levels[0]).euler;
  for (const lv of levels) {
    const chi = topologyEvidence(lv).euler;
    add(`L${lv.level}:euler`, chi === chi0, { chi, expected: chi0 });
  }

  // 3. No coincident-but-distinct vertices (the per-face seam bug).
  for (const lv of levels) {
    const dups = findDuplicatePositions(lv.positions, tol);
    // A fixture may intentionally carry projection seams (the stroke regression).
    const allowed = options.allowedSeams != null ? options.allowedSeams : 0;
    add(`L${lv.level}:no-accidental-seams`, dups.length <= allowed,
      { duplicates: dups.length, allowed });
  }

  // 4. Child vertex count is exactly V + E + F of the parent.
  for (let i = 1; i < levels.length; i++) {
    const p = levels[i - 1];
    const expected = p.positions.length + p.topology.numEdges + p.faces.length;
    add(`L${i}:child-vertex-count`, levels[i].positions.length === expected,
      { got: levels[i].positions.length, expected });
  }

  // 5. Every interior edge is shared by exactly two faces (already implied by
  //    manifold check, but recorded separately so a topology regression names it).
  for (const lv of levels) {
    const topo = lv.topology || buildTopology(lv.positions, lv.faces, { allowBoundary: true });
    const bad = topo.edges.filter((e) => e.faces.length > 2).length;
    add(`L${lv.level}:edge-sharing`, bad === 0, { overShared: bad });
  }

  return { ok: checks.every((c) => c.ok), checks };
}

/**
 * Position and normal accuracy against an independently supplied reference.
 *
 * @param {object} level
 * @param {object} reference
 * @param {Array<[number,number,number]>} reference.positions  per vertex
 * @param {Array<[number,number,number]>} [reference.normals]  per vertex (optional)
 * @param {number[]} [reference.indices]  which vertices to check
 */
export function limitAccuracy(level, reference, options = {}) {
  const indices = reference.indices ||
    Array.from({ length: Math.min(level.positions.length, reference.positions.length) }, (_, i) => i);
  const posTol = options.positionTolerance != null ? options.positionTolerance : 1e-6;
  const angTol = options.normalToleranceDeg != null ? options.normalToleranceDeg : 1.0;

  const lim = evaluateLimits(level, { indices, maxCreaseLevels: 0 });
  const byIndex = new Map(lim.results.map((r) => [r.index, r.position]));

  let posMax = 0, posWorst = -1;
  for (const v of indices) {
    const p = byIndex.get(v) || level.positions[v];
    const d = vdist(p, reference.positions[v]);
    if (d > posMax) { posMax = d; posWorst = v; }
  }

  let angMax = 0, angWorst = -1;
  if (reference.normals) {
    // Compare the limit surface's normal (from incident faces at their limit
    // positions) against the independent reference normal at the same vertex.
    const limPos = new Array(level.positions.length);
    for (let i = 0; i < limPos.length; i++) limPos[i] = level.positions[i];
    for (const r of lim.results) if (r.position) limPos[r.index] = r.position;
    for (const v of indices) {
      if (!reference.normals[v]) continue;
      const nCore = vertexLimitNormal(level, limPos, v);
      const ang = Math.acos(Math.max(-1, Math.min(1, vdot(nCore, reference.normals[v])))) * 180 / Math.PI;
      if (ang > angMax) { angMax = ang; angWorst = v; }
    }
  }

  const posOk = posMax <= posTol;
  const normOk = !reference.normals || angMax <= angTol;
  return {
    ok: posOk && normOk,
    position: { max: posMax, worstVertex: posWorst, tolerance: posTol, ok: posOk },
    normal: { maxAngleDeg: angMax, worstVertex: angWorst, tolerance: angTol, ok: normOk },
  };
}

/**
 * Seam acceptance: the two quantities are reported and required separately.
 * A caller that only cares about one can pass a zero tolerance for the other.
 */
export function seamAcceptance(clean, test, metrics, options = {}) {
  const posTol = options.positionTolerance != null ? options.positionTolerance : 1e-6;
  const normTol = options.normalToleranceDeg != null ? options.normalToleranceDeg : 0.5;
  const topoExpected = options.topologyOk !== false;

  const posOk = metrics.positionDelta.max <= posTol;
  const limOk = metrics.limitPositionDelta.max <= posTol;
  const normOk = metrics.normalDelta.maxAngleDeg <= normTol;
  const ev = topologyEvidence(test);
  const topoOk = topoExpected ? ev.ok : true;

  return {
    ok: posOk && limOk && normOk && topoOk,
    position: {
      current: metrics.positionDelta.max,
      limit: metrics.limitPositionDelta.max,
      tolerance: posTol,
      ok: posOk && limOk,
      worstVertex: metrics.positionDelta.worstVertex,
    },
    normal: {
      maxAngleDeg: metrics.normalDelta.maxAngleDeg,
      tolerance: normTol,
      ok: normOk,
      worstVertex: metrics.normalDelta.worstVertex,
    },
    topology: { ok: topoOk, evidence: ev },
  };
}

/**
 * Stroke acceptance: connection and on-engraving are separate assertions.
 * `expect` is one of 'clean' | 'broken-pen' | 'off-engraving'.
 */
export function strokeAcceptance(diagnosis, expect, options = {}) {
  const driftTol = options.driftTolerance != null ? options.driftTolerance : 1e-6;
  const connected = diagnosis.connected;
  const onEngraving = diagnosis.drift.max <= driftTol;
  const checks = [];
  const add = (name, ok, evidence) => checks.push({ name, ok: !!ok, evidence });

  if (expect === 'clean') {
    add('connected', connected, { breaks: diagnosis.breaks.length });
    add('on-engraving', onEngraving, { drift: diagnosis.drift.max });
  } else if (expect === 'broken-pen') {
    add('disconnected', !connected, { breaks: diagnosis.breaks.length });
    add('geometry-unmoved', onEngraving, { drift: diagnosis.drift.max });
  } else if (expect === 'off-engraving') {
    add('connected', connected, { breaks: diagnosis.breaks.length });
    add('off-engraving', !onEngraving, { drift: diagnosis.drift.max });
  } else {
    throw new Error(`Unknown stroke expectation "${expect}"`);
  }
  return { ok: checks.every((c) => c.ok), checks, signature: diagnosis.signature };
}
