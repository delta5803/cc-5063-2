// Export the delivered mesh (the exact subdivided positions/faces) as OBJ or JSON.
//
// OBJ: quads are written as quads (not triangulated) so a DCC tool receives the
// real topology.  We also write a crease companion file hint separately in JSON.
// JSON: full descriptor including creases, corners and level stats, so the model
// can be reloaded without loss.

export function toOBJ(level, name = 'subdivision') {
  const lines = [];
  lines.push('# ' + name + ' — Catmull-Clark subdivision, level ' + level.level);
  lines.push('# vertices: ' + level.positions.length + ', faces: ' + level.faces.length);
  lines.push('o ' + sanitize(name) + '_L' + level.level);
  for (const p of level.positions) {
    lines.push(`v ${fmt(p[0])} ${fmt(p[1])} ${fmt(p[2])}`);
  }
  for (const f of level.faces) {
    lines.push('f ' + f.map((v) => v + 1).join(' '));
  }
  if (level.creaseMap && level.creaseMap.size) {
    lines.push('# creases (child-level)');
    for (const [key, s] of level.creaseMap) {
      const [a, b] = key.split(':').map(Number);
      lines.push(`# crease ${a + 1} ${b + 1} sharpness=${s}`);
    }
  }
  return lines.join('\n') + '\n';
}

export function toJSON(control, levels, extra = {}) {
  const topLevel = levels[levels.length - 1];
  return JSON.stringify({
    format: 'cc-subdivision-v1',
    name: control.name || 'mesh',
    options: levels[0].options,
    control: {
      positions: control.positions,
      faces: control.faces,
      creases: control.creases || [],
      corners: control.corners || [],
    },
    levels: levels.map((lv) => ({
      level: lv.level,
      vertexCount: lv.positions.length,
      edgeCount: lv.topology ? lv.topology.numEdges : null,
      faceCount: lv.faces.length,
      boundaryEdgeCount: lv.topology ? lv.topology.boundaryEdgeCount : null,
    })),
    mesh: {
      level: topLevel.level,
      positions: topLevel.positions,
      faces: topLevel.faces,
      creases: [...topLevel.creaseMap.entries()].map(([key, sharpness]) => {
        const [a, b] = key.split(':').map(Number);
        return { a, b, sharpness: Number.isFinite(sharpness) ? sharpness : 'inf' };
      }),
      corners: [...topLevel.cornerSet],
    },
    ...extra,
  }, null, 2);
}

export function download(filename, text, mime = 'text/plain') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function fmt(x) {
  // Enough precision for the limit surface, without 17-digit noise.
  return Number.isFinite(x) ? x.toPrecision(10) : '0';
}

function sanitize(s) {
  return s.replace(/[^A-Za-z0-9_-]+/g, '_');
}
