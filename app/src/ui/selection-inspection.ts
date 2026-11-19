/** Project only selected centres; bounded work even for multi-million selections.
 * Markers are a diagnostic overlay, never a change to the document or export.
 */
export const MARKER_LIMIT = 20000;

export function selectionMarkers(centres: Float32Array<ArrayBufferLike>,
  indices: Uint32Array<ArrayBufferLike>, alive: Uint8Array<ArrayBufferLike>,
  matrix: Float32Array<ArrayBufferLike>, width: number, height: number): number[] {
  const points: number[] = [];
  const step = Math.max(1, Math.ceil(indices.length / MARKER_LIMIT));
  for (let k = 0; k < indices.length; k += step) {
    const i = indices[k];
    if (!alive[i]) continue;
    const x = centres[i * 3], y = centres[i * 3 + 1], z = centres[i * 3 + 2];
    const w = matrix[3] * x + matrix[7] * y + matrix[11] * z + matrix[15];
    if (!(w > 1e-6)) continue;
    const cx = matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12];
    const cy = matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13];
    const cz = matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14];
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(cz)
        || Math.abs(cx) > w || Math.abs(cy) > w || Math.abs(cz) > w) continue;
    points.push((cx / w + 1) * width / 2, (1 - cy / w) * height / 2);
  }
  return points;
}
