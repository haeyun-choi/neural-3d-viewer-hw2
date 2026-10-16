// Tiny original, asymmetric fixture. No external model or license-dependent data.
export const asymmetricPoints = [[-2, 1, 0], [-0.5, 1.3, 0.4], [1.5, 0.8, -0.2], [-1.3, -1.2, 0.1], [0.3, -0.6, 0.7], [1.9, -1.7, -0.4]];
export function asymmetricPly(identity = "orientation-region") {
  return Buffer.from(`ply\nformat ascii 1.0\ncomment Synthetic asymmetric ${identity}\nelement vertex ${asymmetricPoints.length}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n${asymmetricPoints.map((p, i) => `${p.join(" ")} ${40 + i * 35} ${210 - i * 20} ${80 + i * 25}`).join("\n")}\n`);
}
