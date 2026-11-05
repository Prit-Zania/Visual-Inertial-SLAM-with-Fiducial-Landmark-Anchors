import cv2
from pathlib import Path
from slam.sensor_api import generate_aruco_svg
from slam.live_capture import ROOM_MARKERS

out_dir = Path("aruco_markers")
out_dir.mkdir(exist_ok=True)

html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>ArUco Markers Sheet (DICT_6X6_250) - 120mm Room Anchors</title>
  <style>
    @page { size: A4 portrait; margin: 10mm; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #0f172a; margin: 0; padding: 20px; }
    .header { text-align: center; margin-bottom: 20px; }
    .header h1 { margin: 0 0 6px 0; font-size: 22px; color: #0f172a; }
    .header p { margin: 0; font-size: 13px; color: #475569; }
    .btn-print { background: #0284c7; color: #fff; border: none; border-radius: 6px; padding: 8px 18px; font-weight: 600; cursor: pointer; margin-top: 10px; font-size: 14px; }
    @media print { .btn-print { display: none; } body { background: #fff; padding: 0; } }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; max-width: 820px; margin: 0 auto; }
    .marker-card { background: #ffffff; border: 2px dashed #94a3b8; border-radius: 8px; padding: 14px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.04); }
    .marker-card svg { width: 100%; max-width: 280px; height: auto; display: block; margin: 0 auto; }
    .marker-info { margin-top: 8px; font-size: 13px; }
    .marker-info strong { color: #0284c7; }
    .marker-loc { font-size: 11px; color: #64748b; font-family: monospace; margin-top: 2px; }
  </style>
</head>
<body>
  <div class="header">
    <h1>ArUco Marker Print Studio &bull; DICT_6X6_250</h1>
    <p>Standard 120mm Room Anchors for Physical Walls &bull; Visual-Inertial SLAM Ground Anchors</p>
    <button class="btn-print" onclick="window.print()">🖨️ Print Marker Sheet (100% Scale)</button>
  </div>
  <div class="grid">
"""

for mid in range(6):
    svg = generate_aruco_svg(mid, 120)
    rm = ROOM_MARKERS.get(mid, {})
    name = rm.get("name", f"Marker #{mid}")
    wall = rm.get("wall", "wall").upper()
    pos = rm.get("pos", [0, 0, 0])
    
    html += f"""    <div class="marker-card">
      {svg}
      <div class="marker-info">
        <strong>#{mid} &bull; {wall} WALL</strong> &bull; {name}
      </div>
      <div class="marker-loc">Surveyed 3D Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}] m | Size: 120mm</div>
    </div>
"""

html += """  </div>
</body>
</html>
"""

out_path = out_dir / "print_markers.html"
out_path.write_text(html, encoding="utf-8")
print(f"Generated {out_path} ({len(html)} bytes)")

