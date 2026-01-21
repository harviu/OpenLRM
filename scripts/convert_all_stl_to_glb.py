import trimesh
import os
from pathlib import Path

# === Configuration ===
input_dir = Path("/mnt/home/lihao/lihao_project/antler_scans")   # folder containing .stl files
output_dir = Path("/mnt/home/lihao/lihao_project/antler_glb")  # where to save .glb files
output_dir.mkdir(parents=True, exist_ok=True)

# === Conversion loop ===
for stl_file in input_dir.glob("*.stl"):
    try:
        # Load mesh
        mesh = trimesh.load(stl_file, force='mesh')
        if mesh.is_empty:
            print(f"⚠️ Skipped empty file: {stl_file.name}")
            continue

        # Construct output filename
        glb_file = output_dir / (stl_file.stem + ".glb")

        # Export to GLB
        mesh.export(glb_file)
        print(f"✅ Converted: {stl_file.name} → {glb_file.name}")

    except Exception as e:
        print(f"❌ Failed to convert {stl_file.name}: {e}")
