# Efficient-maps-for-local-areas
This project extracts object skeletons, detects junction points, segments paths, and reconstructs geometries using cubic splines.


📍 Application Workflow

    📸 Input: Drone or satellite images of a village/town.
    🎛️ Preprocess: Convert to grayscale → binarize.
    🦴 Skeletonize: Reduce roads and paths to skeleton lines.
    🔗 Bridge Detection: Detect all intersections and junctions.
    🧩 Segment and Fit: Fit smooth curves (splines) to every segment.
    🗺️ Reconstruct Map: overlay on actual terrain.

🖼 Sample Input

    Place an input images under `input/` folder.

🧾 Outputs

    The generates the following images under the `output/` directory:
    
    - `skeletonized_all_objects.png` – Skeleton view of all detected objects
    - `detected_bridges.png` – Bridge point markers on skeleton
    - `debug_Segments.png` – Color-highlighted segments
    - `reconstructed_image.png` – Reconstructed image using splines
    - `knot_points.png` – Knot points marked on skeleton
    - `debug_width_exploration.png` – Width debug visualization

📈 Why Use Splines?

    Representing detected paths as parametric splines (instead of raw pixel chains):
      
    - 📏 Enables precise mathematical analysis (length, curvature, etc.)
    - 💾 Reduces memory requirements by representing hundreds of pixels with a few control points
    - 🔄 Facilitates smooth reconstruction, which is important for modeling
      
🔮 Future Scope

    This project sets the foundation for building routing systems and efficient local maps by enabling:
    
     ✅ Source & Destination Allocation
     ✅ Minimum Distance Path Calculation
