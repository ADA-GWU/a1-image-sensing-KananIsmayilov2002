### Stereo Matching with Census Transform

Computes disparity maps from stereo image pairs using the **Census Transform** and simple block matching.

---

## Folder Structure

```text
images/
├── original/
│   ├── dataset1/      (left.jpeg, right.jpeg)
│   ├── dataset2/      (left.jpeg, right.jpeg)
│   └── dataset3/      (left.jpeg, right.jpeg)
└── output/
    ├── dataset1/      (disparity maps)
    ├── dataset2/      (disparity maps)
    └── dataset3/      (disparity maps)
```

The images were captured with a smartphone camera at different times of the day, with:

- **Indoor artificial light** (dataset1 – bookshelf / textured objects at various depths)
- **Outdoor near sunset** (dataset2, dataset3 – objects at multiple distances and natural light)

Typical EXIF settings are similar to:

| Dataset  | ISO | Focal Length | Aperture | Shutter Speed | Lighting Conditions           |
| -------- | --- | ------------ | -------- | ------------- | ----------------------------- |
| dataset1 | 500 | 48mm         | f/1.78   | 1/40s         | Indoor, artificial            |
| dataset2 | 400 | 24mm         | f/1.78   | 1/50s         | Outdoor, 1 hour before sunset |
| dataset3 | 400 | 24mm         | f/1.78   | 1/50s         | Outdoor, 1 hour before sunset |

Inputs are converted to **grayscale (0–255)** and the disparity maps are visualized using the **`inferno` colormap**, where **dark colors ≈ far** and **bright yellow/white ≈ close**.

---

## Installation

Install the required dependencies (Python ≥ 3.8):

```bash
pip install opencv-python numpy matplotlib
```

---

## Usage

Run with default parameters:

```bash
python main.py
```

Or customize via command line:

```bash
python main.py --dataset dataset2 --window_size 51 --max_disparity 96
```

**Arguments (from `main.py`):**

- `--dataset`: which folder in `images/original/` to use (e.g., `dataset1`, `dataset2`, `dataset3`)
- `--window_size`: Census window size (odd integer, controls local neighborhood)
- `--max_disparity`: maximum disparity (pixels) to search along the epipolar line

Output files are named:

```text
images/output/<dataset>/Window_size_<window_size>_<dataset>.png
```

Each output figure shows:

- Left grayscale image
- Right grayscale image
- Disparity map with colorbar (pixel disparity range, `inferno` colormap)

---

## Window Size Comparison (Census Transform)

The **Census Transform** is applied with different window sizes to study the trade‑off between detail and smoothness:

- **Small windows** (e.g., 11): sharper edges and more detail, but more noise.
- **Medium windows** (e.g., 31, 51): good balance of smoothness and detail.
- **Large windows** (e.g., 71, 91): smoother disparity maps, but depth boundaries become blurred.

You can reproduce these comparisons by running `main.py` on the same dataset with different `--window_size` values and inspecting the saved PNGs in `images/output/<dataset>/`.

