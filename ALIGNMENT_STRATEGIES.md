# Alignment Strategies

## Current Method

The initial leaf alignment in `HPX_HADES.py` is mostly rigid/similarity-like:

- fixed resize from RootCam mask space to VNIR canvas scale
- small rotation search
- coarse fixed paste offset
- integer translation grid search

The root refinement now defaults to `ecc_affine`: the leaf-aligned root mask is refined against the root-enhanced VNIR band with OpenCV ECC affine registration. If ECC is unavailable or fails to converge, the script falls back to the previous grid IoU root alignment.

After the final `root_mask` alignment is estimated, the same leaf alignment and root refinement are reused for `main_root_mask`, `lateral_root_mask`, `node_mask`, and `tip_mask`. Each aligned component mask gets its own `*_summary.csv` file with mean, sum, and std spectra.

`root_mask`, `main_root_mask`, and `lateral_root_mask` are treated as skeleton masks: foreground pixels are transformed as centerline points and rasterized back to a one-pixel skeleton in VNIR space. `node_mask` and `tip_mask` are treated as landmark masks: each connected object is transformed by centroid and exported as exactly one VNIR pixel, preserving the number of nodes/tips before and after alignment.

The initial leaf stage still does not estimate affine shear, nonuniform scale, perspective/homography, or local deformation. The root stage can now estimate affine rotation/scale/shear/translation.

One important detail: the grid fallback now uses `iou` rather than raw `intersection`, because raw intersection can choose alignments with many overlapping pixels while not penalizing extra mask pixels.

## Recommended Experiment Order

1. Keep the current method as the baseline.
2. Use `ecc_affine` as the default root-stage refinement.
3. Keep grid IoU as fallback for plants where ECC fails.
4. Benchmark Euclidean and affine refinements after the current coarse transform.
5. Add phase correlation as a fast translation sanity check.
6. Add log-polar phase correlation to test whether rotation/scale mismatch is important.
7. Try chamfer/distance-transform scoring because thin roots lose IoU quickly after small pixel errors.
8. Try skeleton ICP because root centerlines are often a better geometry than thick binary areas.
9. Try MI on root-enhanced intensity bands when binary masks are noisy.
10. Treat ORB/homography as optional: ORB may fail on thin root masks, and homography can overfit root shape differences.

## Added Benchmark Methods

- `phase_correlation` / `translation`: fast FFT-based shift estimate.
- `logpolar_phase_correlation` / `similarity`: estimates rotation and scale, then refines translation.
- `chamfer` / `translation`, `euclidean`, `affine`: minimizes mean distance from transformed root-mask pixels to detected VNIR root pixels. This is often more forgiving and meaningful for thin roots than raw IoU.
- `skeleton_icp` / `euclidean`, `affine`: skeletonizes both masks and iteratively aligns root centerline point sets.
- `grid_iou`, `grid_dice`, `differential_evolution_iou`, `ecc`, `mutual_information`, and `orb_ransac` remain in the comparison.

## Benchmark Script

Use `scripts/benchmark_alignment_strategies.py` to compare methods on the exported bands and saved masks:

```powershell
python scripts\benchmark_alignment_strategies.py `
  --analysis-dir "VNIR2\Analysis\123_17_2025-08-04_11-17-49_exp63_ox10_417r_high_trans_01_VNIR2_Data" `
  --root-analysis-dir "ROOT2_analysis\123_16_exp63_ox10_417r_high_trans_01_ROOT2_Fish Eye Corrected_A0\123_16_exp63_ox10_417r_high_trans_01_ROOT2_Fish Eye Corrected_A0"
```

The script writes:

- `alignment_benchmark\alignment_benchmark.csv`
- per-plant overlay PNGs for each successful method

In the production `HPX_HADES.py` output, the alignment QC overlay uses component masks when available: main root is red, lateral roots are green, nodes are yellow, and tips are magenta. Node and tip markers are enlarged in the PNG for visibility only; the extraction masks remain one pixel per landmark. The peri-root border remains blue.

The `opencv` dependency in `environment.yml` enables ECC. Without OpenCV, ECC methods are skipped and the SciPy/scikit-image methods still run.

Useful speed controls:

- `--skip-slow` skips affine-IoU, chamfer-affine, and MI optimizers.
- `--grid-step 4` makes translation/chamfer grids faster but coarser.
- `--icp-max-points 1000` makes skeleton ICP faster.
