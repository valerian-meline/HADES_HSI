from .common import *
from .io import *
from .preprocessing import *
from .diagnostics import *
from .utils import *
from .exports import *
from .alignment import *
from .master import *


def process_all_images(root_dir):

    root_dir = Path(root_dir)
    bil_files = list(root_dir.rglob("*Data.bil"))

    for bil_path in bil_files:
        try:
            raw, wavelengths = open_bil_with_temp_envi_header(bil_path, bil_path.with_suffix('.hdr'))
            dark_ref = bil_path.parent / bil_path.name.replace("_Data.bil", "_DarkCalibration.bil")
            dark, _ = open_bil_with_temp_envi_header(dark_ref, dark_ref.with_suffix('.hdr'))
            white_ref = bil_path.parent / bil_path.name.replace("_Data.bil", "_WhiteCalibration.bil")
            white, _ = open_bil_with_temp_envi_header(white_ref, white_ref.with_suffix('.hdr'))

            output_dir = bil_path.parent / "processed_images" / bil_path.stem
            output_dir.mkdir(parents=True, exist_ok=True)

            run_params = {
                "analysis_type": "process_all_images",
                "input_files": {
                    "raw_bil": bil_path,
                    "raw_hdr": bil_path.with_suffix('.hdr'),
                    "dark_bil": dark_ref,
                    "white_bil": white_ref,
                },
                "preprocessing": {
                    "mode": "VNIR2",
                    "dark_matching": "average calibration over dark rows only, preserve width x band structure, repeat across raw rows",
                    "zero_clipping": "disabled in analysis path; applied only for visualization/export",
                    "invalid_pixel_repair": "band-wise nearest-valid fill followed by 3x3 local median repair",
                    "spectral_smoothing": {"method": "Savitzky-Golay", "window_length": 11, "polyorder": 2, "passes": 1},
                    "spatial_denoising": {"method": "median_filter", "size": [3, 3]},
                },
                "exports": {
                    "band_png_fixed_max": 500.0,
                    "diagnostic_reference_band": 160,
                },
            }
            save_run_parameters(output_dir, run_params)

            denoised, corrected, repaired, smoothed, dark_flat, dark_flat_smooth = preprocessing_bil(
                raw, dark, white=None, mode='VNIR2'
            )

            npy_dir = output_dir / "npy_export"
            png_dir = output_dir / "png_export"
            npy_dir.mkdir(exist_ok=True)
            png_dir.mkdir(exist_ok=True)

            for i in range(denoised.shape[2]):
                band_out_npy = npy_dir / f"band_{i:03d}.npy"
                band_out_png = png_dir / f"band_{i:03d}.png"
                np.save(band_out_npy, denoised[:, :, i])

                fixed_max = 500.0
                band_vis = np.clip(denoised[:, :, i], 0, fixed_max)
                img_8bit = (band_vis / fixed_max * 255).astype(np.uint8)
                plt.imsave(band_out_png, img_8bit, cmap='gray')

            qc_plot_spectra(
                output_dir,
                wavelengths=wavelengths,
                raw=raw,
                white=white,
                dark=dark,
                corrected=corrected,
                repaired=repaired,
                smoothed=smoothed,
                denoised=denoised,
                dark_flat=dark_flat,
                dark_flat_smooth=dark_flat_smooth
            )

            save_preprocessing_diagnostics(
                output_dir=output_dir,
                raw_cube=raw,
                processed_cube=denoised,
                wavelengths=wavelengths,
                reference_band=min(160, denoised.shape[2] - 1),
                roi_masks=None,
                prefix=bil_path.stem
            )

        except Exception as e:
            print(f"⚠️ Error processing {bil_path.name}: {e}")


def analyze_segmentation_from_mask(mask_path, npy_folder, output_dir):

    mask = np.array(Image.open(mask_path).convert("L")) > 127
    npy_files = sorted(Path(npy_folder).glob("band_*.npy"))
    cube = np.stack([np.load(f) for f in npy_files], axis=-1)

    root_pixels = cube[mask]
    bg_pixels = cube[~mask]

    root_mean = np.mean(root_pixels, axis=0)
    bg_mean = np.mean(bg_pixels, axis=0)

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, root_mean, label='Root Region')
    plt.plot(wavelengths, bg_mean, label='Background')
    plt.title("Fluorescence Intensity: Root vs Background")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "qc_root_vs_background.png")
    plt.close()

    # LDA-based combination
    start_band = 150
    X = cube[:, :, 160:]
    y = mask.ravel().astype(int)
    lda = LinearDiscriminantAnalysis(n_components=1)
    lda_proj = lda.fit_transform(X, y).reshape(cube.shape[0], cube.shape[1])

    thresh = threshold_otsu(lda_proj)
    segmentation = lda_proj > thresh

    jaccard = jaccard_score(mask.ravel(), segmentation.ravel())

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(mask, cmap='gray')
    axs[0].set_title("Manual Mask")
    axs[1].imshow(combined, cmap='gray')
    axs[1].set_title("Combined Contrast Image")
    axs[2].imshow(segmentation, cmap='gray')
    axs[2].set_title(f"Otsu Segmentation\nJaccard={jaccard:.3f}")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "qc_segmentation_comparison.png")
    plt.close()


def process_vnir_with_root_masks(
        vnir2_root_dir,
        root_mask_dir,
        export_all_bands=False,
        export_band_indices=None,
        export_pixel_spectra_csv=False,
        save_preprocessing_plots=False,
        write_master_file=True,
):

    vnir2_root_dir = Path(vnir2_root_dir)
    #vnir1_root_dir = Path(vnir1_root_dir)
    root_mask_dir = Path(root_mask_dir)

    vnir_2_bil_files = list(vnir2_root_dir.rglob("*Data.bil"))
    #vnir_1_bil_files = list(vnir1_root_dir.rglob("*Data.bil"))

    if not vnir_2_bil_files:
        print("❌ No VNIR2 .bil files found.")
        return
    #if not vnir_1_bil_files:
    #    print("❌ No VNIR1 .bil files found.")
    #    return

    root_masks = [
        p for p in root_mask_dir.rglob("root_mask.png")
        if p.is_file() and any(f"plant_{i}" in str(p) for i in range(1, 6))
    ]

    if not root_masks:
        print("❌ No root masks found for plant_1 to plant_5.")
        return

    for vnir_2_bil_path in vnir_2_bil_files:
        try:
            sample_id = extract_vnir_id(vnir_2_bil_path.name)
            if not sample_id:
                print(f"⚠️ Could not extract VNIR ID from {vnir_2_bil_path.name}")
                continue

            matching_masks, previous_masks = find_two_closest_root_mask_rounds(sample_id, root_masks)
            if not matching_masks:
                matching_masks = previous_masks
            if not matching_masks:
                print(f"❌ No root mask found for ID: {sample_id}")
                continue

            #matching_vnir_1_bil_path = find_matching_vnir1(vnir_2_bil_path, vnir1_root_dir)

            # load hypercube, white and dark reference for VNIR2
            raw_vnir2, wavelengths_vnir2 = open_bil_with_temp_envi_header(vnir_2_bil_path, vnir_2_bil_path.with_suffix('.hdr'))
            dark_ref_vnir2 = vnir_2_bil_path.parent / vnir_2_bil_path.name.replace("_Data.bil", "_DarkCalibration.bil")
            white_ref_vnir2 = vnir_2_bil_path.parent / vnir_2_bil_path.name.replace("_Data.bil", "_WhiteCalibration.bil")
            dark_vnir2, _ = open_bil_with_temp_envi_header(dark_ref_vnir2, dark_ref_vnir2.with_suffix('.hdr'))
            white_vnir2, _ = open_bil_with_temp_envi_header(white_ref_vnir2, white_ref_vnir2.with_suffix('.hdr'))

            # preprocess the cube, flip horizontally and pad to a fixed shape if necessary
            denoised_vnir2, corrected_vnir2, repaired_vnir2, smoothed_vnir2, dark_flat_vnir2, dark_flat_smooth_vnir2 = preprocessing_bil(
                raw_vnir2, dark_vnir2, white=None, mode='VNIR2'
            )
            denoised_vnir2 = pad_cube_to_shape(np.fliplr(denoised_vnir2))
            corrected_vnir2 = pad_cube_to_shape(np.fliplr(corrected_vnir2))
            repaired_vnir2 = pad_cube_to_shape(np.fliplr(repaired_vnir2))
            smoothed_vnir2 = pad_cube_to_shape(np.fliplr(smoothed_vnir2))
            raw_vnir2_aligned = pad_cube_to_shape(np.fliplr(raw_vnir2.astype(np.float32)))

            band_index_leaf = 290
            band_index_root = 130
            band_index_root_visible = 180

            # create output directory
            output_dir = Path(str(vnir_2_bil_path).replace("Measurement", "Analysis")).with_suffix('')
            output_dir.mkdir(parents=True, exist_ok=True)

            run_params = {
                "analysis_type": "process_vnir_with_root_masks",
                "input_files": {
                    "raw_bil": vnir_2_bil_path,
                    "raw_hdr": vnir_2_bil_path.with_suffix('.hdr'),
                    "dark_bil": dark_ref_vnir2,
                    "white_bil": white_ref_vnir2,
                },
                "matching_root_masks": [str(p) for p in matching_masks],
                "preprocessing": {
                    "mode": "VNIR2",
                    "dark_matching": "average calibration over dark rows only, preserve width x band structure, repeat across raw rows",
                    "zero_clipping": "disabled in analysis path; applied only for visualization/export",
                    "invalid_pixel_repair": "band-wise nearest-valid fill followed by 3x3 local median repair",
                    "spectral_smoothing": {"method": "Savitzky-Golay", "window_length": 11, "polyorder": 2, "passes": 1},
                    "spatial_denoising": {"method": "median_filter", "size": [3, 3]},
                    "flip_horizontal": True,
                    "pad_shape": [780, 960],
                },
                "band_selection": {
                    "leaf_band_index": band_index_leaf,
                    "root_band_index": band_index_root,
                    "root_visible_band_index": band_index_root_visible,
                    "export_all_bands": export_all_bands,
                    "export_band_indices": export_band_indices,
                },
                "roi_definition": {
                    "peri_root_binary_dilation_iterations": 10,
                    "remove_pixels_above_root_top": True,
                },
                "outputs": {
                    "export_pixel_spectra_csv": export_pixel_spectra_csv,
                    "save_preprocessing_plots": save_preprocessing_plots,
                    "write_master_file": write_master_file,
                },
            }
            save_run_parameters(output_dir, run_params)

            # export the quality control plots
            qc_plot_spectra(
                output_dir,
                wavelengths=wavelengths_vnir2,
                raw=raw_vnir2_aligned,
                white=pad_cube_to_shape(np.fliplr(white_vnir2.astype(np.float32))),
                dark=pad_cube_to_shape(np.fliplr(expand_calibration_to_match(dark_vnir2, raw_vnir2.shape))),
                corrected=corrected_vnir2,
                repaired=repaired_vnir2,
                smoothed=smoothed_vnir2,
                denoised=denoised_vnir2,
                dark_flat=dark_flat_vnir2,
                dark_flat_smooth=dark_flat_smooth_vnir2
            )

            # Export only analysis bands by default; set export_all_bands=True for a full cube PNG dump.
            band_export_indices = export_band_indices or [band_index_root, band_index_root_visible, band_index_leaf]
            export_bands(
                denoised_vnir2,
                output_dir,
                export_all_bands=export_all_bands,
                target_band=band_export_indices,
                wavelengths=wavelengths_vnir2,
            )

            # extract leaf and root specific band images for later alignement
            band_leaf = denoised_vnir2[:, :, band_index_leaf]
            band_leaf_vis = np.clip(band_leaf, 0, None)
            vmin, vmax = np.nanmin(band_leaf_vis), np.nanmax(band_leaf_vis)
            band_leaf = np.zeros_like(band_leaf_vis) if vmax - vmin == 0 else np.clip((band_leaf_vis - vmin) / (vmax - vmin), 0, 1)
            fixed_band_leaf = (band_leaf * 255).astype(np.uint8)
            threshold = threshold_otsu(fixed_band_leaf)
            fixed_band_leaf_binary = fixed_band_leaf > threshold
            fixed_band_leaf_binary = (fixed_band_leaf_binary * 255).astype(np.uint8)

            band_root = denoised_vnir2[:, :, band_index_root]
            band_root = white_tophat(band_root, disk(15))
            band_root_vis = np.clip(band_root, 0, None)
            vmin, vmax = np.nanmin(band_root_vis), np.nanmax(band_root_vis)
            band_root = np.zeros_like(band_root_vis) if vmax - vmin == 0 else np.clip((band_root_vis - vmin) / (vmax - vmin), 0, 1)
            fixed_band_root = (band_root * 255).astype(np.uint8)
            threshold = threshold_otsu(fixed_band_root)
            fixed_band_root_binary = fixed_band_root > threshold
            fixed_band_root_binary = (fixed_band_root_binary * 255).astype(np.uint8)

            band_root_visible = denoised_vnir2[:, :, band_index_root_visible]
            band_root_visible_vis = np.clip(band_root_visible, 0, None)
            vmin, vmax = np.nanmin(band_root_visible_vis), np.nanmax(band_root_visible_vis)
            band_root_visible = np.zeros_like(band_root_visible_vis) if vmax - vmin == 0 else np.clip((band_root_visible_vis - vmin) / (vmax - vmin), 0, 1)

            # # load a process VNIR1
            # # load hypercube, white and dark reference for VNIR2
            # raw_vnir1, wavelengths_vnir1 = open_bil_with_temp_envi_header(matching_vnir_1_bil_path, matching_vnir_1_bil_path.with_suffix('.hdr'))
            # dark_ref_vnir1 = matching_vnir_1_bil_path.parent / matching_vnir_1_bil_path.name.replace("_Data.bil", "_DarkCalibration.bil")
            # white_ref_vnir1 = matching_vnir_1_bil_path.parent / matching_vnir_1_bil_path.name.replace("_Data.bil", "_WhiteCalibration.bil")
            # dark_vnir1, _ = open_bil_with_temp_envi_header(dark_ref_vnir1, dark_ref_vnir1.with_suffix('.hdr'))
            # white_vnir1, _ = open_bil_with_temp_envi_header(white_ref_vnir1, white_ref_vnir1.with_suffix('.hdr'))
            #
            # # preprocess the cube, flip horzontally and pad it to shape if necessary
            # denoised_vnir1, cleaned_vnir1, smoothed_vnir1, dark_flat_vnir1, dark_flat_smooth_vnir1 = preprocessing_bil(raw_vnir1, dark_vnir1, white=white_vnir1, mode = 'VNIR1')
            # denoised_vnir1 = np.fliplr(denoised_vnir1)
            # denoised_vnir1 = pad_cube_to_shape(denoised_vnir1)
            #
            # # create output directory
            # output_dir_vnir1 = Path(str(matching_vnir_1_bil_path).replace("Measurement", "Analysis")).with_suffix('')
            # output_dir_vnir1.mkdir(parents=True, exist_ok=True)
            #
            # # export the quality control plots
            # qc_plot_spectra(
            #     output_dir_vnir1,
            #     wavelengths=wavelengths_vnir1,
            #     raw=raw_vnir1,
            #     white=white_vnir1,
            #     dark=dark_vnir1,
            #     cleaned=cleaned_vnir1,
            #     smoothed=smoothed_vnir1,
            #     denoised=denoised_vnir1,
            #     dark_flat=dark_flat_vnir1,
            #     dark_flat_smooth=dark_flat_smooth_vnir1
            # )
            #
            # # Loop through all bands
            # export_bands(denoised_vnir1, output_dir_vnir1, export_all_bands=True, target_band=160)
            # band_index = 160
            # band = denoised_vnir2[:, :, band_index]
            # vmin, vmax = np.nanmin(band), np.nanmax(band)
            # band_img = np.zeros_like(band) if vmax - vmin == 0 else np.clip((band - vmin) / (vmax - vmin), 0, 1)
            # band_img_8bit = (band_img * 255).astype(np.uint8)

            for matching_mask in matching_masks:
                root_mask = load_binary_mask(matching_mask)
                leaf_mask_link = re.sub("root_mask.png", "shoot_mask.png", str(matching_mask))
                leaf_mask = load_binary_mask(Path(leaf_mask_link))
                related_mask_paths = {
                    "main_root_mask": matching_mask.with_name("main_root_mask.png"),
                    "lateral_root_mask": matching_mask.with_name("lateral_root_mask.png"),
                    "node_mask": matching_mask.with_name("node_mask.png"),
                    "tip_mask": matching_mask.with_name("tip_mask.png"),
                }

                match = re.search(r"plant_[1-5]", str(matching_mask))
                plant_name = match.group() if match else "unknown_plant"
                plant_folder = output_dir / plant_name
                plant_folder.mkdir(parents=True, exist_ok=True)

                plant_run_params = {
                    "analysis_type": "plant_roi_analysis",
                    "plant_name": plant_name,
                    "source_vnir2_bil": vnir_2_bil_path,
                    "root_mask_path": matching_mask,
                    "leaf_mask_path": leaf_mask_link,
                    "related_root_mask_paths": {name: str(path) for name, path in related_mask_paths.items()},
                    "leaf_alignment": {
                        "initial_offset_yx": [-56, -205],
                        "target_size_yx": [901, 1370],
                        "search_radius": 50,
                        "fine_max_shift": 50,
                        "fine_score": "iou",
                    },
                    "root_alignment": {
                        "method": "ecc_affine",
                        "input": "leaf-aligned root mask refined against root-enhanced VNIR band",
                        "ecc_iterations": 300,
                        "ecc_gaussian_kernel": 5,
                        "fallback_method": "grid_iou",
                        "fallback_initial_offset_yx": [0, 0],
                        "fallback_target_size_yx": [780, 960],
                        "fallback_search_radius": 50,
                        "fallback_fine_max_shift": 50,
                        "fallback_fine_score": "iou",
                    },
                    "roi_definition": {
                        "peri_root_binary_dilation_iterations": 10,
                        "remove_pixels_above_root_top": True,
                    },
                }

                # Estimate on leaf (moving) vs fixed (reference)
                leaf_params, aligned_leaf, (dy_best, dx_best), heatmap = estimate_alignment_from_reference(
                    fixed_band_leaf_binary,
                    leaf_mask,
                    initial_offset_yx=(-56, -205),  # (y, x)
                    target_size_yx=(901, 1370),  # (H, W)
                    search_radius=50,
                    fine_max_shift=50,
                    fine_score="iou",
                    return_heatmap=False,
                    save_heatmap_path=None,
                    heatmap_title="Shift score heatmap")
                leaf_params["fixed_shape"] = fixed_band_leaf.shape

                #initial_offset_yx = (-32, -134),  # (y, x)
                #target_size_yx = (869, 1322),  # (H, W)

                # Use a skeleton-aware leaf alignment for the root mask before
                # estimating the final ECC affine root refinement.
                aligned_root_coarse = align_skeleton_mask_to_leaf(root_mask, leaf_params)
                #out_path = plant_folder / f"{plant_name}_aligned_leaf_without.png"
                #Image.fromarray(aligned_root).save(out_path)

                # second alignment using the root mask: default to ECC affine,
                # then fall back to the previous grid IoU search if ECC fails.
                heatmap_outputlink = plant_folder / f"{plant_name}_AlignScoreRoot.png"
                root_params, aligned_root = estimate_root_alignment_default(
                    fixed_band_root,
                    fixed_band_root_binary,
                    aligned_root_coarse,
                    fallback_heatmap_path=heatmap_outputlink,
                    fallback_search_radius=50,
                    fallback_fine_max_shift=50,
                    fallback_fine_score="iou",
                )
                root_output_shape = tuple(root_params.get("fixed_shape", fixed_band_root.shape))
                aligned_root = align_skeleton_mask(root_mask, leaf_params, root_params, root_output_shape)
                aligned_root = (aligned_root * 255).astype(np.uint8)

                aligned_trait_masks, landmark_object_counts = align_related_root_masks(
                    related_mask_paths,
                    leaf_params,
                    root_params,
                )

                plant_run_params["leaf_alignment"]["estimated_transform"] = leaf_params
                plant_run_params["root_alignment"]["estimated_transform"] = root_params
                plant_run_params["related_root_masks"] = {
                    "alignment_reuse": "leaf_alignment estimated from shoot_mask, then root_alignment estimated from root_mask",
                    "root_mask_source_pixel_count": int(np.count_nonzero(root_mask)),
                    "root_mask_aligned_pixel_count": int(np.count_nonzero(aligned_root)),
                    "source_pixel_counts": {
                        name: int(np.count_nonzero(load_binary_mask(path)))
                        for name, path in related_mask_paths.items()
                        if path.exists()
                    },
                    "source_object_counts": {
                        name: count_mask_objects(path)
                        for name, path in related_mask_paths.items()
                        if path.exists()
                    },
                    "aligned_pixel_counts": {
                        name: int(np.count_nonzero(mask)) for name, mask in aligned_trait_masks.items()
                    },
                    "landmark_object_counts_preserved": landmark_object_counts,
                }
                save_run_parameters(plant_folder, plant_run_params)

                # use aligned root mask for ROI spectral extraction
                dilated_zone = binary_dilation(aligned_root, iterations=10)
                dilation_zone_exclusive = dilated_zone.copy()
                dilation_zone_exclusive[aligned_root == 255] = 0
                root_y_top = np.min(np.argwhere(aligned_root)[:, 0]) if np.any(aligned_root) else 0
                dilation_zone_exclusive[:root_y_top, :] = 0
                out_path = plant_folder / f"{plant_name}_OverlayQualityCheck.png"
                save_alignment_overlay(
                    band_root_visible,
                    out_path,
                    aligned_root=aligned_root,
                    trait_masks=aligned_trait_masks,
                    dilation_zone=dilation_zone_exclusive,
                )
                print(f"✅ Saved overlay to: {out_path}")

                diagnostic_roi_masks = {
                    'root': aligned_root.astype(bool),
                    'peri_root': dilation_zone_exclusive.astype(bool),
                }
                diagnostic_roi_masks.update({
                    name: mask.astype(bool) for name, mask in aligned_trait_masks.items()
                })

                save_preprocessing_diagnostics(
                    output_dir=plant_folder,
                    raw_cube=raw_vnir2_aligned,
                    processed_cube=denoised_vnir2,
                    wavelengths=wavelengths_vnir2,
                    reference_band=band_index_root_visible,
                    roi_masks=diagnostic_roi_masks,
                    prefix=plant_name,
                    save_plot_images=save_preprocessing_plots,
                )

                # For exclusive dilation zone
                summarize_and_save_pixel_data(
                    bil_path=vnir_2_bil_path,
                    plant_folder=plant_folder,
                    cube=denoised_vnir2,
                    aligned_mask=dilation_zone_exclusive,
                    wavelengths=wavelengths_vnir2,
                    plant_name=plant_name + "_dilatedzone",
                    root_mask_path=matching_mask
                )

                summarize_and_save_pixel_data(
                    bil_path=vnir_2_bil_path,
                    plant_folder=plant_folder,
                    cube=denoised_vnir2,
                    aligned_mask=aligned_root,
                    wavelengths=wavelengths_vnir2,
                    plant_name=plant_name + "_rootmask",
                    root_mask_path=matching_mask
                )

                for mask_name, aligned_trait_mask in aligned_trait_masks.items():
                    summarize_and_save_pixel_data(
                        bil_path=vnir_2_bil_path,
                        plant_folder=plant_folder,
                        cube=denoised_vnir2,
                        aligned_mask=aligned_trait_mask,
                        wavelengths=wavelengths_vnir2,
                        plant_name=f"{plant_name}_{mask_name}",
                        root_mask_path=related_mask_paths[mask_name]
                    )

                if export_pixel_spectra_csv:
                    export_pixel_spectra(
                        bil_path=vnir_2_bil_path,
                        plant_folder=plant_folder,
                        cube=denoised_vnir2,
                        aligned_mask=dilation_zone_exclusive,
                        wavelengths=wavelengths_vnir2,
                        plant_name=plant_name + "_dilatedzone",
                        root_mask_path=matching_mask,
                        save_as="csv",
                        max_csv_pixels=None)

                    export_pixel_spectra(
                        bil_path=vnir_2_bil_path,
                        plant_folder=plant_folder,
                        cube=denoised_vnir2,
                        aligned_mask=aligned_root,
                        wavelengths=wavelengths_vnir2,
                        plant_name=plant_name + "_rootmask",
                        root_mask_path=matching_mask,
                        save_as="csv",
                        max_csv_pixels=None)

                    for mask_name, aligned_trait_mask in aligned_trait_masks.items():
                        export_pixel_spectra(
                            bil_path=vnir_2_bil_path,
                            plant_folder=plant_folder,
                            cube=denoised_vnir2,
                            aligned_mask=aligned_trait_mask,
                            wavelengths=wavelengths_vnir2,
                            plant_name=f"{plant_name}_{mask_name}",
                            root_mask_path=related_mask_paths[mask_name],
                            save_as="csv",
                            max_csv_pixels=None)

                # ➕ Handle NEW zone based on previous mask

                #prev_candidates = [p for p in previous_masks if plant_name in str(p)] if previous_masks else []

                #if prev_candidates:
                #    prev_mask = skimage.io.imread(prev_candidates[0]) > 127
                #    prev_dilated = binary_dilation(prev_mask, iterations=5)
                #    aligned_prev = align_masks(band_img_8bit, prev_dilated)
                #    dilated_prev = binary_dilation(aligned_prev, iterations=10)
                #    new_zone_exclusive = dilated_prev.copy()
                #    new_zone_exclusive = new_zone_exclusive & (~dilation_zone_exclusive)
                #    root_y_top = np.min(np.argwhere(new_growth)[:, 0]) if np.any(new_growth) else 0
                #    new_zone_exclusive[:root_y_top, :] = 0
                #    new_border = new_zone_exclusive ^ binary_erosion(new_zone_exclusive, iterations=1)
                #    band_rgb[new_border] = [0, 255, 0]  # Green for new growth
                #    Image.fromarray(band_rgb).save(plant_folder / f"{plant_name}_overlay_new_growth_band{band_index}.png")

                #    summarize_and_save_pixel_data(bil_path=bil_path,
                #        plant_folder=plant_folder,
                #        cube=denoised,
                #        aligned_mask=new_zone_exclusive,
                #        wavelengths=wavelengths,
                #        plant_name=plant_name + "_new_growth",
                #        root_mask_path=matching_mask)

            if write_master_file:
                master_path = write_master_summary(output_dir)
                if master_path is not None:
                    print(f"Saved master summary to: {master_path}")
                if export_pixel_spectra_csv:
                    pixel_master_path = write_master_pixel_spectra(output_dir)
                    if pixel_master_path is not None:
                        print(f"Saved master pixel spectra to: {pixel_master_path}")

        except Exception as e:
            print(f"❌ Error processing {vnir_2_bil_path.name}: {e}")
