"""Modular HPX HADES hyperspectral root-analysis pipeline."""

__all__ = [
    "process_vnir_with_root_masks",
    "process_all_images",
    "analyze_segmentation_from_mask",
    "build_master_summary",
    "write_master_summary",
    "write_master_pixel_spectra",
]


def __getattr__(name):
    if name in {"process_vnir_with_root_masks", "process_all_images", "analyze_segmentation_from_mask"}:
        from . import pipeline

        return getattr(pipeline, name)
    if name in {"build_master_summary", "write_master_summary", "write_master_pixel_spectra"}:
        from . import master

        return getattr(master, name)
    raise AttributeError(f"module 'hpx_hades' has no attribute {name!r}")
