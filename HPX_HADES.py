# Purpose: Hyperspectral Fluorescence Root measurement HADES system
# Author: Valerian Meline
# Notes: Compatibility wrapper around the modular hpx_hades package.

from hpx_hades.io import *
from hpx_hades.preprocessing import *
from hpx_hades.diagnostics import *
from hpx_hades.utils import *
from hpx_hades.exports import *
from hpx_hades.alignment import *
from hpx_hades.pipeline import *


if __name__ == "__main__":
    vnir2_root_dir = r"E:\HADES_HPX_04-2026\VNIR2\Measurement"
    root_mask_dir = r"E:\HADES_HPX_04-2026\ROOT2_analysis"

    process_vnir_with_root_masks(vnir2_root_dir, root_mask_dir)
