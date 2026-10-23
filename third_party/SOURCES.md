# Third-party code (copied unchanged)

| File | Upstream | Commit | SHA-256 of file | License |
|---|---|---|---|---|
| `MAGN/uvmd.py` | https://github.com/OsamaAhmad369/MAGN | `6bfae1bf3ebf5af4684382cd12ba23d28e3a6ae8` (2025-12-10) | `32f25dc8b9cb9241fc53e6fabd3a06be4fa0e48b524f7b7c013173e435c8d798` | **none stated in the upstream repo** |
| `MAGN/README_MAGN.md` | same repo, `README.md` | same | – | same |
| `vmdpy/vmdpy.py` | https://github.com/vrcarva/vmdpy | `47ca3e83f3d5acb36d6eb68c224cbcdba9cca098` | `4d1b294bb7e699e2c0b481a358254aa042f51394996048196b5f9f3953fda9ee` | MIT (`vmdpy/LICENSE.txt`) |

`MAGN/uvmd.py` is the UVMD code of

> O. Ahmad, L. Wesemann, F. Waschkowski, Z. Khalid,
> *Robust Spatiotemporal Forecasting Using Adaptive Deep-Unfolded Variational Mode Decomposition*,
> arXiv:2509.00703 (2025). LUMS / Maincode.

The upstream repository has no LICENSE file, so all rights stay with its authors. Keep this repository
private, or ask the authors before publishing it.

`tests/test_vendored_files.py` checks the hashes above, so any edit to these files fails the test suite.
The benchmark never edits `uvmd.py`. `decomp/uvmd_magn.py` loads the model classes from it at run time
and skips the script part (argument parsing, data loading, training on `his.npz`).
