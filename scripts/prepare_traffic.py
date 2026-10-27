"""Build a region file (GBA / GLA / SD, 15-minute, year 2019) from the raw LargeST California files.

    python scripts/prepare_traffic.py --raw ca_his_raw_2019.h5 --meta ca_meta.csv --region sd --out ~/data/LargeST/sd_his_2019.h5

Same steps as the authors' notebooks (MAGN repo, data/ca/process_ca_his.ipynb and data/<region>/generate_*_dataset.ipynb):
keep the region's districts, average the 5-minute values into 15-minute bins, round, fill missing values with 0.
Districts: gba = 4, gla = 7/8/12, sd = 11. Needs roughly 10 GB of RAM if the raw file cannot be read column-wise.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.traffic import REGION_DISTRICTS, prepare_region  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="ca_his_raw_2019.h5 (5-minute, all California sensors)")
    p.add_argument("--meta", required=True, help="ca_meta.csv")
    p.add_argument("--region", required=True, choices=sorted(REGION_DISTRICTS))
    p.add_argument("--out", required=True, help="output, e.g. sd_his_2019.h5")
    a = p.parse_args()
    df = prepare_region(a.raw, a.meta, a.region, a.out)
    print(f"{a.region}: {df.shape[1]} sensors x {df.shape[0]} time steps ({df.shape[0] / 96:.0f} days) -> {a.out}")
    print("expected sensors: gba 2352, gla 3834, sd 716 (paper, Sec. IV)")
