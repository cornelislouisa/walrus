"""Rewrite the ``dataset_name`` root attribute of the WT_old (128x128) Well files.

The salvaged WT_old exports carry ``drosophila_toll[RM9]_sqh_gfp_piv``, but the
morphogenesis runs that trained on this data logged metrics under
``drosophila_wt_sqh_mcherry_piv``. WellDataset reads the name straight from this
attribute, so analysis keys only line up with the wandb history once it is fixed.

Dry run by default; pass ``--apply`` to write.
"""

import argparse
import pathlib

import h5py

DEFAULT_ROOT = pathlib.Path("/data/lcornelis/morphogenesis_data/WT_old")
OLD_NAME = "drosophila_toll[RM9]_sqh_gfp_piv"
NEW_NAME = "drosophila_wt_sqh_mcherry_piv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=DEFAULT_ROOT)
    parser.add_argument("--old-name", default=OLD_NAME)
    parser.add_argument("--new-name", default=NEW_NAME)
    parser.add_argument("--apply", action="store_true", help="write changes to disk")
    args = parser.parse_args()

    files = sorted(args.root.glob("data/*/*.hdf5"))
    if not files:
        raise SystemExit(f"No HDF5 files under {args.root}/data/*/")

    changed = skipped = 0
    for path in files:
        mode = "r+" if args.apply else "r"
        with h5py.File(path, mode) as f:
            current = f.attrs.get("dataset_name")
            if current == args.new_name:
                skipped += 1
                continue
            if current != args.old_name:
                print(f"  SKIP (unexpected name {current!r}): {path.name}")
                skipped += 1
                continue
            if args.apply:
                f.attrs["dataset_name"] = args.new_name
            changed += 1
            print(f"  {'set' if args.apply else 'would set'} {path.parent.name}/{path.name}")

    verb = "updated" if args.apply else "would update"
    print(f"\n{verb} {changed} file(s); {skipped} already correct or skipped")
    if not args.apply:
        print("Dry run only — re-run with --apply to write.")


if __name__ == "__main__":
    main()
