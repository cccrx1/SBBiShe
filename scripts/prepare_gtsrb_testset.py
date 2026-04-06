from _common import DEFAULT_DATA_ROOT, prepare_gtsrb_testset, to_path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare raw GTSRB testset into DatasetFolder-compatible layout.")
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    args = parser.parse_args()

    prepared = prepare_gtsrb_testset(args.data_root)
    print(f"Prepared {prepared} test images into class subdirectories under {to_path(args.data_root) / 'GTSRB' / 'testset'}")


if __name__ == "__main__":
    main()
