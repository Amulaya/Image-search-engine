import argparse
import sys
from pathlib import Path

from src.cluster_generator import ClusterGenerator


def execute_create_cluster_cmd(dataset_dir_path: Path, results_dir_path: Path):
    if not dataset_dir_path.exists():
        print("Dataset directory not found: {}".format(dataset_dir_path.as_posix()))
        sys.exit(1)

    if not results_dir_path.exists():
        print("Results directory not found: {}".format(dataset_dir_path.as_posix()))
        sys.exit(1)

    ClusterGenerator().execute(dataset_dir_path, results_dir_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to the directory that contains the images to be clustered")
    ap.add_argument("-r", "--results", required=True,
                    help="Path to where clustered results will be stored")
    args = vars(ap.parse_args())
    print(args)
    execute_create_cluster_cmd(Path(args["dataset"]), Path(args["results"]))


if __name__ == "__main__":
    main()
