import sys

from train_benign import main as train_benign_main


def main():
    train_benign_main([*sys.argv[1:], "--dataset", "gtsrb"])


if __name__ == "__main__":
    main()
