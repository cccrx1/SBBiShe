import sys

from train_refine import main as train_refine_main


def main():
    train_refine_main([*sys.argv[1:], "--dataset", "gtsrb"])


if __name__ == "__main__":
    main()
