import sys

from eval_refine import main as eval_refine_main


def main():
    eval_refine_main([*sys.argv[1:], "--dataset", "gtsrb"])


if __name__ == "__main__":
    main()
