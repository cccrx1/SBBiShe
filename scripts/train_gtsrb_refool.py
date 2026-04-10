import sys

from train_attack import main as train_attack_main


def main():
    train_attack_main([*sys.argv[1:], "--dataset", "gtsrb", "--attack", "refool"])


if __name__ == "__main__":
    main()
