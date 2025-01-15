# complex.py
from argparse import ArgumentParser
from .kbc.src import preprocess_datasets
from .kbc.src import main as kbc_main
import sys

if __name__ == '__main__':
    # Create the top-level parser
    parser = ArgumentParser(description="Manage KBC-related functions")
    parser.add_argument("command", choices=["preprocess", "train"], help="Command to execute")
    args, unknown_args = parser.parse_known_args()

    # Replace sys.argv with the remaining arguments
    sys.argv = [sys.argv[0]] + unknown_args
    if args.command == "preprocess":
        preprocess_datasets.main()
    elif args.command == "train":
        kbc_main.main()
