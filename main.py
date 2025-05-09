import torch
import torch.optim as optim
import os
import argparse
from scripts.utils import read_config
import sys

def main():

    parser = argparse.ArgumentParser(description="Paddy ML Project")
    parser.add_argument(
        "--task",
        required=True,
        choices=["train", "eval"],
        help="Main operation"
    )

    parser.add_argument(
        "--target",
        required=True,
        choices=["age", "classify_diseases", "variety_id"],
        help="Task target"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the config YAML"
    )

    args = parser.parse_args()
    config = read_config(args.config)

    try:
        # Dynamically import and call function based on task and target
        module_path = f"scripts.{args.task}.{args.task}_{args.target}"
        function_name = f"{args.task}_{args.target}_model"

        module = __import__(module_path, fromlist=[function_name])
        task_function = getattr(module, function_name)
        task_function(config)

    except ImportError:
        print(f"[ERROR] Could not import module: {module_path}")
        sys.exit(1)

    except AttributeError:
        print(f"[ERROR] Function '{function_name}' not found in '{module_path}'")
        sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Failed to execute {args.task}_{args.target}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()