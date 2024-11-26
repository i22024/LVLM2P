import argparse
import sys
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
gemini_policy_path = os.path.join(current_dir, "torch_ac", "algos", "vlm_policy", "gemini_policy_2turn.py")

def update_gemini_policy(api_keys):
    """Update gemini_api_keys in gemini_policy_2turn.py file."""

    with open(gemini_policy_path, "r") as file:
        lines = file.readlines()


    for idx, line in enumerate(lines):
        if line.strip().startswith("gemini_api_keys ="):
            lines[idx] = f"gemini_api_keys = {api_keys}\n"
            break

    with open(gemini_policy_path, "w") as file:
        file.writelines(lines)

parser = argparse.ArgumentParser(description="Update Gemini API keys")
parser.add_argument("--api-keys", nargs="+", required=True, help="List of API keys to add")
args = parser.parse_args()

update_gemini_policy(args.api_keys)
print(f"Updated gemini_api_keys in {gemini_policy_path} to: {args.api_keys}")
