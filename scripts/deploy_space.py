"""Deploy ClassCritique to a Hugging Face Space (Docker SDK).

Creates the Space if needed, uploads exactly the files git would publish (never data/, .env, anchors/style,
caches or source images), and stores the Groq key as an encrypted Space SECRET read from your local .env, so it
is never part of the code.
Model names are set as public Space variables.

    1. Create a free account at https://huggingface.co and an access token with "write" permission
       (Settings -> Access Tokens). Put it in .env as HF_TOKEN=...
    2. python scripts/deploy_space.py      (the Space is named 'classcritique' under your account)
    3. Open https://huggingface.co/spaces/<your-username>/classcritique (first build takes a few minutes).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
SECRETS = ["GROQ_API_KEY"]
VARIABLES = ["PROVIDER", "HOSTED_PROVIDER_APPROVED", "TRANSCRIBE_MODEL", "CONDENSE_MODEL",
             "CONDENSE_PARALLEL_MODELS", "CONDENSE_FALLBACK_MODELS", "JUDGE_MODEL", "JUDGE_FALLBACK_MODELS"]


def publishable_files() -> list[str]:
    """Exactly the files git would publish (tracked + new, minus .gitignore). This keeps the Space identical
    to the GitHub repo and guarantees that data/, .env, anchors/style files and caches are never uploaded."""
    import subprocess

    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [f for f in out.splitlines() if f and (ROOT / f).is_file() and f != "fh.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="classcritique", help="<space-name> or <username>/<space-name>")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    args = ap.parse_args()

    env = {**dotenv_values(ROOT / ".env"), **{k: v for k, v in os.environ.items() if k in ("HF_TOKEN",)}}
    token = next((v for k, v in env.items() if k.upper() == "HF_TOKEN" and v), None)  # any capitalisation
    if not token:
        sys.exit("HF_TOKEN missing: add a Hugging Face write token to .env (see the docstring).")
    if not env.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY missing in .env.")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    if "/" not in args.space:  # just a name: put it under the token owner's account
        args.space = f"{api.whoami()['name']}/{args.space}"
    api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
    print(f"Space ready: https://huggingface.co/spaces/{args.space}")

    for key in SECRETS:
        api.add_space_secret(args.space, key, env[key])          # encrypted, never shown publicly
    for key in VARIABLES:
        if env.get(key):
            api.add_space_variable(args.space, key, env[key])
    api.add_space_variable(args.space, "HOSTED_PROVIDER_APPROVED", "true")
    print("Secrets and variables set:", ", ".join(SECRETS + [k for k in VARIABLES if env.get(k)]))

    files = publishable_files()
    assert not any(f.startswith("data/") or f.endswith((".env", "anchors.json", "style.json")) for f in files)
    api.upload_folder(folder_path=str(ROOT), repo_id=args.space, repo_type="space", allow_patterns=files,
                      commit_message="Deploy ClassCritique")
    print(f"Uploaded {len(files)} files. The Space builds now: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
