import argparse
import tomllib

parser = argparse.ArgumentParser()
parser.add_argument("--tag", "-t", required=False, default="")
args = parser.parse_args()

with open("pyproject.toml", "rb") as f:
    python_version = tomllib.load(f)["project"]["version"]
with open("rust/Cargo.toml", "rb") as f:
    rust_version = tomllib.load(f)["package"]["version"]

if python_version.split(".")[:3] != rust_version.split("."):
    raise ValueError(f"{python_version=} != {rust_version=}")

if args.tag:
    tag = args.tag.replace("v", "")
    if rust_version != tag:
        raise ValueError(f"{rust_version=} != {tag=} ({args.tag=})")
    
