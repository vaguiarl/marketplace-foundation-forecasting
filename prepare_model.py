"""Fetch only the pinned official model configuration/weights and verify them."""
from hashlib import sha256
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
EXPECTED = "ddcda3c7508bf2528087723e98a20707cc04b7f370ae275a9fd88078ddba4f42"

if __name__ == "__main__":
    folder = Path(snapshot_download("amazon/chronos-2", revision=REVISION,
                                   allow_patterns=["config.json", "model.safetensors"]))
    actual = sha256((folder / "model.safetensors").read_bytes()).hexdigest()
    if actual != EXPECTED:
        raise RuntimeError("Pinned model failed SHA256 verification")
    print(f"Pinned official Chronos-2 model ready: {folder}")
