"""Restore archived public models at their recorded Hugging Face revisions."""
from pathlib import Path
import hashlib, json, shutil, tempfile
from huggingface_hub import hf_hub_download
root = Path(__file__).resolve().parent
manifest = json.loads((root / "model-manifest.json").read_text())
for model in manifest["models"]:
    target = root / model["local_dir"]
    with tempfile.TemporaryDirectory(prefix="restore-", dir=root) as tmp:
        for item in model["files"]:
            dest = target / item["local_path"]
            if dest.exists() and dest.stat().st_size == item["bytes"]:
                if len(item["etag"]) != 64 or hashlib.file_digest(dest.open("rb"), "sha256").hexdigest() == item["etag"]:
                    continue
            src = Path(hf_hub_download(repo_id=model["repo_id"], revision=item["revision"], filename=item["remote_path"], local_dir=tmp))
            if src.stat().st_size != item["bytes"]:
                raise RuntimeError(f"Size mismatch: {src}")
            if len(item["etag"]) == 64:
                with src.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != item["etag"]:
                        raise RuntimeError(f"Hash mismatch: {src}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            src.unlink()
    print(f"Restored {model['repo_id']} -> {target}")
