from huggingface_hub import snapshot_download
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

MODELS = {
    "BAAI/bge-m3": os.path.join(MODELS_DIR, "bge-m3"),
    "BAAI/bge-reranker-v2-m3": os.path.join(MODELS_DIR, "bge-reranker-v2-m3"),
}

for repo_id, target_dir in MODELS.items():
    print(f"正在下载 {repo_id} → {target_dir}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✅ {repo_id} 下载完成！")

print("\n✅ 所有模型已就位。")
