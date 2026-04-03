from huggingface_hub import snapshot_download
import os

# 定义目标路径（会自动创建）
target_dir = os.path.expanduser("~/Downloads/models/bge-m3")

print(f"正在启动多线程下载，目标目录: {target_dir}")

# 执行下载
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir=target_dir,
    local_dir_use_symlinks=False,  # 确保是真实文件而非链接
    resume_download=True           # 支持断点续传
)

print("✅ 下载完成！所有模型文件已就位。")