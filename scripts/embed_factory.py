import os
import json
import gc
import re
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel

# --- 1. 环境与显存监控初始化 ---
try:
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    nvml_available = True
except Exception as e:
    print(f"NVML 初始化失败，显存监控不可用: {e}")
    nvml_available = False


def get_vram_usage():
    if not nvml_available:
        return 0
    info = nvmlDeviceGetMemoryInfo(handle)
    return info.used / 1024**3  # 返回 GB


def clean_text(text: str) -> str:
    """清理多余换行符及特殊字符，保持语义纯净"""
    if not text:
        return ""
    text = text.replace("\r", "")
    # 过滤掉非必要的特殊符号，保留中英文及数字
    text = "\n".join([line.strip() for line in text.split("\n") if line.strip()])
    return text


# --- 2. 断点续传与文件序号管理 ---
def get_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            try:
                return int(f.read().strip())
            except ValueError:
                return 0
    return 0


def save_checkpoint(checkpoint_file, current_line):
    tmp_file = checkpoint_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        f.write(str(current_line))
    os.replace(tmp_file, checkpoint_file)


def get_next_file_index(output_dir):
    """扫描目录，自动获取下一个不冲突的文件序号"""
    existing_files = [f for f in os.listdir(output_dir) if f.endswith(".parquet")]
    if not existing_files:
        return 0
    try:
        # 提取 wiki_embeddings_0042.parquet 中的 42
        indices = [
            int(re.findall(r"\d+", f)[-1])
            for f in existing_files
            if re.findall(r"\d+", f)
        ]
        return max(indices) + 1 if indices else 0
    except Exception:
        return len(existing_files)


# --- 3. 主程序 ---
def main():
    # 配置参数
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    input_file = os.path.join(PROJECT_ROOT, "data", "raw", "zhwiki_clean.jsonl")
    output_dir = os.path.join(PROJECT_ROOT, "data", "parquet_output")
    checkpoint_file = os.path.join(PROJECT_ROOT, "data", "checkpoint.txt")
    model_path = os.path.join(PROJECT_ROOT, "models", "bge-m3")

    batch_size = 128  # 3070 8G 显存的稳健配置
    save_interval = 12800  # 每 12800 个 chunk 保存一个 Parquet
    vector_dim = 1024  # BGE-M3 全量维度

    os.makedirs(output_dir, exist_ok=True)

    # 模型初始化 (强制 fp16)
    print(f"正在加载 BGE-M3 模型 (使用 fp16 推理, 维度 {vector_dim})...")
    model = BGEM3FlagModel(model_path, use_fp16=True)

    # 分块器
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=60,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )

    # 严格 Schema 定义: 显式指定 fp16 存储
    schema = pa.schema(
        [
            pa.field("pageid", pa.string()),
            pa.field("title", pa.string()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float16(), vector_dim)),
        ]
    )

    start_line = get_checkpoint(checkpoint_file)
    file_index = get_next_file_index(output_dir)
    print(f"从行号 {start_line} 继续处理，起始文件序号: {file_index:04d}")

    # 预估总行数
    print("正在扫描文件行数...")
    total_lines = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total_lines += 1

    buffer_data = {"pageid": [], "title": [], "content": [], "embedding": []}
    chunk_accumulator = 0

    pbar = tqdm(total=total_lines, initial=start_line, desc="Processing Wiki")

    with open(input_file, "r", encoding="utf-8") as f:
        # 跳过已处理行
        try:
            for _ in range(start_line):
                next(f)
        except StopIteration:
            pass

        batch_texts = []
        batch_metas = []

        for line_idx, line in enumerate(f, start=start_line):
            try:
                data = json.loads(line)
                pageid = str(data.get("pageid", data.get("id", "")))
                title = clean_text(data.get("title", ""))
                raw_text = clean_text(data.get("content", data.get("text", "")))

                if not raw_text:
                    pbar.update(1)
                    continue

                # 执行切片
                chunks = splitter.split_text(raw_text)

                for chunk_text in chunks:
                    # 拼接 Header 增强语义
                    formatted_text = f"条目：{title}\n内容：{chunk_text}"
                    batch_texts.append(formatted_text)
                    batch_metas.append((pageid, title, formatted_text))

                    # 达到 batch_size 开始推理
                    if len(batch_texts) >= batch_size:
                        # 显存监控下的向量计算
                        out = model.encode(
                            batch_texts, batch_size=batch_size, max_length=1024
                        )
                        # 核心：获取 dense_vecs 并强制转为 float16
                        vecs = out["dense_vecs"].astype(np.float16)

                        for i, meta in enumerate(batch_metas):
                            buffer_data["pageid"].append(meta[0])
                            buffer_data["title"].append(meta[1])
                            buffer_data["content"].append(meta[2])
                            # 存入 list 格式，后续由 pyarrow 按照 schema 压回 fp16
                            buffer_data["embedding"].append(vecs[i].tolist())
                            chunk_accumulator += 1

                            # 达到存储间隔，写入 Parquet
                            if chunk_accumulator >= save_interval:
                                out_path = os.path.join(
                                    output_dir,
                                    f"wiki_embeddings_{file_index:04d}.parquet",
                                )
                                table = pa.Table.from_pydict(buffer_data, schema=schema)
                                pq.write_table(table, out_path)

                                # 重置缓冲区
                                buffer_data = {k: [] for k in buffer_data}
                                chunk_accumulator = 0
                                file_index += 1
                                save_checkpoint(checkpoint_file, line_idx + 1)
                                gc.collect()  # 显式回收内存

                        batch_texts = []
                        batch_metas = []

                pbar.update(1)
                if line_idx % 100 == 0:
                    pbar.set_postfix(
                        {"VRAM": f"{get_vram_usage():.2f}GB", "FileIdx": file_index}
                    )

            except Exception as e:
                print(f"\n行 {line_idx} 处理出错: {e}")
                pbar.update(1)
                # 防止异常导致的 batch 堆积
                if len(batch_texts) >= batch_size:
                    batch_texts = []
                    batch_metas = []
                continue

        # --- 4. 收尾工作：处理最后一批数据 ---
        if batch_texts:
            out = model.encode(
                batch_texts, batch_size=len(batch_texts), max_length=1024
            )
            vecs = out["dense_vecs"].astype(np.float16)
            for i, meta in enumerate(batch_metas):
                buffer_data["pageid"].append(meta[0])
                buffer_data["title"].append(meta[1])
                buffer_data["content"].append(meta[2])
                buffer_data["embedding"].append(vecs[i].tolist())
                chunk_accumulator += 1

        if buffer_data["pageid"]:
            out_path = os.path.join(
                output_dir, f"wiki_embeddings_{file_index:04d}.parquet"
            )
            table = pa.Table.from_pydict(buffer_data, schema=schema)
            pq.write_table(table, out_path)
            save_checkpoint(checkpoint_file, total_lines)

    pbar.close()
    print(f"\n🎉 处理完成！所有 Parquet 文件已存入 {output_dir}")


if __name__ == "__main__":
    main()
