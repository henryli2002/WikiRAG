import os
import sys
import gzip
import re
import json
import subprocess
import shutil
import tempfile
import multiprocessing as mp
from pathlib import Path


def setup_dependencies():
    """Ensure required packages are installed."""
    try:
        import opencc
        import wikiextractor
        # 简单检查一下是不是包含修复的版本（比如查看 extract.py 中是否还有异常写法）
    except ImportError:
        print(
            "正在安装必要依赖 (wikiextractor 社区修复版, opencc-python-reimplemented)..."
        )
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "git+https://github.com/santhoshtr/wikiextractor.git",
                "opencc-python-reimplemented",
                "-q",
            ]
        )


def extract_categories(sql_gz_path):
    """
    流式读取 categorylinks.sql.gz，构建 page_id -> [categories] 的字典。
    仅耗费少部分内存。
    """
    print(f"开始解析分类映射文件: {sql_gz_path} ... (这可能需要几分钟，请耐心等待)")
    category_map = {}
    # 匹配 SQL 中的 VALUES (page_id, 'category_name', ...)
    # 匹配数字 id，然后是单引号包裹的内容，内容可以是任意非单引号字符，或者是反斜杠转义的单引号
    pattern = re.compile(b"\\(([0-9]+),'((?:[^']|\\\\')*)'")

    try:
        with gzip.open(sql_gz_path, "rb") as f:
            for line in f:
                if isinstance(line, bytes) and b"INSERT INTO `categorylinks`" in line:
                    for match in pattern.finditer(line):
                        page_id = match.group(1).decode("ascii")
                        # 类别名称可能会有一些特殊字符，忽略无法解码的字符，并将下划线还原为空格，同时处理转义单引号
                        cat_name = (
                            match.group(2)
                            .decode("utf-8", errors="ignore")
                            .replace("\\'", "'")
                            .replace("_", " ")
                        )

                        if page_id not in category_map:
                            category_map[page_id] = []
                        category_map[page_id].append(cat_name)
    except Exception as e:
        print(f"解析分类映射失败: {e}")
        sys.exit(1)

    print(f"分类映射构建完成，共包含 {len(category_map)} 个页面的分类。")
    return category_map


def run_wikiextractor(xml_bz2_path, output_dir):
    """
    调用 wikiextractor 解析 XML 压缩包
    """
    cpu_count = os.cpu_count()
    cores = str(max(1, (cpu_count if cpu_count is not None else 2) - 1))
    print(
        f"开始执行 wikiextractor 提取正文: {xml_bz2_path} ... (启用 {cores} 进程并发，通常耗时 10-20 分钟)"
    )
    cmd = [
        sys.executable,
        "-m",
        "wikiextractor.WikiExtractor",
        xml_bz2_path,
        "-o",
        output_dir,
        "--json",  # 输出为 JSON 格式
        "-q",  # 安静模式
        "-b",
        "10M",  # 文件分块大小，减小体积以产生足够多的小文件供下一步的多进程消费
        "--processes",
        cores,
    ]
    try:
        subprocess.run(cmd, check=True)
        print("wikiextractor 提取完成。")
    except subprocess.CalledProcessError as e:
        print(f"wikiextractor 运行失败: {e}")
        sys.exit(1)


# ==============================================================================
# 多进程处理模块
# ==============================================================================

import typing

# 定义全局变量以供 fork 出的子进程直接读取（写时复制，0内存开销传递大字典）
_GLOBAL_CATEGORY_MAP: dict = {}
_OPENCC_CONVERTER: typing.Any = None


def init_worker(category_map):
    global _GLOBAL_CATEGORY_MAP, _OPENCC_CONVERTER
    import opencc

    _GLOBAL_CATEGORY_MAP = category_map
    _OPENCC_CONVERTER = opencc.OpenCC("t2s")


def process_single_file(file_path):
    out_path = file_path + ".out"
    valid_count = 0
    total_count = 0

    with (
        open(file_path, "r", encoding="utf-8") as in_f,
        open(out_path, "w", encoding="utf-8") as out_f,
    ):
        for line in in_f:
            if not line.strip():
                continue

            total_count += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            pageid = str(data.get("id", ""))
            title = data.get("title", "")
            text = data.get("text", "")

            # 过滤 1: 正文长度 < 50
            if len(text) < 50:
                continue

            # 转换标题并进行 过滤 2: 消歧义页面
            s_title = _OPENCC_CONVERTER.convert(title)
            if "(消歧义)" in s_title:
                continue

            # 标题校验通过后，再进行正文的繁简转换（节省不必要的转化开销）
            s_text = _OPENCC_CONVERTER.convert(text)

            # 获取类别并转换
            categories = _GLOBAL_CATEGORY_MAP.get(pageid, [])
            s_categories = [_OPENCC_CONVERTER.convert(c) for c in categories]

            # 组装最终 JSON 对象
            out_data = {
                "pageid": pageid,
                "title": s_title,
                "content": s_text,
                "categories": s_categories,
            }

            # 写入文件
            out_f.write(json.dumps(out_data, ensure_ascii=False) + "\n")
            valid_count += 1

    return file_path, out_path, total_count, valid_count


def process_and_combine(extracted_dir, category_map, output_file):
    """
    使用多进程池遍历 wikiextractor 输出的 JSON 文件，进行繁简转换、过滤和组装
    """
    print("开始进行清洗、繁简转换与组装 (已启用多进程加速)...")

    tasks = []
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if not file.startswith("wiki_"):
                continue
            tasks.append(os.path.join(root, file))

    cpu_count = os.cpu_count()
    num_workers = max(1, (cpu_count if cpu_count is not None else 2) - 1)

    # 在 macOS 上默认使用 spawn 启动进程，但这会导致 category_map (数 GB) 被重复序列化导致爆内存
    # 由于我们在启动进程前未启用任何多线程，这里显式使用 fork 模式来利用操作系统的写时复制(CoW)特性，实现 0 开销内存共享。
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")

    total_processed = 0
    total_valid = 0

    print(f"共发现 {len(tasks)} 个数据块，将使用 {num_workers} 个工作进程全速处理...")

    with ctx.Pool(
        processes=num_workers, initializer=init_worker, initargs=(category_map,)
    ) as pool:
        results = pool.imap_unordered(process_single_file, tasks)

        # 主进程：实时合并各个子进程的处理结果到最终的 zhwiki.jsonl 中
        with open(output_file, "w", encoding="utf-8") as out_f:
            for i, (orig_path, out_path, t_count, v_count) in enumerate(results, 1):
                total_processed += t_count
                total_valid += v_count

                # 将子进程写入的临时文件内容追加到主文件
                with open(out_path, "r", encoding="utf-8") as tmp_f:
                    shutil.copyfileobj(tmp_f, out_f)

                # 追加完毕后删除子进程的临时文件
                os.remove(out_path)

                if i % 10 == 0 or i == len(tasks):
                    print(f"数据清洗进度: {i}/{len(tasks)} 块...")

    print(
        f"数据清洗完毕！总计处理 {total_processed} 条，保留有效条目 {total_valid} 条。"
    )
    print(f"最终输出文件: {os.path.abspath(output_file)}")


def main():
    xml_file = "zhwiki-latest-pages-articles.xml"
    # 可以先用10000条测试数据验证流程，确认无误后再切换回完整数据
    # 命令为 head -n 10000 zhwiki-latest-pages-articles.xml > test.xml
    # xml_file = "test.xml"
    sql_file = "zhwiki-latest-categorylinks.sql.gz"
    output_file = "zhwiki.jsonl"

    if not os.path.exists(xml_file):
        print(f"错误: 当前目录下找不到维基百科正文数据 '{xml_file}'")
        sys.exit(1)
    if not os.path.exists(sql_file):
        print(f"错误: 当前目录下找不到分类映射数据 '{sql_file}'")
        sys.exit(1)

    # 确保依赖
    setup_dependencies()

    # 第一步：构建内存分类字典
    category_map = extract_categories(sql_file)

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="wiki_extracted_")

    try:
        # 第二步：调用 wikiextractor 提取 XML
        run_wikiextractor(xml_file, temp_dir)

        # 第三步：读取提取内容，清洗、转换、结合分类，最终落盘
        process_and_combine(temp_dir, category_map, output_file)
    finally:
        # 第四步：清理生成的临时目录
        print(f"清理临时目录: {temp_dir}")
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("所有任务完成！")


if __name__ == "__main__":
    main()
