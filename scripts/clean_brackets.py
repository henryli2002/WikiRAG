"""
WikiRAG 清理脚本 - 用于清理和规范化维基百科JSON格式数据
主要功能：移除噪音、HTML标签、维基标记、脚注等无关内容
"""

import json
import re
import os
import html


def final_refine_wiki(input_file: str, output_file: str):
    if not os.path.exists(input_file):
        print(f"❌ 找不到输入文件: {input_file}")
        return

    # ==========================================
    # 🎯 核心正则武器库 (Regex Arsenal)
    # ==========================================

    # 1. 纯标点空壳猎杀器：只删里面全是空格、逗号、分号等无意义符号的括号/书名号
    # 只要里面有一个数字(1889)或汉字，就绝对不碰
    noise_pattern = re.compile(
        r"[ \t\u3000]*(?:[（(][ \t\u3000,，;；、.。]*[）)]|[《][ \t\u3000,，;；、.。]*[》])"
    )

    # 2. 括号内边缘标点修剪器：将 (，内容。) 裁为 (内容)
    leading_punct_pattern = re.compile(r"([（(])[ \t\u3000]*[,，;；、.。]+[ \t\u3000]*")
    trailing_punct_pattern = re.compile(
        r"[ \t\u3000]*[,，;；、.。]+[ \t\u3000]*([）)])"
    )

    # 3. HTML 与 维基暗坑清理器 (安全白名单模式)
    html_pattern = re.compile(
        r"<[/]?(?:templatestyles|ref|br|div|span|sup|sub|math|table|tr|td|th|p)[^>]*>",
        re.IGNORECASE,
    )
    wiki_lang_pattern1 = re.compile(r"\{[Hh]\|[^}]*\}")
    wiki_lang_pattern2 = re.compile(r"-\{[^}]*\}-")

    # 4. 虚线与重定向狙击手 (严格边界：遇到标点、换行或右侧包裹符号即止)
    dash_pattern = re.compile(r"-{4,}")
    redirect_pattern = re.compile(
        r"[《（(]?\s*#(?:重定向|REDIRECT)[ \t\u3000,，;；、.。》）)]*", re.IGNORECASE
    )

    # 5. 脚注、多媒体参数与粗斜体
    footnote_pattern = re.compile(r"\[(?:注\s*)?\d+\]|\[需要引用\]")
    media_pattern = re.compile(
        r"(?:File|Image|文件|图像|thumb)[|:][^\s]+", re.IGNORECASE
    )
    bold_italic_pattern = re.compile(r"'''?")

    # ==========================================

    processed_count = 0
    modified_count = 0
    dropped_count = 0

    print(f"🚀 启动全量清洗流水线: {input_file} -> {output_file} ...")

    with (
        open(input_file, "r", encoding="utf-8") as fin,
        open(output_file, "w", encoding="utf-8") as fout,
    ):
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = data.get("content", "")
            if not text.strip():
                dropped_count += 1
                continue

            original_text = text

            # --- 步骤 A: 格式清洗与宏过滤 ---
            text = html.unescape(text)  # &lt; -> <
            text = html_pattern.sub("", text)
            text = wiki_lang_pattern1.sub("", text)
            text = wiki_lang_pattern2.sub("", text)
            text = redirect_pattern.sub(" ", text)
            text = dash_pattern.sub(" ", text)
            text = footnote_pattern.sub("", text)
            text = media_pattern.sub("", text)
            text = bold_italic_pattern.sub("", text)

            # --- 步骤 B: 括号空壳迭代清洗 ---
            while True:
                new_text = noise_pattern.sub(" ", text)
                if new_text == text:
                    break
                text = new_text

            # --- 步骤 C: 边缘标点手术 ---
            text = leading_punct_pattern.sub(r"\1", text)
            text = trailing_punct_pattern.sub(r"\1", text)

            # --- 步骤 D: 空格压缩与空值拦截 ---
            text = re.sub(r"[ \t\u3000]{2,}", " ", text).strip()

            if not text or len(text) < 20:  # 质量门槛：至少20个字符
                dropped_count += 1
                continue

            data["content"] = text
            if text != original_text:
                modified_count += 1

            # --- 步骤 E: 分类字段“原子化净化” (解决 \\n, \n, 脏前缀) ---
            categories = data.get("categories", [])
            if categories:
                all_shards = []
                for cat in categories:
                    # 核心修改：使用正则同时拆分 换行符(\n) 和 字符串形式的双斜杠(\\n)
                    # re.split(r'\\n|\n', cat) 能确保无论它怎么转义，都能切开
                    shards = re.split(r"\\n|\n", cat)
                    all_shards.extend(shards)

                qualified_shards = []
                for s in all_shards:
                    s_strip = s.strip()

                    # 1. 长度门槛：必须大于 1 (干掉 C, F, A 等)
                    # 2. 内容门槛：必须包含汉字、字母或数字 (干掉 ㏼, ★ 等)
                    if len(s_strip) > 1 and re.search(
                        r"[\u4e00-\u9fa5a-zA-Z0-9]", s_strip
                    ):
                        # 3. 额外过滤：如果拆分后依然带有残留的 \ (比如 "C\" )，顺手洗掉
                        s_strip = s_strip.replace("\\", "").strip()
                        if s_strip and s_strip not in qualified_shards:
                            qualified_shards.append(s_strip)

                data["categories"] = qualified_shards

            # 写入结果
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

            processed_count += 1
            if processed_count % 100000 == 0:
                print(
                    f"📊 已扫描 {processed_count} 篇，拦截空词条 {dropped_count} 篇..."
                )

    print("-" * 40)
    print(f"✅ 清洗任务圆满完成！")
    print(f"📦 有效词条数: {processed_count}")
    print(f"🛠️  优化词条数: {modified_count}")
    print(f"🗑️  剔除空词条: {dropped_count}")
    print(f"💾 结果已保存至: {os.path.abspath(output_file)}")


if __name__ == "__main__":
    final_refine_wiki("../data/raw/zhwiki.jsonl", "../data/raw/zhwiki_clean.jsonl")
