# 数据处理流程

本目录下的脚本用于处理原始的中文维基百科数据。

## 文件生成路径

1.  **原始数据**:
    *   `zhwiki-latest-pages-articles.xml`: 从维基百科下载的原始文章数据，已解压。
    *   `zhwiki-latest-categorylinks.sql.gz`: 从维基百科下载的分类链接数据。

2.  **`auto_wiki_parser.py`**:
    *   **输入**: `zhwiki-latest-pages-articles.xml`
    *   **输出**: `zhwiki.jsonl`
    *   **作用**: 解析 XML 文件，提取文章内容，并将其转换为 jsonl 格式。

3.  **`clean_brackets.py`**:
    *   **输入**: `zhwiki.jsonl`
    *   **输出**: `zhwiki_clean.jsonl`
    *   **作用**: 清理 `zhwiki.jsonl` 文件中的多余括号和格式，得到更干净的文本数据。
