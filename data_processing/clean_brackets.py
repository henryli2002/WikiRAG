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
        print(f"找不到文件: {input_file}")
        return

    # ==========================================
    # 🎯 核心正则武器库
    # ==========================================
    noise_pattern = re.compile(r'[ \t\u3000]*(?:[（(][ \t\u3000,，;；、.。]*[）)]|[《][ \t\u3000,，;；、.。]*[》])')
    leading_punct_pattern = re.compile(r'([（(])[ \t\u3000]*[,，;；、.。]+[ \t\u3000]*')
    trailing_punct_pattern = re.compile(r'[ \t\u3000]*[,，;；、.。]+[ \t\u3000]*([）)])')
    
    html_pattern = re.compile(r'<[/]?(?:templatestyles|ref|br|div|span|sup|sub|math|table|tr|td|th|p)[^>]*>', re.IGNORECASE)
    wiki_lang_pattern1 = re.compile(r'\{[Hh]\|[^}]*\}')
    wiki_lang_pattern2 = re.compile(r'-\{[^}]*\}-')
    dash_pattern = re.compile(r'-{4,}')
    
    # 1. 重定向精准拦截：只匹配纯标点或遇到换行符停止
    redirect_pattern = re.compile(r'[《（(]?\s*#(?:重定向|REDIRECT)[ \t\u3000,，;；、.。》）)]*', re.IGNORECASE)
    
    # 2. 脚注猎杀器 [1], [注 1], [需要引用]
    footnote_pattern = re.compile(r'\[(?:注\s*)?\d+\]|\[需要引用\]')
    
    # 3. 多媒体参数猎杀器 File:, thumb|250px 等
    media_pattern = re.compile(r'(?:File|Image|文件|图像|thumb)[|:][^\s]+', re.IGNORECASE)
    
    # 4. 粗斜体标记清理
    bold_italic_pattern = re.compile(r"'''?")

    # ==========================================
    
    processed_count = 0
    modified_count = 0
    dropped_count = 0 
    
    print(f"🚀 开始执行终极数据清洗与空值拦截: {input_file} -> {output_file} ...")

    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            text = data.get("content", "")
            
            if not text.strip():
                dropped_count += 1
                continue
                
            original_text = text
            
            text = html.unescape(text)
            text = html_pattern.sub('', text)
            text = wiki_lang_pattern1.sub('', text)
            text = wiki_lang_pattern2.sub('', text)
            
            text = redirect_pattern.sub(' ', text)
            text = dash_pattern.sub(' ', text)
            text = footnote_pattern.sub('', text)
            text = media_pattern.sub('', text)
            text = bold_italic_pattern.sub('', text)

            while True:
                new_text = noise_pattern.sub(' ', text)
                if new_text == text: break
                text = new_text

            text = leading_punct_pattern.sub(r'\1', text)
            text = trailing_punct_pattern.sub(r'\1', text)
            text = re.sub(r'[ \t\u3000]{2,}', ' ', text).strip()
            
            if not text or len(text) < 20:
                dropped_count += 1
                continue
                
            data["content"] = text
            if text != original_text:
                modified_count += 1

            categories = data.get("categories", [])
            if categories:
                data["categories"] = list(dict.fromkeys(categories))

            fout.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            processed_count += 1
            if processed_count % 100000 == 0:
                print(f"已处理 {processed_count} 篇有效文章，丢弃了 {dropped_count} 篇空或内容过短的文章...")

    print("-" * 40)
    print(f"✅ 清洗完美收官！")
    print(f"总计保留有效词条: {processed_count} 篇")
    print(f"实际清理并优化正文: {modified_count} 篇")
    print(f"🗑️  成功拦截并丢弃无内容/纯空词条: {dropped_count} 篇")

if __name__ == "__main__":
    final_refine_wiki("zhwiki.jsonl", "zhwiki_clean.jsonl")