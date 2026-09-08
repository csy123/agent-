# -*-coding: utf-8 -*-
# 对txt文件进行中文分词
import jieba
import os
import sys

# 确保可以导入 utils 模块
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from utils import files_processing

# 源文件所在目录（基于脚本所在位置）
source_folder = os.path.join(SCRIPT_DIR, 'journey_to_the_west', 'source')
segment_folder = os.path.join(SCRIPT_DIR, 'journey_to_the_west', 'segment')

# 创建输出目录（如果不存在）
os.makedirs(segment_folder, exist_ok=True)

print(f"源目录: {source_folder}")
print(f"输出目录: {segment_folder}")

# 字词分割，对整个文件内容进行字词分割
def segment_lines(file_list, segment_out_dir, stopwords=[]):
    if len(file_list) == 0:
        print("⚠️  没有找到待处理的文件！")
        return
    for i, file in enumerate(file_list):
        print(f"正在处理 [{i+1}/{len(file_list)}]: {os.path.basename(file)}")
        segment_out_name = os.path.join(segment_out_dir, 'segment_{}.txt'.format(i))
        # 尝试多种编码读取文件
        document = None
        for encoding in ['utf-8', 'gb18030', 'gbk', 'gb2312', 'latin-1']:
            try:
                with open(file, 'r', encoding=encoding) as f:
                    document = f.read()
                print(f"    ✓ 使用编码: {encoding}")
                break
            except (UnicodeDecodeError, LookupError):
                continue
        if document is None:
            print(f"    ❌ 无法解码文件，跳过")
            continue
        # jieba.cut 返回的是生成器，逐词处理
        document_cut = jieba.cut(document)
        sentence_segment = []
        for word in document_cut:
            if word not in stopwords:
                sentence_segment.append(word)
        result = ' '.join(sentence_segment)
        with open(segment_out_name, 'w', encoding='utf-8') as f2:
            f2.write(result)
        print(f"    ✓ 已生成: segment_{i}.txt (共 {len(sentence_segment)} 个词)")

# 对source中的txt文件进行分词，输出到segment目录中
file_list = files_processing.get_files_list(source_folder, postfix='*.txt')
print(f"\n共发现 {len(file_list)} 个文件待分词处理")
segment_lines(file_list, segment_folder)
print(f"\n✅ 分词完成！输出目录: {segment_folder}")
