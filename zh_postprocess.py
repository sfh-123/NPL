import sys
import os
from zh_error_classifier import ZHErrorClassifier

def parse_m2(file_path):
    """解析原有M2文件（复用原代码逻辑）"""
    data = []
    current_sent = None
    current_edits = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("S "):
                if current_sent is not None:
                    data.append((current_sent, current_edits))
                current_sent = line[2:]
                current_edits = []
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) >= 3:
                    span = parts[0]
                    orig_type = parts[1]
                    correction = parts[2]
                    current_edits.append((span, orig_type, correction))
    if current_sent is not None:
        data.append((current_sent, current_edits))
    return data

def postprocess_m2(orig_m2_path, output_m2_path):
    """后处理M2文件，补充细粒度中文错误标注"""
    # 初始化分类器
    classifier = ZHErrorClassifier()
    
    # 解析原有M2文件
    m2_data = parse_m2(orig_m2_path)
    
    # 生成优化后的M2文件
    with open(output_m2_path, "w", encoding="utf-8") as f:
        for sent, edits in m2_data:
            # 写入句子行
            f.write(f"S {sent}\n")
            
            # 处理每个编辑
            for span, orig_type, correction in edits:
                # 精准分类错误类型
                fine_type = classifier.classify_error(sent, correction, span)
                # 获取错误说明
                err_desc = classifier.get_error_desc(fine_type)
                # 写入优化后的编辑行
                edit_line = f"A {span}|||{fine_type}|||{correction}|||JP-Errant-ZH-Opt|||REQUIRED|||{err_desc}|||0\n"
                f.write(edit_line)
            
            # 空行分隔
            f.write("\n")
    
    print(f" 深度优化完成！输出文件：{output_m2_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("用法：python zh_postprocess.py <原有M2文件路径> <优化后M2文件路径>")
        print("示例：python zh_postprocess.py docs/data/GEC_European_Datasets/Chinese/zh_annotated.m2 docs/data/GEC_European_Datasets/Chinese/zh_annotated_fine.m2")
        sys.exit(1)
    
    orig_m2 = sys.argv[1]
    output_m2 = sys.argv[2]
    
    # 检查输入文件是否存在
    if not os.path.exists(orig_m2):
        print(f" 输入文件不存在：{orig_m2}")
        sys.exit(1)
    
    # 执行后处理
    postprocess_m2(orig_m2, output_m2)