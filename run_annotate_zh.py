import os
import sys
import argparse
from typing import List, Tuple, Dict
import stanza
from stanza.models.common.doc import Document, Sentence, Token

# ===================== 还原原有路径配置 =====================
# 与你原本的路径保持一致
INPUT_FILE = "docs/data/GEC_European_Datasets/Chinese/zh.train.auto.m2"
OUTPUT_FILE = "docs/data/GEC_European_Datasets/Chinese/zh_annotated.m2"
STANZA_MODEL_DIR = "./stanza_models"

# 中文错误类型映射
ZH_ERROR_TYPES = {
    "QUANTIFIER": "量词错误",
    "WORD_ORDER": "语序错误",
    "PARTICLE": "助词错误",
    "CLASSIFIER": "类别词错误",
    "POLYPHONE": "多音字错误"
}

class JPErrantZH:
    def __init__(self):
        # 初始化Stanza中文模型
        self.nlp = self.init_stanza()
        self.error_count = 0

    def init_stanza(self) -> stanza.Pipeline:
        """初始化Stanza中文模型（含依赖分析）"""
        print("初始化Stanza中文模型...")
        try:
            nlp = stanza.Pipeline(
                lang="zh",
                dir=STANZA_MODEL_DIR,
                processors="tokenize,pos,lemma,depparse",
                use_gpu=False,
                download_method=None,
                verbose=False,
                logging_level="ERROR"
            )
            print("Stanza中文模型（含依赖分析）加载成功！")
            return nlp
        except Exception as e:
            print(f"Stanza加载失败: {str(e)}")
            print("降级为基础分词模式运行...")
            return None

    def parse_m2_file(self, input_file: str) -> List[Dict]:
        """解析中文M2格式文件"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}\n请确认文件路径是否正确，或将数据集文件放到指定路径下")
        
        data = []
        current_sent = None
        current_edits = []
        
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 处理句子行
            if line.startswith("S "):
                if current_sent is not None:
                    data.append({
                        "sentence": current_sent,
                        "edits": current_edits
                    })
                current_sent = line[2:]
                current_edits = []
            # 处理编辑行
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) >= 3:
                    span = parts[0].strip()
                    error_type = parts[1].strip()
                    correction = parts[2].strip()
                    current_edits.append({
                        "span": span,
                        "error_type": error_type,
                        "correction": correction,
                        "meta": parts[3:] if len(parts) > 3 else []
                    })
                    self.error_count += 1
        
        # 添加最后一个句子
        if current_sent is not None:
            data.append({
                "sentence": current_sent,
                "edits": current_edits
            })
        
        print(f"解析完成 - 共加载 {len(data)} 个句子，{self.error_count} 个标注错误")
        return data

    def analyze_sentence(self, sentence: str) -> Tuple[List[Token], Sentence]:
        """分析中文句子（分词+词性+依赖分析）"""
        if self.nlp:
            doc = self.nlp(sentence)
            if doc.sentences:
                sent = doc.sentences[0]
                return [token for token in sent.tokens], sent
        # 降级分词（空格分词）
        tokens = [Token(text=word) for word in sentence]
        return tokens, None

    def generate_m2_output(self, data: List[Dict], output_file: str):
        """生成中文标注后的M2文件（自动创建输出目录）"""
        # 自动创建输出目录（避免路径不存在报错）
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        with open(output_file, "w", encoding="utf-8") as f:
            for idx, item in enumerate(data):
                sentence = item["sentence"]
                edits = item["edits"]
                
                # 写入句子行
                f.write(f"S {sentence}\n")
                
                # 分析句子（获取分词/依赖信息）
                tokens, sent_analysis = self.analyze_sentence(sentence)
                
                # 写入编辑行
                for edit in edits:
                    span = edit["span"]
                    error_type = edit["error_type"]
                    correction = edit["correction"]
                    meta = edit["meta"]
                    
                    # 补充中文错误类型说明
                    zh_error = ZH_ERROR_TYPES.get(error_type, "未知错误")
                    
                    # 构建M2编辑行（与原格式一致）
                    meta_str = "|||".join(meta) if meta else "-NONE-"
                    edit_line = f"A {span}|||{error_type}|||{correction}|||JP-Errant-ZH|||REQUIRED|||{zh_error}|||0\n"
                    f.write(edit_line)
                
                # 空行分隔
                f.write("\n")
        
        print(f"标注完成 - 输出文件: {output_file}")
        print(f"统计信息 - 总句子数: {len(data)}, 总错误数: {self.error_count}")

    def run(self):
        """主运行函数（使用全局路径配置）"""
        try:
            # 使用全局配置的输入输出路径
            input_file = INPUT_FILE
            output_file = OUTPUT_FILE
            
            # 解析输入文件
            data = self.parse_m2_file(input_file)
            
            # 生成标注输出
            self.generate_m2_output(data, output_file)
            
            return 0
        except Exception as e:
            print(f"运行出错: {str(e)}", file=sys.stderr)
            return 1

def main():
    # 保持极简的主函数（与原代码风格一致）
    annotator = JPErrantZH()
    exit_code = annotator.run()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()