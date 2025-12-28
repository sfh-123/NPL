import sys
import os
import stanza

# ===================== 1. 环境配置 =====================
# 确保jp_errant模块可导入
sys.path.append(os.path.join(os.path.dirname(__file__), "jp_errant"))

# ===================== 2. 核心配置 =====================
# 适配Lang8数据集（修改输入/输出文件名）
INPUT_FILE = "docs/data/GEC_European_Datasets/English/A.train.gold.bea19.m2"
OUTPUT_FILE = "docs/data/GEC_European_Datasets/English/A_annotated.m2"  
# Stanza模型路径（本地缓存）
STANZA_MODEL_DIR = "./stanza_models"
# 支持的错误类型
ERROR_TYPES = {
    "ART": "冠词错误",
    "PREP": "介词错误",
    "VERB": "动词错误",
    "VERB:TENSE": "动词时态错误",
    "VERB:FORM": "动词形式错误",
    "NOUN": "名词错误",
    "ADJ": "形容词错误",
    "ADV": "副词错误",
    "PRON": "代词错误",
    "CONJ": "连词错误",
    "NUM": "数词错误",
    "PUNCT": "标点错误",
    "MORPH": "形态错误",
    "SYNTAX": "句法错误",
    "OTHER": "其他错误"
}

# ===================== 3. 初始化Stanza（离线模式） =====================
def init_stanza():
    """初始化Stanza处理器（优先读取本地模型，无外网依赖）"""
    print("初始化Stanza模型（离线模式）...")
    try:
        # 强制使用本地模型，禁用任何下载，通过logging_level控制日志
        nlp = stanza.Pipeline(
            lang="en",
            dir=STANZA_MODEL_DIR,
            processors="tokenize,pos,lemma,mwt",
            use_gpu=False,
            download_method=None,
            verbose=False,
            logging_level="ERROR"  # 直接通过Pipeline参数关闭日志
        )
        print("Stanza模型加载成功！")
        return nlp
    except Exception as e:
        print(f"Stanza加载失败: {str(e)}")
        print("降级为空格分词模式...")
        return None

# ===================== 4. 精准分词函数（Stanza） =====================
def tokenize_sent(nlp, sent_str):
    """
    使用Stanza进行精准分词，生成JP-Errant要求的结构化对象
    :param nlp: Stanza Pipeline对象
    :param sent_str: 待分词的句子字符串
    :return: 包含text/lemma/pos属性的分词对象
    """
    if not sent_str:
        return None

    # 方案1：Stanza精准分词（优先）
    if nlp is not None:
        try:
            doc = nlp(sent_str)
            # 构造JP-Errant兼容的对象结构
            class WordObj:
                def __init__(self, word):
                    self.text = word.text
                    self.lemma = word.lemma
                    self.pos = word.upos  # 通用词性标注（UPOS）
                    self.idx = word.id    # 单词在句子中的位置

            class SentenceObj:
                def __init__(self, words):
                    self.words = words

            class TokenizedObj:
                def __init__(self, sentences):
                    self.sentences = sentences

            # 提取Stanza分词结果
            sentences = []
            for sent in doc.sentences:
                words = [WordObj(word) for word in sent.words]
                sentences.append(SentenceObj(words))
            return TokenizedObj(sentences)
        except Exception as e:
            print(f"Stanza分词失败: {str(e)}")

    # 方案2：降级为空格分词（兜底）
    class WordObj:
        def __init__(self, text, idx):
            self.text = text
            self.lemma = text.lower()
            self.pos = "UNK"
            self.idx = idx

    class SentenceObj:
        def __init__(self, words):
            self.words = words

    class TokenizedObj:
        def __init__(self, sentences):
            self.sentences = sentences

    word_list = sent_str.split()
    words = [WordObj(word, idx+1) for idx, word in enumerate(word_list)]
    return TokenizedObj([SentenceObj(words)])

# ===================== 5. 核心：错误分类规则 =====================
def classify_edit(edit, orig_sent, cor_sent):
    """
    对编辑对象进行精准错误分类
    :param edit: 包含原始/修正单词的编辑对象
    :param orig_sent: 原始句分词对象
    :param cor_sent: 修正句分词对象
    :return: 错误类型（如ART/PREP/VERB）
    """
    # 提取编辑核心特征
    orig_toks = edit.get("orig_toks", [])
    cor_toks = edit.get("cor_toks", [])
    orig_text = " ".join([tok.text.lower() for tok in orig_toks]) if orig_toks else ""
    cor_text = " ".join([tok.text.lower() for tok in cor_toks]) if cor_toks else ""
    orig_pos = [tok.pos for tok in orig_toks] if orig_toks else []
    cor_pos = [tok.pos for tok in cor_toks] if cor_toks else []
    orig_lemma = [tok.lemma.lower() for tok in orig_toks] if orig_toks else []
    cor_lemma = [tok.lemma.lower() for tok in cor_toks] if cor_toks else []

    # 规则1：冠词错误（ART）- 限定词（DET）+ 冠词词汇
    art_words = {"a", "an", "the"}
    if (orig_text in art_words or cor_text in art_words) and \
       any(pos == "DET" for pos in orig_pos + cor_pos):
        return "ART"

    # 规则2：介词错误（PREP）- 介词（ADP）+ 介词词汇
    prep_words = {"in", "on", "at", "by", "with", "for", "to", "from", "into", "onto"}
    if (orig_text in prep_words or cor_text in prep_words) and \
       any(pos == "ADP" for pos in orig_pos + cor_pos):
        return "PREP"

    # 规则3：动词错误（VERB）- 动词/助动词（VERB/AUX）
    if any(pos in ["VERB", "AUX"] for pos in orig_pos + cor_pos):
        # 子规则：时态错误
        tense_words = {"is", "are", "was", "were", "has", "have", "had"}
        if orig_text in tense_words or cor_text in tense_words:
            return "VERB:TENSE"
        # 子规则：形式错误
        form_words = {"do", "does", "did", "doing", "done", "go", "goes", "went", "gone"}
        if orig_text in form_words or cor_text in form_words:
            return "VERB:FORM"
        # 通用动词错误
        return "VERB"

    # 规则4：形态错误（MORPH）- 词形不同但词性相同
    if orig_lemma != cor_lemma and orig_pos == cor_pos:
        return "MORPH"

    # 规则5：句法错误（SYNTAX）- 位置变化但词汇相同
    if orig_text == cor_text and edit.get("o_start") != edit.get("c_start"):
        return "SYNTAX"

    # 规则6：按词性匹配其他类型
    pos_map = {
        "NOUN": ["NOUN"],
        "ADJ": ["ADJ"],
        "ADV": ["ADV"],
        "PRON": ["PRON"],
        "CONJ": ["CCONJ", "SCONJ"],
        "NUM": ["NUM"],
        "PUNCT": ["PUNCT"]
    }
    for err_type, pos_list in pos_map.items():
        if any(pos in pos_list for pos in orig_pos + cor_pos):
            return err_type

    # 规则7：未匹配到的归为OTHER
    return "OTHER"

# ===================== 6. 处理M2文件（完整流程） =====================
def process_m2_file(nlp, annotator):
    """处理M2文件，生成带精准错误分类的标注结果"""
    print(f"开始处理M2文件: {INPUT_FILE}")
    print("支持的错误类型：" + ", ".join(ERROR_TYPES.keys()))
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        
        current_source = None
        current_source_str = ""
        line_count = 0
        error_count = 0

        for line in f_in:
            line = line.strip()
            line_count += 1

            # 打印进度（每200行）
            if line_count % 200 == 0:
                print(f"已处理 {line_count} 行，已标注 {error_count} 个错误")

            # 处理S行（原始错误句）
            if line.startswith("S "):
                current_source_str = line[2:]
                current_source = tokenize_sent(nlp, current_source_str)
                # 写入原始S行
                f_out.write(f"{line}\n")
                continue

            # 处理A行（修正信息）
            if line.startswith("A ") and current_source:
                a_parts = line.split("|||")
                if len(a_parts) < 3:
                    continue
                
                # 提取修正句
                current_cor_str = a_parts[2]
                current_cor = tokenize_sent(nlp, current_cor_str)
                if not current_cor:
                    continue

                try:
                    # 1. 对齐原始句和修正句
                    alignment = annotator.align(current_source, current_cor)
                    
                    # 2. 遍历对齐结果，生成编辑对象
                    for idx in range(min(len(alignment.orig), len(alignment.cor))):
                        orig_tok = alignment.orig[idx]
                        cor_tok = alignment.cor[idx]
                        
                        # 只处理有差异的编辑
                        if orig_tok.text != cor_tok.text:
                            # 构造编辑对象
                            edit = {
                                "o_start": idx,
                                "o_end": idx + 1,
                                "c_start": idx,
                                "c_end": idx + 1,
                                "orig_toks": [orig_tok],
                                "cor_toks": [cor_tok]
                            }
                            
                            # 3. 精准分类错误类型
                            err_type = classify_edit(edit, current_source, current_cor)
                            error_count += 1
                            
                            # 4. 写入标准M2格式的A行
                            a_line = (
                                f"A {edit['o_start']} {edit['o_end']}|||"
                                f"{err_type}|||"
                                f"{cor_tok.text}|||"
                                f"JP_Errant|||REQUIRED|||-NONE-|||0\n"
                            )
                            f_out.write(a_line)
                except Exception as e:
                    print(f"行{line_count}处理出错: {str(e)}")
                    continue
                
                # 重置当前句
                current_source = None

    # 输出统计信息
    print("\n处理完成！")
    print(f"总计处理行数：{line_count}")
    print(f"总计标注错误：{error_count}")
    print(f"输出文件：{OUTPUT_FILE}")

# ===================== 7. 主函数 =====================
if __name__ == "__main__":
    # 1. 安装依赖提示（仅首次运行）
    print("所需依赖：stanza")
    print("安装命令：pip install stanza")
    
    # 2. 初始化Stanza
    nlp = init_stanza()
    
    # 3. 初始化JP-Errant标注器
    print("初始化JP-Errant标注器...")
    try:
        from jp_errant.annotator import Annotator
        annotator = Annotator(lang="en")
        print("JP-Errant标注器初始化成功！")
    except Exception as e:
        print(f"JP-Errant初始化失败: {str(e)}")
        sys.exit(1)
    
    # 4. 处理M2文件
    process_m2_file(nlp, annotator)
    
    # 5. 验证输出文件
    if os.path.exists(OUTPUT_FILE):
        file_size = os.path.getsize(OUTPUT_FILE) / 1024
        print(f"\n验证通过：输出文件大小 {file_size:.2f} KB")
    else:
        print("\n验证失败：输出文件未生成")