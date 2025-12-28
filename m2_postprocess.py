import sys
import os
import stanza

# ===================== 1. 初始化Stanza（用于分词/词性分析） =====================
def init_stanza():
    try:
        nlp = stanza.Pipeline(
            lang="en",
            dir="./stanza_models",
            processors="tokenize,pos,lemma,mwt",
            use_gpu=False,
            download_method=None,
            verbose=False,
            logging_level="ERROR"
        )
        return nlp
    except Exception as e:
        print(f"Stanza加载失败: {e}，降级为空格分词")
        return None

# ===================== 2. 分词函数（兼容Stanza/空格分词） =====================
def tokenize(nlp, text):
    if not text:
        return []
    if nlp:
        try:
            doc = nlp(text)
            return [word.text for word in doc.sentences[0].words]
        except:
            pass
    return text.split()

# ===================== 3. 细粒度错误分类核心逻辑 =====================
def get_fine_grain_type(coarse_type, orig_text, cor_text):
    """
    从粗粒度类型+编辑内容，生成细粒度类型
    :param coarse_type: 原粗粒度类型（如ART/VERB）
    :param orig_text: 原始错误文本
    :param cor_text: 修正后文本
    :return: 细粒度类型（如FUNCTION:ART/MORPH:TENSE）
    """
    # 优化粗→细映射规则（更全面）
    fine_map = {
        "ART": "FUNCTION:ART",
        "PREP": "LEX:PREP",
        "VERB": "MORPH:VERB",
        "NOUN": "MORPH:NUM",
        "MORPH": "MORPH:OTHER",
        "SYNTAX": "SYNTAX:WO",
        "ADJ": "MORPH:ADJ",
        "default": coarse_type
    }

    # 处理VERB的子类型（更精准）
    if coarse_type == "VERB":
        tense_keywords = {"is", "are", "was", "were", "am"}
        form_keywords = {"do", "does", "did", "doing", "done"}
        if orig_text in tense_keywords or cor_text in tense_keywords:
            return "MORPH:TENSE"
        elif orig_text in form_keywords or cor_text in form_keywords:
            return "MORPH:VERB_FORM"
        else:
            return "MORPH:VERB"
    
    # 其他类型直接映射
    return fine_map.get(coarse_type, fine_map["default"])

# ===================== 4. 解析M2文件（独立实现） =====================
def parse_m2(file_path):
    m2_data = []
    current_sent = None
    current_edits = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("S "):
                if current_sent:
                    m2_data.append((current_sent, current_edits))
                current_sent = line[2:]
                current_edits = []
            elif line.startswith("A "):
                parts = line[2:].split("|||")
                if len(parts) >= 3:
                    span = parts[0].strip()
                    coarse_type = parts[1].strip()
                    cor_text = parts[2].strip()
                    # 补充：保留原始A行的其他字段（如REQUIRED/-NONE-/标注者ID）
                    rest_parts = parts[3:] if len(parts) > 3 else ["REQUIRED", "-NONE-", "0"]
                    current_edits.append((span, coarse_type, cor_text, rest_parts))
    if current_sent:
        m2_data.append((current_sent, current_edits))
    return m2_data

# ===================== 5. 生成细粒度M2文件（修正格式） =====================
def postprocess_m2(coarse_m2, fine_m2):
    nlp = init_stanza()
    m2_data = parse_m2(coarse_m2)
    
    with open(fine_m2, "w", encoding="utf-8") as f:
        for sent, edits in m2_data:
            f.write(f"S {sent}\n")
            orig_tokens = tokenize(nlp, sent)
            for edit in edits:
                span, coarse_type, cor_text, rest_parts = edit
                # 获取原始文本（兼容span越界）
                try:
                    start, end = map(int, span.split())
                    orig_text = " ".join(orig_tokens[start:end]) if 0 <= start < end <= len(orig_tokens) else ""
                except:
                    orig_text = ""
                # 生成细粒度类型
                fine_type = get_fine_grain_type(coarse_type, orig_text, cor_text)
                # 修正：用细粒度类型替换粗粒度类型，符合M2标准格式
                f.write(f"A {span}|||{fine_type}|||{cor_text}|||{'|||'.join(rest_parts[:3])}\n")
            f.write("\n")
    print(f" 生成合规的细粒度M2文件：{fine_m2}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python m2_postprocess.py <coarse_m2> <fine_m2>")
        sys.exit(1)
    postprocess_m2(sys.argv[1], sys.argv[2])