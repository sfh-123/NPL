import sys
import os
import stanza
from collections import defaultdict

# 中文特有错误判定规则（深度优化核心）
ZH_ERROR_RULES = {
    "QUANTIFIER": {  # 量词错误
        "keywords": {"个", "本", "只", "条", "张", "把", "位"},
        "rule": lambda o, c: any(word in o or word in c for word in ZH_ERROR_RULES["QUANTIFIER"]["keywords"])
    },
    "PARTICLE": {  # 的/地/得误用
        "keywords": {"的", "地", "得"},
        "rule": lambda o, c: any(word in o or word in c for word in ZH_ERROR_RULES["PARTICLE"]["keywords"])
    },
    "WORD_ORDER": {  # 语序错误
        "rule": lambda o, c: set(o.strip()) == set(c.strip()) and o != c
    },
    "PREPOSITION": {  # 介词误用（把/被/对/向）
        "keywords": {"把", "被", "对", "向", "在", "从"},
        "rule": lambda o, c: any(word in o or word in c for word in ZH_ERROR_RULES["PREPOSITION"]["keywords"])
    },
    "CLASSIFIER": {  # 类别词错误
        "keywords": {"名", "动", "形", "副"},
        "rule": lambda o, c: any(word in o or word in c for word in ZH_ERROR_RULES["CLASSIFIER"]["keywords"])
    },
    "POLYPHONE": {  # 多音字错误
        "keywords": {"行", "乐", "好", "还"},
        "rule": lambda o, c: any(word in o or word in c for word in ZH_ERROR_RULES["POLYPHONE"]["keywords"])
    }
}

class ZHErrorClassifier:
    def __init__(self):
        # 初始化Stanza中文模型（复用原代码路径）
        self.nlp = self.init_stanza()
        # 学习者文本分词容错规则（优化嘈杂文本处理）
        self.fault_tolerant_rules = {
            "动宾短语": {"打篮球", "看电影", "写作业"},
            "固定搭配": {"总而言之", "众所周知", "一方面"}
        }

    def init_stanza(self):
        """复用原代码的Stanza初始化逻辑"""
        try:
            return stanza.Pipeline(
                lang="zh",
                dir="./stanza_models",
                processors="tokenize,pos,lemma,depparse",
                use_gpu=False,
                download_method=None,
                verbose=False,
                logging_level="ERROR"
            )
        except Exception as e:
            print(f"Stanza加载失败，使用基础分词: {e}")
            return None

    def fault_tolerant_tokenize(self, text):
        """优化学习者文本分词（解决Stanza对嘈杂文本处理不足）"""
        if not self.nlp:
            return text.split()
        
        # 第一步：Stanza基础分词
        doc = self.nlp(text)
        tokens = [token.text for sent in doc.sentences for token in sent.tokens]
        
        # 第二步：应用容错规则（合并固定短语）
        new_tokens = []
        i = 0
        while i < len(tokens):
            # 检查是否匹配固定短语
            matched = False
            for phrase in self.fault_tolerant_rules["动宾短语"] | self.fault_tolerant_rules["固定搭配"]:
                phrase_tokens = phrase.split()
                if i + len(phrase_tokens) <= len(tokens):
                    current_phrase = "".join(tokens[i:i+len(phrase_tokens)])
                    if current_phrase == phrase:
                        new_tokens.append(phrase)
                        i += len(phrase_tokens)
                        matched = True
                        break
            if not matched:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def classify_error(self, orig_sent, cor_sent, span):
        """精准分类中文错误类型（深度优化核心）"""
        # 第一步：容错分词
        orig_tokens = self.fault_tolerant_tokenize(orig_sent)
        cor_tokens = self.fault_tolerant_tokenize(cor_sent)
        
        # 第二步：提取span对应的文本片段
        start, end = map(int, span.split())
        orig_span_text = "".join(orig_tokens[start:end]) if start < len(orig_tokens) else ""
        cor_span_text = cor_sent  # 修正文本
        
        # 第三步：匹配错误规则
        for err_type, rule_info in ZH_ERROR_RULES.items():
            if "rule" in rule_info and rule_info["rule"](orig_span_text, cor_span_text):
                return err_type
        
        # 兜底：未知错误
        return "OTHER"

    def get_error_desc(self, err_type):
        """错误类型中文说明"""
        desc_map = {
            "QUANTIFIER": "量词错误（如：一个书→一本书）",
            "PARTICLE": "助词错误（的/地/得误用）",
            "WORD_ORDER": "语序错误（如：我吃饭在食堂→我在食堂吃饭）",
            "PREPOSITION": "介词错误（把/被/对误用）",
            "CLASSIFIER": "类别词错误",
            "POLYPHONE": "多音字错误",
            "OTHER": "其他错误"
        }
        return desc_map.get(err_type, "未知错误")