import argparse
from collections import defaultdict, Counter
import re

class SimpleGranularityComparer:
    def __init__(self):
        # 存储两类文件的标注统计
        self.stats = {
            "coarse": {
                "total_edits": 0,  # 总错误数
                "cat_count": defaultdict(int),  # 类别-数量
                "other_ratio": 0.0  # OTHER类占比
            },
            "fine": {
                "total_edits": 0,
                "cat_count": defaultdict(int),
                "parent_cat_count": defaultdict(int),  # 细粒度归并后的粗类别计数
                "other_ratio": 0.0
            }
        }
        # 粗细粒度映射关系（细→粗）
        self.fine_to_coarse = defaultdict(str)

    def parse_m2_edits(self, m2_path):
        """解析M2文件，提取所有错误编辑的类别"""
        edits = []
        with open(m2_path, "r", encoding="utf-8") as f:
            blocks = f.read().strip().split("\n\n")
            for block in blocks:
                lines = block.strip().split("\n")
                for line in lines[1:]:  # 跳过S行，只处理A行
                    if line.startswith("A "):
                        parts = line[2:].split("|||")
                        if len(parts) >= 2:
                            cat = parts[1].strip()
                            if cat != "noop":  # 跳过无错误标注
                                edits.append(cat)
        return edits

    def analyze_coarse(self, coarse_m2):
        """分析粗粒度标注文件"""
        edits = self.parse_m2_edits(coarse_m2)
        self.stats["coarse"]["total_edits"] = len(edits)
        self.stats["coarse"]["cat_count"] = Counter(edits)
        
        # 计算OTHER类占比
        other_count = self.stats["coarse"]["cat_count"].get("OTHER", 0)
        if self.stats["coarse"]["total_edits"] > 0:
            self.stats["coarse"]["other_ratio"] = (other_count / self.stats["coarse"]["total_edits"]) * 100
        else:
            self.stats["coarse"]["other_ratio"] = 0.0

    def analyze_fine(self, fine_m2):
        """分析细粒度标注文件，并归并到粗粒度类别"""
        edits = self.parse_m2_edits(fine_m2)
        self.stats["fine"]["total_edits"] = len(edits)
        self.stats["fine"]["cat_count"] = Counter(edits)
        
        # 1. 计算细粒度OTHER类占比
        other_count = self.stats["fine"]["cat_count"].get("OTHER", 0)
        if self.stats["fine"]["total_edits"] > 0:
            self.stats["fine"]["other_ratio"] = (other_count / self.stats["fine"]["total_edits"]) * 100
        else:
            self.stats["fine"]["other_ratio"] = 0.0
        
        # 2. 细粒度归并到粗粒度（如FUNCTION:ART → FUNCTION）
        for fine_cat, count in self.stats["fine"]["cat_count"].items():
            if ":" in fine_cat:
                coarse_cat = fine_cat.split(":")[0]
                self.fine_to_coarse[fine_cat] = coarse_cat
                self.stats["fine"]["parent_cat_count"][coarse_cat] += count
            else:
                # 无细分的类别直接归为自身
                self.fine_to_coarse[fine_cat] = fine_cat
                self.stats["fine"]["parent_cat_count"][fine_cat] += count

    def generate_compare_report(self, output_path=None):
        """生成对比报告"""
        report = []
        report.append("="*80)
        report.append(" 粗/细粒度标注文件对比报告")
        report.append("="*80)
        report.append("")

        # 1. 基础统计
        report.append("【基础标注统计】")
        report.append(f"粗粒度标注总错误数：{self.stats['coarse']['total_edits']}")
        report.append(f"细粒度标注总错误数：{self.stats['fine']['total_edits']}")
        report.append(f"粗粒度OTHER类占比：{self.stats['coarse']['other_ratio']:.2f}%")
        report.append(f"细粒度OTHER类占比：{self.stats['fine']['other_ratio']:.2f}%")
        report.append(f"OTHER类占比下降：{self.stats['coarse']['other_ratio'] - self.stats['fine']['other_ratio']:.2f}个百分点")
        report.append("")

        # 2. 粗粒度类别覆盖对比
        report.append("【粗粒度类别覆盖对比】")
        report.append(f"{'粗类别':<15} {'粗粒度数量':<12} {'细粒度归并数量':<15} {'数量差异':<10}")
        report.append("-"*80)
        
        all_coarse_cats = set(self.stats["coarse"]["cat_count"].keys()) | set(self.stats["fine"]["parent_cat_count"].keys())
        for cat in sorted(all_coarse_cats):
            if cat == "OTHER":
                continue
            c_count = self.stats["coarse"]["cat_count"].get(cat, 0)
            f_count = self.stats["fine"]["parent_cat_count"].get(cat, 0)
            diff = f_count - c_count
            diff_str = f"+{diff}" if diff > 0 else f"{diff}" if diff < 0 else "0"
            report.append(f"{cat:<15} {c_count:<12} {f_count:<15} {diff_str:<10}")
        report.append("")

        # 3. 细粒度类别细分详情
        report.append("【细粒度类别细分详情】")
        report.append(f"{'细粒度类别':<20} {'归属粗类别':<15} {'标注数量':<10}")
        report.append("-"*80)
        for fine_cat, count in sorted(self.stats["fine"]["cat_count"].items()):
            if fine_cat == "OTHER":
                continue
            coarse_cat = self.fine_to_coarse[fine_cat]
            report.append(f"{fine_cat:<20} {coarse_cat:<15} {count:<10}")
        report.append("")

        # 4. 核心结论
        report.append("【核心结论】")
        if self.stats["fine"]["other_ratio"] < self.stats["coarse"]["other_ratio"]:
            report.append(" 细粒度标注有效减少了OTHER类占比，错误分类更精准")
        else:
            report.append(" 细粒度标注未降低OTHER类占比，分类效果无明显提升")
        
        # 计算细分覆盖率（有细分的类别数/总类别数）
        fine_cats_with_sub = [c for c in self.stats["fine"]["cat_count"].keys() if ":" in c]
        sub_coverage = len(fine_cats_with_sub) / len(self.stats["fine"]["cat_count"]) * 100 if self.stats["fine"]["cat_count"] else 0.0
        report.append(f" 细粒度标注细分覆盖率：{sub_coverage:.2f}%（{len(fine_cats_with_sub)}/{len(self.stats['fine']['cat_count'])}）")

        # 拼接并打印报告
        report_str = "\n".join(report)
        print(report_str)

        # 保存报告
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report_str)
            print(f"\n 对比报告已保存至：{output_path}")

def main():
    parser = argparse.ArgumentParser(description="仅用粗/细粒度标注文件完成对比评估（无人工参考）")
    parser.add_argument("--coarse-m2", required=True, help="粗粒度标注M2文件路径")
    parser.add_argument("--fine-m2", required=True, help="细粒度标注M2文件路径")
    parser.add_argument("--output", help="对比报告输出路径（可选）")
    args = parser.parse_args()

    # 初始化对比器
    comparer = SimpleGranularityComparer()

    # 分析粗/细粒度文件
    print(" 解析粗粒度标注文件...")
    comparer.analyze_coarse(args.coarse_m2)
    print(" 解析细粒度标注文件...")
    comparer.analyze_fine(args.fine_m2)

    # 生成对比报告
    comparer.generate_compare_report(output_path=args.output)

if __name__ == "__main__":
    main()