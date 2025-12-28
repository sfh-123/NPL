import sys
from collections import defaultdict

def stat_fine_error_types(fine_m2_path: str):
    """统计细粒度错误类型分布"""
    fine_count = defaultdict(int)
    total_edits = 0

    try:
        with open(fine_m2_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("A "):
                    parts = line.split("|||")
                    if len(parts) >=4:
                        fine_type = parts[3]
                        fine_count[fine_type] += 1
                        total_edits += 1

        # 输出统计结果
        print("===== 细粒度错误类型分布 =====")
        print(f"总错误数：{total_edits}")
        for fine_type, count in sorted(fine_count.items(), key=lambda x: x[1], reverse=True):
            ratio = count / total_edits * 100 if total_edits >0 else 0
            print(f"{fine_type}: {count} ({ratio:.2f}%)")
    except Exception as e:
        print(f"统计失败：{e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stat_fine_types.py <fine_m2_file>")
        sys.exit(1)
    stat_fine_error_types(sys.argv[1])