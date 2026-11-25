"""检查规则配置，不读取对局或模型。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from werewolf_nlp.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description="检查游戏角色配置")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
    except (OSError, ValueError) as exc:
        print(f"配置检查失败：{exc}", file=sys.stderr)
        return 1
    print(f"配置有效：{config['ruleset_id']}，{config['player_count']} 人")
    for role, count in config["role_counts"].items():
        print(f"  {config['role_names_zh'][role]}：{count} 人")
    print("规则细节尚待数据核验。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
