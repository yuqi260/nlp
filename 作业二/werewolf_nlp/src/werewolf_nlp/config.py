"""加载和检查游戏配置。"""

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        config = json.load(file)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("配置必须是 JSON 对象")
    if not isinstance(config.get("ruleset_id"), str) or not config["ruleset_id"].strip():
        raise ValueError("ruleset_id 必须是非空字符串")
    count = config.get("player_count")
    if type(count) is not int or count <= 0:
        raise ValueError("player_count 必须是正整数")
    roles = config.get("role_counts")
    if not isinstance(roles, dict) or not roles:
        raise ValueError("role_counts 必须是非空对象")
    if any(not isinstance(k, str) or not k.strip() or type(v) is not int or v <= 0 for k, v in roles.items()):
        raise ValueError("角色名须为非空字符串，人数须为正整数")
    if sum(roles.values()) != count:
        raise ValueError("角色人数总和必须等于玩家人数")
    names = config.get("role_names_zh")
    if not isinstance(names, dict) or names.keys() != roles.keys() or any(not isinstance(v, str) or not v.strip() for v in names.values()):
        raise ValueError("中文角色名必须覆盖全部角色且非空")
    if config.get("viewpoint") != "public_observer":
        raise ValueError("当前仅支持公开信息旁观者视角")
