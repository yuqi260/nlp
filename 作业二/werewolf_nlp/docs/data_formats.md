# 数据格式约定（初版）

此文档定义模块间的数据边界，不表示已有真实数据。所有记录采用 UTF-8 JSON 或 JSONL；同一局使用稳定的 `game_id`、玩家编号和事件编号。具体来源字段与规则待核验后映射。

## 1. 公开事件 public_events.jsonl

每行一个事件：`game_id`、`event_id`、`day`、`phase`、`order`、`type`、`speaker`、`text`。`speaker` 为玩家编号或 null（系统公告）；`type` 暂定 `speech`、`vote`、`announcement`，投票可额外记录 `target`。`order` 为局内唯一顺序，`text` 保留原文。公告公开死亡不等于公开身份。发言只是玩家声明，不自动视为真实事件。

## 2. 真实身份 labels.jsonl

每行一名玩家：`game_id`、`player_id`、`true_role`。只供训练目标与离线评测使用，独立保存，不能拼接进模型输入。须从可靠的原始身份分配核验，不用玩家自报代替。

## 3. 预测快照 snapshots.jsonl

每行包含 `snapshot_id`、`game_id`、`ruleset_id`、`day`、`cutoff_order`、`public_events`。事件必须属于该局且 `order <= cutoff_order`；公开确认的身份只有在当时规则与公告确实确认后才可输入。预测时点为白天投票及公开结算完成后、下一夜之前。

## 4. 数据划分 splits.jsonl

每行 `game_id`、`split`（`train`、`validation` 或 `test`）。整局及其不同轮次、玩家视角和翻译版本只能属于同一集合；保存划分清单与随机种子。

## 5. 身份评分 scores.jsonl

每行包含 `snapshot_id`、`player_id`、`candidate_role`、`score`、`support_event_ids`、`counter_event_ids`、`explanation`。`score` 为 -2 至 2 的整数，仅表示模型相对匹配评价，不是概率或真实标签。引用必须存在且不晚于快照；规则明确不允许的身份单独标记，不用 -2 冒充排除。相同证据不机械重复累加。

## 6. 身份预测 predictions.jsonl

每行一个快照：`snapshot_id`、`assignments`（玩家编号、预测身份、依据编号）、`alternative_assignments`、`status`、`model_id`、`prompt_version`。身份组合须满足实际配置与公开确认信息。允许并列和信息不足；失败明确记录，不能静默填造结果。逐轮重新评分并计算整体分配，不累加先前分数，也不把先前预测当事实。

## 7. 评价口径

主指标优先统计当时尚未公开确认身份的玩家，全部玩家指标另列；记录每局每轮及各角色结果。真实标签永不放入预测输入；事实引用存在不代表解释一定支持结论，另行人工抽查。
