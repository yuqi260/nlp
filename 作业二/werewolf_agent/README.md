# 狼人杀双系统角色智能体工程（`werewolf_agent`）

本项目对应 NLP 课程作业二：**不微调 LLM，采用“冻结 LLM 语言层（System-1）+ 本地 CPU 训练轻量级席位共享 Belief 信念模块与席位等变指针 RL 策略模块（System-2）”的双系统架构**，实现低成本、可解释、防泄密的中文九人预女猎狼人杀角色智能体。

## 目录结构

- `configs/nine_player_standard.json`：冻结的九人预女猎标准规则（屠边、无警长、女巫首夜可自救且同夜不可双药、猎人吃毒不可开枪）。
- `src/werewolf_agent/`：
  - `config.py`、`protocol.py`、`engine.py`：规则校验、6 类离散动作/64 维发言意图协议、严格视角隔离与双模状态机（`symbolic` / `nl`）。
  - `features.py`、`makto_trajectories.py`：统一 NumPy 特征编码（含无损中文角色自称直通）与 MaKTO 133 局第一人称专家轨迹因果重构。
  - `nn.py`、`models/belief.py`、`models/policy.py`、`models/pointer_policy.py`、`training.py`：纯 NumPy 反向传播/Adam/Masked-PPO，实现席位共享 `BeliefMLP`、扁平 MLP 与席位等变指针网络 `SeatEquivariantPolicyValueNet`。
  - `llm_client.py`、`nl_parser.py`、`nl_speech.py`：百炼 API 客户端（带预算熔断）、混合正则+LLM 意图解析器、低 Token 摘要与防泄密 Guardrail。
  - `agents/`：`RandomAgent`、`RuleBasedAgent`、`BeliefRuleAgent`、`NeuralPolicyAgent`、`PureLLMAgent`、`BeliefLLMAgent`、`DecoupledLLMAgent`、`DualSystemLLMAgent`。
- `scripts/`：M1–M6 全流程可复现脚本。
- `outputs/`：训练权重（`models/`）、M3/M5 实验指标与 142 局真实对局回放（`metrics/`、`m5_eval/`）、4 张矢量 SVG 图表（`figures/`）与单文件离线交互演示网页（`demo/werewolf_agent_demo.html`）。
- `tests/`：22 项单元测试（规则、视角隔离、有限差分梯度校验、Guardrail、端到端 NL）。

## 全流程复现命令

在 `作业二/werewolf_agent/` 目录下运行：

```powershell
# 1. 运行全部 22 项单元测试
py -3 -m unittest discover -s tests -v

# 2. M1：1000 局符号模式压力测试（0 API 费）
py -3 scripts/run_symbolic_benchmark.py --games 1000 --policy rule --seed 42

# 3. M2：重构 MaKTO 第一人称专家轨迹数据集（0 API 费）
py -3 scripts/build_makto_trajectories.py

# 4. M3：本机 CPU 训练 BeliefMLP + Pointer BC + 4,800 局自博弈 PPO（0 API 费）
py -3 scripts/train_m3.py --arch pointer --ppo-iters 100 --games-per-iter 48 --eval-games 300 --seed 42

# 5. M4：NL 试点（默认零成本 Mock；加 --execute 调用真实百炼 API）
py -3 scripts/run_nl_pilot.py --games 2 --matchup ours_vs_pure

# 6. M5：自然语言受控对照评测（默认 Mock；加 --execute 调用真实 API）
py -3 scripts/run_m5_eval.py --suite ablation --games-per-side 10 --workers 8

# 7. M6：生成 4 张矢量实验图表与离线交互演示网页（0 API 费）
py -3 scripts/generate_m6_charts.py
py -3 scripts/build_demo_web.py
```
