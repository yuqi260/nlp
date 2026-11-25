"""OpenAI-compatible Bailian LLM client with strict token/cost accounting and Mock client."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

# RMB per 1,000,000 tokens (Beijing Bailian reference rates; update if provider pricing changes)
MODEL_PRICING_RMB_PER_1M: Dict[str, tuple[float, float]] = {
    "qwen3.5-27b": (0.6, 4.8),
    "qwen-plus": (0.8, 2.0),
    "qwen-turbo": (0.3, 0.6),
    "qwen3.8-max": (2.5, 10.0),
    "deepseek-v4-pro-0813": (2.0, 8.0),
}


def load_api_env(extra_paths: Optional[List[Path]] = None) -> Dict[str, str]:
    """Load WEREWOLF_API_KEY and WEREWOLF_API_BASE_URL from env or .env files without logging secrets."""
    allowed = {"WEREWOLF_API_KEY", "WEREWOLF_API_BASE_URL"}
    root = Path(__file__).resolve().parents[2]
    candidates = list(extra_paths or []) + [
        root / ".env",
        root.parent / "werewolf_nlp" / ".env",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k in allowed and v and not os.environ.get(k):
                os.environ[k] = v

    return {
        "api_key": os.environ.get("WEREWOLF_API_KEY", ""),
        "base_url": os.environ.get(
            "WEREWOLF_API_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
    }


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    finish_reason: str = "stop"


@dataclass
class UsageTracker:
    calls: int = 0
    retries: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_latency_sec: float = 0.0
    estimated_cost_rmb: float = 0.0
    model_names_seen: List[str] = field(default_factory=list)

    def record(self, resp: LLMResponse, pricing: tuple[float, float]) -> None:
        self.calls += 1
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.total_latency_sec += resp.latency_sec
        self.estimated_cost_rmb += (
            resp.prompt_tokens * pricing[0] + resp.completion_tokens * pricing[1]
        ) / 1_000_000.0
        if resp.model and resp.model not in self.model_names_seen:
            self.model_names_seen.append(resp.model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "retries": self.retries,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "total_latency_sec": round(self.total_latency_sec, 3),
            "estimated_cost_rmb": round(self.estimated_cost_rmb, 5),
            "model_names_seen": list(self.model_names_seen),
        }


class BudgetExceededError(RuntimeError):
    """Raised before making a call that would exceed max_budget_rmb."""


class BailianLLMClient:
    """OpenAI-compatible client for Alibaba Cloud Bailian with hard budget protection."""

    def __init__(
        self,
        model: str = "qwen3.5-27b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,
        timeout: float = 60.0,
        max_retries: int = 2,
        max_budget_rmb: float = 2.0,
        tracker: Optional[UsageTracker] = None,
    ) -> None:
        env = load_api_env()
        self.model = model
        self.api_key = api_key or env["api_key"]
        self.base_url = (base_url or env["base_url"]).rstrip("/")
        self.enable_thinking = enable_thinking
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_budget_rmb = max_budget_rmb
        self.pricing = MODEL_PRICING_RMB_PER_1M.get(model, (1.0, 5.0))
        self.tracker = tracker if tracker is not None else UsageTracker()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 180,
        temperature: float = 0.6,
    ) -> LLMResponse:
        if not self.api_key:
            raise RuntimeError("Missing WEREWOLF_API_KEY in environment or .env")
        if self.tracker.estimated_cost_rmb >= self.max_budget_rmb:
            raise BudgetExceededError(
                f"Estimated cost {self.tracker.estimated_cost_rmb:.4f} RMB reached cap {self.max_budget_rmb:.4f} RMB"
            )

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": self.enable_thinking,
        }
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        url = self.base_url + "/chat/completions"

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.tracker.retries += 1
                time.sleep(min(1.5 * attempt, 5.0))
            req = request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with request.urlopen(req, timeout=self.timeout) as resp_fp:
                    raw = json.load(resp_fp)
                latency = time.perf_counter() - t0
                choice = raw["choices"][0]
                text = choice["message"].get("content") or ""
                if not text.strip():
                    raise ValueError("Empty content from API")
                usage = raw.get("usage") or {}
                out = LLMResponse(
                    text=text.strip(),
                    model=str(raw.get("model") or self.model),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_sec=latency,
                    finish_reason=str(choice.get("finish_reason") or "stop"),
                )
                self.tracker.record(out, self.pricing)
                return out
            except error.HTTPError as exc:
                last_err = RuntimeError(f"HTTP {exc.code}")
                if exc.code in (400, 401, 403):
                    break
            except Exception as exc:
                last_err = exc

        self.tracker.failures += 1
        raise RuntimeError(f"LLM API failed after retries: {last_err}")


class MockLLMClient:
    """Zero-cost deterministic Mock LLM for offline end-to-end NL testing & cost estimation."""

    def __init__(self, model: str = "mock-qwen3.5-27b", tracker: Optional[UsageTracker] = None) -> None:
        self.model = model
        self.pricing = MODEL_PRICING_RMB_PER_1M["qwen3.5-27b"]
        self.tracker = tracker if tracker is not None else UsageTracker()

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 180,
        temperature: float = 0.6,
    ) -> LLMResponse:
        sys_txt = messages[0].get("content", "")
        usr_txt = messages[-1].get("content", "")
        est_in = max(20, (len(sys_txt) + len(usr_txt)) // 2)

        # 1. Parser prompt -> JSON intent
        if "发言意图解析器" in sys_txt:
            m_t = re.search(r"([1-9])\s*号", usr_txt)
            tgt = int(m_t.group(1)) if m_t else None
            if "预言家" in usr_txt and tgt:
                itype = "claim_seer_wolf" if ("查杀" in usr_txt or "狼" in usr_txt) else "claim_seer_good"
            elif tgt:
                itype = "claim_villager_accuse"
            else:
                itype, tgt = "stay_low", None
            text = json.dumps({"intent_type": itype, "target": tgt}, ensure_ascii=False)

        # 1b. MaKTO-style Think-before-Speak JSON prompt
        elif "display_role" in sys_txt and "speech" in sys_txt:
            text = json.dumps(
                {
                    "plan": "结合跨轮次笔记保持冷静逻辑表水，避免无故暴露身份。",
                    "display_role": "平民",
                    "vote_target": None,
                    "speech": "我是好人平民，结合目前场上各轮发言与实际票型，信息仍在收敛中，我先理性表水，听后面玩家补充逻辑后再统一归票找狼。",
                },
                ensure_ascii=False,
            )

        # 2. Candidate proposal or discrete decision prompt -> JSON
        elif "candidates" in usr_txt or "legal_action_ids" in usr_txt:
            m_ids = re.search(r"legal_action_ids[\"']?\s*[:：]\s*\[([0-9,\s]+)\]", usr_txt)
            if m_ids:
                ids = [int(x.strip()) for x in m_ids.group(1).split(",") if x.strip()]
                chosen = ids[1] if len(ids) > 1 and ids[0] == 0 else ids[0]
                cands = [chosen] + ([ids[0]] if ids[0] != chosen else (ids[1:2] if len(ids) > 1 else []))
            else:
                chosen, cands = 0, [0]
            if "candidates" in usr_txt:
                text = json.dumps({"candidates": cands[:2], "reason": "mock候选"}, ensure_ascii=False)
            else:
                text = json.dumps({"action_id": chosen, "reason": "mock推理"}, ensure_ascii=False)

        # 3. Speech generation prompt -> concise natural Chinese honoring instruction
        else:
            m_t = re.search(r"([1-9])\s*号", usr_txt.split("【本轮必须执行的战术意图】")[-1])
            t = int(m_t.group(1)) if m_t else None
            inst = usr_txt.split("【本轮必须执行的战术意图】")[-1]
            if "跳预言家" in inst and "查杀" in inst and t:
                text = f"我是预言家，昨晚我查验了{t}号，{t}号是查杀狼人！请大家相信我，今天全票把{t}号放逐出局。"
            elif "跳预言家" in inst and "金水" in inst and t:
                text = f"我跳预言家，昨晚验了{t}号是金水好人，请好人先认下{t}号，后面听前置位和后置位发言再定归票。"
            elif "跳女巫" in inst and "救" in inst and t:
                text = f"我跳女巫报信息，之前我用解药救过{t}号，{t}号是我的银水，今天不要错推{t}号。"
            elif "跳女巫" in inst and "毒" in inst and t:
                text = f"我拍女巫身份，{t}号发言破绽很大，我怀疑{t}号是狼，准备对{t}号用毒或请大家今天出{t}号。"
            elif "猎人" in inst and t:
                text = f"我拍猎人身份，{t}号在我视角里狼面非常大，今天建议大家重点投{t}号，若我出局会考虑带走{t}号。"
            elif "点踩" in inst and t:
                text = f"我是好人平民，结合目前发言和票型，我觉得{t}号非常可疑，建议今天集中票型投出{t}号。"
            elif "站边" in inst and t:
                text = f"我是平民表水，我觉得{t}号发言逻辑比较清晰，目前我愿意站边相信{t}号是好人。"
            else:
                text = "我是平民，目前场上信息还不够明朗，我先表水过麦，等后面神职起跳再根据发言和票型判断。"

        est_out = max(10, len(text) // 2)
        resp = LLMResponse(
            text=text,
            model=self.model,
            prompt_tokens=est_in,
            completion_tokens=est_out,
            latency_sec=0.001,
        )
        self.tracker.record(resp, self.pricing)
        return resp
