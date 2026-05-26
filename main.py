"""
B2B 智能获客自动化系统 — LangGraph MVP
LLM: OpenRouter + DeepSeek V4 Flash (via ChatOpenAI-compatible endpoint)
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

BAIEN_CONTEXT = """
百恩 (BAIEN) 是一家专注于工业自动化与智能制造解决方案的 B2B 科技企业。
核心产品：PLC 编程服务、产线数字化改造、预测性维护 SaaS。
目标客户：汽车零部件、电子制造、物流仓储等行业的中大型制造企业。
价值主张：帮助客户降低停机损失 15%+，缩短产线部署周期 30%。
""".strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_NAME = "deepseek/deepseek-v4-flash"


def load_dotenv(path: str = ".env") -> None:
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def build_llm() -> ChatOpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")

    return ChatOpenAI(
        model=MODEL_NAME,
        openai_api_base=OPENROUTER_BASE_URL,
        openai_api_key=api_key,
        temperature=0.4,
        extra_body={"reasoning": {"enabled": True}},
        default_headers={
            "HTTP-Referer": "https://github.com/ryanisnew/B2B-Sales-Agent",
            "X-Title": "B2B-Sales-Agent",
        },
    )


# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    company_list: list[str]
    current_company: str
    lead_info: dict
    generated_pitch: dict
    review_passed: bool
    corrections_feedback: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"Could not parse JSON from model output:\n{text}")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def lead_research_node(state: AgentState) -> dict:
    """Mock: pick a target company and fabricate a contact profile."""
    companies = state.get("company_list") or [
        "Valeo",
        "Bosch",
        "Continental",
        "Magna",
    ]
    target = "Valeo" if "Valeo" in companies else companies[0]

    lead_info = {
        "company": target,
        "contact_name": "Marie Dubois",
        "title": "Head of Manufacturing Engineering",
        "industry": "Automotive Tier-1 Supplier",
        "pain_points": [
            "Legacy PLC code slowing new line ramp-up",
            "Unplanned downtime on assembly cells",
        ],
        "recent_signal": "Posted about Industry 4.0 roadmap on LinkedIn last week",
    }

    return {
        "company_list": companies,
        "current_company": target,
        "lead_info": lead_info,
        "review_passed": False,
        "corrections_feedback": "",
    }


def pitch_generation_node(state: AgentState) -> dict:
    """Generate a personalized outreach pitch using the OpenRouter LLM."""
    llm = build_llm()
    lead = state["lead_info"]
    feedback = state.get("corrections_feedback", "").strip()

    feedback_block = ""
    if feedback:
        feedback_block = f"""
上一轮合规审核未通过，请根据以下反馈修改话术：
{feedback}
"""

    system = SystemMessage(
        content=(
            "你是百恩 (BAIEN) 的资深 B2B 销售文案专家。"
            "根据线索信息撰写一封简洁、专业、个性化的中文开发信（邮件正文）。"
            "必须提及对方公司与具体痛点，并自然衔接百恩的价值主张。"
            "只输出 JSON，不要 markdown 代码块。格式："
            '{"subject": "...", "body": "...", "call_to_action": "..."}'
        )
    )
    human = HumanMessage(
        content=f"""
百恩公司背景：
{BAIEN_CONTEXT}

目标线索：
{json.dumps(lead, ensure_ascii=False, indent=2)}
{feedback_block}
请生成开发信 JSON。
"""
    )

    response = llm.invoke([system, human])
    pitch = _extract_json(response.content)

    return {
        "generated_pitch": pitch,
        "review_passed": False,
    }


def 合规审核_node(state: AgentState) -> dict:
    """LLM critic: check personalization and compliance; return Yes/No."""
    llm = build_llm()
    lead = state["lead_info"]
    pitch = state.get("generated_pitch", {})

    system = SystemMessage(
        content=(
            "你是 B2B 外联合规与质量审核员。"
            "检查开发信是否：1) 提及目标客户公司与联系人相关信息；"
            "2) 无夸大承诺、无敏感/歧视性表述；3) 语气专业克制。"
            "只输出 JSON："
            '{"decision": "Yes" 或 "No", "feedback": "若 No 则给出具体修改建议，Yes 则为空字符串"}'
        )
    )
    human = HumanMessage(
        content=f"""
线索信息：
{json.dumps(lead, ensure_ascii=False, indent=2)}

待审核开发信：
{json.dumps(pitch, ensure_ascii=False, indent=2)}

百恩合规要求：不得承诺具体 ROI 数字除非有合同依据；不得虚构客户案例。
"""
    )

    response = llm.invoke([system, human])
    review = _extract_json(response.content)

    decision = str(review.get("decision", "No")).strip().lower()
    passed = decision in ("yes", "y", "通过", "true")

    return {
        "review_passed": passed,
        "corrections_feedback": "" if passed else str(review.get("feedback", "需要更具体的个性化与合规表述。")),
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def route_after_review(state: AgentState) -> Literal["pitch_generation_node", "__end__"]:
    if state.get("review_passed"):
        return "__end__"
    return "pitch_generation_node"


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("lead_research_node", lead_research_node)
    graph.add_node("pitch_generation_node", pitch_generation_node)
    graph.add_node("合规审核_node", 合规审核_node)

    graph.add_edge(START, "lead_research_node")
    graph.add_edge("lead_research_node", "pitch_generation_node")
    graph.add_edge("pitch_generation_node", "合规审核_node")
    graph.add_conditional_edges(
        "合规审核_node",
        route_after_review,
        {
            "pitch_generation_node": "pitch_generation_node",
            "__end__": END,
        },
    )

    return graph.compile()


def main() -> None:
    load_dotenv()
    app = build_graph()

    print("=" * 60)
    print("B2B 智能获客自动化系统 — LangGraph MVP")
    print(f"Model: {MODEL_NAME} @ OpenRouter")
    print("=" * 60)

    result = app.invoke(
        {
            "company_list": ["Valeo", "Bosch", "Continental"],
            "current_company": "",
            "lead_info": {},
            "generated_pitch": {},
            "review_passed": False,
            "corrections_feedback": "",
        }
    )

    print("\n[线索信息]")
    print(json.dumps(result["lead_info"], ensure_ascii=False, indent=2))

    print("\n[生成的开发信]")
    print(json.dumps(result["generated_pitch"], ensure_ascii=False, indent=2))

    print(f"\n[合规审核] 通过: {result['review_passed']}")
    if result.get("corrections_feedback"):
        print(f"反馈: {result['corrections_feedback']}")

    print("\n完成。")


if __name__ == "__main__":
    main()
