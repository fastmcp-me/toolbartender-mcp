from __future__ import annotations

from typing import Dict, List

from fastmcp import FastMCP
from pydantic import BaseModel

from models import Plan
from planner import create_plan


# -------------------------
# IO models
# -------------------------
class PlanCreateInput(BaseModel):
    goal: str
    available_tools: List[str] = []


class PlanCreateOutput(BaseModel):
    plan: Plan


class PlanValidateInput(BaseModel):
    plan: Plan
    available_tools: List[str] = []


class PlanValidateOutput(BaseModel):
    ok: bool
    issues: List[str]
    missing_tools: List[str]


class PlanRenderPromptInput(BaseModel):
    plan: Plan
    available_tools: List[str] = []


class PlanRenderPromptOutput(BaseModel):
    ok: bool
    missing_tools: List[str]
    prompt: str


class PlanExplainInput(BaseModel):
    plan: Plan


class PlanExplainOutput(BaseModel):
    summary: str


# -------------------------
# helpers
# -------------------------
_TOOL_RATIONALE_PREFIX = {
    "calendar.": "캘린더 관련 작업을 처리하기 위해 사용합니다.",
    "map.": "이동 경로/시간을 조회하기 위해 사용합니다.",
    "paper.": "논문 검색/요약을 수행하기 위해 사용합니다.",
    "news.": "뉴스 검색/요약을 수행하기 위해 사용합니다.",
    "email.": "이메일 전송을 위해 사용합니다.",
    "kakao.": "카카오 채널/카톡 전송을 위해 사용합니다.",
}

_TOOL_RATIONALE_EXACT: Dict[str, str] = {
    "calendar.read": "목표 날짜의 일정/가용 여부를 확인하기 위해 사용합니다.",
    "map.route": "출발지→도착지 이동 경로/소요시간을 조회하기 위해 사용합니다.",
}


def _rationale(tool_name: str) -> str:
    if tool_name in _TOOL_RATIONALE_EXACT:
        return _TOOL_RATIONALE_EXACT[tool_name]
    for pfx, msg in _TOOL_RATIONALE_PREFIX.items():
        if tool_name.startswith(pfx):
            return msg
    return "이 작업을 수행하기 위해 사용합니다."


def _is_write_tool(tool_name: str) -> bool:
    # 보수적으로 write로 간주
    suffixes = (".create", ".write", ".update", ".delete", ".send")
    return tool_name.endswith(suffixes)


def _step_to_text(tool_name: str, args: dict) -> str:
    if tool_name == "calendar.read":
        return f"- calendar.read: date={args.get('date')}"
    if tool_name == "map.route":
        frm = args.get("from")
        to = args.get("to")
        depart_time = args.get("depart_time")
        if depart_time:
            return f"- map.route: {frm} → {to} (depart_time={depart_time})"
        return f"- map.route: {frm} → {to}"
    return f"- {tool_name}: args={args}"


# -------------------------
# MCP server
# -------------------------
mcp = FastMCP(
    "toolbartender",
    # FastMCP는 description 인자가 버전에 따라 없을 수 있어 instructions만 사용
    instructions="Mixing tools. Serving intent.",
)


@mcp.tool(
    name="plan_create", 
    description="사용자의 복합 자연어 목표에서 실행 순서와 MCP 도구 목록을 추출해 단계별 계획(JSON)을 생성합니다.",
)
def plan_create(input: PlanCreateInput) -> PlanCreateOutput:
    plan = create_plan(goal=input.goal, available_tools=input.available_tools)
    return PlanCreateOutput(plan=plan)



@mcp.tool(name="plan_validate", description="생성된 실행 계획(plan)이 현재 사용 가능한 MCP 도구로 실제 실행 가능한지 검증합니다. 필요한 도구가 누락되었거나, 실행 전에 사용자 확인이 필요한 작업이 있는지 등을 점검하여 실행 가능 여부와 주의사항(issues), 누락된 도구 목록을 반환합니다.")
def plan_validate(input: PlanValidateInput) -> PlanValidateOutput:
    available = set(input.available_tools or [])
    used = [s.tool_name for s in input.plan.steps]
    missing = sorted({t for t in used if t not in available})

    issues: List[str] = []
    if not available:
        issues.append("available_tools가 비어 있습니다: 활성화된 도구가 없으므로, 해당 환경에서는 계획을 실행할 수 없습니다.")
    if missing:
        issues.append(f"누락된 도구: {', '.join(missing)}")
    if not input.plan.steps:
        issues.append("plan.steps가 비어 있습니다: 실행할 단계가 없습니다 (필요한 도구가 활성화되지 않은 것일 수 있습니다).")

    ok = (len(missing) == 0) and bool(available) and bool(input.plan.steps)
    return PlanValidateOutput(ok=ok, issues=issues, missing_tools=missing)


@mcp.tool(
    name="plan_render_prompt",
    description=(
        "실행 에이전트나 LLM이 계획(plan)을 안전하게 실행할 수 있도록 실행용 프롬프트를 생성합니다. "
        "도구 사용 규칙, 실행 순서, 오류 발생 시 대응 방법 등 안전한 실행을 위한 안내 문장을 제공합니다."
    ),
)
def plan_render_prompt(input: PlanRenderPromptInput) -> PlanRenderPromptOutput:
    plan = input.plan
    available = set(input.available_tools or [])

    used_tools = [s.tool_name for s in plan.steps]
    used_unique = sorted(set(used_tools))

    if not available:
        missing = used_unique
    else:
        missing = sorted({t for t in used_unique if t not in available})

    ok = (len(missing) == 0) and bool(plan.steps) and bool(available)

    steps_text = (
        "\n".join(
            [
                f"{i+1}. {s.tool_name}  args={s.args}" + (f"  on_fail={s.on_fail}" if s.on_fail else "")
                for i, s in enumerate(plan.steps)
            ]
        )
        if plan.steps
        else "(no steps)"
    )

    # confirmation 리스트 + 휴리스틱(write tool)
    confirm_set = set(plan.required_confirmations or [])
    write_tools = [t for t in used_unique if _is_write_tool(t)]
    if write_tools:
        confirm_set.add("Write-like tools detected: " + ", ".join(write_tools))
    confirm_text = "\n".join(f"- {c}" for c in sorted(confirm_set)) if confirm_set else "(none)"

    prompt = f"""You are an execution agent that follows a provided MCP plan.

## Goal
{plan.intent}

## Available Tools (enabled)
{", ".join(sorted(available)) if available else "(unknown/empty)"}

## Plan Steps (execute sequentially)
{steps_text}

## Safety / Confirmation Gates
- Before calling any step that can modify data (create/write/update/delete/send) OR appears in required_confirmations:
  1) STOP and ask the user for explicit confirmation.
  2) Only proceed if the user confirms.
required_confirmations:
{confirm_text}

## Execution Rules
1) Do NOT invent tools. Only call the exact tool_name listed in each step.
2) Use args exactly as provided. Do not add new required fields unless the tool returns a schema error.
3) Execute steps in order. After each tool call, summarize the result in 1-3 bullets.
4) If a tool call fails:
   - If on_fail is provided, execute on_fail as the fallback.
   - Otherwise, stop and report the error + what you need from the user.
5) If any step tool is missing from available tools:
   - Stop. Tell the user which tools to enable: {", ".join(missing) if missing else "(none)"}.
6) Final output:
   - Provide a concise summary of all step results
   - Provide next action suggestions

Execution hint (from plan):
{plan.execution_hint or "(none)"}
"""

    return PlanRenderPromptOutput(ok=ok, missing_tools=missing, prompt=prompt)



@mcp.tool(
    name="plan_explain",
    description="실행 계획을 사용자에게 한국어로 설명합니다: 어떤 도구를 왜 사용하는지, 필요한 확인 사항은 무엇인지 안내합니다.",
)
def plan_explain(input: PlanExplainInput) -> PlanExplainOutput:
    plan = input.plan

    lines: List[str] = []
    lines.append(f"목표: {plan.intent}")

    if not plan.steps:
        lines.append("실행할 단계가 없습니다. (사용 가능한 도구 목록이 비었거나, 목표(goal) 해석에 실패한 경우일 수 있습니다.)")
    else:
        lines.append("실행 단계:")
        for i, s in enumerate(plan.steps, 1):
            lines.append(f"{i}) {s.tool_name} — {_rationale(s.tool_name)}")
            lines.append(_step_to_text(s.tool_name, s.args))

    if plan.required_confirmations:
        lines.append("확인 필요:")
        for c in plan.required_confirmations:
            lines.append(f"- {c}")

    if plan.assumptions:
        lines.append("가정/전제:")
        for a in plan.assumptions:
            lines.append(f"- {a}")

    return PlanExplainOutput(summary="\n".join(lines))


if __name__ == "__main__":
    # FastMCP streamable HTTP: transport="http", 기본 경로는 /mcp/ 이며 path로 커스텀 가능
    # https://gofastmcp.com/deployment/http
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=3333,
        path="/mcp",
    )
