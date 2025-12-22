from __future__ import annotations

from typing import Dict, List, Optional

from fastmcp import FastMCP
from pydantic import BaseModel, Field

from models import Plan
from planner import create_plan, parse_goal


# -------------------------
# Output models (tool results)
# -------------------------
class ParseGoalOutput(BaseModel):
    intent_type: str
    date_token: str
    time_hhmm: Optional[str] = None
    origin: Optional[str] = None
    destination: Optional[str] = None
    title: Optional[str] = None
    email: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class PlanCreateOutput(BaseModel):
    plan: Plan


class PlanValidateOutput(BaseModel):
    ok: bool
    issues: List[str] = Field(default_factory=list)
    missing_tools: List[str] = Field(default_factory=list)


class PlanRenderPromptOutput(BaseModel):
    ok: bool
    missing_tools: List[str] = Field(default_factory=list)
    prompt: str


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

# NOTE:
# PlayMCP AI 채팅은 tool call arguments를 자동으로 구성해야 하므로,
# input을 중첩(BaseModel 파라미터 1개) 형태로 받으면 LLM이 쉽게 실패합니다.
# 따라서 아래 Tool API는 "flat arguments"를 우선으로 설계합니다.
#
# MCP 스펙 관점에서도 tools/call의 params.arguments는 inputSchema에 맞는 object를 기대합니다.
# (goal: string 같은 단순한 required 필드가 가장 안정적)


# -------------------------
# 1) Parse: goal만 받아서 slot/intent를 추출
# -------------------------
@mcp.tool(
    name="tb_parse_goal",
    description="자연어 goal에서 intent/slot(출발/도착/시간/날짜 등)을 추출합니다. (PlayMCP 친화: goal 문자열 1개 입력)",
)
def tb_parse_goal(goal: str) -> ParseGoalOutput:
    data = parse_goal(goal)
    return ParseGoalOutput(**data)


# -------------------------
# 2) Plan: goal(+optional available_tools)로 실행 계획 생성
# -------------------------
@mcp.tool(
    name="tb_plan_create",
    description=(
        "자연어 goal에서 실행 순서와 도구 호출 단계를 포함한 plan(JSON)을 생성합니다. "
        "available_tools를 제공하면 해당 도구만 사용하도록 제한합니다(엄격 모드)."
    ),
)
def tb_plan_create(goal: str, available_tools: Optional[List[str]] = None) -> PlanCreateOutput:
    strict = bool(available_tools)
    plan = create_plan(goal=goal, available_tools=available_tools, strict_available_tools=strict)
    return PlanCreateOutput(plan=plan)


# -------------------------
# 3) Validate/Render/Explain: 실행 가능성 점검 + 실행 프롬프트 + 사용자 설명
# -------------------------
@mcp.tool(
    name="tb_plan_validate",
    description=(
        "생성된 plan이 현재 available_tools로 실행 가능한지 검증합니다. "
        "available_tools가 비어 있으면 plan.steps에 포함된 도구 전체를 missing으로 반환합니다."
    ),
)
def tb_plan_validate(plan: Plan, available_tools: Optional[List[str]] = None) -> PlanValidateOutput:
    available = set(available_tools or [])
    used = [s.tool_name for s in plan.steps]
    missing = sorted({t for t in used if (not available) or (t not in available)})

    issues: List[str] = []
    if not available_tools:
        issues.append("available_tools가 비어 있습니다: 실행 환경에서 활성화된 도구 목록을 전달하지 못해 missing 판단을 보수적으로 처리했습니다.")
    if missing:
        issues.append(f"누락된 도구(또는 미확인): {', '.join(missing)}")
    if not plan.steps:
        issues.append("plan.steps가 비어 있습니다: 실행할 단계가 없습니다 (goal 해석 실패 또는 도구 제한 때문일 수 있습니다).")

    ok = (len(missing) == 0) and bool(plan.steps) and bool(available_tools)
    return PlanValidateOutput(ok=ok, issues=issues, missing_tools=missing)


@mcp.tool(
    name="tb_plan_render_prompt",
    description=(
        "실행 에이전트/LLM이 plan을 안전하게 실행하도록 실행용 프롬프트를 생성합니다. "
        "도구 사용 규칙, 실행 순서, 오류 대응, 사용자 확인 게이트를 포함합니다."
    ),
)
def tb_plan_render_prompt(plan: Plan, available_tools: Optional[List[str]] = None) -> PlanRenderPromptOutput:
    available = set(available_tools or [])
    used_tools = [s.tool_name for s in plan.steps]
    used_unique = sorted(set(used_tools))

    if not available_tools:
        missing = used_unique
    else:
        missing = sorted({t for t in used_unique if t not in available})

    ok = (len(missing) == 0) and bool(plan.steps) and bool(available_tools)

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
    name="tb_plan_explain",
    description="plan을 사용자에게 한국어로 설명합니다: 어떤 도구를 왜 사용하는지, 필요한 확인 사항은 무엇인지 안내합니다.",
)
def tb_plan_explain(plan: Plan) -> PlanExplainOutput:
    lines: List[str] = []
    lines.append(f"목표: {plan.intent}")

    if not plan.steps:
        lines.append("실행할 단계가 없습니다. (goal 해석 실패 또는 도구 제한/미제공 때문일 수 있습니다.)")
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
