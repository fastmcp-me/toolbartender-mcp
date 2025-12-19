# Tools

> 주의: PlayMCP 등록 제약 때문에, ToolBartender가 “노출하는 MCP tool name”은
> `^[a-zA-Z0-9_-]+$`를 만족하도록 `plan_create` 같은 형태를 권장합니다.

## 1) plan_create
### Input
- goal: string
- available_tools: string[]

### Output
- plan: { plan_id, intent, steps, assumptions, required_confirmations, execution_hint }

## 2) plan_validate
### Input
- plan
- available_tools

### Output
- ok: boolean
- issues: string[]
- missing_tools: string[]

## 3) plan_render_prompt
### 목적
LLM 실행 에이전트가 steps를 “순서대로” 호출하도록 실행 프롬프트를 생성합니다.

### Output
- ok, missing_tools, prompt

## 4) plan_explain
### 목적
사용자에게 “왜 이 도구들을 쓰는지 / 무엇을 확인해야 하는지” 요약합니다.

---

## Example plan (simplified)
기존 README에 있는 예시를 그대로 사용해도 됩니다.
