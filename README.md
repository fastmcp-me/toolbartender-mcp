# ToolBartender (MCP)

PlayMCP에서 여러 MCP 도구를 **안전하게 조합**해서 실행할 수 있도록,<br>
`goal` + `available_tools`를 입력으로 받아 **실행 계획(plan)** 을 만드는 MCP 서버입니다.

# PoC 링크
http://54.180.123.112/mcp

## 제공 도구

- `plan.create`
  - 입력: `{ goal: str, available_tools: [str] }`
  - 출력: `{ plan: { plan_id, intent, steps, assumptions, required_confirmations, execution_hint } }`

- `plan.validate`
  - 입력: `{ plan: Plan, available_tools: [str] }`
  - 출력: `{ ok, issues, missing_tools }`

- `plan.render_prompt`
  - 입력: `{ plan: Plan, available_tools: [str] }`
  - 출력: `{ ok, missing_tools, prompt }`
  - LLM 실행 에이전트가 **step을 순서대로 호출**하게 만드는 지시문을 생성합니다.

- `plan.explain`
  - 입력: `{ plan: Plan }`
  - 출력: `{ summary }`
  - 사용자에게 “무슨 도구를 왜 쓰는지 / 무엇을 확인해야 하는지”를 요약해줍니다.

## 로컬 실행

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py
# 기본: http://localhost:3333/mcp
```

서버는 Streamable HTTP로 뜹니다. FastMCP는 기본 경로가 `/mcp/`이고,<br> 
이 프로젝트는 `path="/mcp"`로 맞춰둔 상태입니다.

## Inspector로 로컬 테스트 (PlayMCP 역할 흉내)

MCP Inspector는 Node 기반 도구입니다. GitHub 문서 기준으로 아래처럼 실행합니다.

```bash
# UI 실행
npx @modelcontextprotocol/inspector
# 브라우저: http://localhost:6274
```

Inspector UI에서 **서버 URL**을 `http://localhost:3333/mcp` 로 연결해<br>
도구 목록/스키마/호출을 확인합니다.

## PlayMCP 등록/테스트 흐름

1) 서버를 외부에서 접근 가능한 URL로 공개 (예: ngrok, Cloud Run 등)  
2) PlayMCP → “새로운 MCP 서버 등록” → MCP Server URL에 `https://<public-host>/mcp` 입력  
3) “임시 등록” 상태로 저장 → “MCP 상세 미리보기” → “AI 채팅에 적용”으로 **나만 테스트**  
4) 충분히 검증되면 심사 요청 → 게시

> 중요한 점: “내 서버가 PlayMCP 도구함 목록을 API로 긁어오는 구조”가 아니라,
> PlayMCP(클라이언트)가 “내 서버 URL로 도구 호출”을 하는 구조입니다.
