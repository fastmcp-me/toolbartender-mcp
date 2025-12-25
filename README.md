[![Add to Cursor](https://fastmcp.me/badges/cursor_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)
[![Add to VS Code](https://fastmcp.me/badges/vscode_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)
[![Add to Claude](https://fastmcp.me/badges/claude_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)
[![Add to ChatGPT](https://fastmcp.me/badges/chatgpt_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)
[![Add to Codex](https://fastmcp.me/badges/codex_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)
[![Add to Gemini](https://fastmcp.me/badges/gemini_dark.svg)](https://fastmcp.me/MCP/Details/1550/toolbartender)

# ToolBartender 🍸
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![CI Status](https://github.com/sungminwoo0612/toolbartender-mcp/actions/workflows/deploy.yml/badge.svg)](https://github.com/sungminwoo0612/toolbartender-mcp/actions/workflows/deploy.yml)
[![Deploy (EC2)](https://img.shields.io/badge/Deploy-EC2-orange.svg)](https://aws.amazon.com/ec2/)

**Planner MCP that turns natural-language goals into safe, structured execution plans (JSON)**

ToolBartender는 사용자의 자연어 요청을 **하나의 goal**로 받아,<br>
**어떤 MCP 도구를 어떤 순서로 호출해야 하는지** 실행 계획(plan)을 생성하는 **Planner MCP**입니다.<br>

실제 도구 실행은 하지 않고, LLM/실행 에이전트가 **안전하고 예측 가능하게** 실행하도록<br>
`steps / assumptions / required_confirmations / execution_hint`를 포함한 JSON을 반환합니다.<br>

> 한 줄 요약: **ToolBartender = “도구 실행”이 아니라 “도구 조합 계획”을 만드는 MCP**


## PoC Endpoint
`http://mcp.toolbartender.dev/mcp` (Streamable HTTP / SSE)<br>  



## Keywords
`planner`, `safe execution`, `tool orchestration`, `PlayMCP`


## What it does
다음과 같은 복합 요청을 하나의 goal로 받아 plan을 생성합니다.
- 이동 계획 (예: “오늘 오후 8시 판교에서 강남으로 이동”)
- 일정 조회/조정
- 정보 탐색 및 요약
- 결과 전달(이메일/메신저 등)

### Output Plan includes
- `steps`: 사용할 MCP 도구 + 실행 순서
- `assumptions`: 전제 조건
- `required_confirmations`: 사용자 확인이 필요한 작업(특히 write 계열)
- `execution_hint`: 실행 에이전트용 가이드


## Exposed MCP tools (this server)
PlayMCP 등록 제약(정규식) 때문에 **ToolBartender가 노출하는 tool name은 ASCII로 고정**을 권장합니다.

- `plan_create` : goal + available_tools → plan 생성
- `plan_validate`: plan이 현재 컨텍스트(available_tools)에서 실행 가능한지 검증
- `plan_render_prompt`: LLM 실행 에이전트가 steps를 “순서대로” 호출하도록 프롬프트 생성
- `plan_explain`: 사용자에게 plan을 쉽게 설명

> 내부 `plan.steps[*].tool_name`은 다른 MCP들의 도구 이름이므로
> (예: `calendar.read`, `map.route`) 그대로 두는 구조가 자연스럽습니다.

도구 I/O, 스키마, 예시는 `docs/02_tools.md` 참고.


## Quickstart
### 1) Local run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/main.py
# Open http://localhost:3333/mcp in your browser
```

### 2) Inspector로 로컬 테스트
```bash
npx @modelcontextprotocol/inspector
# http://localhost:6274
```
Inspector UI에서 서버 URL을 http://localhost:3333/mcp로 연결합니다. 

### 3) PlayMCP 통합 테스트
- public URL 필요(ngrok/배포)
- PlayMCP → “새로운 MCP 서버 등록” → https://<public-host>/mcp 입력 → 임시 등록 → AI 채팅에 적용
흐름은 기존 README에 이미 있습니다. 


## Docs
- docs/00_quickstart.md : 5분 셋업
- docs/01_concept.md : Planner MCP 컨셉 / 철학 / 로드맵
- docs/02_tools.md : tool별 입력/출력/예시 + naming 규칙
- docs/03_playmcp.md : PlayMCP 등록 팁 / 심사 관점 체크리스트
- docs/04_deploy.md : 배포(nginx/SSE 포함) + 도메인/Cloudflare 포인트
- docs/05_troubleshooting.md : 405/핸드셰이크/SSE 관련 이슈 모음

## License
MIT
