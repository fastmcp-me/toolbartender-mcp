# Concept

## ToolBartender는 무엇인가
ToolBartender는 “도구 실행”이 아니라 “도구 조합 계획”을 만드는 Planner MCP입니다.<br>
goal + available_tools를 받아, 실행 에이전트가 순차 실행 가능한 JSON plan을 생성합니다.

## 핵심 설계
### 1) Planner Only
- ToolBartender는 도구를 직접 실행하지 않음
- 결과는 plan 생성/검증/설명/프롬프트 생성에 한정

### 2) Safe-by-design
- create/update/delete 등 write 성격 작업은 `required_confirmations`로 분리

### 3) Catalog 입력 방식(현실적인 통합 모델)
PlayMCP의 도구함을 “조회”하는 게 아니라,<br>
클라이언트(PlayMCP/사용자)가 available_tools(이름/설명/스키마)를 전달하고<br>
ToolBartender는 그 컨텍스트에서 plan 품질을 끌어올리는 역할을 하는 구조가 가장 설득력 있습니다.

## 로드맵(권장)
- P0(필수): plan_create / plan_validate / plan_render_prompt 품질 강화
- P1(차별화): plan_refine(실패 복구) + trace_export(리포트)