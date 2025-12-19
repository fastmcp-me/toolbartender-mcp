# PlayMCP Integration Guide

## 핵심 오해 방지
PlayMCP는 “사용자의 서버 도구를 가져다 쓰는 곳”이지,<br>
사용자의 서버가 PlayMCP 도구함을 API로 긁어오는 구조가 아닙니다.

## 테스트 3단계
1) 로컬 단독 테스트: 서버가 살아있고 plan_create가 응답하는지
2) Inspector 테스트: schema/tools 노출 확인
3) PlayMCP 통합 테스트: public URL로 임시 등록 후 실제 호출 확인

## PlayMCP 등록 시 체크리스트
- MCP Server URL: `https://<public-host>/mcp`
- Tool name 규칙 준수: `^[a-zA-Z0-9_-]+$`
- tool description/용도는 한국어로 작성 권장(PlayMCP UX에 맞춤)
