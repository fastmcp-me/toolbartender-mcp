# Deployment Notes

## 목표
- public HTTPS endpoint: `https://mcp.toolbartender.dev/mcp`
- SSE(Streamable HTTP) 동작 보장

## 권장 구성(요약)
- App: FastMCP server (port 3333)
- Reverse proxy: nginx
- TLS: Cloudflare(Full/Strict) 또는 Let's Encrypt
- DNS: Cloudflare A/AAAA 레코드 + 프록시 여부는 SSE 안정성 기준으로 선택

## nginx 포인트
- SSE는 버퍼링/타임아웃에 민감하므로
  - proxy_buffering off
  - 적절한 proxy_read_timeout
  - Connection 헤더 처리
같은 설정이 필요할 수 있음(환경에 따라).<br>
(상세는 프로젝트의 실제 nginx conf에 맞춰 업데이트 권장)
