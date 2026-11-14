#!/bin/sh
# A scripted MCP stdio server for test_mcp_stdio. Usage: scripted_server.sh LOG
# Every line the client sends is appended to LOG. Between the expected requests it
# sends what a real server may send: a non-JSON line, a notification, a request from
# the server, a response to another id. It answers the second tools/call with a
# JSON-RPC error that echoes $QA_M1_TEST_API_KEY, and exits before answering the third.
log="$1"
: > "$log"

read -r line; printf '%s\n' "$line" >> "$log"                 # initialize (id 0)
printf '%s\n' 'this line is not JSON'
printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"hi"}}'
printf '%s\n' '{"jsonrpc":"2.0","id":"srv-1","method":"roots/list"}'
read -r line; printf '%s\n' "$line" >> "$log"                 # our -32601 answer to srv-1
printf '%s\n' '{"jsonrpc":"2.0","id":99,"result":{}}'
printf '%s\n' '{"jsonrpc":"2.0","id":0,"result":{"protocolVersion":"2025-03-26","capabilities":{"tools":{}},"serverInfo":{"name":"scripted","version":"1.0"}}}'
read -r line; printf '%s\n' "$line" >> "$log"                 # notifications/initialized

read -r line; printf '%s\n' "$line" >> "$log"                 # tools/list (id 1)
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"one","description":"first","inputSchema":{"type":"object"}}],"nextCursor":"p2"}}'
read -r line; printf '%s\n' "$line" >> "$log"                 # tools/list (id 2, cursor p2)
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"two","inputSchema":{"type":"object","properties":{"x":{"type":"string"}}}}]}}'

read -r line; printf '%s\n' "$line" >> "$log"                 # tools/call one (id 3)
printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"a"},{"type":"text","text":"b"}],"isError":false}}'
read -r line; printf '%s\n' "$line" >> "$log"                 # tools/call two (id 4)
printf '{"jsonrpc":"2.0","id":4,"error":{"code":-32602,"message":"bad key %s"}}\n' "$QA_M1_TEST_API_KEY"
read -r line; printf '%s\n' "$line" >> "$log"                 # tools/call one (id 5): no answer
exit 0
