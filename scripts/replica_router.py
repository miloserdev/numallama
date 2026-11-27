#!/usr/bin/env python3
"""Sticky HTTP router for numallama NUMA replicas.

    replica_router.py LISTEN_PORT BACKEND [BACKEND ...]
    BACKEND = host:port

Routes by x-session-id / x-opencode-session-id / x-conversation-id or the JSON
session_id/conversation_id, so a conversation keeps its KV cache on one replica.
Requests without a session use round-robin. Streaming responses (SSE) are relayed
chunk-by-chunk; non-streaming responses are buffered and get Content-Length.
"""
import argparse
import http.client
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BACKENDS = []
STICKY = {}
LOCK = threading.Lock()
RR = 0
SESSION_HEADERS = ("x-session-id", "x-opencode-session-id", "x-conversation-id")


def parse_backend(s):
    host, _, port = s.rpartition(":")
    if not host or not port.isdigit():
        raise argparse.ArgumentTypeError("backend must be host:port")
    return host, int(port)


def session_of(headers, body):
    for h in SESSION_HEADERS:
        v = headers.get(h)
        if v:
            return v.strip()
    try:
        o = json.loads(body.decode("utf-8")) if body else None
        if isinstance(o, dict):
            for k in ("session_id", "conversation_id"):
                v = o.get(k)
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return ""


def pick(sid):
    global RR
    with LOCK:
        if sid and sid in STICKY:
            return STICKY[sid]
        # Round-robin for new conversations; a hash can leave replicas idle.
        idx = RR % len(BACKENDS)
        RR += 1
        if sid:
            STICKY[sid] = idx
        return idx


def mark_down(idx):
    with LOCK:
        for k, v in list(STICKY.items()):
            if v == idx:
                del STICKY[k]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("[router] " + (fmt % args) + "\n")

    def _proxy(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        if n < 0 or n > (64 << 20):
            self.send_error(413)
            return
        body = self.rfile.read(n) if n else b""
        sid = session_of(self.headers, body)
        start = pick(sid)
        last = None
        for k in range(len(BACKENDS)):
            idx = (start + k) % len(BACKENDS)
            host, port = BACKENDS[idx]
            conn = None
            try:
                conn = http.client.HTTPConnection(host, port, timeout=600)
                hdrs = {k2: v for k2, v in self.headers.items()
                        if k2.lower() not in ("host", "connection", "content-length", "expect", "transfer-encoding")}
                hdrs["Content-Length"] = str(len(body))
                hdrs["Connection"] = "close"
                conn.request(self.command, self.path, body=body, headers=hdrs)
                resp = conn.getresponse()
                stream = (b'"stream":true' in body.replace(b" ", b"").lower() or
                          "text/event-stream" in (resp.getheader("Content-Type") or ""))
                if stream:
                    self.send_response_only(resp.status, resp.reason)
                    for k2, v in resp.getheaders():
                        if k2.lower() not in ("connection", "content-length", "transfer-encoding"):
                            self.send_header(k2, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    self.send_response_only(resp.status, resp.reason)
                    for k2, v in resp.getheaders():
                        if k2.lower() not in ("connection", "content-length", "transfer-encoding"):
                            self.send_header(k2, v)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                self.close_connection = True
                return
            except (OSError, http.client.HTTPException) as e:
                last = e
            finally:
                if conn is not None:
                    conn.close()
        if last is not None:
            sys.stderr.write(f"[router] all backends failed: {last}\n")
        self.send_error(502)

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("backends", nargs="+", type=parse_backend)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    global BACKENDS
    BACKENDS = a.backends
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"[router] {a.host}:{a.port} -> {', '.join(f'{h}:{p}' for h, p in BACKENDS)}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
