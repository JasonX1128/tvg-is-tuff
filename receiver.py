#!/usr/bin/env python3
"""Simple local HTTP receiver for testing webhooks.

Run:
    python3 receiver.py

It logs incoming POST payloads to stdout and returns 204.
"""
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Receiver(BaseHTTPRequestHandler):
    def _respond(self, code=204):
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain; charset=utf-8')
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length) if length else b''
        ct = self.headers.get('Content-Type', '')
        try:
            if 'application/json' in ct:
                parsed = json.loads(body.decode('utf-8')) if body else None
                pretty = json.dumps(parsed, indent=2) if parsed is not None else '<empty>'
            else:
                pretty = body.decode('utf-8', errors='replace')
        except Exception as e:
            pretty = f'<error parsing body: {e}>'

        print('--- WEBHOOK RECEIVED ---')
        print(f'Path: {self.path}')
        print('Headers:')
        for k, v in self.headers.items():
            print(f'  {k}: {v}')
        print('Body:')
        print(pretty)
        print('------------------------')

        self._respond(204)

    def log_message(self, format, *args):
        # keep logs minimal (we already print payload)
        pass

if __name__ == '__main__':
    port = 9000
    server = HTTPServer(('0.0.0.0', port), Receiver)
    print(f'Receiver listening on http://0.0.0.0:{port}/ — press Ctrl-C to quit')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down')
        server.server_close()
