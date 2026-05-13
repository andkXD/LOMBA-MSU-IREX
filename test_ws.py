import websocket
import json
import threading
import time

def on_message(ws, msg):
    data = json.loads(msg)
    print(f"Type: {data.get('type')}")
    if data.get('type') == 'connected':
        print("  ✅ WebSocket Connected & Ready!")
    elif data.get('type') == 'decision':
        for lid, d in data['data']['decisions'].items():
            print(f"  {d['lane_name']}: {d['signal']} {d['green_duration']}s")

def on_open(ws):
    print("✅ WebSocket Connected!")
    def send_ping():
        while True:
            time.sleep(10)
            try:
                ws.send(json.dumps({"type": "ping"}))
            except:
                break
    threading.Thread(target=send_ping, daemon=True).start()

def on_error(ws, err):
    print(f"❌ Error: {err}")

def on_close(ws, code, msg):
    print(f"🔌 Disconnected (code={code}) — reconnecting in 3s...")

def run():
    while True:
        ws = websocket.WebSocketApp(
            "ws://localhost:8000/ws/live",
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()
        time.sleep(3)  # tunggu sebentar lalu reconnect

run()