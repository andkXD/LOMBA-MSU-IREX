"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC Backend — FastAPI Main Application                    ║
║  File: main.py                                               ║
║                                                              ║
║  Cara jalankan:                                              ║
║  pip install fastapi uvicorn paho-mqtt influxdb-client       ║
║             python-dotenv websockets numpy                   ║
║  uvicorn main:app --reload --host 0.0.0.0 --port 8000        ║
║                                                              ║
║  Docs otomatis: http://localhost:8000/docs                   ║
╚══════════════════════════════════════════════════════════════╝

ENDPOINT SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
POST /api/decide          → Manual input → NERIC decision
POST /api/simulate        → Auto-simulate N cycles
POST /api/emergency/route → Set emergency pre-routing
DELETE /api/emergency/route → Clear emergency route
GET  /api/status          → System status
GET  /api/history/sensors → Sensor history from InfluxDB
GET  /api/history/decisions → Decision history
WS   /ws/live             → Real-time WebSocket stream
"""

import asyncio
import json
import logging
import random
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Optional, Dict, Set

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import (
    ManualInputRequest, EmergencyRouteRequest, SimulationRequest,
    NERICDecisionResponse, SystemStatusResponse, EmergencyRouteResponse,
    ScenarioType, NERICConfig
)
from brain import NERICBrain
from influx_service import InfluxService
from mqtt_service import MQTTService

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)
logger = logging.getLogger("neric.api")


# ══════════════════════════════════════════════════════════════
# APPLICATION STATE (singleton services)
# ══════════════════════════════════════════════════════════════

brain   = NERICBrain()
influx  = InfluxService()
mqtt    = MQTTService()

# WebSocket connection manager
class ConnectionManager:
    """Kelola semua koneksi WebSocket aktif dari React clients."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.add(ws)
        logger.info(f"WebSocket connected. Total: {len(self.active)}")

    def disconnect(self, ws: WebSocket) -> None:
        self.active.discard(ws)
        logger.info(f"WebSocket disconnected. Total: {len(self.active)}")

    async def broadcast(self, data: dict) -> None:
        """Broadcast ke semua client yang terhubung."""
        if not self.active:
            return
        msg = json.dumps(data)
        dead = set()
        for ws in self.active.copy():
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.active.discard(ws)

ws_manager = ConnectionManager()


# ══════════════════════════════════════════════════════════════
# LIFESPAN (startup / shutdown)
# ══════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup dan shutdown hooks."""
    logger.info("═" * 50)
    logger.info("  NERIC Backend Starting...")
    logger.info("═" * 50)

    # Connect ke InfluxDB
    influx_ok = influx.connect()
    logger.info(f"  InfluxDB: {'✓ Connected' if influx_ok else '✗ Not available (mock mode)'}")

    # Connect ke MQTT broker
    mqtt_ok = mqtt.connect()
    logger.info(f"  MQTT:     {'✓ Connected' if mqtt_ok else '✗ Not available (simulation mode)'}")

    # Set MQTT callback untuk trigger NERIC saat ada data ESP32
    async def on_sensor_from_esp32(lane_id: int, count: int, payload: dict):
        """Dipanggil saat ESP32 kirim data sensor via MQTT."""
        sensor_data = mqtt.get_latest_sensor_data()
        result = brain.decide(sensor_data)

        # Simpan ke InfluxDB
        influx.write_sensor_data(brain.get_neuron_data(), result.cycle)
        influx.write_decisions(result)

        # Publish signal balik ke semua ESP32
        for lane_id, decision in result.decisions.items():
            mqtt.publish_signal(
                lane_id=lane_id,
                signal=decision.signal.value,
                duration=decision.green_duration,
                emergency=result.emergency,
            )

        # Broadcast ke React dashboard via WebSocket
        await ws_manager.broadcast({
            "type":  "decision",
            "data":  result.dict(),
            "neurons": brain.get_neuron_data(),
        })

    mqtt.on_sensor_data = lambda lid, cnt, p: asyncio.create_task(
        on_sensor_from_esp32(lid, cnt, p)
    )

    logger.info("  NERIC Brain: ✓ Ready")
    logger.info("═" * 50)
    logger.info("  API Docs: http://localhost:8000/docs")
    logger.info("  WebSocket: ws://localhost:8000/ws/live")
    logger.info("═" * 50)

    yield  # ← Aplikasi berjalan di sini

    # Shutdown
    logger.info("NERIC Backend shutting down...")
    mqtt.disconnect()
    influx.close()


# ══════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="NERIC API",
    description="""
    ## NERIC — Neuromorphic-Inspired Responsive Intersection Controller

    Backend API untuk sistem manajemen lalu lintas berbasis AI neuromorphic.

    ### Fitur Utama
    - **Manual Input**: Kirim data kepadatan kendaraan → dapatkan keputusan NERIC
    - **WebSocket**: Stream real-time keputusan ke React dashboard
    - **Emergency Routing**: Pre-set rute ambulans → green corridor otomatis
    - **Simulation**: Jalankan simulasi otomatis berbagai skenario
    - **History**: Query data historis dari InfluxDB
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# CORS: izinkan React development server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# HELPER: Generate scenario data
# ══════════════════════════════════════════════════════════════

def generate_scenario_data(scenario: ScenarioType) -> List[int]:
    """Generate sensor data sintetis berdasarkan skenario."""
    MAX = NERICConfig.MAX_VEHICLES

    if scenario == ScenarioType.RUSH_HOUR:
        return [int(np.clip(random.gauss(16, 1.5), 0, MAX)) for _ in range(4)]

    elif scenario == ScenarioType.EMERGENCY:
        return [
            int(np.clip(random.gauss(18, 0.8), 17, MAX)),
            int(np.clip(random.gauss(5,  2.0), 0,  MAX)),
            int(np.clip(random.gauss(4,  1.5), 0,  MAX)),
            int(np.clip(random.gauss(3,  1.0), 0,  MAX)),
        ]
    elif scenario == ScenarioType.IMBALANCED:
        return [
            int(np.clip(random.gauss(17, 1.5), 0, MAX)),
            int(np.clip(random.gauss(2,  1.0), 0, MAX)),
            int(np.clip(random.gauss(3,  1.0), 0, MAX)),
            int(np.clip(random.gauss(1,  0.8), 0, MAX)),
        ]
    elif scenario == ScenarioType.NIGHT:
        return [int(np.clip(random.gauss(2, 1), 0, MAX)) for _ in range(4)]
    else:  # NORMAL / CUSTOM
        return [int(np.clip(random.gauss(b, 2.5), 0, MAX))
                for b in [8, 7, 4, 3]]


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get("/", tags=["General"])
async def root():
    """Health check endpoint."""
    return {
        "service":   "NERIC Backend",
        "version":   "1.0.0",
        "status":    "running",
        "timestamp": datetime.now().isoformat(),
        "docs":      "/docs",
        "websocket": "/ws/live",
    }


# ── CORE DECISION ENDPOINT ────────────────────────────────────

@app.post(
    "/api/decide",
    response_model=NERICDecisionResponse,
    tags=["Core"],
    summary="Input manual → keputusan NERIC",
    description="""
    Endpoint utama untuk input-based testing.

    Kirim jumlah kendaraan per jalur → NERIC Brain proses
    dengan algoritma neuromorphic LIF → return keputusan lampu.

    **Ini yang dipakai React dashboard saat user geser slider!**
    """
)
async def decide(request: ManualInputRequest) -> NERICDecisionResponse:
    """
    Proses data sensor manual → keputusan NERIC.

    Input:
    - vehicle_counts: [utara, selatan, timur, barat] (0-20 per jalur)
    - simulated_hour: override jam (opsional, untuk testing threshold)
    - scenario: label skenario (opsional, hanya untuk metadata)
    """
    # Validasi input
    if len(request.vehicle_counts) != 4:
        raise HTTPException(
            status_code=422,
            detail="vehicle_counts harus berisi tepat 4 nilai (Utara, Selatan, Timur, Barat)"
        )

    for i, count in enumerate(request.vehicle_counts):
        if not 0 <= count <= NERICConfig.MAX_VEHICLES:
            raise HTTPException(
                status_code=422,
                detail=f"vehicle_counts[{i}] harus antara 0-{NERICConfig.MAX_VEHICLES}"
            )

    # Proses neuromorphic
    result = brain.decide(
        sensor_data=request.vehicle_counts,
        simulated_hour=request.simulated_hour,
    )

    # Simpan ke InfluxDB (async, tidak block response)
    influx.write_sensor_data(brain.get_neuron_data(), result.cycle)
    influx.write_decisions(result)

    # Publish ke ESP32 via MQTT
    for lane_id, decision in result.decisions.items():
        mqtt.publish_signal(
            lane_id=lane_id,
            signal=decision.signal.value,
            duration=decision.green_duration,
            emergency=result.emergency,
        )

    # Broadcast ke React dashboard via WebSocket
    await ws_manager.broadcast({
        "type":    "decision",
        "data":    result.dict(),
        "neurons": brain.get_neuron_data(),
    })

    return result


@app.post(
    "/api/simulate",
    tags=["Core"],
    summary="Jalankan simulasi otomatis",
    description="Generate N siklus simulasi dengan skenario tertentu. Berguna untuk demo dan testing."
)
async def simulate(request: SimulationRequest) -> dict:
    """
    Jalankan simulasi otomatis N siklus.
    Setiap siklus di-broadcast ke WebSocket.
    """
    if request.n_cycles > 100:
        raise HTTPException(
            status_code=422,
            detail="Maksimal 100 siklus per request simulasi"
        )

    results = []
    for i in range(request.n_cycles):
        sensor_data = generate_scenario_data(request.scenario)

        result = brain.decide(
            sensor_data=sensor_data,
            simulated_hour=request.simulated_hour,
        )

        influx.write_sensor_data(brain.get_neuron_data(), result.cycle)
        influx.write_decisions(result)

        await ws_manager.broadcast({
            "type":    "simulation_step",
            "step":    i + 1,
            "total":   request.n_cycles,
            "data":    result.dict(),
            "neurons": brain.get_neuron_data(),
        })

        results.append({
            "cycle":     result.cycle,
            "emergency": result.emergency,
            "decisions": {
                k: {
                    "signal":         v.signal.value,
                    "green_duration": v.green_duration,
                    "priority_rank":  v.priority_rank,
                }
                for k, v in result.decisions.items()
            }
        })

        # Jeda kecil agar WebSocket bisa stream dengan smooth
        await asyncio.sleep(0.3)

    return {
        "message":    f"Simulasi {request.n_cycles} siklus selesai",
        "scenario":   request.scenario.value,
        "total_cycles": request.n_cycles,
        "results":    results,
        "brain_stats": brain.get_stats(),
    }


# ── EMERGENCY ROUTING ─────────────────────────────────────────

@app.post(
    "/api/emergency/route",
    response_model=EmergencyRouteResponse,
    tags=["Emergency"],
    summary="Set emergency pre-routing (ambulans)",
    description="""
    Aktifkan green corridor untuk rute ambulans.

    Sebelum ambulans berangkat, pengemudi set rute via aplikasi.
    NERIC akan proaktif hijau-kan semua lampu di sepanjang rute.
    """
)
async def set_emergency_route(request: EmergencyRouteRequest) -> EmergencyRouteResponse:
    """Aktivasi emergency pre-routing untuk ambulans."""

    if not request.route_lanes:
        raise HTTPException(status_code=422, detail="route_lanes tidak boleh kosong")

    for lane in request.route_lanes:
        if not 0 <= lane <= 3:
            raise HTTPException(
                status_code=422,
                detail=f"Lane ID {lane} tidak valid. Gunakan 0=Utara, 1=Selatan, 2=Timur, 3=Barat"
            )

    # Aktivasi di NERIC Brain
    brain.set_emergency_route(request.route_lanes)

    # Log ke InfluxDB
    influx.write_emergency_event(
        event_type="pre_routing_activated",
        lane_ids=request.route_lanes,
        details={
            "destination":    request.destination,
            "priority_level": request.priority_level,
        }
    )

    # Broadcast ke semua client
    await ws_manager.broadcast({
        "type":        "emergency_route_activated",
        "route_lanes": request.route_lanes,
        "destination": request.destination,
    })

    # Estimasi waktu clearance (GREEN_MAX per jalur dalam rute)
    estimated_seconds = len(request.route_lanes) * NERICConfig.GREEN_MAX

    return EmergencyRouteResponse(
        success=True,
        message=f"Green corridor aktif menuju {request.destination}",
        route_lanes=request.route_lanes,
        estimated_clearance_seconds=estimated_seconds,
        affected_intersections=len(request.route_lanes),
    )


@app.delete(
    "/api/emergency/route",
    tags=["Emergency"],
    summary="Batalkan emergency route"
)
async def clear_emergency_route() -> dict:
    """Batalkan emergency pre-routing yang aktif."""
    brain.clear_emergency_route()
    await ws_manager.broadcast({"type": "emergency_route_cleared"})
    return {"success": True, "message": "Emergency route dibatalkan"}


@app.post(
    "/api/emergency/advance",
    tags=["Emergency"],
    summary="Maju ke jalur berikutnya dalam rute"
)
async def advance_emergency_route() -> dict:
    """
    Maju ke persimpangan berikutnya dalam rute ambulans.
    Dipanggil saat ambulans sudah melewati satu persimpangan.
    """
    has_more = brain.advance_route()
    await ws_manager.broadcast({
        "type":     "emergency_route_advanced",
        "has_more": has_more,
        "step":     brain.route_step if has_more else "completed",
    })
    return {
        "success":  True,
        "has_more": has_more,
        "message":  "Maju ke jalur berikutnya" if has_more else "Rute selesai",
    }


# ── STATUS & STATS ────────────────────────────────────────────

@app.get(
    "/api/status",
    tags=["Monitoring"],
    summary="Status sistem NERIC"
)
async def get_status() -> dict:
    """Ambil status lengkap sistem NERIC."""
    stats = brain.get_stats()
    return {
        "running":          True,
        "timestamp":        datetime.now().isoformat(),
        "mqtt_connected":   mqtt.connected,
        "influxdb_connected": influx.connected,
        "ws_clients":       len(ws_manager.active),
        **stats,
    }


@app.get(
    "/api/neurons",
    tags=["Monitoring"],
    summary="Data neuron saat ini"
)
async def get_neurons() -> dict:
    """Ambil data lengkap semua neuron (V_m, spike, history, dll)."""
    return {
        "cycle":   brain.cycle_count,
        "neurons": brain.get_neuron_data(),
    }


# ── HISTORY (InfluxDB queries) ────────────────────────────────

@app.get(
    "/api/history/sensors",
    tags=["History"],
    summary="Riwayat data sensor"
)
async def get_sensor_history(
    lane_name: Optional[str] = None,
    minutes: int = 10
) -> dict:
    """
    Query riwayat data sensor dari InfluxDB.

    Parameters:
    - lane_name: filter per jalur (opsional)
    - minutes: berapa menit ke belakang (default 10)
    """
    if minutes > 60:
        raise HTTPException(status_code=422, detail="Maksimal 60 menit")

    data = influx.query_recent_sensors(lane_name, minutes)
    return {
        "lane_name": lane_name or "all",
        "minutes":   minutes,
        "count":     len(data),
        "data":      data,
    }


@app.get(
    "/api/history/avg-green",
    tags=["History"],
    summary="Rata-rata durasi hijau vs timer statis"
)
async def get_avg_green(minutes: int = 30) -> dict:
    """
    Perbandingan rata-rata durasi hijau NERIC vs timer statis.
    Digunakan untuk panel efisiensi di React dashboard.
    """
    data = influx.query_avg_green_by_lane(minutes)
    static_baseline = 20  # Timer statis konvensional: 20 detik flat

    return {
        "minutes":          minutes,
        "static_baseline":  static_baseline,
        "neric_avg":        data,
        "emergency_count":  influx.query_emergency_count(),
    }


# ── WEBSOCKET ─────────────────────────────────────────────────

@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """
    WebSocket endpoint untuk streaming real-time ke React.

    React client connect ke ws://localhost:8000/ws/live
    dan menerima update setiap kali NERIC membuat keputusan.

    Message Types yang dikirim server:
    ──────────────────────────────────
    { type: "connected", ... }         → Konfirmasi koneksi
    { type: "decision", data: {...} }  → Keputusan NERIC baru
    { type: "simulation_step", ... }   → Langkah simulasi
    { type: "emergency_route_activated" } → Emergency route aktif
    { type: "heartbeat" }              → Keepalive ping

    Message yang diterima dari client:
    ──────────────────────────────────
    { type: "ping" }        → Client keepalive
    { type: "request_state" } → Minta state terkini
    """
    await ws_manager.connect(ws)

    # Kirim state awal saat pertama connect
    await ws.send_text(json.dumps({
        "type":    "connected",
        "message": "NERIC WebSocket connected",
        "cycle":   brain.cycle_count,
        "neurons": brain.get_neuron_data(),
        "stats":   brain.get_stats(),
    }))

    # Task heartbeat — kirim ping setiap 30 detik
    async def heartbeat():
        while True:
            try:
                await asyncio.sleep(30)
                await ws.send_text(json.dumps({"type": "heartbeat"}))
            except Exception:
                break

    heartbeat_task = asyncio.create_task(heartbeat())

    try:
        while True:
            # Terima message dari client
            raw = await ws.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "ping":
                await ws.send_text(json.dumps({"type": "pong"}))

            elif msg.get("type") == "request_state":
                await ws.send_text(json.dumps({
                    "type":    "state",
                    "cycle":   brain.cycle_count,
                    "neurons": brain.get_neuron_data(),
                    "stats":   brain.get_stats(),
                }))

            elif msg.get("type") == "manual_input":
                # Client bisa trigger keputusan langsung via WebSocket juga
                counts = msg.get("vehicle_counts", [5, 5, 5, 5])
                result = brain.decide(counts)
                await ws.send_text(json.dumps({
                    "type":    "decision",
                    "data":    result.dict(),
                    "neurons": brain.get_neuron_data(),
                }))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        heartbeat_task.cancel()
        ws_manager.disconnect(ws)


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "true").lower() == "true",
        log_level="info",
    )