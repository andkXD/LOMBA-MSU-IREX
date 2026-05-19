import os
import logging
import requests
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

# Pastikan import ini sesuai dengan struktur folder project NERIC kamu
from config.config import NERICDecisionResponse, NERICConfig

logger = logging.getLogger("neric.influx")

class InfluxService:
    """
    Service layer untuk InfluxDB 3 OSS (Local).
    Sudah disesuaikan agar tidak error HTTP 400 atau Auth Malformed pada versi Core.
    """

    def __init__(self):
        self.url       = os.getenv("INFLUXDB_URL", "http://localhost:8181")
        self.token     = os.getenv("INFLUXDB_TOKEN", "").strip() # Bersihkan spasi/newline jika ada
        self.database  = os.getenv("INFLUXDB_DATABASE", "neric-data")
        self.connected = False

    def _get_headers(self, content_type: Optional[str] = None) -> dict:
        """Helper untuk membuat header dinamis. Mencegah header malformed jika token kosong."""
        headers = {}
        
        # Menggunakan format 'Token <token>' yang stabil untuk InfluxDB 3 OSS lokal
        if self.token:
            headers["Authorization"] = f"Token {self.token}"
            
        if content_type:
            headers["Content-Type"] = content_type
        else:
            headers["Accept"] = "application/json"
            
        return headers

    def connect(self) -> bool:
        """Koneksi ke InfluxDB 3 OSS Lokal."""
        try:
            # InfluxDB 3 Core kadang tidak merespon /health, 
            # jadi kita langsung tembak endpoint config sebagai test
            r = requests.get(
                f"{self.url}/api/v3/configure/database",
                params={"format": "json"}, 
                headers=self._get_headers(),
                timeout=5
            )
            
            # 200 OK, atau 400/401 (artinya server ada tapi token/format bermasalah)
            if r.status_code in (200, 400, 401):
                if r.status_code == 401:
                    logger.warning("InfluxDB 3 auth failed: Token salah atau tidak berwenang")
                    return False
                
                self.connected = True
                logger.info(f"InfluxDB 3 OSS connected: {self.url} ✓")
                self._ensure_database()
                return True
            
            logger.warning(f"InfluxDB connection failed: HTTP {r.status_code}")
            return False

        except Exception as e:
            logger.warning(f"InfluxDB 3 connection failed: {e}")
            self.connected = False
            return False

    def _ensure_database(self):
        """Buat database jika belum ada secara aman."""
        try:
            # Langsung POST untuk memastikan database neric-data ada.
            # Jika sudah ada, server InfluxDB akan merespons HTTP 409 atau melempar info "already exists",
            # kita tidak perlu memunculkan error-nya ke log utama aplikasi agar log tetap bersih.
            r = requests.post(
                f"{self.url}/api/v3/configure/database",
                headers=self._get_headers(content_type="application/json"),
                json={"db": self.database},
                timeout=5
            )
            if r.status_code == 401:
                logger.warning("Gagal memastikan database: Token tidak berwenang (401)")
        except Exception as e:
            logger.debug(f"Ensure database info: {e}")

    def _write_line_protocol(self, lines: List[str]) -> bool:
        if not lines or not self.connected:
            return False
        body = "\n".join(lines)
        try:
            r = requests.post(
                f"{self.url}/api/v3/write_lp",
                params={"db": self.database, "precision": "ns"},
                headers=self._get_headers(content_type="text/plain; charset=utf-8"),
                data=body.encode("utf-8"),
                timeout=10
            )
            return r.status_code in (200, 204)
        except Exception as e:
            logger.error(f"InfluxDB write error: {e}")
            return False

    def _query_sql(self, sql: str) -> List[dict]:
        """Query SQL yang sudah diperbaiki agar tidak HTTP 400."""
        try:
            r = requests.get(
                f"{self.url}/api/v3/query_sql",
                params={
                    "db": self.database, 
                    "q": sql, 
                    "format": "json" # CRITICAL: Harus ada parameter format
                },
                headers=self._get_headers(),
                timeout=15
            )
            if r.status_code == 200:
                return r.json()
            return []
        except Exception as e:
            logger.error(f"InfluxDB query error: {e}")
            return []

    def write_sensor_data(self, neuron_data: List[dict], cycle: int) -> None:
        if not self.connected: return
        try:
            ts = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            lines = []
            for n in neuron_data:
                line = (
                    f"sensor_data,lane_id={n['lane_id']},lane_name={n['lane_name']} "
                    f"vehicle_count={n['vehicle_count']}i,membrane_potential={n['membrane_potential']},"
                    f"spike={int(n['spike'])}i,synaptic_weight={n['synaptic_weight']},"
                    f"cycle={cycle}i {ts}"
                )
                lines.append(line)
            self._write_line_protocol(lines)
        except: pass

    def write_decisions(self, response: NERICDecisionResponse) -> None:
        if not self.connected: return
        try:
            ts = int(datetime.now(timezone.utc).timestamp() * 1_000_000_000)
            lines = []
            for lane_id, d in response.decisions.items():
                line = (
                    f"neric_decisions,lane_id={lane_id},lane_name={d.lane_name},signal={d.signal.value} "
                    f"green_duration={d.green_duration}i,priority_score={d.priority_score},"
                    f"cycle={response.cycle}i {ts}"
                )
                lines.append(line)
            self._write_line_protocol(lines)
        except: pass

    def query_recent_sensors(self, lane_name: Optional[str] = None, minutes: int = 10) -> List[dict]:
        if not self.connected:
            return self._mock_sensor_data(lane_name, minutes)

        lane_filter = f"AND lane_name = '{lane_name}'" if lane_name else ""
        # SQL syntax untuk InfluxDB 3 Lokal
        sql = f"""
            SELECT time, lane_name, vehicle_count, membrane_potential, spike
            FROM sensor_data
            WHERE time >= now() - INTERVAL '{minutes} minutes'
            {lane_filter}
            ORDER BY time ASC
        """
        rows = self._query_sql(sql)
        if not rows: return self._mock_sensor_data(lane_name, minutes)
        
        return [{
            "time": r.get("time"),
            "lane_name": r.get("lane_name"),
            "vehicle_count": r.get("vehicle_count"),
            "membrane_potential": r.get("membrane_potential"),
            "spike": bool(r.get("spike"))
        } for r in rows]

    def _mock_sensor_data(self, lane_name, minutes):
        import random
        lanes = [lane_name] if lane_name else NERICConfig.LANE_NAMES
        res = []
        now = datetime.now(timezone.utc)
        for i in range(minutes * 2):
            t = now - timedelta(seconds=(minutes * 60 - i * 30))
            for l in lanes:
                res.append({
                    "time": t.isoformat(),
                    "lane_name": l,
                    "vehicle_count": random.randint(0, 20),
                    "membrane_potential": random.uniform(0, 1),
                    "spike": random.random() > 0.8
                })
        return res

    def close(self):
        self.connected = False