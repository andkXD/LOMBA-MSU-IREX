"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC Backend — InfluxDB Service                            ║
║  File: influx_service.py                                     ║
║                                                              ║
║  Handles semua operasi database:                             ║
║  - Write sensor data per siklus                              ║
║  - Write keputusan NERIC                                     ║
║  - Query historis untuk dashboard                            ║
║  - Write emergency events                                    ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import logging
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional

from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influxdb_client.client.exceptions import InfluxDBError

from config import NERICDecisionResponse, NERICConfig

logger = logging.getLogger("neric.influx")


class InfluxService:
    """
    Service layer untuk semua operasi InfluxDB.

    InfluxDB dipilih karena:
    - Didesain khusus untuk time-series data (sensor IoT)
    - Query sangat cepat untuk data berurutan waktu
    - Retention policy built-in (auto-hapus data lama)
    - Line protocol sangat efisien untuk write banyak data

    STRUKTUR DATA:
    ─────────────────────────────────────────────────────────
    Measurement: sensor_data
      Tags:   lane_id, lane_name
      Fields: vehicle_count, membrane_potential, spike,
              spike_strength, synaptic_weight, avg_density
      Time:   nanosecond precision

    Measurement: neric_decisions
      Tags:   lane_id, lane_name, signal
      Fields: green_duration, priority_rank, priority_score,
              threshold, emergency
      Time:   nanosecond precision

    Measurement: emergency_events
      Tags:   event_type
      Fields: lane_ids, route_lanes, duration_seconds
      Time:   nanosecond precision
    """

    def __init__(self):
        self.url    = os.getenv("INFLUXDB_URL",    "http://localhost:8086")
        self.token  = os.getenv("INFLUXDB_TOKEN",  "")
        self.org    = os.getenv("INFLUXDB_ORG",    "neric-org")
        self.bucket = os.getenv("INFLUXDB_BUCKET", "neric-data")
        self.connected = False
        self.client = None
        self.write_api = None
        self.query_api = None

    def connect(self) -> bool:
        """
        Inisialisasi koneksi ke InfluxDB Cloud.
        Menggunakan requests API langsung karena InfluxDB Cloud
        tidak support endpoint /health seperti versi lokal.
        Return True jika berhasil, False jika gagal.
        """
        try:
            self.client = InfluxDBClient(
                url=self.url,
                token=self.token,
                org=self.org,
            )

            # ── Test koneksi via REST API langsung ────────────
            # InfluxDB Cloud tidak support /health endpoint
            # Gunakan /api/v2/buckets sebagai health check
            r = requests.get(
                f"{self.url}/api/v2/buckets",
                headers={
                    "Authorization": f"Token {self.token}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )

            if r.status_code == 200:
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
                self.query_api = self.client.query_api()
                self.connected = True
                logger.info(f"InfluxDB Cloud connected: {self.url} ✓")
                return True
            elif r.status_code == 401:
                logger.warning("InfluxDB auth failed: Token tidak valid")
                return False
            elif r.status_code == 403:
                logger.warning("InfluxDB auth failed: Permission tidak cukup")
                return False
            else:
                logger.warning(f"InfluxDB connection failed: HTTP {r.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            logger.warning(f"InfluxDB tidak dapat dijangkau: {self.url}")
            logger.info("Running without InfluxDB — data will not be persisted")
            self.connected = False
            return False
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {e}")
            logger.info("Running without InfluxDB — data will not be persisted")
            self.connected = False
            return False

    def write_sensor_data(
        self,
        neuron_data: List[dict],
        cycle: int
    ) -> None:
        """
        Tulis data sensor dari semua jalur ke InfluxDB.
        Dipanggil setiap siklus NERIC (setiap kali decide() dipanggil).
        """
        if not self.connected:
            return
        try:
            points = []
            for neuron in neuron_data:
                point = (
                    Point("sensor_data")
                    .tag("lane_id",   str(neuron["lane_id"]))
                    .tag("lane_name", neuron["lane_name"])
                    .field("vehicle_count",      neuron["vehicle_count"])
                    .field("membrane_potential", neuron["membrane_potential"])
                    .field("spike",              int(neuron["spike"]))
                    .field("spike_strength",     neuron["spike_strength"])
                    .field("synaptic_weight",    neuron["synaptic_weight"])
                    .field("avg_density",        neuron["avg_density"])
                    .field("cycle",              cycle)
                    .time(datetime.now(timezone.utc), WritePrecision.NANOSECONDS)
                )
                points.append(point)

            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=points
            )
        except InfluxDBError as e:
            logger.error(f"InfluxDB write_sensor_data error: {e}")

    def write_decisions(self, response: NERICDecisionResponse) -> None:
        """
        Tulis keputusan NERIC ke InfluxDB.
        Menyimpan signal, durasi, skor prioritas per jalur.
        """
        if not self.connected:
            return
        try:
            points = []
            for lane_id, decision in response.decisions.items():
                point = (
                    Point("neric_decisions")
                    .tag("lane_id",   str(lane_id))
                    .tag("lane_name", decision.lane_name)
                    .tag("signal",    decision.signal.value)
                    .field("green_duration",  decision.green_duration)
                    .field("priority_rank",   decision.priority_rank)
                    .field("priority_score",  decision.priority_score)
                    .field("threshold",       decision.threshold)
                    .field("emergency",       int(response.emergency))
                    .field("cycle",           response.cycle)
                    .time(datetime.now(timezone.utc), WritePrecision.NANOSECONDS)
                )
                points.append(point)

            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=points
            )
        except InfluxDBError as e:
            logger.error(f"InfluxDB write_decisions error: {e}")

    def write_emergency_event(
        self,
        event_type: str,
        lane_ids: List[int],
        details: dict = None
    ) -> None:
        """
        Tulis emergency event ke InfluxDB.
        Berguna untuk analisis post-mortem dan laporan.
        """
        if not self.connected:
            return
        try:
            point = (
                Point("emergency_events")
                .tag("event_type", event_type)
                .field("lane_ids",  str(lane_ids))
                .field("details",   str(details or {}))
                .time(datetime.now(timezone.utc), WritePrecision.NANOSECONDS)
            )
            self.write_api.write(
                bucket=self.bucket,
                org=self.org,
                record=point
            )
        except InfluxDBError as e:
            logger.error(f"InfluxDB write_emergency_event error: {e}")

    def query_recent_sensors(
        self,
        lane_name: Optional[str] = None,
        minutes: int = 10
    ) -> List[dict]:
        """
        Query data sensor N menit terakhir.
        Digunakan oleh React dashboard untuk tampilkan grafik historis.
        """
        if not self.connected:
            return self._mock_sensor_data(lane_name, minutes)

        lane_filter = f'|> filter(fn: (r) => r["lane_name"] == "{lane_name}")' \
                      if lane_name else ""

        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "sensor_data")
          {lane_filter}
          |> pivot(
               rowKey: ["_time", "lane_id", "lane_name"],
               columnKey: ["_field"],
               valueColumn: "_value"
             )
          |> sort(columns: ["_time"])
          |> limit(n: 500)
        """
        try:
            result = []
            tables = self.query_api.query(query, org=self.org)
            for table in tables:
                for record in table.records:
                    result.append({
                        "time":               record.get_time().isoformat(),
                        "lane_name":          record.values.get("lane_name"),
                        "vehicle_count":      record.values.get("vehicle_count", 0),
                        "membrane_potential": record.values.get("membrane_potential", 0),
                        "spike":              bool(record.values.get("spike", 0)),
                        "synaptic_weight":    record.values.get("synaptic_weight", 0.5),
                    })
            return result
        except Exception as e:
            logger.error(f"InfluxDB query error: {e}")
            return self._mock_sensor_data(lane_name, minutes)

    def query_avg_green_by_lane(self, minutes: int = 60) -> List[dict]:
        """
        Query rata-rata durasi hijau per jalur dalam N menit terakhir.
        """
        if not self.connected:
            return [
                {"lane_name": n, "avg_green": 20.0, "count": 0}
                for n in NERICConfig.LANE_NAMES
            ]

        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: -{minutes}m)
          |> filter(fn: (r) => r["_measurement"] == "neric_decisions")
          |> filter(fn: (r) => r["_field"] == "green_duration")
          |> group(columns: ["lane_name"])
          |> mean()
        """
        try:
            result = []
            tables = self.query_api.query(query, org=self.org)
            for table in tables:
                for record in table.records:
                    result.append({
                        "lane_name": record.values.get("lane_name"),
                        "avg_green": round(record.get_value(), 1),
                    })
            return result
        except Exception as e:
            logger.error(f"InfluxDB query avg_green error: {e}")
            return []

    def query_emergency_count(self, hours: int = 24) -> int:
        """Hitung jumlah emergency event dalam N jam terakhir."""
        if not self.connected:
            return 0
        query = f"""
        from(bucket: "{self.bucket}")
          |> range(start: -{hours}h)
          |> filter(fn: (r) => r["_measurement"] == "emergency_events")
          |> count()
        """
        try:
            tables = self.query_api.query(query, org=self.org)
            for table in tables:
                for record in table.records:
                    return int(record.get_value())
        except Exception:
            pass
        return 0

    def _mock_sensor_data(
        self,
        lane_name: Optional[str],
        minutes: int
    ) -> List[dict]:
        """
        Fallback mock data saat InfluxDB tidak tersedia.
        Agar React dashboard tetap bisa ditampilkan saat development.
        """
        import random
        from datetime import timedelta

        lanes = [lane_name] if lane_name else NERICConfig.LANE_NAMES
        result = []
        now = datetime.now(timezone.utc)

        for i in range(minutes * 6):
            t = now - timedelta(seconds=(minutes * 60 - i * 10))
            for lane in lanes:
                result.append({
                    "time":               t.isoformat(),
                    "lane_name":          lane,
                    "vehicle_count":      random.randint(2, 18),
                    "membrane_potential": round(random.uniform(0, 0.8), 3),
                    "spike":              random.random() > 0.7,
                    "synaptic_weight":    round(random.uniform(0.3, 0.8), 3),
                })
        return result

    def close(self) -> None:
        """Tutup koneksi InfluxDB saat shutdown."""
        if self.client:
            self.client.close()
            logger.info("InfluxDB connection closed")