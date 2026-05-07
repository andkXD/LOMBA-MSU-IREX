"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC Backend — MQTT Service                                ║
║  File: mqtt_service.py                                       ║
║                                                              ║
║  Handles komunikasi IoT dengan ESP32:                        ║
║  - Subscribe data sensor dari ESP32                          ║
║  - Publish keputusan sinyal balik ke ESP32                   ║
║  - Auto-reconnect jika koneksi terputus                      ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import json
import logging
import threading
from typing import Callable, Optional, Dict
import paho.mqtt.client as mqtt

logger = logging.getLogger("neric.mqtt")


class MQTTService:
    """
    MQTT client untuk komunikasi dengan ESP32 IoT nodes.

    TOPIC STRUCTURE:
    ────────────────────────────────────────────────────
    neric/sensor/{lane_id}  → ESP32 publish data sensor
    neric/signal/{lane_id}  → Server publish keputusan
    neric/emergency         → Broadcast mode darurat
    neric/status            → Server publish status sistem
    neric/heartbeat/{id}    → ESP32 heartbeat ping

    MESSAGE FORMAT:
    ────────────────────────────────────────────────────
    Sensor (ESP32 → Server):
    {
      "lane_id": 0,
      "vehicle_count": 12,
      "timestamp": "2026-05-15T08:30:00"
    }

    Signal (Server → ESP32):
    {
      "lane_id": 0,
      "signal": "GREEN",
      "duration": 45,
      "emergency": false
    }
    """

    def __init__(self):
        self.broker   = os.getenv("MQTT_BROKER", "localhost")
        self.port     = int(os.getenv("MQTT_PORT", 1883))
        self.username = os.getenv("MQTT_USERNAME", "")
        self.password = os.getenv("MQTT_PASSWORD", "")
        self.connected = False
        self.client   = None
        self._lock    = threading.Lock()

        # Latest sensor data dari ESP32 (lane_id → count)
        self.latest_sensor: Dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        # Callback yang dipanggil saat data sensor baru masuk
        self.on_sensor_data: Optional[Callable] = None

    def connect(self) -> bool:
        """
        Inisialisasi koneksi MQTT.
        Return True jika berhasil.
        """
        try:
            self.client = mqtt.Client(
                client_id="neric-server",
                clean_session=True
            )

            # Auth jika dikonfigurasi
            if self.username:
                self.client.username_pw_set(self.username, self.password)

            # Callbacks
            self.client.on_connect    = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_message    = self._on_message

            # Connect dengan timeout 5 detik
            self.client.connect(self.broker, self.port, keepalive=60)

            # Start loop di background thread
            self.client.loop_start()
            logger.info(f"MQTT connecting to {self.broker}:{self.port}...")
            return True
        except Exception as e:
            logger.warning(f"MQTT connection failed: {e}")
            logger.info("Running without MQTT — using simulated sensor data")
            return False

    def _on_connect(self, client, userdata, flags, rc):
        """Callback saat berhasil connect ke broker."""
        if rc == 0:
            self.connected = True
            logger.info("MQTT connected ✓")

            # Subscribe ke semua topic sensor dari ESP32
            client.subscribe("neric/sensor/#", qos=1)
            client.subscribe("neric/heartbeat/#", qos=0)
            logger.info("Subscribed to neric/sensor/#")
        else:
            codes = {
                1: "Protocol version mismatch",
                2: "Client ID rejected",
                3: "Broker unavailable",
                4: "Bad credentials",
                5: "Not authorized",
            }
            logger.error(f"MQTT connect failed: {codes.get(rc, f'Unknown error {rc}')}")

    def _on_disconnect(self, client, userdata, rc):
        """Callback saat disconnect — auto-reconnect."""
        self.connected = False
        if rc != 0:
            logger.warning(f"MQTT disconnected unexpectedly (rc={rc}), reconnecting...")
            # Paho mqtt auto-reconnect via loop_start()

    def _on_message(self, client, userdata, msg):
        """
        Callback saat menerima pesan dari ESP32.
        Parse topic → update latest_sensor → trigger callback.
        """
        try:
            topic = msg.topic
            payload = json.loads(msg.payload.decode())

            # ── Handle sensor data ────────────────────────────
            if topic.startswith("neric/sensor/"):
                lane_id = int(topic.split("/")[-1])
                count   = int(payload.get("vehicle_count", 0))

                with self._lock:
                    self.latest_sensor[lane_id] = count

                logger.debug(f"Sensor update: Lane {lane_id} = {count} vehicles")

                # Trigger callback ke FastAPI untuk proses neuromorphic
                if self.on_sensor_data:
                    self.on_sensor_data(lane_id, count, payload)

            # ── Handle heartbeat ──────────────────────────────
            elif topic.startswith("neric/heartbeat/"):
                unit_id = topic.split("/")[-1]
                logger.debug(f"Heartbeat from ESP32-{unit_id}")

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON from {msg.topic}: {msg.payload}")
        except Exception as e:
            logger.error(f"MQTT message handler error: {e}")

    def publish_signal(
        self,
        lane_id: int,
        signal: str,
        duration: int,
        emergency: bool = False
    ) -> None:
        """
        Publish keputusan sinyal lampu ke ESP32 yang sesuai.

        ESP32 akan menerima ini dan mengaktuasi LED traffic light.
        """
        if not self.connected or not self.client:
            return

        topic = f"neric/signal/{lane_id}"
        payload = json.dumps({
            "lane_id":   lane_id,
            "signal":    signal,
            "duration":  duration,
            "emergency": emergency,
        })

        try:
            self.client.publish(topic, payload, qos=1)
            logger.debug(f"Published to {topic}: {signal} {duration}s")
        except Exception as e:
            logger.error(f"MQTT publish error: {e}")

    def publish_emergency(self, emergency_lanes: list, route: list = None) -> None:
        """Broadcast emergency status ke semua ESP32."""
        if not self.connected or not self.client:
            return

        payload = json.dumps({
            "emergency":      True,
            "emergency_lanes": emergency_lanes,
            "route":           route or [],
        })
        self.client.publish("neric/emergency", payload, qos=1)

    def publish_status(self, status: dict) -> None:
        """Broadcast status sistem ke semua subscriber."""
        if not self.connected or not self.client:
            return
        self.client.publish(
            "neric/status",
            json.dumps(status),
            qos=0
        )

    def get_latest_sensor_data(self) -> list:
        """
        Ambil data sensor terbaru dari semua ESP32.
        Digunakan sebagai input ke NERIC Brain saat ada sensor nyata.
        """
        with self._lock:
            return [
                self.latest_sensor.get(i, 0)
                for i in range(4)
            ]

    def disconnect(self) -> None:
        """Tutup koneksi MQTT saat shutdown."""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            logger.info("MQTT disconnected")