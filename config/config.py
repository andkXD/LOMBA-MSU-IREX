"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC Backend — Config & Data Models                        ║
║  File: config.py                                             ║
╚══════════════════════════════════════════════════════════════╝
"""

from pydantic import BaseModel
from typing import List, Optional, Dict
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv("config/.env")


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════

class SignalState(str, Enum):
    GREEN  = "GREEN"
    YELLOW = "YELLOW"
    RED    = "RED"

class ScenarioType(str, Enum):
    NORMAL     = "NORMAL"
    RUSH_HOUR  = "RUSH_HOUR"
    EMERGENCY  = "EMERGENCY"
    IMBALANCED = "IMBALANCED"
    NIGHT      = "NIGHT"
    CUSTOM     = "CUSTOM"


# ══════════════════════════════════════════════════════════════
# NERIC SYSTEM CONFIG
# ══════════════════════════════════════════════════════════════

class NERICConfig:
    # Lane setup
    NUM_LANES    = 4
    LANE_NAMES   = ["Utara", "Selatan", "Timur", "Barat"]
    MAX_VEHICLES = int(os.getenv("MAX_VEHICLES", 20))

    # Green light duration (seconds)
    GREEN_MIN       = int(os.getenv("GREEN_MIN", 10))
    GREEN_MAX       = int(os.getenv("GREEN_MAX", 60))
    YELLOW_DURATION = 3

    # LIF Neuron parameters
    ALPHA           = 0.70   # Weight for new input
    BETA            = 0.30   # Weight for previous V_m
    LEAK_RATE       = 0.08   # Membrane leak per cycle
    REFRACTORY      = 3      # Cycles neuron rests after spike

    # Adaptive threshold by time-of-day
    THRESHOLD_RUSH   = 0.32
    THRESHOLD_NORMAL = 0.55
    THRESHOLD_NIGHT  = 0.72
    RUSH_HOURS       = [(7, 9), (12, 13), (16, 19)]
    NIGHT_HOURS      = [(22, 24), (0, 5)]

    # Emergency detection
    EMERGENCY_RATIO  = float(os.getenv("EMERGENCY_RATIO", 0.85))

    # Priority score weights (must sum to ~1.0)
    W_DENSITY  = 0.40
    W_SPIKE    = 0.30
    W_TREND    = 0.15
    W_HISTORY  = 0.15

    # E-I Network
    EXCITATORY_STRENGTH = 0.12
    INHIBITORY_STRENGTH = 0.08

    # History window
    HISTORY_WINDOW = 15


# ══════════════════════════════════════════════════════════════
# REQUEST MODELS (FastAPI input)
# ══════════════════════════════════════════════════════════════

class ManualInputRequest(BaseModel):
    """
    Input manual dari React dashboard.
    Pengguna menggeser slider → dikirim ke /api/decide
    """
    vehicle_counts: List[int]       # [utara, selatan, timur, barat]
    simulated_hour: Optional[int] = None   # Override jam untuk testing
    scenario: Optional[ScenarioType] = ScenarioType.CUSTOM

    class Config:
        json_schema_extra = {
            "example": {
                "vehicle_counts": [12, 5, 8, 3],
                "simulated_hour": 8,
                "scenario": "CUSTOM"
            }
        }


class EmergencyRouteRequest(BaseModel):
    """
    Request pre-routing ambulans dari aplikasi mobile / dashboard.
    Sebelum ambulans berangkat, set rute dulu.
    """
    origin_lane: int        # 0=Utara, 1=Selatan, 2=Timur, 3=Barat
    destination: str        # Nama rumah sakit / tujuan
    route_lanes: List[int]  # Urutan jalur yang akan dilalui
    priority_level: int = 1 # 1=ambulans, 2=pemadam, 3=polisi

    class Config:
        json_schema_extra = {
            "example": {
                "origin_lane": 0,
                "destination": "RS Unair Surabaya",
                "route_lanes": [0, 2, 1],
                "priority_level": 1
            }
        }


class SimulationRequest(BaseModel):
    """
    Request untuk jalankan simulasi otomatis N siklus.
    Berguna untuk demo dan testing.
    """
    n_cycles:  int = 10
    scenario:  ScenarioType = ScenarioType.NORMAL
    simulated_hour: Optional[int] = None


# ══════════════════════════════════════════════════════════════
# RESPONSE MODELS (FastAPI output)
# ══════════════════════════════════════════════════════════════

class LaneDecision(BaseModel):
    """Keputusan untuk satu jalur."""
    lane_id:            int
    lane_name:          str
    signal:             SignalState
    green_duration:     int
    priority_rank:      int
    priority_score:     float
    reason:             str
    vehicle_count:      int
    membrane_potential: float
    spike:              bool
    spike_strength:     float
    trend:              str
    avg_density:        float
    synaptic_weight:    float
    refractory:         bool
    threshold:          float


class NERICDecisionResponse(BaseModel):
    """Response lengkap dari NERIC Brain per siklus."""
    cycle:              int
    timestamp:          str
    decisions:          Dict[int, LaneDecision]
    emergency:          bool
    emergency_lanes:    List[int]
    threshold:          float
    period:             str
    scenario:           str


class SystemStatusResponse(BaseModel):
    """Status sistem NERIC secara keseluruhan."""
    running:            bool
    cycle_count:        int
    emergency_mode:     bool
    emergency_events:   int
    mqtt_connected:     bool
    influxdb_connected: bool
    uptime_seconds:     float
    lane_stats:         Dict[str, dict]


class EmergencyRouteResponse(BaseModel):
    """Response setelah emergency route diset."""
    success:            bool
    message:            str
    route_lanes:        List[int]
    estimated_clearance_seconds: int
    affected_intersections: int