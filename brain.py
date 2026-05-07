"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC Backend — Neuromorphic Brain Engine                   ║
║  File: brain.py                                              ║
║                                                              ║
║  Berisi implementasi lengkap:                                ║
║  - LIF Neuron Model                                          ║
║  - Excitatory-Inhibitory Network                             ║
║  - Adaptive Threshold Controller                             ║
║  - Priority Calculator                                       ║
║  - NERIC Brain (koordinator utama)                           ║
╚══════════════════════════════════════════════════════════════╝
"""

import numpy as np
from collections import deque
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import asyncio
import logging

from config import NERICConfig, LaneDecision, NERICDecisionResponse, SignalState

logger = logging.getLogger("neric.brain")


# ══════════════════════════════════════════════════════════════
# LIF NEURON
# ══════════════════════════════════════════════════════════════

class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron — Model satu jalur persimpangan.

    Setiap jalur = satu neuron biologis:
    V_m naik saat kendaraan padat → spike saat V_m ≥ threshold
    → reset → refractory → siap spike lagi
    """

    def __init__(self, lane_id: int, name: str, cfg: NERICConfig):
        self.lane_id = lane_id
        self.name    = name
        self.cfg     = cfg

        # ── LIF State ───────────────────────────────────────
        self.V_m:              float = 0.0
        self.spike:            bool  = False
        self.spike_strength:   float = 0.0
        self.refractory_timer: int   = 0

        # ── Data ────────────────────────────────────────────
        self.vehicle_count: int = 0
        self.history:       deque = deque(maxlen=cfg.HISTORY_WINDOW)
        self.Vm_history:    deque = deque(maxlen=100)
        self.spike_history: deque = deque(maxlen=100)

        # ── STDP (Synaptic plasticity) ───────────────────────
        self.synaptic_weight: float = 0.5
        self.total_spikes:    int   = 0

    def update(self,
               vehicle_count: int,
               threshold: float,
               external_input: float = 0.0) -> None:
        """
        Update satu siklus neuron LIF.

        Formula: V_m(t+1) = α×I_sensor + β×V_m(t) + external_input - leak
        """
        self.vehicle_count = vehicle_count
        self.history.append(vehicle_count)

        # Refractory: neuron istirahat setelah spike
        if self.refractory_timer > 0:
            self.refractory_timer -= 1
            self.spike        = False
            self.spike_strength = 0.0
            self.Vm_history.append(0.0)
            self.spike_history.append(0)
            return

        # Normalisasi input sensor → 0.0–1.0
        I_sensor = vehicle_count / self.cfg.MAX_VEHICLES

        # Integrate: akumulasi membrane potential
        self.V_m = (
            self.cfg.ALPHA * I_sensor
            + self.cfg.BETA  * self.V_m
            + external_input
        )
        self.V_m = float(np.clip(self.V_m, 0.0, 1.0))

        # Leak: kebocoran membran
        self.V_m = max(0.0, self.V_m - self.cfg.LEAK_RATE)

        # Cek spike
        if self.V_m >= threshold:
            self.spike          = True
            self.spike_strength = self.V_m
            self.V_m            = 0.0
            self.refractory_timer = self.cfg.REFRACTORY
            self.total_spikes  += 1
            # STDP: perkuat sinapsis saat spike (Hebbian learning)
            self.synaptic_weight = min(1.0, self.synaptic_weight + 0.02)
        else:
            self.spike          = False
            self.spike_strength = 0.0
            # STDP: lemahkan perlahan saat tidak spike
            self.synaptic_weight = max(0.1, self.synaptic_weight - 0.005)

        self.Vm_history.append(self.V_m)
        self.spike_history.append(1 if self.spike else 0)

    def get_avg_density(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0

    def get_trend(self) -> Tuple[str, float]:
        """Deteksi tren kepadatan via regresi linear."""
        if len(self.history) < 4:
            return ("STABIL →", 0.0)
        y = np.array(list(self.history)[-8:])
        x = np.arange(len(y))
        slope = float(np.polyfit(x, y, 1)[0])
        if slope > 0.2:   return ("NAIK ↑",   slope)
        if slope < -0.2:  return ("TURUN ↓",  slope)
        return ("STABIL →", slope)

    def get_trend_score(self) -> float:
        label, _ = self.get_trend()
        if "NAIK"   in label: return 0.2
        if "STABIL" in label: return 0.1
        return 0.0

    def is_emergency(self) -> bool:
        return self.vehicle_count >= self.cfg.MAX_VEHICLES * self.cfg.EMERGENCY_RATIO

    def to_dict(self) -> dict:
        """Serialisasi ke dict untuk API response & InfluxDB."""
        trend_label, trend_slope = self.get_trend()
        return {
            "lane_id":            self.lane_id,
            "lane_name":          self.name,
            "vehicle_count":      self.vehicle_count,
            "membrane_potential": round(self.V_m, 4),
            "spike":              self.spike,
            "spike_strength":     round(self.spike_strength, 4),
            "refractory":         self.refractory_timer > 0,
            "synaptic_weight":    round(self.synaptic_weight, 4),
            "avg_density":        round(self.get_avg_density(), 2),
            "trend":              trend_label,
            "trend_slope":        round(trend_slope, 3),
            "total_spikes":       self.total_spikes,
            "Vm_history":         list(self.Vm_history)[-20:],
            "spike_history":      list(self.spike_history)[-20:],
        }


# ══════════════════════════════════════════════════════════════
# EXCITATORY-INHIBITORY NETWORK
# ══════════════════════════════════════════════════════════════

class EINetwork:
    """
    Jaringan Excitatory-Inhibitory antar neuron.

    Jalur spike → excite diri sendiri + inhibit jalur lain.
    Meniru lateral inhibition di korteks visual otak.
    """

    def __init__(self, neurons: List[LIFNeuron], cfg: NERICConfig):
        self.neurons = neurons
        self.cfg     = cfg

    def compute(self) -> Dict[int, float]:
        """Hitung sinyal E-I untuk setiap jalur."""
        signals = {n.lane_id: 0.0 for n in self.neurons}

        for spiking in [n for n in self.neurons if n.spike]:
            # Excitatory ke diri sendiri
            signals[spiking.lane_id] += (
                self.cfg.EXCITATORY_STRENGTH * spiking.spike_strength
            )
            # Inhibitory ke jalur lain
            for other in self.neurons:
                if other.lane_id != spiking.lane_id:
                    signals[other.lane_id] -= (
                        self.cfg.INHIBITORY_STRENGTH * spiking.spike_strength
                    )

        # Clamp ke range aman
        return {
            k: float(np.clip(v, -0.3, 0.3))
            for k, v in signals.items()
        }


# ══════════════════════════════════════════════════════════════
# ADAPTIVE THRESHOLD CONTROLLER
# ══════════════════════════════════════════════════════════════

class ThresholdController:
    """
    Threshold adaptif berdasarkan jam.
    Jam sibuk = threshold rendah = sistem lebih sensitif.
    """

    def __init__(self, cfg: NERICConfig):
        self.cfg = cfg

    def get(self, hour: Optional[int] = None) -> float:
        h = hour if hour is not None else datetime.now().hour
        for s, e in self.cfg.RUSH_HOURS:
            if s <= h < e:
                return self.cfg.THRESHOLD_RUSH
        for s, e in self.cfg.NIGHT_HOURS:
            if s <= h < e:
                return self.cfg.THRESHOLD_NIGHT
        return self.cfg.THRESHOLD_NORMAL

    def get_period(self, hour: Optional[int] = None) -> str:
        h = hour if hour is not None else datetime.now().hour
        for s, e in self.cfg.RUSH_HOURS:
            if s <= h < e:
                return f"JAM SIBUK ({s:02d}:00–{e:02d}:00)"
        for s, e in self.cfg.NIGHT_HOURS:
            if s <= h < e:
                return f"JAM MALAM ({s:02d}:00–{e:02d}:00)"
        return f"JAM NORMAL ({h:02d}:xx)"


# ══════════════════════════════════════════════════════════════
# PRIORITY CALCULATOR
# ══════════════════════════════════════════════════════════════

class PriorityCalculator:
    """Hitung priority score dan konversi ke durasi hijau."""

    def __init__(self, cfg: NERICConfig):
        self.cfg = cfg

    def score(self, neuron: LIFNeuron) -> float:
        cfg = self.cfg
        density  = neuron.vehicle_count / cfg.MAX_VEHICLES
        spike    = neuron.spike_strength
        trend    = neuron.get_trend_score()
        history  = neuron.get_avg_density() / cfg.MAX_VEHICLES
        stdp     = neuron.synaptic_weight * 0.05

        total = (
            cfg.W_DENSITY * density
            + cfg.W_SPIKE   * spike
            + cfg.W_TREND   * trend
            + cfg.W_HISTORY * history
            + stdp
        )
        return float(np.clip(total, 0.0, 1.0))

    def to_green_duration(self, score: float) -> int:
        cfg = self.cfg
        raw = cfg.GREEN_MIN + score * (cfg.GREEN_MAX - cfg.GREEN_MIN)
        return int(np.clip(raw, cfg.GREEN_MIN, cfg.GREEN_MAX))

    def rank(self, neurons: List[LIFNeuron]) -> List[Tuple[int, float, LIFNeuron]]:
        scored = [(self.score(n), n) for n in neurons]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(i+1, s, n) for i, (s, n) in enumerate(scored)]


# ══════════════════════════════════════════════════════════════
# NERIC BRAIN — Koordinator Utama
# ══════════════════════════════════════════════════════════════

class NERICBrain:
    """
    Otak pusat NERIC.

    Menerima data sensor → proses neuromorphic →
    hasilkan keputusan lampu → kirim ke MQTT & InfluxDB.
    """

    def __init__(self):
        self.cfg       = NERICConfig()
        self.neurons   = [
            LIFNeuron(i, self.cfg.LANE_NAMES[i], self.cfg)
            for i in range(self.cfg.NUM_LANES)
        ]
        self.ei_net    = EINetwork(self.neurons, self.cfg)
        self.threshold = ThresholdController(self.cfg)
        self.priority  = PriorityCalculator(self.cfg)

        # State
        self.cycle_count:    int  = 0
        self.emergency_mode: bool = False
        self.emergency_lanes: List[int] = []
        self.start_time:     float = datetime.now().timestamp()

        # Emergency routing state
        self.active_route:   Optional[List[int]] = None
        self.route_step:     int = 0

        # Stats
        self.stats = {
            "emergency_events": 0,
            "total_spikes":     [0] * 4,
            "avg_green":        [[] for _ in range(4)],
        }

        logger.info("NERIC Brain initialized ✓")

    def decide(
        self,
        sensor_data: List[int],
        simulated_hour: Optional[int] = None
    ) -> NERICDecisionResponse:
        """
        Proses utama: sensor data → keputusan lampu.

        Return NERICDecisionResponse yang siap dikirim ke:
        - React frontend via WebSocket
        - InfluxDB untuk storage
        - MQTT untuk aktuasi ESP32
        """
        self.cycle_count += 1

        # ── Step 1: Threshold adaptif ────────────────────────
        threshold = self.threshold.get(simulated_hour)
        period    = self.threshold.get_period(simulated_hour)

        # ── Step 2: E-I signals dari siklus sebelumnya ───────
        ei_signals = self.ei_net.compute()

        # ── Step 3: Update semua neuron (paralel) ────────────
        for i, neuron in enumerate(self.neurons):
            neuron.update(
                vehicle_count=int(np.clip(sensor_data[i], 0, self.cfg.MAX_VEHICLES)),
                threshold=threshold,
                external_input=ei_signals[neuron.lane_id]
            )

        # ── Step 4: Emergency detection ──────────────────────
        emergency_lanes = [
            n.lane_id for n in self.neurons if n.is_emergency()
        ]
        self.emergency_mode  = len(emergency_lanes) > 0
        self.emergency_lanes = emergency_lanes
        if self.emergency_mode:
            self.stats["emergency_events"] += 1

        # ── Step 5: Priority ranking ──────────────────────────
        ranked = self.priority.rank(self.neurons)

        # ── Step 6: Build decisions ───────────────────────────
        decisions: Dict[int, LaneDecision] = {}

        # Cek apakah ada emergency route aktif
        if self.active_route and self.route_step < len(self.active_route):
            decisions = self._apply_emergency_routing(ranked, threshold, period)
        elif self.emergency_mode:
            decisions = self._emergency_decisions(ranked, threshold, period, emergency_lanes)
        else:
            decisions = self._normal_decisions(ranked, threshold, period)

        # ── Step 7: Update stats ──────────────────────────────
        for i, neuron in enumerate(self.neurons):
            if neuron.spike:
                self.stats["total_spikes"][i] += 1
            d = decisions[i]
            self.stats["avg_green"][i].append(d.green_duration)

        logger.debug(f"Cycle #{self.cycle_count} | Emergency: {self.emergency_mode}")

        return NERICDecisionResponse(
            cycle=self.cycle_count,
            timestamp=datetime.now().isoformat(),
            decisions=decisions,
            emergency=self.emergency_mode,
            emergency_lanes=emergency_lanes,
            threshold=round(threshold, 3),
            period=period,
            scenario="EMERGENCY" if self.emergency_mode else "NORMAL",
        )

    def _build_decision(
        self,
        neuron: LIFNeuron,
        signal: SignalState,
        green_dur: int,
        rank: int,
        score: float,
        reason: str,
        threshold: float,
    ) -> LaneDecision:
        trend_label, _ = neuron.get_trend()
        return LaneDecision(
            lane_id=neuron.lane_id,
            lane_name=neuron.name,
            signal=signal,
            green_duration=green_dur,
            priority_rank=rank,
            priority_score=round(score, 4),
            reason=reason,
            vehicle_count=neuron.vehicle_count,
            membrane_potential=round(neuron.V_m, 4),
            spike=neuron.spike,
            spike_strength=round(neuron.spike_strength, 4),
            trend=trend_label,
            avg_density=round(neuron.get_avg_density(), 2),
            synaptic_weight=round(neuron.synaptic_weight, 4),
            refractory=neuron.refractory_timer > 0,
            threshold=round(threshold, 3),
        )

    def _normal_decisions(self, ranked, threshold, period) -> Dict[int, LaneDecision]:
        decisions = {}
        for rank, score, neuron in ranked:
            dur = self.priority.to_green_duration(score)
            sig = SignalState.GREEN if rank == 1 else SignalState.RED
            reason = (
                f"Prioritas #{rank} — Skor neuromorphic: {score:.3f}"
                if rank > 1 else
                f"Prioritas utama — Skor: {score:.3f} | Spike: {'Ya' if neuron.spike else 'Tidak'}"
            )
            decisions[neuron.lane_id] = self._build_decision(
                neuron, sig, dur, rank, score, reason, threshold
            )
        return decisions

    def _emergency_decisions(
        self, ranked, threshold, period, emergency_lanes
    ) -> Dict[int, LaneDecision]:
        decisions = {}
        for rank, score, neuron in ranked:
            if neuron.lane_id in emergency_lanes:
                sig = SignalState.GREEN
                dur = self.cfg.GREEN_MAX
                reason = f"⚠️ DARURAT — Kepadatan kritis ({neuron.vehicle_count}/{self.cfg.MAX_VEHICLES})"
                r = 1
            else:
                sig = SignalState.RED
                dur = self.cfg.GREEN_MAX + self.cfg.YELLOW_DURATION
                reason = "Menunggu — Mode darurat aktif"
                r = rank
            decisions[neuron.lane_id] = self._build_decision(
                neuron, sig, dur, r, score, reason, threshold
            )
        return decisions

    def _apply_emergency_routing(
        self, ranked, threshold, period
    ) -> Dict[int, LaneDecision]:
        """
        Green corridor untuk rute ambulans yang sudah di-preset.
        Jalur di rute → GREEN, jalur lain → RED.
        """
        current_lane = self.active_route[self.route_step]
        decisions = {}
        for rank, score, neuron in ranked:
            if neuron.lane_id == current_lane:
                sig = SignalState.GREEN
                dur = self.cfg.GREEN_MAX
                reason = f"🚑 GREEN CORRIDOR — Rute ambulans (step {self.route_step+1}/{len(self.active_route)})"
                r = 1
            else:
                sig = SignalState.RED
                dur = self.cfg.GREEN_MAX
                reason = "🚑 Menunggu — Koridor ambulans aktif"
                r = rank
            decisions[neuron.lane_id] = self._build_decision(
                neuron, sig, dur, r, score, reason, threshold
            )
        return decisions

    def set_emergency_route(self, route_lanes: List[int]) -> None:
        """Aktivasi emergency pre-routing."""
        self.active_route = route_lanes
        self.route_step   = 0
        logger.info(f"Emergency route set: {route_lanes}")

    def advance_route(self) -> bool:
        """Maju ke jalur berikutnya dalam rute. Return False jika rute selesai."""
        if self.active_route is None:
            return False
        self.route_step += 1
        if self.route_step >= len(self.active_route):
            self.active_route = None
            self.route_step   = 0
            logger.info("Emergency route completed")
            return False
        return True

    def clear_emergency_route(self) -> None:
        """Reset emergency route."""
        self.active_route = None
        self.route_step   = 0

    def get_neuron_data(self) -> List[dict]:
        """Ambil semua data neuron untuk WebSocket streaming."""
        return [n.to_dict() for n in self.neurons]

    def get_stats(self) -> dict:
        """Statistik sistem untuk status endpoint."""
        uptime = datetime.now().timestamp() - self.start_time
        lane_stats = {}
        for i, neuron in enumerate(self.neurons):
            green_times = self.stats["avg_green"][i]
            lane_stats[neuron.name] = {
                "total_spikes":    self.stats["total_spikes"][i],
                "avg_green_time":  round(float(np.mean(green_times)) if green_times else 0, 1),
                "avg_density":     round(neuron.get_avg_density(), 1),
                "synaptic_weight": round(neuron.synaptic_weight, 3),
            }
        return {
            "cycle_count":      self.cycle_count,
            "emergency_mode":   self.emergency_mode,
            "emergency_events": self.stats["emergency_events"],
            "uptime_seconds":   round(uptime, 1),
            "lane_stats":       lane_stats,
            "active_route":     self.active_route,
        }