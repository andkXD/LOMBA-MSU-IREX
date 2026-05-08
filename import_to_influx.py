import os
import csv
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv

# REVISI 1: Jalur load_dotenv diarahkan ke folder config
load_dotenv("config/.env")

# ── Konfigurasi InfluxDB 3 OSS dari .env ─────────────────────
INFLUX_URL      = os.getenv("INFLUXDB_URL", "http://localhost:8181")
INFLUX_TOKEN    = os.getenv("INFLUXDB_TOKEN", "")
INFLUX_DATABASE = os.getenv("INFLUXDB_DATABASE", "neric-data")

# ── Mapping Junction → Nama Jalur NERIC ──────────────────────
JUNCTION_MAP = {
    "1": "Utara",
    "2": "Selatan",
    "3": "Timur",
    "4": "Barat",
}

# REVISI 2: Jalur file CSV diarahkan ke folder data
CSV_FILE   = "data/traffic.csv" 
BATCH_SIZE = 500   
DELAY_SEC  = 0.1   
MAX_RETRY  = 3

def parse_datetime(dt_str: str):
    """Mengonversi string datetime menjadi nanosecond timestamp."""
    try:
        dt = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000_000)
    except Exception:
        return None

def check_connection() -> bool:
    """Test koneksi ke InfluxDB 3 OSS sebelum import."""
    print("🔍 Mengecek koneksi ke InfluxDB 3 OSS...")
    try:
        r = requests.get(f"{INFLUX_URL}/health", timeout=5)
        if r.status_code == 200:
            print(f"✅ InfluxDB 3 OSS terhubung di {INFLUX_URL}!\n")
            return True
        
        r = requests.get(
            f"{INFLUX_URL}/api/v3/configure/database",
            params={"format": "json"},
            headers={"Authorization": f"Bearer {INFLUX_TOKEN}"},
            timeout=5
        )
        if r.status_code in (200, 400, 401):
            if r.status_code == 401:
                print("❌ Token tidak valid! Cek kembali .env kamu.")
                return False
            print(f"✅ InfluxDB 3 OSS terdeteksi aktif!\n")
            return True
        return False
    except Exception as e:
        print(f"❌ Error koneksi: {e}")
        return False

def write_batch(lines: list, batch_num: int) -> bool:
    """Mengirim data batch ke InfluxDB."""
    body = "\n".join(lines)
    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.post(
                f"{INFLUX_URL}/api/v3/write_lp",
                params={"db": INFLUX_DATABASE, "precision": "ns"},
                headers={"Authorization": f"Bearer {INFLUX_TOKEN}"},
                data=body.encode("utf-8"),
                timeout=30
            )
            if r.status_code in (200, 204):
                return True
            time.sleep(DELAY_SEC)
        except Exception:
            time.sleep(DELAY_SEC)
    return False

def import_csv():
    """Proses utama membaca CSV dan mengirim ke InfluxDB."""
    if not os.path.exists(CSV_FILE):
        print(f"❌ File '{CSV_FILE}' tidak ditemukan!")
        print(f"💡 Pastikan file traffic.csv ada di dalam folder 'data/'")
        return

    with open(CSV_FILE, "r") as f:
        total_rows = sum(1 for _ in f) - 1

    print(f"📊 Total data: {total_rows:,} baris")
    print(f"⏳ Memulai import ke '{INFLUX_DATABASE}'...")

    batch = []
    total_written = 0
    batch_count = 0

    with open(CSV_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Fungsi parse_datetime sekarang sudah didefinisikan di atas
            ts = parse_datetime(row.get("DateTime", ""))
            if not ts: continue
            
            junc = row.get("Junction", "1")
            raw_val = int(row.get("Vehicles", 0))
            norm_val = min(20, max(0, raw_val // 10))
            lane = JUNCTION_MAP.get(junc, f"Lane{junc}")

            line = f"sensor_data,lane_name={lane},lane_id={int(junc)-1},source=kaggle_real vehicle_count={norm_val}i,vehicle_count_raw={raw_val}i {ts}"
            batch.append(line)

            if len(batch) >= BATCH_SIZE:
                batch_count += 1
                if write_batch(batch, batch_count):
                    total_written += len(batch)
                    print(f" ✅ Batch #{batch_count} — {total_written:,}/{total_rows:,} ({total_written/total_rows*100:.1f}%)")
                batch = []
                time.sleep(DELAY_SEC)

    if batch:
        write_batch(batch, batch_count + 1)
        total_written += len(batch)
    
    print(f"\n🎉 Selesai! Berhasil import {total_written:,} data.")

if __name__ == "__main__":
    print("\n" + "🚦" * 15)
    print(" NERIC — IMPORT DATA LOKAL")
    print("🚦" * 15 + "\n")

    if check_connection():
        import_csv()
        print("\n✅ Langkah selanjutnya: jalankan backend dengan uvicorn!")
    