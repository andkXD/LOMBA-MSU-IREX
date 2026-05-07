"""
╔══════════════════════════════════════════════════════════════╗
║  NERIC — Import Traffic Data ke InfluxDB Cloud               ║
║  File: import_to_influx.py                                   ║
║  v3 — Fixed: timestamp digeser ke masa kini                  ║
║                                                              ║
║  Kenapa timestamp digeser?                                   ║
║  InfluxDB Cloud Free Plan hanya terima data 30 hari terakhir ║
║  Data CSV asli dari 2015 → digeser ke hari ini               ║
║  Pola lalu lintas tetap REAL, hanya waktunya disesuaikan     ║
║                                                              ║
║  Cara pakai:                                                 ║
║  1. Taruh traffic.csv di folder yang sama                    ║
║  2. Pastikan .env sudah terisi token InfluxDB                ║
║  3. Jalankan: python import_to_influx.py                     ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import csv
import time
import requests
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Konfigurasi InfluxDB dari .env ────────────────────────────
INFLUX_URL    = os.getenv("INFLUXDB_URL",    "https://us-east-1-1.aws.cloud2.influxdata.com")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "")
INFLUX_ORG    = os.getenv("INFLUXDB_ORG",    "Teazzi")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "neric-data")

# ── Mapping Junction → Nama Jalur NERIC ──────────────────────
JUNCTION_MAP = {
    "1": "Utara",
    "2": "Selatan",
    "3": "Timur",
    "4": "Barat",
}

CSV_FILE   = "traffic.csv"
BATCH_SIZE = 50    # Kecil supaya tidak kena rate limit
DELAY_SEC  = 3     # Delay 3 detik antar batch
MAX_RETRY  = 3     # Retry maksimal 3x

# ── Timestamp Shifting ────────────────────────────────────────
# Data CSV asli mulai dari 2015-11-01
# Kita geser ke 29 hari lalu dari sekarang
# Sehingga data masuk dalam retention period InfluxDB Cloud (30 hari)
DATASET_START = datetime(2015, 11, 1, tzinfo=timezone.utc)
IMPORT_START  = datetime.now(timezone.utc) - timedelta(days=29)


def check_connection() -> bool:
    """Test koneksi ke InfluxDB Cloud sebelum import."""
    print("🔍 Mengecek koneksi ke InfluxDB Cloud...")
    try:
        r = requests.get(
            f"{INFLUX_URL}/api/v2/buckets",
            headers={
                "Authorization": f"Token {INFLUX_TOKEN}",
                "Content-Type":  "application/json"
            },
            timeout=10
        )
        if r.status_code == 200:
            print("✅ InfluxDB Cloud terhubung!\n")
            return True
        print(f"❌ Gagal konek: HTTP {r.status_code}")
        return False
    except Exception as e:
        print(f"❌ Error koneksi: {e}")
        return False


def parse_datetime(dt_str: str) -> int | None:
    """
    Konversi datetime CSV → Unix nanoseconds dengan timestamp shifting.

    Contoh:
    - Data asli  : 2015-11-01 08:00:00 (jam sibuk pagi)
    - Digeser ke : 2026-04-08 08:00:00 (tetap jam sibuk pagi!)

    Pola lalu lintas (rush hour, malam, dll) tetap terjaga.
    Hanya tahunnya yang berubah supaya masuk retention period.
    """
    try:
        dt_original = datetime.strptime(dt_str.strip(), "%Y-%m-%d %H:%M:%S")
        dt_original = dt_original.replace(tzinfo=timezone.utc)

        # Hitung selisih waktu dari awal dataset
        delta = dt_original - DATASET_START

        # Geser ke masa kini
        dt_shifted = IMPORT_START + delta

        # Skip data yang melebihi waktu sekarang
        if dt_shifted > datetime.now(timezone.utc):
            return None

        return int(dt_shifted.timestamp() * 1_000_000_000)

    except ValueError:
        return None


def write_batch(lines: list, batch_num: int) -> bool:
    """
    Kirim batch data ke InfluxDB dengan retry logic.
    Kalau 429 (rate limit) → tunggu lebih lama lalu retry.
    Kalau 400 (bad request) → print error detail untuk debug.
    """
    body = "\n".join(lines)

    for attempt in range(1, MAX_RETRY + 1):
        try:
            r = requests.post(
                f"{INFLUX_URL}/api/v2/write",
                params={
                    "org":       INFLUX_ORG,
                    "bucket":    INFLUX_BUCKET,
                    "precision": "ns"
                },
                headers={
                    "Authorization": f"Token {INFLUX_TOKEN}",
                    "Content-Type":  "text/plain; charset=utf-8"
                },
                data=body.encode("utf-8"),
                timeout=30
            )

            if r.status_code == 204:
                # Sukses!
                return True

            elif r.status_code == 429:
                # Rate limit — tunggu makin lama tiap retry
                wait = DELAY_SEC * (attempt * 5)
                print(f"  ⚠️  Rate limit! Tunggu {wait}s ({attempt}/{MAX_RETRY})...")
                time.sleep(wait)

            elif r.status_code == 400:
                # Bad request — print detail untuk debug
                print(f"  ❌ Bad Request: {r.text[:200]}")
                return False  # Tidak perlu retry, format memang salah

            elif r.status_code == 401:
                print(f"  ❌ Token tidak valid! Cek .env kamu.")
                return False

            else:
                print(f"  ⚠️  HTTP {r.status_code} — retry {attempt}/{MAX_RETRY}")
                time.sleep(DELAY_SEC)

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Timeout! Retry {attempt}/{MAX_RETRY}...")
            time.sleep(DELAY_SEC * 2)

        except Exception as e:
            print(f"  ⚠️  Error: {e} — retry {attempt}/{MAX_RETRY}")
            time.sleep(DELAY_SEC)

    return False


def import_csv():
    """Main function: baca CSV → geser timestamp → import ke InfluxDB."""

    if not os.path.exists(CSV_FILE):
        print(f"❌ File '{CSV_FILE}' tidak ditemukan!")
        print(f"   Download dari:")
        print(f"   https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset")
        return

    print(f"📂 File          : {CSV_FILE}")

    # Hitung total baris
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        total_rows = sum(1 for _ in f) - 1

    # Info timestamp shifting
    import_end = datetime.now(timezone.utc)
    print(f"📊 Total data    : {total_rows:,} baris")
    print(f"⏰ Timestamp asli: mulai 2015-11-01")
    print(f"⏰ Digeser ke    : {IMPORT_START.strftime('%Y-%m-%d')} s/d {import_end.strftime('%Y-%m-%d')}")
    print(f"📦 Batch size    : {BATCH_SIZE} baris/batch")
    print(f"⏱️  Delay         : {DELAY_SEC}s antar batch")

    total_batches = (total_rows // BATCH_SIZE) + 1
    est_minutes   = (total_batches * DELAY_SEC) // 60
    print(f"⏳ Estimasi waktu: ~{est_minutes} menit\n")

    batch         = []
    total_written = 0
    total_skipped = 0
    total_failed  = 0
    batch_count   = 0

    print("⏳ Memulai import...")
    print("-" * 55)

    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                dt_str       = row.get("DateTime", "").strip()
                junction     = row.get("Junction", "").strip()
                vehicles_str = row.get("Vehicles", "0").strip()

                # Validasi field
                if not dt_str or not junction:
                    total_skipped += 1
                    continue

                # Parse & geser timestamp
                timestamp_ns = parse_datetime(dt_str)
                if timestamp_ns is None:
                    # Data ini melebihi waktu sekarang → skip
                    total_skipped += 1
                    continue

                # Parse vehicle count
                try:
                    vehicle_count = int(float(vehicles_str))
                except (ValueError, TypeError):
                    total_skipped += 1
                    continue

                # Normalisasi ke skala NERIC 0-20
                # Data asli bisa ratusan kendaraan/jam → scale down
                vehicle_normalized = min(20, max(0, vehicle_count // 10))

                # Mapping junction → jalur NERIC
                lane_name = JUNCTION_MAP.get(junction, f"Lane{junction}")
                lane_id   = int(junction) - 1  # 0-indexed

                # Buat InfluxDB Line Protocol
                line = (
                    f"sensor_data,"
                    f"lane_name={lane_name},"
                    f"lane_id={lane_id},"
                    f"source=kaggle_real "
                    f"vehicle_count={vehicle_normalized}i,"
                    f"vehicle_count_raw={vehicle_count}i "
                    f"{timestamp_ns}"
                )
                batch.append(line)

                # Kirim batch kalau sudah penuh
                if len(batch) >= BATCH_SIZE:
                    batch_count += 1
                    success = write_batch(batch, batch_count)

                    if success:
                        total_written += len(batch)
                        pct = total_written / total_rows * 100
                        print(
                            f"  ✅ Batch #{batch_count:3d} — "
                            f"{total_written:6,}/{total_rows:,} "
                            f"({pct:.1f}%)"
                        )
                    else:
                        total_failed += len(batch)
                        print(f"  ❌ Batch #{batch_count:3d} gagal!")

                    batch = []
                    time.sleep(DELAY_SEC)  # Delay antar batch

            except Exception:
                total_skipped += 1
                continue

        # Kirim sisa batch terakhir
        if batch:
            batch_count += 1
            success = write_batch(batch, batch_count)
            if success:
                total_written += len(batch)
                print(
                    f"  ✅ Batch #{batch_count:3d} — "
                    f"{total_written:,}/{total_rows:,} (100%)"
                )
            else:
                total_failed += len(batch)

    # Laporan akhir
    print(f"\n{'='*55}")
    print(f"  📊 HASIL IMPORT")
    print(f"{'='*55}")
    print(f"  ✅ Berhasil : {total_written:,} baris")
    print(f"  ❌ Gagal    : {total_failed:,} baris")
    print(f"  ⚠️  Dilewati : {total_skipped:,} baris")
    print(f"  📦 Batch    : {batch_count}")
    print(f"{'='*55}")

    if total_written > 0:
        print(f"\n🎉 Import selesai!")
        print(f"   Cek di InfluxDB Cloud → Data Explorer → sensor_data")
    else:
        print(f"\n❌ Tidak ada data yang berhasil diimport.")


def verify_import():
    """Verifikasi data sudah masuk ke InfluxDB."""
    print("\n🔍 Verifikasi data di InfluxDB...")

    # Query 30 hari terakhir (sesuai data yang kita import)
    query = f"""
    from(bucket: "{INFLUX_BUCKET}")
      |> range(start: -30d)
      |> filter(fn: (r) => r["_measurement"] == "sensor_data")
      |> filter(fn: (r) => r["_field"] == "vehicle_count")
      |> count()
    """
    try:
        r = requests.post(
            f"{INFLUX_URL}/api/v2/query",
            params={"org": INFLUX_ORG},
            headers={
                "Authorization": f"Token {INFLUX_TOKEN}",
                "Content-Type":  "application/vnd.flux",
                "Accept":        "application/csv"
            },
            data=query.encode("utf-8"),
            timeout=30
        )
        if r.status_code == 200 and r.text:
            lines = [l for l in r.text.strip().split("\n")
                     if l and not l.startswith("#")]
            for line in lines[1:]:
                parts = line.split(",")
                if parts and parts[-1].strip().isdigit():
                    count = int(parts[-1].strip())
                    print(f"  ✅ Total records di InfluxDB: {count:,}")
                    print(f"  🎉 Data siap dipakai NERIC Backend!")
                    return
        print("  ⚠️  Belum ada data (coba cek manual di Data Explorer)")
    except Exception as e:
        print(f"  ⚠️  Verifikasi error: {e}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "🚦" * 20)
    print("  NERIC — Import Traffic Data ke InfluxDB Cloud v3")
    print("🚦" * 20 + "\n")

    if not check_connection():
        exit(1)

    import_csv()
    verify_import()

    print("\n✅ Done! Langkah selanjutnya:")
    print("   1. uvicorn main:app --reload --port 8000")
    print("   2. Buka http://localhost:8000/docs")
    print("   3. Test endpoint /api/history/sensors")