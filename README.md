# NERIC: Neural-Inspired Traffic Controller

**Implementation of Neuromorphic-Inspired AI to Optimize Urban Traffic Flow Based on IoT Ecosystems**

NERIC is an advanced traffic management system designed to address the inefficiencies of traditional time-based traffic signaling. By emulating human cognitive processes—specifically attention and prioritization—NERIC dynamically adjusts traffic signals in real-time based on high-density IoT sensor data.

---

## Authors
* **Muhammad Rizki Fahrezi**
* **Aurel Zalsabilla**
* **Arya Satya Andhika Akbar**
* **Bima Akbar Rizqullah**

*Information Systems Study Program, Universitas Airlangga, Indonesia.*

---

## System Overview

Traditional traffic systems often fail to adapt to sudden spikes in vehicle volume, leading to economic inefficiency and increased carbon emissions. NERIC introduces a Neuromorphic AI approach that:
1.  **Assesses** real-time traffic conditions via an IoT ecosystem.
2.  **Predicts** potential congestion points using neural-inspired logic.
3.  **Optimizes** signal durations dynamically to improve vehicle throughput.

## Technical Architecture

The system is built upon a robust, scalable backend architecture designed for low-latency processing of time-series data:

* **Core Framework:** FastAPI (Python) for high-performance asynchronous API handling.
* **Database:** InfluxDB 3.0 (Open Source) for high-ingestion time-series storage.
* **Messaging Protocol:** MQTT via Mosquitto Broker for seamless IoT sensor integration.
* **Engine:** Neuromorphic-inspired Spiking Neural Network (SNN) logic for adaptive decision making.

## Prerequisites

To deploy the NERIC backend locally, ensure the following components are installed:
* Python 3.10 or higher
* InfluxDB 3 OSS (Local Instance)
* Mosquitto MQTT Broker

## Installation and Deployment

### 1. Repository Setup
```bash
git clone [https://github.com/yourusername/neric-backend.git](https://github.com/yourusername/neric-backend.git)
cd neric-backend

2. Environment Configuration
Create a .env file in the root directory and configure the following variables:

Cuplikan kode
INFLUXDB_URL=http://localhost:8181
INFLUXDB_TOKEN=your_administrative_token
INFLUXDB_DATABASE=neric-data
MQTT_BROKER=localhost
MQTT_PORT=1883

3. Virtual Environment and Dependencies
Bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

pip install -r requirements.txt

4. Running the Service
Ensure InfluxDB and Mosquitto services are active, then execute:

Bash
uvicorn main:app --reload

Data Methodology
NERIC utilizes a combination of historical traffic datasets and real-time sensor inputs. During the development phase, the system is validated using the Traffic Prediction Dataset to simulate various high-density urban intersection scenarios. Experimental findings indicate a significant reduction in average wait times and improved vehicle flow compared to static signaling methods.

© 2026 NERIC Team - Universitas Airlangga