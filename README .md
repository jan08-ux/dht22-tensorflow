#  ThermalMind 


<p align="center">
  <strong>A fully edge-native IoT system that streams live predictions vs actual sensor readings to a  browser dashboard — all without any cloud inference.</strong>
</p>

---

##  Table of Contents

- [Features](#-features)
- [Hardware Requirements](#-hardware-requirements)
- [Project Structure](#-project-structure)
- [Deep Code Analysis](#-deep-code-analysis)
- [ML Model Details](#-ml-model-details)
- [Getting Started](#-getting-started)
- [Future Work](#-Future-work)

---

No cloud. No external API. Pure edge AI.

---

## Features

- **On-device ML inference** using TensorFlow Lite Micro — zero cloud dependency
- **DHT22 integration** — reads temperature every 2 seconds with NaN-guard fallback
- **Time-series forecasting** — 10-step lookback window, min-max normalized
- **Live SSE stream** — browser dashboard updates in real time without polling
- **Interactive Chart.js dashboard** — zoom, pan, scroll, range filter
- **Responsive UI** — desktop, tablet, and mobile layouts via CSS Grid
- **Circular data buffer** — stores last 100 readings with millisecond timestamps
- **FreeRTOS-safe data writes** — `portMUX` critical section protects shared state
- **Auto-reconnect** — browser reconnects to SSE stream on disconnect (3 s retry)
- **Visual error tracking** — live absolute error card comparing ML vs. sensor

---

## Hardware Requirements

| Component | Details |
|---|---|
| Microcontroller | ESP32-S3 DevKitC-1 |
| Sensor | DHT22 (AM2302) Temperature & Humidity |
| Data Pin | GPIO **4** |
| Power | 3.3V or 5V via USB |
| WiFi | 2.4 GHz network |

---

## Project Structure

```
dht22-tensorflow/
├── DHT22/
│   ├── src/
│   │   ├── main.cpp          # Core firmware: sensor → inference → web server
│   │   └── model_data.h      # TFLite model as C byte array (~39 KB)
│   ├── platformio.ini        # PlatformIO build config
│   └── README                # PlatformIO test runner placeholder
└── README.md                 # This file
```

---

## ML Model Details

### Training Pipeline 

1. Collect temperature readings from a DHT22 sensor .
2. Fit `MinMaxScaler` over the full dataset → record `temp_min`, `temp_max`.
3. Build sliding windows of length 10 as X, next value as y.
4. Train a regression model (Dense/LSTM).

### Scaler Constants

```cpp
const float temp_min = 16.792248f;  // dataset minimum
const float temp_max = 33.143745f;  // dataset maximum
```
> ⚠️ If you retrain the model on new data, you **must** update these constants in `main.cpp` to match the new scaler fit.

---

## Getting Started

### Prerequisites

- [PlatformIO IDE](https://platformio.org/) (VSCode extension recommended)


### Installation

**1. Clone the repository**
```bash
git clone https://github.com/jan08-ux/dht22-tensorflow.git
cd dht22-tensorflow/DHT22
```

**2. Set WiFi credentials** in `src/main.cpp`:
```cpp
const char* ssid     = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
```

**3. (Optional) Retrain the model** — if you want to train on your own data, run the Colab notebook, export `model_data.h`, and replace `src/model_data.h`.

**4. Build & Upload**
```bash
pio run --target upload
```

**5. Monitor serial output**
```bash
pio device monitor
```

**6. Open the dashboard**

Find the IP address in the serial monitor output and navigate to:
```
http://<ESP32_IP_ADDRESS>/
```


---

## Future Work

**Potential Improvements:**
- [ ] Implement WiFiManager for captive-portal WiFi setup
- [ ] Replace `AllOpsResolver` with selective op registration
- [ ] Add SPIFFS/LittleFS to persist readings across reboots
- [ ] Add MQTT publishing for Home Assistant / Node-RED integration
- [ ] Implement anomaly detection alerts via WebSocket push

---




