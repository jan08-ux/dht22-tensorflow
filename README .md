# 🌡️ ThermalMind — On-Device AI Temperature Forecasting with ESP32-S3 & TensorFlow Lite Micro

<p align="center">
  <img src="https://img.shields.io/badge/Platform-ESP32--S3-red?style=for-the-badge&logo=espressif&logoColor=white"/>
  <img src="https://img.shields.io/badge/Framework-Arduino%20%2B%20PlatformIO-blue?style=for-the-badge&logo=arduino&logoColor=white"/>
  <img src="https://img.shields.io/badge/ML-TensorFlow%20Lite%20Micro-orange?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Sensor-DHT22-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dashboard-Live%20Web%20UI-purple?style=for-the-badge&logo=googlechrome&logoColor=white"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey?style=for-the-badge"/>
</p>

<p align="center">
  <strong>A fully edge-native IoT system that trains a time-series ML model in Google Colab, converts it to TFLite, deploys it onto an ESP32-S3, and streams live predictions vs. actual sensor readings to a beautiful browser dashboard — all without any cloud inference.</strong>
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Project Structure](#-project-structure)
- [Deep Code Analysis](#-deep-code-analysis)
- [ML Model Details](#-ml-model-details)
- [Web Dashboard](#-web-dashboard)
- [Getting Started](#-getting-started)
- [Configuration](#-configuration)
- [Serial Monitor Output](#-serial-monitor-output)
- [Known Limitations & Future Work](#-known-limitations--future-work)

---

## 🔍 Overview

**ThermalMind** demonstrates a complete end-to-end TinyML pipeline:

1. **Data Collection** — DHT22 temperature sensor sampled every 2 seconds on an ESP32-S3.
2. **Model Training** — A time-series regression model trained in Google Colab on real sensor data.
3. **Edge Deployment** — The trained `.tflite` model is converted to a C byte array (`model_data.h`) and flashed directly onto the ESP32-S3.
4. **On-Device Inference** — TensorFlow Lite Micro runs the model using the last 10 temperature readings as a sliding-window input to predict the *next* temperature value.
5. **Live Dashboard** — An AsyncWebServer hosted on the ESP32-S3 streams predictions and actuals in real-time via Server-Sent Events (SSE) to a Chart.js web dashboard accessible from any device on the same WiFi network.

No cloud. No external API. Pure edge AI.

---

## ✨ Features

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

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ESP32-S3 DevKitC-1                          │
│                                                                     │
│  ┌──────────┐    GPIO4    ┌──────────────────────────────────────┐  │
│  │  DHT22   │ ──────────► │         Arduino loop()               │  │
│  │ Sensor   │             │  readTemperature() every 2s          │  │
│  └──────────┘             │  constrain(t, 0°C, 60°C)             │  │
│                           └────────────────┬─────────────────────┘  │
│                                            │                        │
│                                            ▼                        │
│                           ┌──────────────────────────────────────┐  │
│                           │     Circular History Buffer           │  │
│                           │     float temp_history[10]            │  │
│                           │     ring pointer: history_index       │  │
│                           └────────────────┬─────────────────────┘  │
│                                            │                        │
│                                            ▼                        │
│                           ┌──────────────────────────────────────┐  │
│                           │     Min-Max Normalization             │  │
│                           │     range: [16.79°C – 33.14°C]       │  │
│                           │     → input tensor [1 × 10] float32  │  │
│                           └────────────────┬─────────────────────┘  │
│                                            │                        │
│                                            ▼                        │
│                           ┌──────────────────────────────────────┐  │
│                           │     TFLite Micro Inference            │  │
│                           │     model_data.h  (~39 KB)            │  │
│                           │     tensor_arena: 15 KB               │  │
│                           │     → output scalar float32           │  │
│                           └────────────────┬─────────────────────┘  │
│                                            │                        │
│                                            ▼                        │
│                           ┌──────────────────────────────────────┐  │
│                           │     Inverse Scale + Store             │  │
│                           │     predicted_temps[100] circular     │  │
│                           │     portMUX critical section          │  │
│                           └────────────────┬─────────────────────┘  │
│                                            │                        │
│                                            ▼                        │
│                           ┌──────────────────────────────────────┐  │
│                           │     AsyncWebServer (port 80)          │  │
│                           │     GET /       → HTML dashboard      │  │
│                           │     GET /events → SSE JSON stream     │  │
│                           └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                      │ WiFi
                                      ▼
                        ┌─────────────────────────┐
                        │   Browser Dashboard      │
                        │   Chart.js live chart    │
                        │   SSE auto-reconnect     │
                        │   Zoom / Pan / Scroll    │
                        └─────────────────────────┘
```

---

## 🔧 Hardware Requirements

| Component | Details |
|---|---|
| Microcontroller | ESP32-S3 DevKitC-1 |
| Sensor | DHT22 (AM2302) Temperature & Humidity |
| Data Pin | GPIO **4** |
| Power | 3.3V or 5V via USB |
| WiFi | 2.4 GHz network |

**Wiring:**

```
DHT22 Pin 1 (VCC)  → ESP32-S3 3.3V
DHT22 Pin 2 (DATA) → ESP32-S3 GPIO 4  (with 10kΩ pull-up to 3.3V)
DHT22 Pin 4 (GND)  → ESP32-S3 GND
```

---

## 📁 Project Structure

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

## 🔬 Deep Code Analysis

### `platformio.ini`

```ini
[env:esp32-s3-devkitc-1]
platform = espressif32
board    = esp32-s3-devkitc-1
framework = arduino
```

- **Target**: ESP32-S3 DevKitC-1 — chosen for its extra RAM/flash over standard ESP32, necessary for holding the TFLite tensor arena and the embedded HTML page simultaneously.
- **`lib_deps`**:
  - `adafruit/DHT sensor library @ ^1.4.4` — DHT22 driver.
  - `spaziochirale/Chirale_TensorFlowLite @ 2.0.0` — TFLite Micro port for ESP32.
  - `AsyncTCP` + `ESPAsyncWebServer` — non-blocking async HTTP + SSE server.
- **`build_flags`**:
  - `-DTF_LITE_MICRO_ALLOW_REPEATED_OPS` — allows the TFLite Micro op resolver to register ops more than once (required by some model topologies).
  - `-O2` — compiler optimization level 2 for speed.
  - `-Wl,--gc-sections` — linker removes unused code sections, shrinking binary size.
- **`board_build.partitions = huge_app.csv`** — expands the app partition to fit the large binary (model weights + HTML + TFLite runtime).
- **`upload_speed = 921600`** — maximum UART upload speed for fast flashing.

---

### `main.cpp` — Line-by-Line Breakdown

#### Includes & Globals (Lines 1–61)

```cpp
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
```
TFLite Micro headers. `all_ops_resolver.h` registers every built-in op — convenient but costs ~30–50 KB of flash. For production, replace with `MicroMutableOpResolver` and register only the ops your model uses.

```cpp
const int N_STEPS = 10;
float temp_history[N_STEPS];
int history_index = 0;
```
Circular ring buffer. `history_index` is the **next write position** (not the oldest). After each write, it advances modulo 10. When reading for inference, `fill_input_from_history()` starts from `history_index` (oldest entry) and wraps around — yielding a correctly time-ordered sequence.

```cpp
const float temp_min = 16.792248f;
const float temp_max = 33.143745f;
```
These exact float values were computed from the training dataset in Google Colab using `MinMaxScaler`. They must match the scaler used during training exactly — any mismatch causes the model to receive out-of-distribution inputs, degrading predictions.

```cpp
const int kTensorArenaSize = 15 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
```
A 15 KB static scratch buffer allocated in global RAM. TFLite Micro uses this for all intermediate tensor activations during inference. This size is tuned to fit the model's activation memory footprint — too small causes `AllocateTensors()` to fail.

#### HTML Dashboard (Lines 64–662)

The entire dashboard is embedded as a raw string literal in `PROGMEM` — stored in flash rather than RAM. This is a critical optimization: the ESP32-S3 has ~512 KB of RAM but 8–16 MB of flash, so keeping the 5 KB+ HTML page in flash prevents RAM exhaustion.

Key dashboard sections:
- **Stats grid** — 4 live cards: Predicted (ML), Actual (DHT22), Error, Reading#.
- **Chart.js canvas** — 550 px tall line chart with two datasets, both styled with bezier smoothing (`tension: 0.5`).
- **Zoom plugin** — mouse-wheel zoom + Ctrl+drag pan via `chartjs-plugin-zoom`.
- **SSE client** — `EventSource('/events')` for true server-push; no polling loop.

#### Helper Functions (Lines 664–688)

```cpp
void add_to_history(float temp) {
  temp_history[history_index] = temp;
  history_index = (history_index + 1) % N_STEPS;
}
```
Standard ring buffer write. Overwrites the oldest slot on each call.

```cpp
void fill_input_from_history() {
  int idx = history_index;  // oldest entry
  for (int i = 0; i < N_STEPS; i++) {
    float t_scaled = (temp_history[idx] - temp_min) / (temp_max - temp_min);
    t_scaled = constrain(t_scaled, 0.0f, 1.0f);
    input->data.f[i] = t_scaled;
    idx = (idx + 1) % N_STEPS;
  }
}
```
Reads the ring buffer oldest-to-newest, applies min-max normalization, clamps to [0, 1] to guard against out-of-range sensor readings, then writes directly into the TFLite input tensor's float data pointer.

```cpp
void store_data(float predicted, float actual) {
  predicted_temps[reading_index] = predicted;
  actual_temps[reading_index]    = actual;
  timestamps[reading_index]      = millis() / 1000UL;
  reading_index = (reading_index + 1) % MAX_READINGS;
  if (reading_count < MAX_READINGS) reading_count++;
}
```
Another ring buffer, this one for storing the last 100 predicted/actual pairs with Unix-like timestamps (seconds since boot).

#### WiFi & Server Setup (Lines 690–725)

```cpp
WiFi.mode(WIFI_STA);
WiFi.begin(ssid, password);
```
Station mode (not AP mode). Includes a 15-second timeout watchdog loop that resets the timer and keeps retrying rather than hanging indefinitely.

```cpp
events.onConnect([](AsyncEventSourceClient *client) {
  Serial.println("Client connected to /events");
});
server.addHandler(&events);
```
SSE endpoint registered on `/events`. Any browser opening this path receives a persistent HTTP connection over which the server pushes newline-delimited JSON events.

#### `setup()` (Lines 728–778)

The initialization sequence is carefully ordered:
1. DHT22 init + pre-fill history with first real reading (falls back to 25°C on NaN).
2. TFLite model load → schema version check → `AllocateTensors()` → get input/output tensor pointers.
3. WiFi connect → HTTP server start.

The static allocation of `MicroInterpreter` inside `setup()` is intentional — it avoids heap fragmentation by keeping the interpreter in the stack frame of `setup()`, which persists for the lifetime of the program (Arduino's `setup()` frame never returns in the FreeRTOS task model).

#### `loop()` (Lines 781–833)

```cpp
float temperature_now = dht.readTemperature();
if (isnan(temperature_now)) { delay(2000); return; }
temperature_now = constrain(temperature_now, 0.0f, 60.0f);
```
NaN guard (DHT22 can return NaN on read failures) + physical sanity clamp.

```cpp
add_to_history(temperature_now);
fill_input_from_history();
interpreter->Invoke();
```
The core inference pipeline: update history → normalize into tensor → run model.

```cpp
float pred_scaled = output->data.f[0];
float temperature_pred = pred_scaled * (temp_max - temp_min) + temp_min;
temperature_pred = constrain(temperature_pred, temp_min, temp_max);
```
Inverse min-max transform: converts the model's normalized [0,1] output back to Celsius. The final `constrain` prevents physically nonsensical extrapolations.

```cpp
portENTER_CRITICAL(&mux);
store_data(temperature_pred, temperature_now);
portEXIT_CRITICAL(&mux);
```
FreeRTOS critical section. The AsyncWebServer runs on a separate FreeRTOS task and could theoretically read `predicted_temps[]` concurrently — this mutex prevents torn reads/writes.

```cpp
String json = "{\"predicted\": " + String(temperature_pred, 2) +
              ", \"actual\": "   + String(temperature_now,  2) +
              ", \"reading\": "  + String(reading_number)      + "}";
events.send(json.c_str(), "message", millis());
delay(2000);
```
Constructs a minimal JSON payload and pushes it to all connected SSE clients. The `millis()` argument sets the SSE `id:` field, enabling browser reconnection to resume from the last event ID.

---

### `model_data.h`

A ~39 KB C header containing the TFLite FlatBuffer model as an `unsigned char` array. Key facts extracted from the binary:

| Property | Value |
|---|---|
| FlatBuffer magic | `TFL3` |
| TensorFlow version | 2.19.0 |
| TFLite runtime version | 1.5.0 |
| Input tensor name | `input_layer` |
| Output tensor name | `output_0` |
| Input shape | `[1, 10]` float32 |
| Output shape | `[1, 1]` float32 |
| Model size | ~39 KB |

The model is a time-series regression network (likely LSTM or Dense layers) trained in Google Colab to predict `T[t+1]` from `[T[t-9] … T[t]]`.

---

## 📊 ML Model Details

### Training Pipeline (Google Colab — not included in repo)

1. Collect temperature readings from a DHT22 sensor (or simulate).
2. Fit `MinMaxScaler` over the full dataset → record `temp_min`, `temp_max`.
3. Build sliding windows of length 10 as X, next value as y.
4. Train a regression model (Dense/LSTM).
5. Export to `.tflite` using `tf.lite.TFLiteConverter`.
6. Convert `.tflite` binary to C array:
   ```bash
   xxd -i model.tflite > model_data.h
   ```
7. Drop `model_data.h` into the `src/` directory.

### Scaler Constants

```cpp
const float temp_min = 16.792248f;  // dataset minimum
const float temp_max = 33.143745f;  // dataset maximum
```
> ⚠️ If you retrain the model on new data, you **must** update these constants in `main.cpp` to match the new scaler fit.

---

## 🖥️ Web Dashboard

The dashboard is hosted entirely on the ESP32-S3 at `http://<device-ip>/`.

| Element | Detail |
|---|---|
| Predicted card | Current ML model output in °C |
| Actual card | Live DHT22 reading in °C |
| Error card | Absolute difference `\|predicted − actual\|` |
| Reading # | Total inference count since boot |
| Chart | 50-point sliding window, auto-scroll toggle |
| Zoom/Pan | Mouse-wheel zoom, Ctrl+drag pan |
| Range Filter | Set custom X-axis min/max reading numbers |
| Data Table | Last 10 readings with predicted/actual/error |
| Actual delay | Actual line rendered 5 s after predicted for visual clarity |
| SSE reconnect | Auto-retry every 3 s on disconnect |

---

## 🚀 Getting Started

### Prerequisites

- [PlatformIO IDE](https://platformio.org/) (VSCode extension recommended)
- ESP32-S3 DevKitC-1 board
- DHT22 sensor + 10 kΩ pull-up resistor
- USB-C cable
- 2.4 GHz WiFi network

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

## ⚙️ Configuration

| Constant | Location | Default | Description |
|---|---|---|---|
| `DHTPIN` | `main.cpp` | `4` | GPIO pin for DHT22 data |
| `N_STEPS` | `main.cpp` | `10` | Lookback window (must match model input) |
| `MAX_READINGS` | `main.cpp` | `100` | Circular log size |
| `kTensorArenaSize` | `main.cpp` | `15360` bytes | TFLite scratch memory |
| `temp_min / temp_max` | `main.cpp` | `16.79 / 33.14` | MinMaxScaler bounds from training |
| `DELAY_MS` | HTML JS | `5000` ms | Delay before plotting actual data |
| `Y_MIN / Y_MAX` | HTML JS | `20 / 40` | Chart Y-axis display range |
| Sampling interval | `loop()` | `2000` ms | `delay(2000)` at end of loop |

---

## 📟 Serial Monitor Output

```
================================
ESP32-S3 Temperature Predictor
Live WebSocket Dashboard
================================
Connecting to WiFi: MyNetwork
..........
✓ WiFi connected!
IP address: 192.168.1.42
Loading model...
✓ Model loaded!
✓ HTTP server started
================================
Open browser: http://192.168.1.42/
================================

Reading #1  | Predicted: 27.34°C | Actual: 27.18°C
Reading #2  | Predicted: 27.41°C | Actual: 27.22°C
Reading #3  | Predicted: 27.39°C | Actual: 27.31°C
```

---

## ⚡ Known Limitations & Future Work

**Current Limitations:**
- WiFi credentials are hardcoded — no OTA WiFi config (e.g., WiFiManager).
- `AllOpsResolver` wastes ~30–50 KB of flash; should be replaced with `MicroMutableOpResolver` for production.
- No HTTPS — dashboard is served over plain HTTP.
- No humidity prediction (DHT22 provides humidity too, currently unused).
- Model scaler constants must be manually updated on retrain.
- No persistent storage — all readings reset on reboot.

**Potential Improvements:**
- [ ] Add humidity forecasting (second model output)
- [ ] Implement WiFiManager for captive-portal WiFi setup
- [ ] Replace `AllOpsResolver` with selective op registration
- [ ] Add SPIFFS/LittleFS to persist readings across reboots
- [ ] Export CSV download button on the dashboard
- [ ] Add MQTT publishing for Home Assistant / Node-RED integration
- [ ] Implement anomaly detection alerts via WebSocket push
- [ ] OTA firmware updates

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">
  Built with ❤️ on the ESP32-S3 · TensorFlow Lite Micro · PlatformIO
</p>
