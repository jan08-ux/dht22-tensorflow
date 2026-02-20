

#include <Arduino.h>
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>

#include <DHT.h>
#include "model_data.h"

// TFLite Micro includes
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

// ========== USER WIFI CREDENTIALS ==========
const char* ssid = "";
const char* password = "";

// ========== DHT22 CONFIG ==========
#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

// ========== TIME SERIES CONFIG ==========
const int N_STEPS = 10;
float temp_history[N_STEPS];
int history_index = 0;

// ========== DATA LOGGING ==========
const int MAX_READINGS = 100;
float predicted_temps[MAX_READINGS];
float actual_temps[MAX_READINGS];
unsigned long timestamps[MAX_READINGS];
int reading_count = 0;
int reading_index = 0;
unsigned long reading_number = 0;

// ========== SCALER (from Colab) ==========
const float temp_min = 16.792248f;
const float temp_max = 33.143745f;
const float y_min = 16.792248f;
const float y_max = 33.143745f;

// ========== TFLITE SETUP ==========
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;

  tflite::AllOpsResolver resolver;
  const int kTensorArenaSize = 15 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
}

// ========== WEB SERVER ==========
AsyncWebServer server(80);
AsyncEventSource events("/events");

portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// ========== HTML PAGE WITH LIVE UPDATES ==========
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Temperature Predictor - Professional Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-zoom@2.1.0/dist/chartjs-plugin-zoom.min.js"></script>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
      min-height: 100vh;
      padding: 20px;
    }

    .container {
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      overflow: hidden;
    }

    .header {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 40px 30px;
      text-align: center;
    }

    .header h1 {
      font-size: 32px;
      margin-bottom: 10px;
      font-weight: 700;
    }

    .header p {
      font-size: 16px;
      opacity: 0.95;
    }

    .status-dot {
      display: inline-block;
      width: 12px;
      height: 12px;
      background: #4ade80;
      border-radius: 50%;
      margin-right: 8px;
      animation: pulse 1.5s infinite;
    }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.5; }
    }

    .stats-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 20px;
      padding: 30px;
      background: #f0f4f8;
    }

    .stat-card {
      background: white;
      padding: 25px;
      border-radius: 10px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.08);
      border-left: 5px solid;
      transition: transform 0.3s;
    }

    .stat-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 8px 20px rgba(0,0,0,0.12);
    }

    .stat-card.predicted { border-left-color: #667eea; }
    .stat-card.actual { border-left-color: #51cf66; }
    .stat-card.error { border-left-color: #ff6b6b; }
    .stat-card.reading { border-left-color: #ffa94d; }

    .stat-label {
      font-size: 12px;
      color: #666;
      text-transform: uppercase;
      letter-spacing: 1px;
      font-weight: 700;
      margin-bottom: 8px;
    }

    .stat-value {
      font-size: 42px;
      font-weight: 700;
      font-family: 'Monaco', monospace;
      margin-bottom: 5px;
    }

    .stat-card.predicted .stat-value { color: #667eea; }
    .stat-card.actual .stat-value { color: #51cf66; }
    .stat-card.error .stat-value { color: #ff6b6b; }
    .stat-card.reading .stat-value { color: #ffa94d; }

    .stat-unit {
      font-size: 13px;
      color: #999;
      font-weight: 500;
    }

    .chart-section {
      padding: 40px 30px;
      background: white;
    }

    .section-title {
      font-size: 24px;
      font-weight: 700;
      color: #333;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      gap: 10px;
    }

    .controls-row {
      display: flex;
      gap: 15px;
      margin-bottom: 25px;
      flex-wrap: wrap;
      align-items: center;
    }

    .btn {
      padding: 10px 18px;
      background: #667eea;
      color: white;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      font-size: 13px;
      font-weight: 600;
      transition: all 0.3s;
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .btn:hover {
      background: #764ba2;
      transform: translateY(-2px);
      box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
    }

    .btn:active {
      transform: translateY(0);
    }

    .btn.secondary {
      background: #e9ecef;
      color: #333;
    }

    .btn.secondary:hover {
      background: #dee2e6;
    }

    .range-control {
      display: flex;
      gap: 10px;
      align-items: center;
      background: #f0f4f8;
      padding: 12px 16px;
      border-radius: 8px;
    }

    .range-input {
      width: 70px;
      padding: 8px 10px;
      border: 1px solid #ddd;
      border-radius: 6px;
      font-size: 13px;
    }

    .chart-container {
      position: relative;
      height: 550px;
      background: #fafbfc;
      border: 2px solid #e9ecef;
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 30px;
    }

    .chart-info {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
      margin-bottom: 20px;
    }

    .info-box {
      background: #f0f4f8;
      padding: 15px;
      border-radius: 8px;
      font-size: 13px;
    }

    .info-label {
      color: #666;
      font-weight: 600;
      margin-bottom: 5px;
    }

    .info-value {
      font-size: 16px;
      font-weight: 700;
      font-family: 'Monaco', monospace;
    }

    .data-section {
      padding: 40px 30px;
      background: #f0f4f8;
    }

    .data-grid {
      display: grid;
      gap: 12px;
      max-height: 400px;
      overflow-y: auto;
    }

    .data-row {
      background: white;
      padding: 16px;
      border-radius: 8px;
      border-left: 4px solid #667eea;
      display: grid;
      grid-template-columns: 60px 1fr 1fr 1fr;
      gap: 15px;
      align-items: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.05);
      transition: all 0.3s;
    }

    .data-row:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.1);
      transform: translateX(5px);
    }

    .data-row-label {
      font-weight: 700;
      color: #999;
      font-size: 13px;
    }

    .data-value {
      font-family: 'Monaco', monospace;
      font-weight: 600;
      font-size: 14px;
    }

    .predicted-val { color: #667eea; }
    .actual-val { color: #51cf66; }
    .error-val { color: #ff6b6b; }

    .toggle-switch {
      display: flex;
      align-items: center;
      gap: 10px;
      background: #f0f4f8;
      padding: 10px 16px;
      border-radius: 8px;
      cursor: pointer;
      user-select: none;
    }

    .toggle-switch input {
      cursor: pointer;
      width: 18px;
      height: 18px;
    }

    @media (max-width: 1024px) {
      .stats-grid { grid-template-columns: repeat(2, 1fr); }
      .chart-container { height: 400px; }
      .data-row { grid-template-columns: 50px 1fr 1fr; }
    }

    @media (max-width: 768px) {
      .header h1 { font-size: 24px; }
      .stat-value { font-size: 32px; }
      .stats-grid { grid-template-columns: 1fr; }
      .chart-container { height: 300px; }
      .data-row { grid-template-columns: 1fr; }
      .controls-row { flex-direction: column; }
      .range-control { flex-direction: column; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1><span class="status-dot"></span>🌡️ Temperature Predictor</h1>
      <p>Live ML Model Predictions vs Actual DHT22 Sensor Readings</p>
    </div>

    <div class="stats-grid">
      <div class="stat-card predicted">
        <div class="stat-label">📊 Predicted (ML)</div>
        <div class="stat-value" id="predicted">--</div>
        <div class="stat-unit">°C</div>
      </div>
      <div class="stat-card actual">
        <div class="stat-label">🌡️ Actual (DHT22)</div>
        <div class="stat-value" id="actual">--</div>
        <div class="stat-unit">°C</div>
      </div>
      <div class="stat-card error">
        <div class="stat-label">⚠️ Error</div>
        <div class="stat-value" id="error">--</div>
        <div class="stat-unit">°C</div>
      </div>
      <div class="stat-card reading">
        <div class="stat-label">📈 Reading #</div>
        <div class="stat-value" id="reading-num">0</div>
        <div class="stat-unit">samples</div>
      </div>
    </div>

    <div class="chart-section">
      <div class="section-title">📉 Real-Time Temperature Analysis</div>

      <div class="controls-row">
        <button class="btn" id="prevBtn">← Previous</button>
        <button class="btn" id="nextBtn">Next →</button>
        <button class="btn secondary" id="resetZoomBtn">↺ Reset View</button>
        <label class="toggle-switch">
          <input type="checkbox" id="autoScroll" checked>
          <span style="font-weight: 600; font-size: 13px;">Auto-scroll to Latest</span>
        </label>
      </div>

      <div class="range-control">
        <label style="font-weight: 600; font-size: 13px;">X-Axis Range:</label>
        <input type="number" id="rangeMin" class="range-input" placeholder="Min" value="1">
        <span style="font-weight: 600;">to</span>
        <input type="number" id="rangeMax" class="range-input" placeholder="Max" value="50">
        <button class="btn" id="applyRangeBtn">Apply</button>
      </div>

      <div class="chart-info">
        <div class="info-box">
          <div class="info-label">📊 Predicted Line Status</div>
          <div class="info-value">Live • Blue Line</div>
        </div>
        <div class="info-box">
          <div class="info-label">🌡️ Actual Line Status</div>
          <div class="info-value">Delayed 5s • Green Line</div>
        </div>
        <div class="info-box">
          <div class="info-label">📈 Total Readings</div>
          <div class="info-value" id="totalReadings">0</div>
        </div>
      </div>

      <div class="chart-container">
        <canvas id="tempChart"></canvas>
      </div>
    </div>

    <div class="data-section">
      <div class="section-title">📋 Latest Readings (Last 10)</div>
      <div class="data-grid" id="dataList"></div>
    </div>
  </div>

  <script>
    let chart = null;
    let predictedData = [];
    let actualData = [];
    let readingNumbers = [];
    let allDataPoints = [];
    let totalReadings = 0;

    const DELAY_MS = 5000;
    const Y_MIN = 20;
    const Y_MAX = 40;

    function initChart() {
      const ctx = document.getElementById('tempChart').getContext('2d');
      chart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: readingNumbers,
          datasets: [
            {
              label: 'Predicted (ML Model)',
              data: predictedData,
              borderColor: '#667eea',
              backgroundColor: 'rgba(102, 126, 234, 0.12)',
              borderWidth: 4,
              fill: true,
              tension: 0.5,
              pointRadius: 7,
              pointBackgroundColor: '#667eea',
              pointBorderColor: '#fff',
              pointBorderWidth: 3,
              pointHoverRadius: 9,
              pointHoverBorderWidth: 3
            },
            {
              label: 'Actual (DHT22)',
              data: actualData,
              borderColor: '#51cf66',
              backgroundColor: 'rgba(81, 207, 102, 0.12)',
              borderWidth: 4,
              fill: true,
              tension: 0.5,
              pointRadius: 7,
              pointBackgroundColor: '#51cf66',
              pointBorderColor: '#fff',
              pointBorderWidth: 3,
              pointHoverRadius: 9,
              pointHoverBorderWidth: 3
            }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          animation: { duration: 0 },
          interaction: { intersect: false, mode: 'index' },
          plugins: {
            legend: {
              display: true,
              position: 'top',
              labels: {
                font: { size: 14, weight: 'bold' },
                padding: 20,
                usePointStyle: true,
                pointStyle: 'circle'
              }
            },
            zoom: {
              zoom: {
                wheel: { enabled: true, speed: 0.15 },
                pinch: { enabled: true },
                mode: 'xy'
              },
              pan: {
                enabled: true,
                mode: 'xy',
                modifierKey: 'ctrl'
              }
            }
          },
          scales: {
            y: {
              title: { display: true, text: 'Temperature (°C)', font: { size: 13, weight: 'bold' } },
              min: Y_MIN,
              max: Y_MAX,
              ticks: { stepSize: 1, font: { size: 12, weight: '600' } },
              grid: { color: 'rgba(0,0,0,0.08)', drawBorder: true }
            },
            x: {
              title: { display: true, text: 'Reading Number', font: { size: 13, weight: 'bold' } },
              grid: { color: 'rgba(0,0,0,0.08)', drawBorder: true },
              ticks: { font: { size: 12, weight: '600' } }
            }
          }
        }
      });
    }

    function connectToServer() {
      const eventSource = new EventSource('/events');

      eventSource.onmessage = function(event) {
        const data = JSON.parse(event.data);

        document.getElementById('predicted').textContent = data.predicted.toFixed(2);
        document.getElementById('actual').textContent = data.actual.toFixed(2);
        document.getElementById('error').textContent = Math.abs(data.predicted - data.actual).toFixed(2);
        document.getElementById('reading-num').textContent = data.reading;

        totalReadings = data.reading;
        document.getElementById('totalReadings').textContent = totalReadings;

        readingNumbers.push(data.reading);
        predictedData.push(data.predicted);

        if (readingNumbers.length > 50) {
          readingNumbers.shift();
          predictedData.shift();
          actualData.shift();
        }

        if (!chart) initChart();

        chart.data.labels = readingNumbers;
        chart.data.datasets[0].data = predictedData;
        chart.data.datasets[1].data = actualData;
        chart.update();

        allDataPoints.push(data);
        if (allDataPoints.length > 10) allDataPoints.shift();
        updateDataList();

        setTimeout(() => {
          actualData.push(data.actual);
          if (actualData.length > 50) {
            actualData.shift();
          }
          chart.data.datasets[1].data = actualData;
          chart.update();
        }, DELAY_MS);

        if (document.getElementById('autoScroll').checked) {
          chart.resetZoom();
        }
      };

      eventSource.onerror = function() {
        eventSource.close();
        setTimeout(connectToServer, 3000);
      };
    }

    function updateDataList() {
      const list = document.getElementById('dataList');
      list.innerHTML = '';

      for (let i = allDataPoints.length - 1; i >= 0; i--) {
        const point = allDataPoints[i];
        const error = Math.abs(point.predicted - point.actual);

        const row = document.createElement('div');
        row.className = 'data-row';
        row.innerHTML = `
          <div class="data-row-label">#${point.reading}</div>
          <div><span style="color: #999;">Pred:</span> <span class="data-value predicted-val">${point.predicted.toFixed(2)}°C</span></div>
          <div><span style="color: #999;">Act:</span> <span class="data-value actual-val">${point.actual.toFixed(2)}°C</span></div>
          <div><span style="color: #999;">Err:</span> <span class="data-value error-val">${error.toFixed(2)}°C</span></div>
        `;
        list.appendChild(row);
      }
    }

    document.getElementById('nextBtn').addEventListener('click', () => {
      const xScale = chart.scales.x;
      const min = xScale.min;
      const max = xScale.max;
      const range = max - min;
      chart.scales.x.min = min + range * 0.25;
      chart.scales.x.max = max + range * 0.25;
      chart.update();
    });

    document.getElementById('prevBtn').addEventListener('click', () => {
      const xScale = chart.scales.x;
      const min = xScale.min;
      const max = xScale.max;
      const range = max - min;
      chart.scales.x.min = Math.max(0, min - range * 0.25);
      chart.scales.x.max = Math.max(range, max - range * 0.25);
      chart.update();
    });

    document.getElementById('resetZoomBtn').addEventListener('click', () => {
      chart.resetZoom();
    });

    document.getElementById('applyRangeBtn').addEventListener('click', () => {
      const min = parseFloat(document.getElementById('rangeMin').value) || 1;
      const max = parseFloat(document.getElementById('rangeMax').value) || 50;
      chart.scales.x.min = min;
      chart.scales.x.max = max;
      chart.update();
    });

    window.addEventListener('load', () => {
      initChart();
      connectToServer();
    });
  </script>
</body>
</html>

)rawliteral";

// ========== HELPERS ==========
void add_to_history(float temp) {
  temp_history[history_index] = temp;
  history_index = (history_index + 1) % N_STEPS;
}

void fill_input_from_history() {
  int idx = history_index;
  for (int i = 0; i < N_STEPS; i++) {
    float t = temp_history[idx];
    float t_scaled = (t - temp_min) / (temp_max - temp_min);
    t_scaled = constrain(t_scaled, 0.0f, 1.0f);
    input->data.f[i] = t_scaled;
    idx = (idx + 1) % N_STEPS;
  }
}

void store_data(float predicted, float actual) {
  predicted_temps[reading_index] = predicted;
  actual_temps[reading_index] = actual;
  timestamps[reading_index] = millis() / 1000UL;

  reading_index = (reading_index + 1) % MAX_READINGS;
  if (reading_count < MAX_READINGS) reading_count++;
}

// ========== WiFi & Server setup ==========
void connectWiFi() {
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);

  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
    if (millis() - start > 15000) {
      Serial.println("\nWiFi connect timeout, retrying...");
      start = millis();
    }
  }
  Serial.println("\n✓ WiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
}

void setupServer() {
  // Serve HTML page
  server.on("/", HTTP_GET, [](AsyncWebServerRequest *request){
    request->send(200, "text/html", index_html);
  });
 
  // EventSource for live updates
  events.onConnect([](AsyncEventSourceClient *client) {
    Serial.println("Client connected to /events");
  });

  server.addHandler(&events);
  server.begin();
  Serial.println("✓ HTTP server started");
}

// ========== SETUP ==========
void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("\n================================");
  Serial.println("ESP32-S3 Temperature Predictor");
  Serial.println("Live WebSocket Dashboard");
  Serial.println("================================");

  // DHT init
  dht.begin();
  delay(1000);

  float init_temp = dht.readTemperature();
  if (isnan(init_temp)) init_temp = 25.0f;
  for (int i = 0; i < N_STEPS; i++) temp_history[i] = init_temp;

  // Load TFLite model
  Serial.println("Loading model...");
  model = tflite::GetModel(model_tflite);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch!");
    while (1) delay(1000);
  }

  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed!");
    while (1) delay(1000);
  }

  input = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("✓ Model loaded!");

  // Connect WiFi + start server
  connectWiFi();
  setupServer();

  Serial.println("================================");
  Serial.println("Open browser: http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
  Serial.println("================================\n");

  delay(500);
}

// ========== MAIN LOOP ==========
void loop() {
  // Read sensor
  float temperature_now = dht.readTemperature();
  if (isnan(temperature_now)) {
    Serial.println("ERROR: Failed to read DHT22");
    delay(2000);
    return;
  }

  temperature_now = constrain(temperature_now, 0.0f, 60.0f);

  // Update history & fill input
  add_to_history(temperature_now);
  fill_input_from_history();

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("ERROR: Model invoke failed");
    delay(2000);
    return;
  }

  // Get prediction
  float pred_scaled = output->data.f[0];
  float temperature_pred = pred_scaled * (temp_max - temp_min) + temp_min;
  temperature_pred = constrain(temperature_pred, temp_min, temp_max);

  // Increment reading number
  reading_number++;

  // Store under lock
  portENTER_CRITICAL(&mux);
  store_data(temperature_pred, temperature_now);
  portEXIT_CRITICAL(&mux);

  // Print to serial (PREDICTED, ACTUAL format)
  Serial.print("Reading #");
  Serial.print(reading_number);
  Serial.print(" | Predicted: ");
  Serial.print(temperature_pred, 2);
  Serial.print("°C | Actual: ");
  Serial.print(temperature_now, 2);
  Serial.println("°C");

  // Send live update to all connected browsers
  String json = "{\"predicted\": " + String(temperature_pred, 2) + 
                ", \"actual\": " + String(temperature_now, 2) + 
                ", \"reading\": " + String(reading_number) + "}";
  events.send(json.c_str(), "message", millis());

  delay(2000); // sampling interval

}
