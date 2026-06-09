# AI-Powered Multi-Agent Reinforcement Learning Traffic Control System

This project is an advanced, production-grade, AI-driven traffic signal control system. It integrates computer vision (**YOLOv8**), Multi-Agent Reinforcement Learning (**MARL** with **PPO**), and physical edge computing (**ESP32-CAM** and **ESP32 Microcontroller**) to dynamically optimize traffic flow at a 4-way intersection, featuring emergency vehicle preemption.

---

## 🏗️ System Architecture

The project consists of three main components: **Perception** (vehicle detection from live HTTP video feeds), **Decision-Making** (reinforcement learning or emergency rule override), and **Actuation** (serial communication to the ESP32 physical traffic lights).

```mermaid
graph TD
    subgraph Perception (Edge Cameras)
        C_N[ESP32-CAM North] -->|HTTP Live Stream| TC[traffic_system.py / app.py]
        C_E[ESP32-CAM East] -->|HTTP Live Stream| TC
        C_S[ESP32-CAM South] -->|HTTP Live Stream| TC
        C_W[ESP32-CAM West] -->|HTTP Live Stream| TC
    end

    subgraph Core AI Controller (Python Host)
        TC -->|Captured Frames| YOLO[YOLOv8 Object Detector]
        YOLO -->|Vehicle counts, Queue length, Speed, Emergency detection| EM{Emergency Check}
        EM -->|🚨 Emergency Detected| EO[Emergency Preemption Override]
        EM -->|✅ No Emergency| PPO[Stable-Baselines3 PPO MARL Agent]
        
        EO -->|Force Priority Phase| SC[Serial Controller]
        PPO -->|Proportional Signal Green Timings| SC
    end

    subgraph Actuation (Physical Hardware)
        SC -->|Serial: 115200 Baud| ESP[ESP32 Traffic Controller]
        ESP -->|GPIO Pin Output| LED[4-Way Traffic LEDs]
        ESP -->|READY Handshake Signal| SC
    end

    subgraph User Dashboard
        TC -->|Real-time Metrics & Streams| Flask[Flask App: app.py]
        Flask -->|Socket / REST endpoints| Web[Live Web UI]
    end
```

---

## 🚀 Key Components & Features

### 1. Computer Vision Perception (`yolov8/`)
* **Custom YOLOv8 Model (`best1.pt`):** Custom-trained to detect and classify standard vehicles (cars, trucks, buses, motorcycles) and emergency vehicles (ambulances, fire trucks, police cars).
* **Multi-Camera Management (`camera_stream.py`):** Spawns concurrent background threads to grab frames via HTTP snapshot endpoints (`/capture`, `/jpg`) or MJPEG streams from 4 independent ESP32-CAM modules.
* **Vehicle Tracking & Metric Extraction (`yolo_detector.py`):** Employs YOLOv8 tracking (with persistent track IDs) to monitor vehicle movements. It calculates:
  * **Vehicle Count:** The number of active vehicles in each lane.
  * **Queue Length:** Vehicles static or moving below a velocity threshold.
  * **Average Speed Estimate:** Pixel distance shifts mapped to approximate real-world speed ($m/s$).
  * **Emergency Detection:** Real-time flagging of emergency vehicle presence.

### 2. Multi-Agent Reinforcement Learning (MARL) (`rl/`)
* **PPO MARL Model (`ppo_marl_v3.zip`):** Developed using Stable-Baselines3. The intersection operates as a Multi-Agent environment where each lane's signal behaves as an agent cooperating to minimize global intersection wait times.
* **Observation Space (22-Dimensional):**
  * Current active signal phase index (normalized).
  * Current phase elapsed duration (normalized, capped at 120s).
  * Queue lengths (normalized by 30) for North, East, South, West.
  * Vehicle counts (normalized by 40) for North, East, South, West.
  * Cumulative waiting times (normalized by 500) for North, East, South, West.
  * Vehicle densities (normalized by 0.15) for North, East, South, West.
  * Average speeds (normalized by 15.0) for North, East, South, West.
* **Action Space:** Continuous/discrete decisions to `KEEP` the current green phase active or `SWITCH` to the lane with the highest waiting queue.
* **Sumo Simulation Training (`sumo_marl_env_v3.py`):** Trained in **SUMO** (Simulation of Urban MObility) with TraCI over thousands of cycles to handle asymmetric traffic peaks, reward throughput, and penalize queue build-ups.

### 3. Emergency Preemption Override
* Features a rule-based safety override. If YOLOv8 detects an emergency vehicle in any lane, the system bypasses the PPO model's normal decision-making and immediately switches the signal to **GREEN** for that specific lane for 20 seconds, preventing delays.

### 4. ESP32 Microcontroller Actuator (`esp32_traffic_controller/`)
* Runs non-blocking firmware on an ESP32 microcontroller to drive physical traffic light LEDs (Red, Yellow, Green for North, East, South, West).
* **Bidirectional Serial Protocol (115200 Baud):**
  * **Incoming commands:** Single-character priority lane commands (e.g., `N\n`, `E\n`, `S\n`, `W\n`) or full sequential cycle timings (e.g., `N:15,E:20,S:10,W:15\n`).
  * **Outgoing status:** Responds with state updates (`GREEN: N`, `YELLOW: N`) and reports `READY` upon completing a sequence, signaling the Python system to capture the next set of camera feeds.

### 5. Flask Live Monitoring Dashboard (`app.py`)
* Web-based interface displaying the annotated camera frames with YOLOv8 bounding boxes.
* Shows current vehicle counts, calculated green times, active signal states, and logs of RL actions.
* Allows comparisons between fixed-time scheduling, Stable RL control, and RL with emergency preemption overrides in the SUMO simulation.

---

## 📂 Project Directory Structure

```
├── app.py                      # Flask web application dashboard
├── traffic_system.py           # Production integration pipeline (runs cam -> detection -> serial)
├── traffic_controller.py       # Core controller class linking YOLOv8, MARL, and cameras
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (configured for clean repo commits)
│
├── yolov8/                     # YOLOv8 assets and modules
│   ├── best1.pt                # Custom trained YOLOv8 model weights
│   ├── yolo_detector.py        # Vehicle detector with persistent object tracking
│   └── camera_stream.py        # Multithreaded ESP32-CAM HTTP/MJPEG stream manager
│
├── rl/                         # Reinforcement Learning modules
│   ├── models/                 # Saved models
│   │   └── ppo_marl_v3.zip     # Production cooperating PPO MARL model
│   ├── sumo_marl_env_v3.py     # Custom Gymnasium environment with SUMO and TraCI
│   ├── train_marl_v3.py        # MARL training pipeline
│   └── sim_runner.py           # Runs fixed-time vs RL simulation modes for evaluation
│
├── sumo/                       # SUMO simulation networks
│   └── stage8_asymmetric/      # Intersection network topology, routes, configurations
│
├── esp32_traffic_controller/
│   └── esp32_traffic_controller.ino  # Non-blocking firmware for ESP32 light actuation
│
├── templates/
│   └── index.html              # Frontend live dashboard layout
│
└── detection_results/          # Evaluation results
    ├── cycle216_N.jpg          # Sample detection output - North lane (216th cycle)
    ├── cycle216_E.jpg          # Sample detection output - East lane (216th cycle)
    ├── cycle216_S.jpg          # Sample detection output - South lane (216th cycle)
    └── cycle216_W.jpg          # Sample detection output - West lane (216th cycle)
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
1. **SUMO:** Download and install [SUMO](https://eclipse.dev/sumo/) on your system. Make sure you set the `SUMO_HOME` environment variable:
   * **Windows:** Add `C:\Program Files (x86)\Eclipse\Sumo` to your path and environment variables.
2. **Python:** Python 3.8 - 3.10 is recommended (for Stable-Baselines3 and PyTorch compatibility).

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. ESP32 Hardware Assembly
1. **Cameras:** Flash standard streaming/snapshot firmware on 4 ESP32-CAMs. Update their IP addresses in `yolov8/camera_stream.py` and `traffic_system.py`.
2. **Controller:** Open `esp32_traffic_controller/esp32_traffic_controller.ino` in Arduino IDE. Connect the LEDs to the mapped GPIO pins, flash the ESP32, and note its COM port (default: `COM7`).

---

## 🏃 Running the System

### 1. Flask Live Dashboard
To start the live web dashboard interface:
```bash
python app.py
```
Open `http://localhost:5000` in your web browser. You can trigger simulation comparisons or launch the live camera control mode.

### 2. Production Traffic Actuator (Command Line)
To run the automated loop connecting real cameras, model inference, and the serial ESP32 microcontroller directly:
```bash
python traffic_system.py --port COM7 --conf 0.3
```

### 3. Test Controller offline (Synthetic Mode)
To verify model loading, detection, and decision-making logic without physical hardware connected:
```bash
python traffic_controller.py --test
```
This runs the controller for 20 cycles using synthetic video frames and displays dynamic green allocation.
