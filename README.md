# AI-Powered Traffic Control System

This project is a comprehensive AI-driven traffic signal control system that uses computer vision, edge computing, and Multi-Agent Reinforcement Learning (MARL) to optimize traffic flow and prioritize emergency vehicles in real-time.

## Key Features
- **Real-Time Vehicle Detection:** Utilizes a custom-trained **YOLOv8** model to accurately detect and classify cars and emergency vehicles from live camera feeds.
- **Hardware Integration:** Integrates with four **ESP32** cameras (North, East, South, West) streaming live video via HTTP.
- **Intelligent Traffic Control:** Employs a **Multi-Agent Reinforcement Learning (MARL)** approach using Stable-Baselines3 (PPO) to dynamically calculate optimal traffic signal timings based on real-time vehicle counts and queue lengths.
- **Emergency Vehicle Preemption:** Features a rule-based override system to immediately grant green lights to detected emergency vehicles, minimizing response times.
- **Hardware Traffic Controller:** Communicates with an ESP32 microcontroller via Serial to actuate physical traffic lights based on the AI's decisions.
- **Live Web Dashboard:** Includes a **Flask-based web application** (`app.py`) for live monitoring, displaying real-time video feeds with bounding boxes, current queue lengths, and active traffic light states.
- **Simulation Environment:** Uses **SUMO** (Simulation of Urban MObility) and TraCI for training the reinforcement learning agents in a highly realistic virtual environment before real-world deployment.

## Tech Stack
- **AI/ML:** Python, YOLOv8 (Ultralytics), Stable-Baselines3 (PPO), PyTorch
- **Hardware:** ESP32-CAM, ESP32 Microcontroller, Serial Communication
- **Simulation:** SUMO, TraCI
- **Web App:** Flask, HTML, CSS, JavaScript
- **Data Processing:** OpenCV, NumPy

## How It Works
1. **Perception:** Four ESP32 cameras capture live traffic footage and stream it to the main controller.
2. **Detection:** The YOLOv8 model processes the frames to detect cars and emergency vehicles, outputting bounding boxes and confidence scores.
3. **Decision:** The MARL agent receives the current queue lengths and vehicle counts as state inputs and predicts the optimal green light durations for each lane.
4. **Action:** The system sends commands via Serial to the ESP32 traffic controller, which physically changes the traffic lights.
5. **Monitoring:** The entire process can be monitored in real-time via the local Flask web dashboard.

## Installation & Setup
1. **Install SUMO:** Download and install SUMO for your OS.
2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Hardware Setup:** 
   - Flash the ESP32 cameras with standard streaming firmware.
   - Flash the main ESP32 traffic controller with the code in `esp32_traffic_controller/`.
4. **Run the System:**
   - To start the web server and live detection:
     ```bash
     python app.py
     ```
   - To run the main traffic controller logic directly:
     ```bash
     python traffic_controller.py
     ```

## Project Structure
- `app.py`: Flask web application for the live dashboard.
- `traffic_controller.py`: Core logic integrating YOLOv8, MARL, and hardware communication.
- `rl/`: Reinforcement learning training scripts, environments (`sumo_marl_env.py`), and saved models.
- `sumo/`: SUMO network files, routes, and configuration for training.
- `yolov8/`: Custom trained YOLOv8 model weights (`best.pt`).
- `esp32_traffic_controller/`: Firmware for the physical traffic light microcontroller.
- `detection_results/`: Sample images showing the model's detection capabilities.
