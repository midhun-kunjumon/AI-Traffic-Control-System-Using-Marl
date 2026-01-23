# AI-Based Traffic Signal Control using PPO (SUMO)

## Project Status
✔ Single-agent PPO implemented  
✔ Emergency vehicle priority (rule-based)  
⏳ Multi-Agent Reinforcement Learning (MARL) – in progress

## Tech Stack
- SUMO
- Python
- Stable-Baselines3 (PPO)
- TraCI

## How It Works
- Traffic junction simulated in SUMO
- PPO agent controls traffic signals
- Emergency vehicles override PPO decisions

## How to Run
1. Install SUMO
2. Install dependencies:
   pip install -r requirements.txt
3. Train PPO agent:
   python rl/train.py
4. Evaluate:
   python rl/evaluate.py

## MARL Work
MARL implementation will be added inside `/marl` folder.
