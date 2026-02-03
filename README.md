# DOOM Reinforcement Learning Agent

An autonomous agent trained to survive, navigate, and eliminate enemies in the classic DOOM environment using Deep Reinforcement Learning (PPO).

https://github.com/user-attachments/assets/c0aa70db-6e78-4074-a433-bbd2da6f44ba


This project implements a Deep Reinforcement Learning agent capable of mastering complex Doom scenarios (Deadly Corridor, Defend the Center, Deathmatch).

Unlike basic implementations, this project uses Curriculum Learning to progressively train the agent on increasing difficulty levels and employs a Strategy Pattern for dynamic reward calculation, ensuring modularity and scalable code structure.

## Key Features

- **Proximal Policy Optimization (PPO):** Uses a Convolutional Neural Network (CNN) policy to interpret visual data from the game engine.

- **Curriculum Learning:** Automated training pipeline that increases scenario difficulty (stages) based on timesteps/performance, preventing the agent from getting stuck in local optima early on.

- **Computer Vision Processing:**

  - **Grayscale & Resize:** Downscales frames to 100x160 to optimize training speed.

  - **Frame Stacking:** Stacks 4 consecutive frames so the agent can perceive motion and velocity (not just static images).

- **Software Engineering Patterns:**

  - **Strategy Pattern for Rewards:** Reward functions are decoupled from the main environment loop. The agent dynamically selects the correct reward strategy (deadly_corridor, defend_center, etc.) based on the configuration loaded.

## Project Structure

    ├── models/                  # Saved models and checkpoints
    ├── logs/                    # Tensorboard logs for training metrics
    ├── deadlyCorridor.py        # MAIN SCRIPT: Training logic & Environment wrapper
    ├── demo_visualization.ipynb # Notebook for loading models and rendering video/GIFs
    ├── requirements.txt         # Dependencies
    └── README.md                # Documentation

## Installation

Clone the repository:

    git clone https://github.com/tu-usuario/Doom-RL-Agent.git
    cd Doom-RL-Agent

Install dependencies: 

    pip install -r requirements.txt

## Usage
### 1. Training the Agent

To start the training process with Curriculum Learning:

    python deadlyCorridor.py

The script will automatically iterate through difficulty stages defined in the stages list.
### 2. Watching the Agent Play

Open the Jupyter Notebook _demo_visualization.ipynb_ to load a trained model and watch it play in real-time or generate a video file.

## Reward Engineering

The agent's behavior is shaped by custom reward functions tailored to each scenario. Using a Strategy Pattern, the environment selects the appropriate function at runtime:

**Deadly Corridor:** Heavily penalizes damage taken, rewards forward movement (delta_x), and gives massive bonuses for kills.

**Defend the Center:** Focuses on ammo conservation and survival duration.

**Deathmatch:** Similarly to Deadly Corridor penalizes damage taken, rewards movement (delta_x. delta_z), and gives bonuses for kills.

### Code snippet example of the dynamic strategy selection
```
if 'deadly_corridor' in config:
    self.reward_function = self.deadly_corridor_reward
elif 'defend_the_center' in config:
    self.reward_function = self.defend_center_reward     
elif 'deathmatch' in config:
        self.reward_function = self.deathmatch_reward
```

##  Requirements

    Python 3.8+

    vizdoom

    stable-baselines3

    torch

    opencv-python

    gymnasium

## Author

Joan Guerrero S. - Software Developer & AI Specialist [LinkedIn](https://www.linkedin.com/in/jg-chez/)