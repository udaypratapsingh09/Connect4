# 🎮 Connect Four AI — Minimax vs MCTS

## 📌 Project Overview

This project implements a fully playable **Connect Four** game with two fundamentally different AI approaches:

* **Minimax with Alpha-Beta Pruning**
* **Monte Carlo Tree Search (MCTS)**

The goal is to perform a **comparative analysis of deterministic vs probabilistic decision-making** in game AI.

* Built entirely in **Python**
* UI implemented using **Pygame**
* No web stack, no JavaScript, no backend

---

## ⚖️ Algorithm Comparison

| Feature   | Minimax + Alpha-Beta | MCTS                           |
| --------- | -------------------- | ------------------------------ |
| Nature    | Deterministic        | Probabilistic                  |
| Strategy  | Game tree search     | Simulation-based               |
| Tuning    | Depth                | Number of simulations          |
| Heuristic | Required             | Not required                   |
| Scaling   | Limited by depth     | Improves with more simulations |

---

## 🎮 Game Modes

* AI vs AI *(Primary experimental mode)*
* Human vs Minimax
* Human vs MCTS

---

## 🧠 Why Pygame?

Using **Pygame** keeps everything in Python:

* No HTTP calls
* No JSON serialization
* No frontend/backend split

This makes the project cleaner and focused on AI rather than web development.

---

## 🚫 No Training Required

This project does **not use machine learning**:

* Minimax evaluates the game tree in real-time
* MCTS performs simulations at decision time

---

## 🏗️ Project Architecture

```
game/
  ├── board.py
  ├── rules.py

ai/
  ├── minimax.py
  ├── mcts.py
  ├── node.py

utils/
  ├── evaluation.py

interface/
  ├── pygame_ui.py

experiments/
  ├── run_tests.py
  ├── results.csv

main.py
requirements.txt
```

---

## ⚙️ Setup & Installation

```bash
pip install -r requirements.txt
```

### Run Game

```bash
python main.py
```

### Run Experiments (Headless)

```bash
python experiments/run_tests.py
```

---

## 🔬 Experiment Design

Each configuration runs **50+ games**:

| Minimax Depth | MCTS Simulations |
| ------------- | ---------------- |
| 3             | 200              |
| 4             | 500              |
| 5             | 1000             |
| 6             | 2000             |

### Metrics Collected

* Win counts (Minimax / MCTS)
* Draws
* Average move time

---

## 📊 Results & Analysis

### Key Observations

* Minimax dominates at **low depths**
* MCTS improves with **more simulations**
* A **crossover point** exists where MCTS outperforms Minimax
* Minimax hits a **practical ceiling (~depth 5–6)**

### Expected Conclusion

* MCTS (200 sims) ≈ Minimax (depth 3–4)
* MCTS (2000 sims) > Minimax (depth 6)

---

## 📈 Graphs Generated

* Win rate comparison (bar chart)
* Move time vs depth/simulations
* Strength vs time tradeoff

---

## 🧠 Key Concepts

* Deterministic vs probabilistic decision-making
* Heuristic evaluation limitations
* UCT (Upper Confidence Bound for Trees)
* Exploration vs exploitation
* Branching factor constraints

---

## ⚠️ Important Notes

* Run **at least 50 games per configuration**
* MCTS requires sufficient simulations to stabilize
* Experiment runner is **critical for grading**

---

## 📦 Requirements

* numpy
* pygame
* matplotlib
* pandas

---

## 🚀 Final Notes

This project focuses on **algorithmic comparison**, not UI complexity.

The **experiment results and analysis** are the most important part of the submission.
