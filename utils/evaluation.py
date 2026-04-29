import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("experiments\\results.csv")

x = np.arange(len(df))  # positions
width = 0.35            # bar width

plt.figure()

plt.bar(x - width/2, df['minimax_win_pct'], width, label='Minimax')
plt.bar(x + width/2, df['mcts_win_pct'], width, label='MCTS')

plt.xlabel("Configuration (Depth vs Sims)")
plt.ylabel("Win Percentage")
plt.title("Minimax vs MCTS Win % Comparison")

plt.xticks(x, df['config'], rotation=30)
plt.legend()

plt.tight_layout()
# plt.show()
plt.savefig("win_percentage_comparison.png")

plt.figure()
plt.plot(df['minimax_depth'], df['avg_minimax_time'], marker='o', label='Minimax Time')
plt.plot(df['minimax_depth'], df['avg_mcts_time'], marker='o', label='MCTS Time')

plt.xlabel("Minimax Depth")
plt.ylabel("Time (seconds)")
plt.title("Computation Time")

plt.legend()
# plt.show()
plt.savefig("computation_time_comparison.png")