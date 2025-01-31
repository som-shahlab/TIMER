
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

df = pd.read_csv('medalign/clinician-reviewed-model-responses.csv')

df['model_rank'] = pd.to_numeric(df['model_rank'], errors='coerce')

average_ranks = df.groupby('model_name')['model_rank'].mean().sort_values()

print("Average Model Ranks:")
print(average_ranks)

rank_stats = df.groupby('model_name')['model_rank'].agg(['mean', 'median', 'std', 'count'])
rank_stats = rank_stats.sort_values('mean')

print("\nDetailed Rank Statistics:")
print(rank_stats)

plt.figure(figsize=(12, 6))
average_ranks.plot(kind='bar')
plt.title('Average Model Ranks')
plt.xlabel('Model Name')
plt.ylabel('Average Rank')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))

figure_path = os.path.join(script_dir, 'average_model_ranks.png')

plt.savefig(figure_path, dpi=300, bbox_inches='tight')
print(f"\nFigure saved as: {figure_path}")

plt.close()
