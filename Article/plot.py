import matplotlib.pyplot as plt
import numpy as np

# Data from the table
emotions = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
BU3DFE = [400, 400, 400, 400, 100, 400, 400]
CalD3R_MenD3s = [881, 1101, 554, 1197, 3422, 1186, 375]
Bosphorus = [116, 145, 101, 182, 98, 102, 158]
Global = [1397, 1646, 1055, 1779, 3620, 1688, 933]

# Plot for BU3DFE, CalD3R&MenD3s, and Bosphorus
fig, ax = plt.subplots(figsize=(10, 6))
width = 0.2  # width of the bars
x = np.arange(len(emotions))  # label locations

# Plot bars
bars1 = ax.bar(x - width, BU3DFE, width, label='BU3DFE')
bars2 = ax.bar(x, CalD3R_MenD3s, width, label='CalD3R&MenD3s')
bars3 = ax.bar(x + width, Bosphorus, width, label='Bosphorus')

# Add numbers on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)
for bar in bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)
for bar in bars3:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)

# Zoom in by adjusting y-axis limit
ax.set_ylim(0, max(BU3DFE + CalD3R_MenD3s + Bosphorus) + 100)

# Labels and title
#ax.set_xlabel('Emotions')
#ax.set_ylabel('Count')
#ax.set_title('Comparison of BU3DFE, CalD3R&MenD3s, and Bosphorus datasets')
ax.set_xticks(x)
ax.set_xticklabels(emotions)
ax.legend(fontsize='small')

plt.tight_layout()
plt.show()

# Plot for Global dataset
fig, ax = plt.subplots(figsize=(10, 6))

# Plot Global bar chart
bars_global = ax.bar(emotions, Global, color='purple')

# Add numbers on top of bars for Global dataset
for bar in bars_global:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, int(yval), va='bottom', ha='center', fontsize=10)

# Labels and title
#ax.set_xlabel('Emotions')
#ax.set_ylabel('Count')
#ax.set_title('Global dataset')

plt.tight_layout()
plt.show()
