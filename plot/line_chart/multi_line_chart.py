import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame with the provided data
data = {
    'PoG Factor': [0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001],
    'Face Factor': [ 5, 10, 15, 20, 5, 10, 15, 20, 5, 10, 15, 20],
    'PoG(pixel)': [ 138.07, 141.30, 126.35, 112.49, 134.51, 136.19, 139.58, 127.22, 130.85, 124.90, 121.99, 134.64]
}

df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 6))

# Group by 'PoG Factor' and plot
for pog_factor in df['PoG Factor'].unique():
    subset = df[df['PoG Factor'] == pog_factor]
    plt.plot(subset['Face Factor'], subset['PoG(pixel)'], marker='o', label=f'PoG Factor {pog_factor}')
    
    # Annotate each point with the value
    for i, row in subset.iterrows():
        plt.annotate(f"{row['PoG(pixel)']:.2f}", 
                        (row['Face Factor'], row['PoG(pixel)']), 
                        textcoords="offset points", 
                        xytext=(0, 5), 
                        ha='center')


plt.axhline(y=133.63, color='k', linestyle='--', label='Baseline (133.63 pixel)')

plt.title('PoG Factor vs Face Factor vs PoG(pixel)')
plt.xlabel('Face Factor')
plt.ylabel('PoG(pixel)')
plt.legend(title='PoG Factor')
# plt.grid(True)
plt.show()