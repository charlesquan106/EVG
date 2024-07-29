import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



#### Face Factor ####
# factors = [ 5, 10, 15, 20, 30, 50]
# pixels = [ 139.68, 119.71, 123.08, 121.86, 131.36, 133.37]



#### PoG Factor ####
factors = [0.001, 0.01, 0.1]
pixels = [128.28, 137.45, 129.12]

plt.figure(figsize=(10, 6))
plt.plot(factors, pixels, marker='o', linestyle='-', color='k', label='PoG(pixel)')

plt.axhline(y=133.63, color='k', linestyle='--', label='Baseline (133.63 pixel)')



#### Face Factor ####
# plt.xticks(range(0, 60, 5))


#### PoG Factor ####
plt.xticks([0.001, 0.01, 0.1])
plt.xscale('log')


# Use ScalarFormatter to avoid scientific notation
formatter = ticker.ScalarFormatter()
formatter.set_scientific(False)
formatter.set_useOffset(False)
plt.gca().xaxis.set_major_formatter(formatter)



for i, txt in enumerate(pixels):
    plt.annotate(f'{txt}', (factors[i], pixels[i]), textcoords="offset points", xytext=(0,10), ha='center')


plt.title('PoG Factor vs PoG(pixel)')

#### Face Factor ####
# plt.xlabel('Face Factor')

#### PoG Factor ####
plt.xlabel('PoG Factor (log scale)')


plt.ylabel('PoG(pixel)')
plt.legend()



plt.show()