import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.stats as stats

num_requests = 1000

time_responses_1_10 = np.random.uniform(1, 10, size=int(num_requests * 0.8))
time_responses_30_100 = np.random.uniform(30,100, size=int(num_requests* 0.2))

all_response= np.concatenate([time_responses_1_10, time_responses_30_100])

plt.hist(all_response,bins=30, edgecolor ='black', alpha = 0.7, color='skyblue')
plt.title('Rozkład czasu odpowiedzi na zapytania')
plt.xlabel('Czas odpowiedzi (sekundy)')
plt.ylabel('Liczba zapytań')
plt.grid(True)

n = 30

samples = [np.mean(random.choices(all_response, k=n)) for _ in range(num_requests)]

mean_sample = np.mean(samples)
std_sample = np.std(samples)

plt.figure()

plt.hist(samples, bins=30, edgecolor='black', alpha=0.7, color='salmon', density=True, label="Histogram średnich z próbek")

xmin, xmax = plt.xlim()  

x = np.linspace(xmin, xmax, 100)
y = stats.norm.pdf(x, mean_sample, std_sample)  

plt.plot(x, y, 'r-', label='Rozkład normalny')

plt.title('Histogram średnich z próbek oraz teoretyczny rozkład normalny')
plt.xlabel('Średnia z prób')
plt.ylabel('Częstotliwość')
plt.legend()

plt.show()

