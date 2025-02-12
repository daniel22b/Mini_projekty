import math
import matplotlib.pyplot as plt
SQRT_TWO_PI = math.sqrt(2 * math.pi)

# def normal_pdf(x: float, mi: float=0, sigma: float = 1)-> float:
#     return math.exp(-(x-mi) ** 2 / 2 / sigma **2) / (SQRT_TWO_PI * sigma)

xs = [x /10 for x in range(-50,50)]

# plt.plot(xs,[normal_pdf(x, sigma=1) for x in xs],'-',label='mi=0, sigma=1')
# plt.plot(xs,[normal_pdf(x, sigma=2) for x in xs],'--',label='mi=0, sigma=2')
# plt.plot(xs,[normal_pdf(x, sigma=0.5) for x in xs],':',label='mi=0, sigma=0.5')
# plt.plot(xs,[normal_pdf(x, mi= -1) for x in xs],'-.',label='mi= -1, sigma=1')

# plt.legend()
# plt.title("WKyres roznych rozkladow normalnych")
# plt.show() 

# #------------------------------------------------------------------------------

def normal_cdf(x: float, mi: float=0, sigma: float = 1)-> float:
    return (1 + math.erf((x - mi)/math.sqrt(2)/ sigma)) /2

# plt.plot(xs,[normal_cdf(x, sigma=1) for x in xs],'-',label='mi=0, sigma=1')
# plt.plot(xs,[normal_cdf(x, sigma=2) for x in xs],'--',label='mi=0, sigma=2')
# plt.plot(xs,[normal_cdf(x, sigma=0.5) for x in xs],':',label='mi=0, sigma=0.5')
# plt.plot(xs,[normal_cdf(x, mi= -1) for x in xs],'-.',label='mi= -1, sigma=1')

# plt.legend(loc = 4)
# plt.title("Dystrybuanty roznych rozkladow normalnych")
# # plt.show()

# # #------------------------------------------------------------------------------

# def inverse_normal_cdf(p: float,
#                        mi: float = 0,
#                        sigma: float = 1,
#                        tolerance: float = 0.00001) -> float:
#     if mi != 0 or sigma != 1:
#         return mi + sigma * inverse_normal_cdf(p, tolerance=tolerance)

#     low_z = -10.0
#     hi_z = 10.0
#     while hi_z - low_z > tolerance:
#         mid_z = (low_z + hi_z) /2
#         mid_p = normal_cdf(mid_z)
#         if mid_p < p:
#             low_z = mid_z
#         else:
#             hi_z = mid_z
        
#     return mid_z


# import scipy.stats as stats
# def z_score(x: float, mi:float, sigma:float)-> float:
#     return (x - mi)/ sigma

# p = stats.norm.cdf(z_score(185, 190, 5))

# print(p)

import numpy as np
import matplotlib.pyplot as plt
# import scipy.stats as stats

# mu = 20
# sigma = 3

# # Generowanie punktów x
# x_values = np.arange(mu - 4*sigma, mu + 4*sigma, sigma)
# plt.xticks(x_values)
# normal_dist = stats.norm(mu, sigma)

# cdf_dist = normal_dist.cdf(x_values)


# # Rysowanie punktów
# plt.plot(x_values,cdf_dist, "r--", label=' mu= 20 , sigma= 3')
# plt.title('Punkty generowane przez np.linspace')
# plt.xlabel('x')
# plt.ylabel('y = 0')
# plt.grid(True)
# plt.legend()
# plt.show()

pop_mean = 5
pop_std = 3

n = 50
num_samples = 1000

samples = np.random.uniform(0,10,size=(num_samples,n))

sample_mean = np.mean(samples, axis=1)\

# plt.hist(sample_mean, bins=30, density=True, alpha=0.7, color='g')
# plt.title('Rozkład średnich próbki (n=50)')
# plt.xlabel('Średnia próbki')
# plt.ylabel('Prawdopodobieństwo')
# plt.grid(True)
# plt.show()
import random
def bernoulli_trial(p: float) -> int:
    return 1 if random.random() < p else 0

def binomial(n: int, p:float) -> int:
    return sum(bernoulli_trial(p)for _ in range(n))

from collections import Counter

def make_hist(p: float, n:int, num_points: int)-> None:
    data = [binomial(n,p) for _ in range(num_points)]

    histogram = Counter(data)
    plt.bar([x- 0.3 for x in histogram.keys()],
            [v/num_points for v in histogram.values()], 
            0.9,
            color='0.75')
    
    mi = p * n
    sigma = math.sqrt(n * p * (1 - p))

    xs = range(min(data), max(data) + 1)
    ys = [normal_cdf(i+0.5 ,mi, sigma) - normal_cdf(i-0.5, mi, sigma)
        for i in xs]

    plt.plot(xs,ys)
    plt.title("Rozklad dwumianu a przyblizenie rozkladu normalnego")
    plt.show()

make_hist(0.75, 100, 100000)


