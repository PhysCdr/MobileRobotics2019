import numpy as np
import timeit
import matplotlib.pyplot as plt

def gaussian(mu, sigma, x):
        return 1 / np.sqrt(2 * np.pi) / sigma * np.exp(-(x - mu)**2 / 2 / sigma ** 2)

def rand_norm1(mu, sigma):
        # average of 12
    samples = 2 * sigma * (np.random.rand(12) - 0.5)
    return np.sum(samples) / 2 + mu


def rand_norm2(mu, sigma):
        # rejection sampling
    f = 0
    fmax = 1 / np.sqrt(2 * np.pi) / sigma
    y = fmax
    while f <= y:
        x = 2 * 5*sigma * (np.random.rand() - 0.5) + mu
        y = np.random.rand() * fmax
        f = gaussian(mu, sigma, x)

    return x


def rand_norm3(mu, sigma):
    # Box-Muller
    u1 = np.random.rand()
    u2 = np.random.rand()

    return sigma * (np.cos(2 * np.pi * u1) * np.sqrt(-2 * np.log(u2))) + mu


if __name__ == '__main__':
    mu = 1
    sigma = 2
    num = 10000
    print(timeit.timeit(lambda: rand_norm1(mu, sigma), number=num) / num * 1e6, 'musec')
    print(timeit.timeit(lambda: rand_norm2(mu, sigma), number=num) / num * 1e6, 'musec')
    print(timeit.timeit(lambda: rand_norm3(mu, sigma), number=num) / num * 1e6, 'musec')
    print(timeit.timeit(lambda: sigma * np.random.randn() + mu, number=num) / num * 1e6, 'musec')

    s1 = [rand_norm1(mu, sigma) for i in range(num)]
    s2 = [rand_norm2(mu, sigma) for i in range(num)]
    s3 = [rand_norm3(mu, sigma) for i in range(num)]

    plt.hist(s1, bins=50, label='1', density=True)
    plt.hist(s2, bins=50, label='2', density=True)
    plt.hist(s3, bins=50, label='3', density=True)
    x = np.linspace(-5*sigma+mu, 5*sigma+mu, 100)
    plt.plot(x, gaussian(mu, sigma, x))
    plt.legend()
    plt.show()
