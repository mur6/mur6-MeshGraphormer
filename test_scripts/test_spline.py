import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline, make_interp_spline


def cheb_nodes(N):
    jj = 2.*np.arange(N) + 1
    x = np.cos(np.pi * jj / 2 / N)[::-1]
    return x

def main():
    x = cheb_nodes(20)
    y = np.sqrt(1 - x**2)
    print(x)
    print(y)
    b = make_interp_spline(x, y)

    # np.allclose(b(x), y)
    l, r = [(2, 0.0)], [(2, 0.0)]
    b_n = make_interp_spline(x, y, bc_type=(l, r))  # or, bc_type="natural"
    # np.allclose(b_n(x), y)

    #x0, x1 = x[0], x[-1]
    #np.allclose([b_n(x0, 2), b_n(x1, 2)], [0, 0])

    phi = np.linspace(0, 2.*np.pi, 40)
    print(f"phi: {phi}")
    r = 0.3 + np.cos(phi)
    x, y = r*np.cos(phi), r*np.sin(phi)

    spl = make_interp_spline(phi, np.c_[x, y])

    phi_new = np.linspace(0, 2.*np.pi, 100)
    x_new, y_new = spl(phi_new).T

    plt.plot(x, y, 'o')
    plt.plot(x_new, y_new, '-')
    plt.show()

main()



