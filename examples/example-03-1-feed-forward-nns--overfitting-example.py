#!/usr/bin/python3
"""
-- Feed-Forward Neural Networks
---- A Brief Digression: Overfitting
------ A Practical Example of Overfitting
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.optimize import curve_fit

import importlib
set_style = importlib.import_module("ADL-Book-2nd-Ed.modules.style_setting").set_style


def poly_func_2(t, p_0, p_1, p_2):
    return p_0 + p_1 * t + p_2 * (t ** 2)


# Generate the dataset
x = np.arange(-5.0, 5.0, 0.05, dtype=np.float64)
y = poly_func_2(x, 1, 2, 3) + 10.0 * np.random.normal(0, 1, size=len(x))

fp = set_style().set_general_style_parameters()
plt.figure()
plt.scatter(x, y, color='blue', linewidths=3)
plt.ylabel('y', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('x', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(-50, 110)
plt.xlim(-6, 6)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-1-1.svg', bbox_inches='tight')


def poly_func_1(t, p_0, p_1):
    return p_0 + p_1 * t


popt, pcov = curve_fit(poly_func_1, x, y)
plt.figure()
plt.scatter(x, y, color='blue', linewidths=3)
plt.plot(x, poly_func_1(x, *popt), lw=3, color='red')
plt.ylabel('y', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('x', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(-50, 110)
plt.xlim(-6, 6)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-1-2.svg', bbox_inches='tight')

popt, pcov = curve_fit(poly_func_2, x, y)
print("popt", popt)  # popt [1.94410716 1.96932697 2.89079599]
plt.figure()
plt.scatter(x, y, color='blue', linewidths=3)
plt.plot(x, poly_func_2(x, *popt), lw=3, color='red')
plt.ylabel('y', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('x', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(-50, 110)
plt.xlim(-6, 6)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-1-3.svg', bbox_inches='tight')


def poly_func_11(t, p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7, p_8, p_9, p_10, p_11):
    return p_0 + p_1 * t + p_2 * t**2 + p_3 * t**3 + p_4 * t**4 + p_5 * t**5 + p_6 * t**6 + p_7 * t**7 + p_8 * t**8 + p_9 * t**9 + p_10 * t**10 + p_11 * t**11


popt, pcov = curve_fit(poly_func_11, x, y)
plt.figure()
plt.scatter(x, y, color='blue', linewidths=3)
plt.plot(x, poly_func_11(x, *popt), lw=3, color='red')
plt.ylabel('y', fontproperties=fm.FontProperties(fname=fp))
plt.xlabel('x', fontproperties=fm.FontProperties(fname=fp))
plt.ylim(-50, 110)
plt.xlim(-6, 6)
plt.axis(True)
# plt.show()
plt.savefig('../figures/figure-03-1-4.svg', bbox_inches='tight')


