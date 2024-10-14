import numpy as np
from src.utilities.figure_generation import LinePlotter

lo = -10
hi = 10
num_points = 1000
x = np.linspace(lo,hi,num_points)
relu = np.max([np.zeros(num_points),x], axis=0)
sigmoid = 1 / (1+np.exp(-x))
tanh = np.tanh(x)

plotter = LinePlotter(title='ReLU',
                      xlabel='x',
                      ylabel='y')
plotter.make_plot({'relu':[x,relu]})
plotter.save_plot('relu.pdf')

plotter = LinePlotter(title='Sigmoid',
                      xlabel='x',
                      ylabel='y')
plotter.make_plot({'sigmoid':[x,sigmoid]})
plotter.save_plot('sigmoid.pdf')

plotter = LinePlotter(title='Tanh',
                      xlabel='x',
                      ylabel='y')
plotter.make_plot({'tanh':[x,tanh]})
plotter.save_plot('tanh.pdf')

