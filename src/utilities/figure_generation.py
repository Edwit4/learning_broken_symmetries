import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from abc import ABC, abstractmethod

POINT_SCALE = 150
LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 22
LEGEND_FONTSIZE = 12
TICK_FONTSIZE = 14

class Plotter(ABC):

    def __init__(self, title=None, xlabel=None, ylabel=None, xticks=None,
                 yticks=None, legend_col=1, legend=True, point_scale=POINT_SCALE,
                 label_fontsize=LABEL_FONTSIZE, title_fontsize=TITLE_FONTSIZE,
                 legend_fontsize=LEGEND_FONTSIZE, tick_fontsize=TICK_FONTSIZE):

        self.label_fontsize = label_fontsize
        self.title_fontsize = title_fontsize
        self.legend_fontsize = legend_fontsize
        self.tick_fontsize = tick_fontsize
        self.point_scale = point_scale
        self.legend_col = legend_col
        self.legend = legend
        if point_scale is None:
            point_scale = 1

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title, fontsize=title_fontsize)
        self.ax.set_xlabel(xlabel, fontsize=label_fontsize)
        self.ax.set_ylabel(ylabel, fontsize=label_fontsize)
        self.ax.tick_params(labelsize=tick_fontsize)
        if xticks is not None:
            self.ax.set_xticks(xticks)
        if yticks is not None:
            self.ax.set_yticks(yticks)

    @abstractmethod
    def make_plot(self, data, **kwargs):
        pass

    def save_plot(self, filepath, bbox_inches='tight', **kwargs):
        self.fig.savefig(filepath, bbox_inches=bbox_inches, **kwargs)

    def show_plot(self):
        self.fig.show()

class Hist2DPlotter(Plotter):

    def __init__(self, *args, clabel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.clabel = clabel

    def make_plot(self, data_xy, **kwargs):

        self.h,self.xedges,self.yedges,self.im = \
            self.ax.hist2d(data_xy[0], data_xy[1], **kwargs)

        if self.clabel is not None:
            self.cb = self.fig.colorbar(self.im)
            self.cb.set_label(self.clabel, fontsize=self.label_fontsize)
            self.cb.ax.tick_params(labelsize=self.tick_fontsize)

class ScatterPlotter(Plotter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_plot(self, data_dict, weight_dict=None, **kwargs):

        for key in data_dict:
            if weight_dict is not None:
                weights = weight_dict[key]*self.point_scale
            else:
                weights = None
            self.ax.scatter(data_dict[key][0], data_dict[key][1], label=key,
                            s=weights, **kwargs)

        if (len(data_dict) > 1) and self.legend:
            if self.legend_col is not None:
                self.ax.legend(fontsize=self.legend_fontsize,ncol=self.legend_col)
            else:
                self.ax.legend(fontsize=self.legend_fontsize)

class LinePlotter(Plotter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_plot(self, data_dict, error_dict=None, color_dict=None,
                  style_dict=None, legend_error=None,
                  **kwargs):

        if error_dict is None:
            error_dict = {key:None for key in data_dict.keys()}

        self.keys = list(data_dict.keys())
        legend_handles = []
        legend_labels = []
        for key in data_dict:
            if color_dict is not None:
                if color_dict[key] is not None:
                    color = color_dict[key]
            else:
                color = None
            if style_dict is not None:
                if style_dict[key] is not None:
                    style = style_dict[key]
            else:
                style = None
            self.line = self.ax.plot(data_dict[key][0], data_dict[key][1], label=key, 
                         color=color, linestyle=style, **kwargs)
            if error_dict[key] is not None:
                current_color = self.ax.get_lines()[-1].get_color()
                self.ax.fill_between(data_dict[key][0],
                                     np.asarray(data_dict[key][1])+np.asarray(error_dict[key]),
                                     np.asarray(data_dict[key][1])-np.asarray(error_dict[key]),
                                     alpha=0.25, color=current_color)
                self.fill = self.ax.fill(np.nan, np.nan, alpha=0.25,
                                         color=current_color)
                legend_handles.append((self.fill[0], self.line[0]))
                legend_labels.append(key)

        if (len(data_dict) > 1) and self.legend:
            if legend_error is not None:
                if self.legend_col is not None:
                    self.ax.legend(legend_handles, legend_labels, 
                                   fontsize=self.legend_fontsize,
                                   ncol=self.legend_col)
                else:
                    self.ax.legend(legend_handles, legend_labels,
                                   fontsize=self.legend_fontsize)
            else:
                if self.legend_col is not None:
                    self.ax.legend(fontsize=self.legend_fontsize, ncol=self.legend_col)
                else:
                    self.ax.legend(fontsize=self.legend_fontsize)

class ImPlotter(Plotter):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def make_plot(self, image, **kwargs):

        self.ax.imshow(image, **kwargs)
