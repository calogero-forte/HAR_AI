"""
This module contains utilities to plot several types of graphs.
"""

import math
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

########################################

def plot_xy_series(
    x_i: np.ndarray,
    y_i: np.ndarray,
    fs_i: float = 1.0,
    title_i: str = 'time_series',
    xlabel_i: str = 'x',
    ylabel_i: str = 'y'
) -> None:
    """
    Plot a X-Y graph. It can be uses to plot a time series of 
    frequency f Hz, sampled at fs Hz

    Parameters
    ----------
    x_i : array_like
        x-axis values
    y_i : array_like
        y-axis values
    fs_i : float
        sampling frequency in Hz. Default is 1.0 Hz
    title_i : str
        title of the plot. Default is 'time_series'
    xlabel_i : str
        label of the x-axis. Default is 'x'
    ylabel_i : str
        label of the y-axis. Default is 'y'

    Return
    -------
    None
    """
    import matplotlib.pyplot as plt

    # Create a figure of 6x6 inches and resolution 250 ppi 
    fig = plt.figure(figsize=[5.4, 3.8],facecolor='skyblue', edgecolor='black',dpi=300) 
    # Create an axes object that fullfils the figure (0, 0, 1, 1). Default projection is rectilier.
    ax = fig.add_axes([0,0,1,1],xlabel=xlabel_i,ylabel=ylabel_i,title=title_i)

    # y axes creation
    if(fs_i == 1.0):
        y_i = np.linspace( start=0, stop=x_i.shape[0], num=x_i.shape[0] * int( np.ceil( fs_i ) ) )
    #Plot
    ax.plot(x_i, y_i, color="blue")
    plt.show()

########################################

def plot_heatmap(
    data_i: np.ndarray,
    title_i: str = 'Heatmap',
    xlabel_i: str = 'x_label',
    ylabel_i: str = 'y_label',
    color_i: str = 'coolwarm',
    format_i: str = '.2f'
) -> None:
    """
    Plot an Heat-map of the matrix given as data_i input.

    Parameters
    ----------
    data_i : array_like
        data matrix to plot
    title_i : str
        title of the plot. Default is 'Heatmap'
    xlabel_i: str
        label of the x-axis. Default is 'x_label'
    ylabel_i: str
        label of the y-axis. Default is 'y_label'
    color_i : str
        color map to use. Default is 'coolwarm'
    format_i : str
        format of the values to plot. Default is '.2f'

    Return
    -------
    None
    """
    

    fig = plt.figure(figsize=[5.4, 3.8], dpi=300)
    ax = fig.add_axes([0,0,1,1])
    sns.heatmap(data_i, annot=True, cmap=color_i, fmt=format_i)
    plt.title(title_i)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.tight_layout()
    plt.show()

########################################

def plot_scatter(
  data_i: np.ndarray,
  classes_i: np.ndarray = None,
  labels_i: list[str] = None,
  marker_i: str = None,
  title_i: str = 'Scatter plot',
  xlabel_i: str = 'xlabel',
  ylabel_i: str = 'ylabel'
) -> None:
    """
    Plot the Scatter plot of the data_i.

    Parameters
    ----------
    data_i : array_like
        data matrix to plot. Data are arranged by rows,
        columns are the axis (x, y)
    classes_i: array_like, default=None
        list of classes to plot. Default is None.
        If classes is not None, labels shall be provided.
    labels_i : list[str]        labels of the plot. Default is None.
        If Labels is not None, a mask is appled to the rows of
        the data_i and only the rows of data_i and a scatter is plotted
        for each lablel (i.e. a class).
    marker_i : str
        marker to use. Default is None

    title_i : str
        title of the plot. Default is 'Scatter plot'
    xlabel_i : str
        label of the x-axis. Default is 'xlabel'
    ylabel_i : str
        label of the y-axis. Default is 'ylabel'
    Return
    ------
    None
    """

    plt.figure(figsize=[6, 4.5], dpi=300)
    plt.xlabel(xlabel_i)
    plt.ylabel(ylabel_i)
    plt.title(title_i)

    if(labels_i != None):

        for idx, l in enumerate(labels_i):
            plt.scatter(x=data_i[ idx == classes_i, 0], y=data_i[ idx == classes_i, 1], 
                marker=marker_i, label=l, alpha=0.6)
        plt.legend()

    else:
        plt.scatter(x=data_i[:, 0], y=data_i[:, 1], marker=marker_i)

    plt.tight_layout()
    plt.show()


########################################

def plot_scatter_3d(
  data_i: np.ndarray,
  classes_i: np.ndarray = None,
  labels_i: list[str] = None,
  marker_i: str = None,
  title_i: str = '3D Scatter plot',
  xlabel_i: str = 'xlabel',
  ylabel_i: str = 'ylabel',
  zlabel_i: str = 'zlabel'
) -> None:
    """
    Plot the 3D Scatter plot of the data_i.

    Parameters
    ----------
    data_i : array_like
        data matrix to plot. Data are arranged by rows,
        columns are the axis (x, y, z)
    classes_i: array_like, default=None
        list of classes to plot. Default is None.
        If classes is not None, labels shall be provided.
    labels_i : list[str]
        labels of the plot. Default is None.
        If Labels is not None, a mask is appled to the rows of
        the data_i and only the rows of data_i and a scatter is plotted
        for each lablel (i.e. a class).
    marker_i : str
        marker to use. Default is None

    title_i : str
        title of the plot. Default is '3D Scatter plot'
    xlabel_i : str
        label of the x-axis. Default is 'xlabel'
    ylabel_i : str
        label of the y-axis. Default is 'ylabel'
    zlabel_i : str
        label of the z-axis. Default is 'zlabel'
    Return
    ------
    None
    """

    fig = plt.figure(figsize=[6, 4.5], dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xlabel_i)
    ax.set_ylabel(ylabel_i)
    ax.set_zlabel(zlabel_i)
    ax.set_title(title_i)

    if(labels_i != None):

        for idx, l in enumerate(labels_i):
            ax.scatter(xs=data_i[ idx == classes_i, 0], ys=data_i[ idx == classes_i, 1], zs=data_i[ idx == classes_i, 2], 
                marker=marker_i, label=l, alpha=0.6)
        ax.legend()

    else:
        ax.scatter(xs=data_i[:, 0], ys=data_i[:, 1], zs=data_i[:, 2], marker=marker_i)

    plt.tight_layout()
    plt.show()

################################################################################

if __name__ == '__main__':

    f = 10
    fs = 1000
    t = np.linspace(0, 1, int( 1 * fs ) )
    y = np.sin(2 * np.pi * f * t)
    X = np.random.normal(loc=0, scale=1, size=(4, 4))

    plot_xy_series(t, y, fs, 'time_series', 'time', 'amplitude')
    
    plot_heatmap(X, title_i='Normal')

    plot_scatter(X)

    X_3d = np.random.normal(loc=0, scale=1, size=(4, 3))
    plot_scatter_3d(X_3d)


    
    