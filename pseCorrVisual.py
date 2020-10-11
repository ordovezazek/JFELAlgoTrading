
# PSEI EXPERIMENT

from fastquant import get_pse_data

import pandas as pd
import pandas_datareader.data as web

import matplotlib.pyplot as plt
from matplotlib import style

import datetime as dt
import os
import numpy as np
import bs4 as bs
import pickle #easily save list of companies
import requests

import traceback

style.use('ggplot')

def visualize_data():
    df = pd.read_csv('READ_WRITE/psei_joined_closes.csv')

    #data manip
    df_corr = df.corr() #determine the correlation of every column with every other column
    df_corr.to_csv('READ_WRITE/PSEIcorr.csv') # save to local csv

    #viz
    data1 = df_corr.values  #get numpy array of just the values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111) #1x1-plot1

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)    #create the heatmap
    fig1.colorbar(heatmap1)                             #side-bar for color scale
    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)    #set ticks for company recognition
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)

    ax1.invert_yaxis()                                  #remove top gap and place xaxis at the top for readability
    ax1.xaxis.tick_top()

    column_labels = df_corr.columns                     # set labels
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)                  
    ax1.set_yticklabels(row_labels)

    plt.xticks(rotation=90)                             # rotate ticks for readability
    heatmap1.set_clim(-1,1)                             # set color limit in the -1:1 range for heatmap
    plt.tight_layout()                                  # clean up

    plt.savefig("Visualizations/PSEI_correlations.png", dpi = (300))       # save to png
    plt.show()

    # df['MMM'].plot()
    # plt.show()

visualize_data()