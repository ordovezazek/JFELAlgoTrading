import datetime as dt

import pandas as pd
import pandas_datareader.data as web

from mplfinance.original_flavor import candlestick_ohlc

import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates

style.use('ggplot')

# time period
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)

#===================================================================================
## _Dataframe init_

# df = web.DataReader('TSLA', 'yahoo', start, end)

# #create csv
# df.to_csv('tsla.csv') 

#read from csv
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

#===================================================================================
## _Algo_

#100ma:
#df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean() #rolling (moving average)

df_ohlc = df['Adj Close'].resample('10D').ohlc() # resample period (every 10 days)
df_volume = df['Volume'].resample('10D').sum()

#df.dropna(inplace=True) #ignore NaN
df_ohlc.reset_index(inplace=True) #reset index

df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num) #convert date->mdates

#test:
#print(df_ohlc.head())
#print(df.tail())

#===================================================================================
## _Visualization_

ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1) #create subplot

ax2 = plt.subplot2grid((6,1), (5,0), rowspan=5, colspan=1, sharex=ax1)

#plot & show:
#df.plot() #general

#df['Adj Close'].plot() #specific

#ax1.plot(df.index, df['Adj Close']) #specialized
#ax1.plot(df.index, df['100ma'])
#ax2.bar(df.index, df['Volume'])

ax1.xaxis_date() #converts the axis from the raw mdate numbers to dates
candlestick_ohlc(ax1, df_ohlc.values, width=5, colorup='g') #graph candlestick graph
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values, 0) #graph an unobstructed volume graph

plt.show()
