#%% ========================================================================================

# TEST CELL

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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## _Dataframe init_

# df = web.DataReader('TSLA', 'yahoo', start, end)

# #create csv
# df.to_csv('tsla.csv') 

#read from csv
df = pd.read_csv('tsla.csv', parse_dates=True, index_col=0)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


# %% ========================================================================================

# read from the S&P 500
import datetime as dt
import os

import pandas as pd
import pandas_datareader.data as web

import bs4 as bs
import pickle #easily save list of companies
import requests

#Scrape data off wikipedia with Beautiful Soup
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml") #turns source code into a BeautifulSoup object 
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]: #each row, after the header row
        ticker = row.findAll('td')[0].text #first td becomes soup object
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:    #serializes Python objects
        pickle.dump(tickers, f) 

    print(tickers)
    return tickers

# save_sp500_tickers() # use only during first succesful run

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:    
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f: #read serialized objects
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'): #create directory to save dataset
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime.now()

    for ticker in tickers:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)): # just in case your connection breaks, we'd like to save our progress!
            df = web.DataReader(ticker, 'yahoo', start, end)
            # df.reset_index(inplace=True)
            # df.set_index("Date", inplace=True)
            # df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker)) #add each ticker to dataset directory
        else:
            print('Already have {}'.format(ticker))

get_data_from_yahoo()

# %% ========================================================================================
