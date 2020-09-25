#%% ========================================================================================

# TEST & REF CELL

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
# df.to_csv('READ_WRITE/tsla.csv') 

#read from csv
df = pd.read_csv('READ_WRITE/tsla.csv', parse_dates=True, index_col=0)

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

# S&P 500 Data Collection ~ Correlation Visualization

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

style.use('ggplot')

# Scrape data off wikipedia with Beautiful Soup
def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml") #turns source code into a BeautifulSoup object 
    table = soup.find('table', {'class':'wikitable sortable'})

    tickers = []
    
    for row in table.findAll('tr')[1:]: #each row, after the header row
        ticker = row.findAll('td')[0].text #first td becomes soup object
        tickers.append(ticker.strip())  #remove "\n"

    with open("sp500tickers.pickle", "wb") as f:    #serializes Python objects
        pickle.dump(tickers, f) 

    # print(tickers)
    return tickers

# Get full data from Yahoo based on saved pickle 
def get_data_from_yahoo(reload_sp500=False): #change condition if data collection was edited
    if reload_sp500:    
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f: #read serialized objects
            tickers = pickle.load(f)

    if not os.path.exists('stock_dfs'): #check for directory to save dataset 
        os.makedirs('stock_dfs')

    start = dt.datetime(2000,1,1)
    end = dt.datetime.now()

    for ticker in tickers[0:19]:
        try:
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)): # just in case your connection breaks, we'd like to save our progress!
                df = web.DataReader(ticker, 'yahoo', start, end)
                # df.reset_index(inplace=True)
                # df.set_index("Date", inplace=True)
                # df = df.drop("Symbol", axis=1)
                df.to_csv('READ_WRITE/stock_dfs/{}.csv'.format(ticker)) #add each ticker to dataset directory
            else:
                print('Already have {}'.format(ticker))

            print(ticker + ' added')

        except:
            print(ticker + ' skipped')
            continue

# Add data into an organized data frame
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)[0:19]

    main_df = pd.DataFrame()    #initialize an empty dataframe

    for count,ticker in enumerate(tickers):
        df = pd.read_csv('READ_WRITE/stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        # df['{}_HL_pct_diff'.format(ticker)] = (df['High'] - df['Low']) / df['Low']    #% diff of high low
        # df['{}_daily_pct_chng'.format(ticker)] = (df['Close'] - df['Open']) / df['Open']  #daily % change

        df.rename(columns={'Adj Close':ticker}, inplace=True)   #rename Adj Close column to ticker name
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)  #drop not needed columns

        if main_df.empty:   #use current df if main_df is still empty
            main_df = df
        else:              #otherwise join dfs
            main_df = main_df.join(df, how='outer')

        if count % 10 == 0: #tracker
            print(count)

    # print(main_df.tail())
    main_df.to_csv('READ_WRITE/sp500_joined_closes.csv')

# Data Manipulation & Visualization
def visualize_data():
    df = pd.read_csv('READ_WRITE/sp500_joined_closes.csv')

    #data manip
    df_corr = df.corr() #determine the correlation of every column with every other column
    df_corr.to_csv('READ_WRITE/sp500corr.csv') # save to local csv

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

    plt.savefig("Visualizations/s&p500_correlations.png", dpi = (300))       # save to png
    plt.show()

    # df['MMM'].plot()
    # plt.show()

# exe:
# save_sp500_tickers()
# get_data_from_yahoo()
# compile_data()
visualize_data()

# TO DO NOTES:
# 1.) get full correlation instead of just 20 companies

# %% ========================================================================================

# S&P 500 Machine Learning Experiment

from statistics import mean

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

from collections import Counter # show the distributions of classes both in the dataset and in the algorithm's predictions

from sklearn import svm, neighbors    #sklearn: machine learning framework - [0]: Support Vector Machine - [1]: K Nearest Neighbors
# from sklearn.model_selection import cross_validate  # [0]: create shuffled training and testing samples 
from sklearn.model_selection import train_test_split # [0]: create shuffled training and testing samples 
from sklearn.ensemble import VotingClassifier, RandomForestClassifier # [0]: give classifiers the ability to vote - [1]: classifier

# Pre-process data for ML
def process_data_for_labels(ticker):
    hm_days = 7 #days into the future
    df = pd.read_csv('READ_WRITE/sp500_joined_closes.csv', index_col=0)    #read in the data for the close prices for all companies
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)    #replace Na with 0

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]   #.shift will shift a column up or down

    df.fillna(0, inplace=True)
    return tickers, df

# Crate FEATURESET & LABELS for machine learning algorithm:

# Business Rules:
# if the price rises more than 2% in the next 7 days, we're going to say that's a buy. 
# If it drops more than 2% in the next 7 days, that's a sell. If it doesn't do either of those, 
# then it's not moving enough, and we're going to just hold whatever our position is. 
# If we have shares in that company, we do nothing, we keep our position.

# ~ LABELS
def buy_sell_hold(*args):   # needed to map to dataframe column | using args to take any number of columns 
    cols = [c for c in args]    # args will be those future price change columns
    requirement = 0.02
    for col in cols:        # A -1 as sell, 0 as hold, and 1 as a buy
        if col > requirement:
            return 1
        elif col < requirement:
            return -1
    return 0    
# Caveat: price might go up 2%, then suddenly fall 2% vice versa etc. which algo does not account for

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)   #process data

    # create "target" column ~ LABELS
    df['{}_target'.format(ticker)] = list(map( buy_sell_hold, 
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)] ))    # optimize later on

    # get the distribution
    vals = df['{}_target'.format(ticker)].values.tolist()   
    str_vals = [str(i) for i in vals]
    print('Data spread:',Counter(str_vals))

    # Clean up data
    df.fillna(0, inplace=True)  #replace totally missing data with 0
    df = df.replace([np.inf, -np.inf], np.nan)  # convert infinite values to NaNs and drop
    df.dropna(inplace=True)

    # Get FEATURES : % change of every company for that day
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values                          # FEATURESETS
    y = df['{}_target'.format(ticker)].values   # LABELS

    return X, y, df

# Machine Learning
def do_ml(ticker):
    X, y ,df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)  # shuffle data and create training and testing samples
    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)   #take Xdata:featuresets and fit to ydata:labels
    confidence = clf.score(X_test, y_test)  # get % accuracy (1.0 is 100% and 0.1 is 10% accurate)

    print('accuracy:',confidence)
    predictions = clf.predict(X_test)   # X_test data prediction
    print('predicted class counts:',Counter(predictions))   # output distribution
    
    return confidence
    
# process_data_for_labels('MMM')
# extract_featuresets('MMM')
do_ml('ABT')

#run against all tickers
# with open("sp500tickers.pickle","rb") as f:
#     tickers = pickle.load(f)

# accuracies = []
# for count,ticker in enumerate(tickers):

#     if count%10==0:
#         print(count)

#     accuracy = do_ml(ticker)
#     accuracies.append(accuracy)
#     print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))


# %%



