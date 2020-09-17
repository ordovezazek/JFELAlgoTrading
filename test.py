import datetime as dt
import matplotlib.pyplot as pyplot
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web

style.use('ggplot')

# period
start = dt.datetime(2000,1,1)
end = dt.datetime(2016,12,31)

# dataframe init
df = web.DataReader('TSLA', 'yahoo', start, end)

# print(df.head())
# print(df.tail())
