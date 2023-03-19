import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
import time


import geopandas as gpd
import folium

import streamlit as st
from streamlit_folium import st_folium

import yfinance as yf


st.title("SEC visualisation")

path = '/Users/matthesfogtmann/Downloads/SEC data/'

def getDir(path):
    lst = []
    for i in os.listdir(path):
        if len(i)==6 and "q" in i:
            lst.append(i)
    return lst

def getFinacialData(filename=path+'2022q4'):
    data = {i:pd.read_csv(f"{filename}{i}.txt",sep="\t") for i in ["tag","num","pre","sub"]}
    
    #drop things
    data["tag"].drop(columns=["version","custom","abstract","datatype","iord","crdr"],inplace=True)
    data["num"].drop(columns=["footnote"],inplace=True)
    data["pre"].drop(columns=["version","negating"],inplace=True)
    data["sub"].drop(columns=['stprba', 'zipba',
       'bas1', 'bas2', 'baph', 'countryma', 'stprma', 'cityma', 'zipma',
       'mas1', 'mas2', 'ein'],inplace=True)
    
    #make mapper for company names
    
    def mapper(colname):
        return {i:j for i,j in zip(data["sub"]["adsh"],data["sub"][colname])}
    
    def inverseMapper(colname):
        return {j:i for i,j in zip(data["sub"]["adsh"],data["sub"][colname])}
    
    data["num"]["name"] = data["num"]["adsh"].map(mapper("name"))
    data["pre"]["name"] = data["pre"]["adsh"].map(mapper("name"))
    
    data["num"].drop(columns=["version","coreg","adsh"],inplace=True)
    data["pre"].drop(columns=["adsh","plabel","inpth","rfile"],inplace=True)
    
    
    #get some sampels of company
    names = sorted(list(data["num"]["name"].unique()))[:20]
    stmts = data["pre"]["stmt"].unique()
    
    
    def f(data, names):
        dic = dict()
        for name in tqdm(names):
            #get dfs where name is equal to name
            df = data["num"][data["num"]["name"]==name].copy()
            df_stmt = data["pre"][data["pre"]["name"]==name].copy()
            
            #creat of company dic to load finacial data onto
            dic[name] = dict()
            
            # creat stmts for each company
            for stmt in stmts:
                dic[name][stmt] = dict()
            # fill more data
            
            for tag in set(df["tag"]):
                df_stmt_temp = df_stmt[df_stmt["tag"]==tag]
                for stmt in set(df_stmt_temp["stmt"]):
                    df_temp = df[df["tag"]==tag]
                    
                    df_temp = df_temp[df_temp["ddate"]==max(df_temp["ddate"])]
                    dic[name][stmt][list(df_temp["tag"])[0]]=df_temp["value"].values[0]
            
               
        return dic
    return f(data, names)
    #return pd.DataFrame.from_dict(f(data, names)).T

def combineReports(reports):
    reformed_dic = {}

    for report, level1 in reports.items():
        for company,level2 in level1.items():
            for stmt, level3 in level2.items():
                for tag, value in level3.items():
                    key = (company,stmt,tag)
                    if key not in reformed_dic:
                        reformed_dic[key] = {}
                    reformed_dic[key][report] = value
                    
    return pd.DataFrame.from_dict(reformed_dic, orient='index')

def comapnyFolium(company,info_df):
    company_df = info_df[company]
    longitude = company_df.iloc[3]
    latitude = company_df.iloc[2]
    st.write(" -\n".join([company[0],company_df.iloc[0][0]]))
    loc = folium.Map(location=[longitude,latitude], zoom_start=10)
    folium.Marker(
        [longitude, latitude], 
        popup=company[0], 
        tooltip=company[0]
    ).add_to(loc)
    
    # call to render Folium map in Streamlit
    map = st_folium(loc,width=800)

    return map

def companyDataframe(company,statement2look):
    data = df[company[0]][statement2look].T.sort_index(axis=1, ascending=False)
    return st.dataframe(data)

@np.vectorize
def timeString2float(x="2020q2"):
    lst = x.split("q")
    return int(lst[0])+(int(lst[1])-1)*0.25

@np.vectorize
def timeString2date(x="2020q2"):
    lst = x.split("q")
    return datetime.strptime(f"{lst[0]}-{int(lst[1])*3}-15","%Y-%m-%d")

def getTicker(company_name):
    
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
    data = res.json()

    company_code = data['quotes'][0]['symbol']
    return company_code

def getCompanyInfo():
    # sort from oldest to newest
    files = [i for i in getDir(path)]
    files.sort(reverse=True)
    info_dic = dict()
    for report in files:
        df = pd.read_csv(f"{path}/{i}/sub.txt",sep="\t")
        companies = sorted(list(set(df["name"])))

        for company in companies[:40]:
            if company not in info_dic:
                info_dic[company] = dict()
                data = df[df["name"]==company].sort_values(["form","period","filed"]).iloc[0]

                # get location
                loc = ", ".join([str(data["bas1"]),str(data["cityba"]),str(data["stprma"])])

                cord = gpd.tools.geocode(loc)
                cord = cord.to_crs(epsg = 5070)  # convert to Conus Albers
                info_dic[company]["long"] = float(cord["geometry"].centroid.x)
                info_dic[company]["lat"] = float(cord["geometry"].centroid.y)

                # get ticker
                try:
                    info_dic[company]["ticker"] = getTicker(company)
                except:
                    pass

            else:
                pass

    return pd.DataFrame.from_dict(info_dic)

df = pd.read_csv("/Users/matthesfogtmann/Documents/GitHub/OpenBloom/data/SEC_data.csv",index_col=[0,1,2],header=[0]).T
info_df = pd.read_csv('/Users/matthesfogtmann/Documents/GitHub/OpenBloom/data/company_info.csv')
options = sorted(set([i[0] for i in set(list(df.columns))]))
company = st.multiselect(label="Search for company",options=options, max_selections=1)

stmt_dic = {"BS" : "Balance Sheet", "IS" : "Income Statement", "CF" : "Cash Flow", "EQ" : "Equity",
 "CI": "Comprehensive Income", "UN" : "Unclassifiable Statement", "CP" :"Cover Page"}

inv_stmt_dic = {v: k for k, v in stmt_dic.items()}
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
statement2look = inv_stmt_dic[st.radio(options=inv_stmt_dic.keys(),label= "Statement")]

# try to plot company location on map
try:
    comapnyFolium(company,info_df)
except:
    pass

# try to plot company financial data
try:
    companyDataframe(company,statement2look)
except:
    st.write("No data")




#if len(companies)>0:
    #st.dataframe(df[companies[0]][['IS', 'CI', 'UN']])

#tags_options = set([i for i in np.array(list(df.index))[:,0]])



def plotCompanyTag(companies="1 800 FLOWERS COM INC",tags="Assets"):

    @np.vectorize
    def timeString2float(x="2020q2"):
        lst = x.split("q")
        num = int(lst[0])+(int(lst[1])-1)*0.25
        return num


    if len(tags)>1:
        fig, ax = plt.subplots(len(tags),1,figsize=(10,8),dpi=300)

        for i in range(len(tags)):
            for company in companies:
                y = df[company][tags[i]].values
                x = timeString2float(np.array(df[company][tags[i]].index))

                sort_index = np.argsort(x)

                x = x[sort_index]
                y = y[sort_index]

                ax[i].plot(x,y,label=company)

                ax[i].legend()
                ax[i].set_ylabel(tags[i])
    else:
        fig, ax = plt.subplots(1,1,figsize=(16,6),dpi=300)
        for company in companies:
            y = df[company][stmts_selected[0]][tags[0]].values
            
            x = timeString2date(np.array(df[company][stmts_selected[0]][tags[0]].index))
            x = pd.to_datetime(x)

            sort_index = np.argsort(x)

            x = x[sort_index]
            y = y[sort_index]

            ax.plot(x,y,label=company)

            ax.legend()
            ax.set_ylabel(tags[0])
    fig.tight_layout()
    st.pyplot(fig)

    
#if len(companies)>0:
    #tags = st.multiselect(label="What tag",options=tags_options)
    #if len(tags)>0:

        #plotCompanyTag(companies,tags)


def plotStock(company_df,stock,period,interval,window,placeholder="No"):
    stock = company_df.iloc[0].values[0]
    df = yf.download(tickers=stock,period=period,interval=interval)

    WINDOW = window
    df['sma'] = df['Close'].rolling(WINDOW).mean()
    df['std'] = df['Close'].rolling(WINDOW).std(ddof = 0)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                vertical_spacing=0.1, subplot_titles=('Stock Price', 'Volume'), 
                row_width=[0.2, 0.7])

    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'], name = 'market data'), 
                row=1, col=1)
    
    fig.add_trace(go.Line(x=df.index,
                             y=df['Close'],
                            name = 'market data fill',
                            visible="legendonly",
                            fill='tozeroy'), 
                row=1, col=1)

    # Moving average
    fig.add_trace(go.Scatter(x = df.index,
                            y = df['sma'],
                            line_color = 'black',
                            name = 'sma',
                            visible="legendonly"), 
                row=1, col=1)

    # Upper Bound
    fig.add_trace(go.Scatter(x = df.index,
                            y = df['sma'] + (df['std'] * 2),
                            line_color = 'yellow',
                            name = 'upper band',
                            opacity = 0.2,
                            visible="legendonly",
                            legendgroup='bounds'), 
                row=1, col=1)

    # Lower Bound fill in between with parameter 'fill': 'tonexty'
    fig.add_trace(go.Scatter(x = df.index,
                            y = df['sma'] - (df['std'] * 2),
                            line_color = 'orange',
                            fill = 'tonexty',
                            name = 'lower band',
                            opacity = 0.2,
                            visible="legendonly",
                            legendgroup='bounds'), 
                row=1, col=1)
    
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], showlegend=False, name = 'volume'), row=2, col=1)


    fig.update_layout(
        title= str(stock)+' Live Share Price:',
        yaxis_title='Stock Price (USD per Shares)',
        template="ggplot2")           

    fig.update_xaxes(rangebreaks=[dict(bounds=[16, 9.5], pattern="hour"), dict(bounds=["sat", "mon"])],
                    rangeslider_visible=False,
                    )
    if placeholder=="No":
        st.plotly_chart(fig)
    else:
        placeholder.plotly_chart(fig)
    return df[['Close',"Open"]]

company_df = info_df[company]

yf.pdr_override() 

try:
    stock = company_df.iloc[0].values[0]
except:
    pass



try:

    if type(stock) == str:
        
        period = st.radio(options=["1d (Live)","5d","1mo","3mo","6mo","ytd","1y","5y","max"],label="Period",index=1)

        window = st.number_input("Window for rolling mean",value=50)
        metric_placeholder = st.empty()
        placeholder = st.empty()

        while period == "1d (Live)":
            df = plotStock(company_df,stock,"1d","1m",window,placeholder)
            metric_placeholder.metric(label=f"{stock} price", value=round(df["Close"].iloc[-1], 2), delta=f"{ round(df['Close'].iloc[-1]-df['Open'].iloc[0],2)} ({round((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0] * 100, 2)}%)")
            if period != "1d (Live)":
                break
                

        if period == "5d":
            plotStock(company_df,stock,"5d","5m",window,placeholder)
        
        elif period == "1mo":
            plotStock(company_df,stock,"1mo","5m",window)

        elif period == "3mo":
            plotStock(company_df,stock,"3mo","1h",window)

        elif period == "6mo":
            plotStock(company_df,stock,"6mo","1h",window)

        elif period == "ytd":
            plotStock(company_df,stock,"ytd","1h",window)

        elif period == "1y":
            plotStock(company_df,stock,"1y","1h",window)
            
        elif period == "5y":
            plotStock(company_df,stock,"5y","1mo",window)

        elif period == "max":
            plotStock(company_df,stock,"max","1mo",window)
except:
    pass