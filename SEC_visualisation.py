import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import plotly.graph_objs as go 
from plotly.subplots import make_subplots
import time

import feedparser
from bs4 import BeautifulSoup
import urllib
from dateparser import parse as parse_date
import requests

from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder

import geopandas as gpd
import folium

import streamlit as st
from streamlit_folium import st_folium

import yfinance as yf


st.set_page_config(page_title = "OpenBloom",layout="wide")

st.markdown("""
<style>
body {
    background-color: #F5F5F5;
}
</style>
""", unsafe_allow_html=True)


st.title("OpenBloom")

path = '/Users/matthesfogtmann/Downloads/SEC data/'

def getDir(path):
    lst = []
    for i in os.listdir(path):
        if len(i)==6 and "q" in i:
            lst.append(i)
    return lst

# this code directly copied from https://github.com/kotartemiy/pygooglenews/blob/master/pygooglenews/__init__.py, because it wouldn't work unless
class GoogleNews:
    def __init__(self, lang = 'en', country = 'US'):
        self.lang = lang.lower()
        self.country = country.upper()
        self.BASE_URL = 'https://news.google.com/rss'

    def __top_news_parser(self, text):
        """Return subarticles from the main and topic feeds"""
        try:
            bs4_html = BeautifulSoup(text, "html.parser")
            # find all li tags
            lis = bs4_html.find_all('li')
            sub_articles = []
            for li in lis:
                try:
                    sub_articles.append({"url": li.a['href'],
                                         "title": li.a.text,
                                         "publisher": li.font.text})
                except:
                    pass
            return sub_articles
        except:
            return text

    def __ceid(self):
        """Compile correct country-lang parameters for Google News RSS URL"""
        return '?ceid={}:{}&hl={}&gl={}'.format(self.country,self.lang,self.lang,self.country)

    def __add_sub_articles(self, entries):
        for i, val in enumerate(entries):
            if 'summary' in entries[i].keys():
                entries[i]['sub_articles'] = self.__top_news_parser(entries[i]['summary'])
            else:
                entries[i]['sub_articles'] = None
        return entries

    def __scaping_bee_request(self, api_key, url):
        response = requests.get(
            url="https://app.scrapingbee.com/api/v1/",
            params={
                "api_key": api_key,
                "url": url,
                "render_js": "false"
            }
        )
        if response.status_code == 200:
            return response
        if response.status_code != 200:
            raise Exception("ScrapingBee status_code: "  + str(response.status_code) + " " + response.text)

    def __parse_feed(self, feed_url, proxies=None, scraping_bee = None):

        if scraping_bee and proxies:
            raise Exception("Pick either ScrapingBee or proxies. Not both!")

        if proxies:
            r = requests.get(feed_url, proxies = proxies)
        else:
            r = requests.get(feed_url)

        if scraping_bee:
            r = self.__scaping_bee_request(url = feed_url, api_key = scraping_bee)
        else:
            r = requests.get(feed_url)


        if 'https://news.google.com/rss/unsupported' in r.url:
            raise Exception('This feed is not available')

        d = feedparser.parse(r.text)

        if not scraping_bee and not proxies and len(d['entries']) == 0:
            d = feedparser.parse(feed_url)

        return dict((k, d[k]) for k in ('feed', 'entries'))

    def __search_helper(self, query):
        return urllib.parse.quote_plus(query)

    def __from_to_helper(self, validate=None):
        try:
            validate = parse_date(validate).strftime('%Y-%m-%d')
            return str(validate)
        except:
            raise Exception('Could not parse your date')



    def top_news(self, proxies=None, scraping_bee = None):
        """Return a list of all articles from the main page of Google News
        given a country and a language"""
        d = self.__parse_feed(self.BASE_URL + self.__ceid(), proxies=proxies, scraping_bee=scraping_bee)
        d['entries'] = self.__add_sub_articles(d['entries'])
        return d

    def topic_headlines(self, topic: str, proxies=None, scraping_bee=None):
        """Return a list of all articles from the topic page of Google News
        given a country and a language"""
        #topic = topic.upper()
        if topic.upper() in ['WORLD', 'NATION', 'BUSINESS', 'TECHNOLOGY', 'ENTERTAINMENT', 'SCIENCE', 'SPORTS', 'HEALTH']:
            d = self.__parse_feed(self.BASE_URL + '/headlines/section/topic/{}'.format(topic.upper()) + self.__ceid(), proxies = proxies, scraping_bee=scraping_bee)

        else:
            d = self.__parse_feed(self.BASE_URL + '/topics/{}'.format(topic) + self.__ceid(), proxies = proxies, scraping_bee=scraping_bee)

        d['entries'] = self.__add_sub_articles(d['entries'])
        if len(d['entries']) > 0:
            return d
        else:
            raise Exception('unsupported topic')

    def geo_headlines(self, geo: str, proxies=None, scraping_bee=None):
        """Return a list of all articles about a specific geolocation
        given a country and a language"""
        d = self.__parse_feed(self.BASE_URL + '/headlines/section/geo/{}'.format(geo) + self.__ceid(), proxies = proxies, scraping_bee=scraping_bee)

        d['entries'] = self.__add_sub_articles(d['entries'])
        return d

    def search(self, query: str, helper = True, when = None, from_ = None, to_ = None, proxies=None, scraping_bee=None):
        """
        Return a list of all articles given a full-text search parameter,
        a country and a language

        :param bool helper: When True helps with URL quoting
        :param str when: Sets a time range for the artiles that can be found
        """

        if when:
            query += ' when:' + when

        if from_ and not when:
            from_ = self.__from_to_helper(validate=from_)
            query += ' after:' + from_

        if to_ and not when:
            to_ = self.__from_to_helper(validate=to_)
            query += ' before:' + to_

        if helper == True:
            query = self.__search_helper(query)

        search_ceid = self.__ceid()
        search_ceid = search_ceid.replace('?', '&')

        d = self.__parse_feed(self.BASE_URL + '/search?q={}'.format(query) + search_ceid, proxies = proxies, scraping_bee=scraping_bee)

        d['entries'] = self.__add_sub_articles(d['entries'])
        return d


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

def comapnyFolium(company,info_df,cols):
    company_df = info_df[company]
    longitude = company_df.iloc[3]
    latitude = company_df.iloc[2]

    loc = folium.Map(location=[longitude,latitude], zoom_start=10,)
    folium.Marker(
        [longitude, latitude], 
        popup=company[0], 
        tooltip=company[0]
    ).add_to(loc)
    
    with cols[0]:
    # call to render Folium map in Streamlit
        map = st_folium(loc,width=300,height=300)

    return map

def companyDataframeSelect(company,statement2look,df):
    data = df[company[0]][statement2look].T.sort_index(axis=1, ascending=False)
    data['Tag'] = data.index

    data = data.reset_index(drop=True)

    # Move the 'index' column to the first position
    data = data[['Tag'] + [col for col in data.columns if col != 'Tag']]
    gb = GridOptionsBuilder.from_dataframe(data)
    gb.configure_side_bar() #Add a sidebar
    gb.configure_selection('multiple', use_checkbox=True, groupSelectsChildren="Group checkbox select children") #Enable multi-row selection
    gb.configure_column("Tag", pinned="left", auto_size=True)
    gridOptions = gb.build()

    grid_response = AgGrid(
        data,
        gridOptions=gridOptions,
        data_return_mode='AS_INPUT',
        update_mode='MODEL_CHANGED', 
        fit_columns_on_grid_load=False,
        theme='streamlit',
        enable_enterprise_modules=True, 
        width='100%',
        reload_data=True,
        
    )

    return pd.DataFrame(grid_response["selected_rows"]).set_index("Tag").drop("_selectedRowNodeInfo",axis=1)

def companyDataframe(company,statement2look,df):
    df = df[company[0]][statement2look].T.sort_index(axis=1, ascending=False)
    
    return st.dataframe(df)

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



df = pd.read_csv("data/SEC_data.csv",index_col=[0,1,2],header=[0]).T
info_df = pd.read_csv('data/company_info.csv')
options = sorted(set([i[0] for i in set(list(df.columns))]))

cols = st.columns((6,15,7))

with cols[0]:
    company = st.multiselect(label="Search for company",options=options, max_selections=1)

# try to plot company location on map
try:
    comapnyFolium(company,info_df,cols)
except:
    pass


def plotCompanyTag(companies, rows):
    @np.vectorize
    def timeString2date(x="2020q2"):
        lst = x.split("q")
        return datetime.strptime(f"{lst[0]}-{int(lst[1])*3}-15", "%Y-%m-%d")

    if len(rows) > 1:
        fig = make_subplots(rows=len(rows), cols=1, shared_xaxes=True, vertical_spacing=0.05)

        for i in range(len(rows)):
            y = df.loc[rows[i]]
            x = df.columns[::-1]
            fig.add_trace(go.Line(x=x, y=y, name=rows[i], showlegend=False), row=i + 1, col=1)

            fig.update_yaxes(title_text=rows[i], row=i + 1, col=1)

    else:
        fig = go.Figure()
        y = df.loc[rows[0]]
        x = df.columns[::-1]
        fig.add_trace(go.Line(x=x, y=y, name=rows[0]))

        fig.update_yaxes(title_text=rows[0])

    fig.update_layout(height=1000, width=1000, template='ggplot2')
    st.plotly_chart(fig)


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
    
    fig.add_trace(go.Scatter(x=df.index,
                             y=df['Close'],
                            name = 'market data fill',
                            visible="legendonly",
                            fill='tonexty'
                            ), 
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
with cols[1]:
    try:
        if type(stock) == str:
            metric_placeholder = st.empty()
            placeholder = st.empty()
            period_place = st.empty()
            period = period_place.radio(options=["1d (Live)","5d","1mo","3mo","6mo","ytd","1y","5y","max"],label="Period",index=1,horizontal=True)

            #window = st.number_input("Window for rolling mean",value=50)
            window = 50

            while period == "1d (Live)":
                df = plotStock(company_df,stock,"1d","1m",window,placeholder)
                metric_placeholder.metric(label=f"{stock} price", value=round(df["Close"].iloc[-1], 2), delta=f"{ round(df['Close'].iloc[-1]-df['Open'].iloc[0],2)} ({round((df['Close'].iloc[-1] - df['Open'].iloc[0]) / df['Open'].iloc[0] * 100, 2)}%)")
                if period != "1d (Live)":
                    break
                    

            if period == "5d":
                plotStock(company_df,stock,"5d","5m",window,placeholder)
            
            elif period == "1mo":
                plotStock(company_df,stock,"1mo","5m",window,placeholder)

            elif period == "3mo":
                plotStock(company_df,stock,"3mo","1h",window,placeholder)

            elif period == "6mo":
                plotStock(company_df,stock,"6mo","1h",window,placeholder)

            elif period == "ytd":
                plotStock(company_df,stock,"ytd","1h",window,placeholder)

            elif period == "1y":
                plotStock(company_df,stock,"1y","1h",window,placeholder)
                
            elif period == "5y":
                plotStock(company_df,stock,"5y","1mo",window,placeholder)

            elif period == "max":
                plotStock(company_df,stock,"max","1mo",window,placeholder)


            
    except:
        pass


try:
    with cols[2]:
        
        dic = dict()
        news_list = list()
        gn = GoogleNews()
        s = gn.search(f'{company} {stock} stock news')
        news_dict = dict()
        for entry in s["entries"]:
            news_dict[entry['published']] = f"[{entry['title']}]({entry['link']})"
        sorted_dict = dict(sorted(news_dict.items(), key=lambda x: datetime.strptime(x[0], '%a, %d %b %Y %H:%M:%S %Z'),reverse=True))          
        for news in sorted_dict.values():
            news_list.append(news)
       

        with st.expander(f"{company[0]} news"):
            news_content = st.empty()
            
            news_content.markdown("\n \n".join(news_list[:10]), unsafe_allow_html=True)

            button_placeholder = st.empty()

            view_more_button = button_placeholder.button("View More")

            if view_more_button:
                news_content.markdown("\n \n".join(news_list), unsafe_allow_html=True)
                
                view_less_button = button_placeholder.button("View Less")

                if view_less_button:
                    news_content.markdown("\n \n".join(news_list[:10]), unsafe_allow_html=True)
                    
                    button_placeholder.button("View More")



except:
    pass

#fig = plt.figure()
#plt.plot(df[company[0]]["BS"].index,df[company[0]]["BS"]["Assets"].values)
#st.pyplot(fig)
#plt.close()

stmt_dic = {"BS" : "Balance Sheet", "IS" : "Income Statement", "CF" : "Cash Flow", "EQ" : "Equity",
"CI": "Comprehensive Income", "UN" : "Unclassifiable Statement", "CP" :"Cover Page"}

inv_stmt_dic = {v: k for k, v in stmt_dic.items()}

try:
    if type(company[0]) == str:
        with st.expander(f"{company[0]} finacial Data"):
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
            statement2look = inv_stmt_dic[st.radio(options=inv_stmt_dic.keys(),label= "Statement")]

            # try to plot company financial data
            df = companyDataframeSelect(company,statement2look,df)
            plotCompanyTag(df,df.index.values.tolist())
    else:
        pass
except:
    pass
