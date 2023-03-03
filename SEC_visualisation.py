import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

import geopandas as gpd

import streamlit as st

st.title("SEC visualisation")

import streamlit as st
import pandas as pd

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

@np.vectorize
def timeString2float(x="2020q2"):
    lst = x.split("q")
    return int(lst[0])+(int(lst[1])-1)*0.25

@np.vectorize
def timeString2date(x="2020q2"):
    lst = x.split("q")
    return f"{lst[0]}-{int(lst[1])*3}-15"

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



options = [i for i in set(np.array(list(df.columns))[:,0])]
companies = st.multiselect(label="Search for company",options=options)


stmt_dic = {"BS" : "Balance Sheet", "IS" : "Income Statement", "CF" : "Cash Flow", "EQ" : "Equity",
 "CI": "Comprehensive Income", "UN" : "Unclassifiable Statement", "CP" :"Cover Page"}

inv_stmt_dic = {v: k for k, v in stmt_dic.items()}
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
statement2look = inv_stmt_dic[st.radio(options=inv_stmt_dic.keys(),label= "Statement")]

if len(companies)==1:
    try:
        data = df[companies[0]][statement2look].T
        st.write(data)
    except KeyError:
        st.write("No data")

else:
    cols = st.columns(len(companies))
    for company in companies:
        try:
            data = df[company][statement2look].T
            st.subheader(company)
            st.write(data)
        except KeyError:
            st.write("No data")  
st.write(len(companies))

if len(companies)>0:
    st.dataframe(df[companies[0]][['IS', 'CI', 'UN']])

tags_options = set([i for i in np.array(list(df.index))[:,0]])



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

    
    
if len(companies)>0:
    tags = st.multiselect(label="What tag",options=tags_options)
    if len(tags)>0:

        plotCompanyTag(companies,tags)