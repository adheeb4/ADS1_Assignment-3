# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:02:53 2023

@author: adheeb
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import matplotlib.pyplot as plt
from sklearn import preprocessing
import scipy.optimize as opt
import numpy as np
import err_ranges as err

def expoFunc(x, a, b):
    return a**(x+b)


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t-1960)) 
    
    return f


def read_data(filename):
    data = pd.read_csv(filename, skiprows=4)
    data_transposed = data.set_index("Country Name").transpose()
    data_transposed = data_transposed.drop(index=["Country Code",
                                                  "Indicator Name",
                                                  "Indicator Code"])
    return data, data_transposed

def filter_data(data):
    data = data[(data["Country Name"] == "Australia") |
                (data["Country Name"] == "Brazil") |
                (data["Country Name"] == "Bahrain") |
                (data["Country Name"] == "Switzerland") |
                (data["Country Name"] == "China") |
                (data["Country Name"] == "Spain") |
                (data["Country Name"] == "France") |
                (data["Country Name"] == "United Kingdom") |
                (data["Country Name"] == "India") |
                (data["Country Name"] == "Ireland") |
                (data["Country Name"] == "Sri Lanka") |
                (data["Country Name"] == "Nigeria") |
                (data["Country Name"] == "Saudi Arabia") |
                (data["Country Name"] == "Singapore") |
                (data["Country Name"] == "Sweden") |
                (data["Country Name"] == "Ukraine") |
                (data["Country Name"] == "United States")]
    data = data[Years]
    return data


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t-1950)) 
    
    return f


    
CO2_data, CO2_data_transposed = read_data("CO2 Emission mt per capita.csv")
pop_data, pop_data_transposed = read_data("Population.csv")
Years = ["Country Name", "1990", "1992", "1994", "1996", "1998", "2000", "2002", "2004",
         "2006", "2008", "2010", "2012", "2014", "2016", "2018","2019"]


CO2_data_filtered = filter_data(CO2_data)
pop_data_filtered = filter_data(pop_data)
CO2_data_filtered["Mean"] = CO2_data_filtered.mean(axis=1)
pop_data_filtered["Mean"] = pop_data_filtered.mean(axis=1)
dataframe = pd.DataFrame(columns=["Country Name", "CO2", "Population"])
dataframe["Country Name"] = CO2_data_filtered["Country Name"]
dataframe["CO2"] = CO2_data_filtered["Mean"]
dataframe["Population"] = pop_data_filtered["Mean"]


plt.scatter(dataframe["CO2"], dataframe["Population"], s=20)
plt.show()

plt.figure(figsize=(20,20))
sns.pairplot(dataframe[["CO2", "Population"]])
plt.title("Pairplot of CO2 Emission(mt per capita\n vs Total Population ", size=16,
          pad=170)
plt.show()
#________________________________________________________________________________________
df_fit = dataframe[["CO2","Population"]].copy()
df_fit = preprocessing.normalize(df_fit, axis=0)
df_fit = pd.DataFrame(df_fit)
print(df_fit.describe())

for ic in range(2, 17):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))

# Plot for three clusters
kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)
labels = kmeans.labels_
cen = kmeans.cluster_centers_
plt.figure(figsize=(6.0, 6.0))
plt.scatter(df_fit[0], df_fit[1], c=labels, cmap="Accent")
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("CO2")
plt.ylabel("Population")
plt.title("K-mean Clustering Plot CO2 vs Population")
plt.show()
#________________________________________________________________________________________




































"""
def scatter_plot_fitting():
    
    The function is used to fit the curve over the cluster for the same data.
    

    xaxis = CO2_data_transposed["Australia"]
    yaxis=  CO2_data_transposed["China"]
    popt, pcov = opt.curve_fit(expoFunc, xaxis, yaxis, p0=[1, 0])
    ab_opt, bc_opt = popt
    x_mod = np.linspace(min(xaxis), max(xaxis), 100)
    y_mod = expoFunc(x_mod, ab_opt, bc_opt)

    # plot for scattering after fitting the curve
    plt.scatter(xaxis, yaxis)
    plt.plot(x_mod, y_mod, color='r')
    plt.title('Scatter plot with the curve fitting', fontsize=20, color='purple')
    plt.ylabel('Renewable energy-->', fontsize=16)
    plt.xlabel('Electricity production--->', fontsize=16)
    plt.savefig("urve_and_Cluster.png")
    plt.show()
"""
