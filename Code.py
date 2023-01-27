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
from sklearn import preprocessing
import scipy.optimize as opt
import numpy as np
import err_ranges as err


def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    f = scale * np.exp(growth * (t-1960))
    return f


def read_data(filename):
    """This function takes filename as argument,
    reads the csv file and returns one normal
    dataframe and one date frame with countries as columns"""
    data = pd.read_csv(filename, skiprows=4)
    data_transposed = data.set_index("Country Name").transpose()
    data_transposed = data_transposed.drop(index=["Country Code",
                                                  "Indicator Name",
                                                  "Indicator Code"])
    return data, data_transposed


def filter_data(data):
    """This functions takes dataframe as argument and rturns the
    filtered dataframe"""
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


def fit_error():
    """This functions fits the data, plots error range and
    saves the graph image as png file"""
    param, covar = opt.curve_fit(exp_growth, pop_data_transposed["Year"],
                                 pop_data_transposed["China"],
                                 p0=(6.6707e+08, 0.03))
    plt.figure()
    pop_data_transposed["fit"] = exp_growth(pop_data_transposed["Year"],
                                            *param)
    pop_data_transposed.plot("Year", ["China", "fit"])

    year = np.arange(1960, 2020)
    sigma = np.sqrt(np.diag(covar))
    low, up = err.err_ranges(year, exp_growth, param, sigma)
    plt.fill_between(year, low, up, color="red",
                     alpha=0.2, label="Error Range")
    plt.title("Fit & Error Range")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.savefig("Fit & Error", dpi=300, bbox_inches="tight")
    plt.show()


def forcast():
    """This functions plots the forcast graph and saves the image as png"""
    year = np.arange(1960, 2031)
    forecast = exp_growth(year, *param)
    pop_data_transposed.plot("Year", ["China"])
    plt.plot(year, forecast, label="Forecast")
    plt.title("Forcast from 1960 to 2031")
    plt.xlabel("Year")
    plt.ylabel("Population")
    plt.legend()
    plt.savefig("Forcast", dpi=300, bbox_inches="tight")
    plt.show()


def scatter():
    """This function plots the scatter plt and saves the image as png"""
    plt.figure()
    plt.scatter(dataframe["CO2"], dataframe["Population"], s=20)
    plt.title("Scatter Plot of CO2(mt per capita) vs Total Population ",
              size=16, pad=30)
    plt.xlabel("CO2")
    plt.ylabel("Population")
    plt.savefig("Scatter", dpi=300, bbox_inches="tight")
    plt.show()


def pairplot():
    """This function plots the pairplot and saves the image as png"""
    plt.figure(figsize=(20, 20))
    sns.pairplot(dataframe[["CO2", "Population"]])
    plt.title("Pairplot of CO2 Emission(mt per capita)\n vs Total Population ",
              size=16, pad=170)
    plt.xlabel("CO2")
    plt.ylabel("Population")
    plt.savefig("Pair Plot", dpi=300, bbox_inches="tight")
    plt.show()


def kmeans_cluster():
    """This finction plots the K-means cluster and saves the image as png"""
    plt.figure(figsize=(6.0, 6.0))
    plt.scatter(df_fit[0], df_fit[1], c=labels, cmap="Accent")
    for ic in range(2):
        xc, yc = cen[ic, :]
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel("CO2")
    plt.ylabel("Population")
    plt.title("K-mean Clustering Plot CO2 vs Population")
    plt.savefig("K-Means", dpi=300, bbox_inches="tight")
    plt.show()


# Reading the data using read_data() function
CO2_data, CO2_data_transposed = read_data("CO2 Emission mt per capita.csv")
pop_data, pop_data_transposed = read_data("Population.csv")

# Creating a lsit of years for filtering the data
Years = ["Country Name", "1990", "1992", "1994", "1996", "1998",
         "2000", "2002", "2004", "2006", "2008", "2010", "2012",
         "2014", "2016", "2018", "2019"]

# Filtering the data using filter_data() function
CO2_data_filtered = filter_data(CO2_data)
pop_data_filtered = filter_data(pop_data)
# Creating a new column having the mean value of the corresponding row
CO2_data_filtered["Mean"] = CO2_data_filtered.mean(axis=1)
pop_data_filtered["Mean"] = pop_data_filtered.mean(axis=1)
# Creating a new blank dataframe having 3 columns
dataframe = pd.DataFrame(columns=["Country Name", "CO2", "Population"])
# Populating the columns of dataframe
dataframe["Country Name"] = CO2_data_filtered["Country Name"]
dataframe["CO2"] = CO2_data_filtered["Mean"]
dataframe["Population"] = pop_data_filtered["Mean"]

# creating copy of dataframe for fitting
df_fit = dataframe[["CO2", "Population"]].copy()
# normalizing the values
df_fit = preprocessing.normalize(df_fit, axis=0)
# converting normalized value to dataframe
df_fit = pd.DataFrame(df_fit)
# printing the attributes of dataframe
print(df_fit.describe())

for ic in range(2, 17):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(df_fit, labels))

kmeans = cluster.KMeans(n_clusters=2)
kmeans.fit(df_fit)
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# Converting transposed data of column to numeric
pop_data_transposed["China"] = pd.to_numeric(pop_data_transposed["China"])
pop_data_transposed["Year"] = pop_data_transposed.index
length = len(pop_data_transposed)
pop_data_transposed = pop_data_transposed.drop(pop_data_transposed.index
                                               [length-1])
pop_data_transposed["Year"] = pd.to_numeric(pop_data_transposed["Year"])

year = np.arange(1960, 2020)

# finding param and covar using fit
param, covar = opt.curve_fit(exp_growth, pop_data_transposed["Year"],
                             pop_data_transposed["China"],
                             p0=(6.6707e+08, 0.03))

# Calling all the functions
scatter()
pairplot()
kmeans_cluster()
fit_error()
forcast()





































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
