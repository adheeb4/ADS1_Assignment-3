# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 23:02:53 2023

@author: adheeb
"""
import matplotlib.pyplot as plt
import pandas as pd


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
                (data["Country Name"] == "China") |
                (data["Country Name"] == "United Kingdom") |
                (data["Country Name"] == "India") |
                (data["Country Name"] == "United States")]
    data = data[Years]
    return data
    
    
forest_area_data, forest_area_data_transposed = read_data("Forest Area.csv")
CO2_data, CO2_data_transposed = read_data("Population.csv")
Years = ["1990","1992","1994","1996","1998","2000","2002",
         "2004","2006","2008","2010","2012","2014","2016","2018"]
Countries_list = ["Australia", "Brazil", "China",
                  "United Kingdom", "India", "United States"]
CO2_data_filtered = filter_data(CO2_data)
forest_area_data_filtered = filter_data(forest_area_data)


plt.scatter(CO2_data_filtered["1990"],CO2_data_filtered["1996"])