import json
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
import numpy as np
import math
import re

studentid = os.path.basename(sys.modules[__name__].__file__)


def log(question, output_df, other):
    print("--------------- {}----------------".format(question))

    if other is not None:
        print(question, other)
    if output_df is not None:
        df = output_df.head(5).copy(True)
        for c in df.columns:
            df[c] = df[c].apply(lambda a: a[:20] if isinstance(a, str) else a)

        df.columns = [a[:10] + "..." for a in df.columns]
        print(df.to_string())


def question_1(exposure, countries):
    """
    :param exposure: the path for the exposure.csv file
    :param countries: the path for the Countries.csv file
    :return: df1
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # find same countries with different names:
    # print(exposure[~exposure['Country'].isin(countries.Country)])
    # print(countries[~countries['Country'].isin(exposure.Country)])
    # then use google to check whether they match

    # Find same countries: {'country from exposure.csv': 'country from Country.csv'}
    same_countries = {'Palestine': 'Palestinian Territory',
                      'North Macedonia': 'Macedonia',
                      'Moldova Republic of': 'Moldova',
                      'C�te d\'Ivoire': 'Ivory Coast',
                      'Korea DPR': 'North Korea',
                      'Korea Republic of': 'South Korea',
                      'Brunei Darussalam': 'Brunei',
                      'Russian Federation': 'Russia',
                      'Lao PDR': 'Laos',
                      'United States of America': 'United States',
                      'Viet Nam': 'Vietnam',
                      'Congo DR': 'Democratic Republic of the Congo',
                      'Congo': 'Republic of the Congo',
                      'Cabo Verde': 'Cape Verde',
                      'Eswatini': 'Swaziland'}

    # Read and preprocess exposure.csv
    exposure = pd.read_csv(exposure, sep=';', low_memory=False)
    exposure.dropna(subset=['country'], inplace=True)
    exposure.rename(columns={'country': 'Country'}, inplace=True)
    exposure.replace(same_countries, inplace=True)

    # Read Countries.csv
    countries = pd.read_csv(countries)

    # merge, set index and sort
    df1 = pd.merge(exposure, countries, on='Country')
    df1.set_index('Country', inplace=True)
    df1.sort_index(inplace=True)
    #################################################

    log("QUESTION 1", output_df=df1, other=df1.shape)
    return df1


def question_2(df1):
    """
    :param df1: the dataframe created in question 1
    :return: df2
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # col_name: which column to deal with
    # split data in cities and get a list of dict
    # convert list of dict into list of json
    # convert list of json into dataframe and remove duplicate rows
    # return mean value of col_name
    def get_column(cities, col_name):
        lst = []
        for city in cities.split('|||'):
            lst.append(json.loads(city))
        df = pd.DataFrame(lst)
        df.drop_duplicates(subset=['Country', 'Latitude', 'Longitude'], inplace=True)

        return df[col_name].mean()

    df2 = df1

    # add new row avg_latitude and avg_longitude
    df2['avg_latitude'] = df1['Cities'].apply(get_column, args=('Latitude',))
    df2['avg_longitude'] = df1['Cities'].apply(get_column, args=('Longitude',))
    #################################################

    log("QUESTION 2", output_df=df2[["avg_latitude", "avg_longitude"]], other=df2.shape)
    return df2


def question_3(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df3
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # compute distance based on haversine formula
    def compute_distance(df):
        Wuhan_latitude = 30.5928
        Wuhan_longitude = 114.3055
        latitude = df['avg_latitude']
        longitude = df['avg_longitude']
        R = 6373
        d = 2 * R * math.asin(math.sqrt(
            math.sin(math.radians(Wuhan_latitude - latitude) / 2) ** 2 + math.cos(math.radians(Wuhan_latitude)) * math.cos(math.radians(latitude)) * math.sin(
                math.radians(Wuhan_longitude - longitude) / 2) ** 2))
        return d

    df3 = df2

    # compute distances and sort df3 by them
    df3['distance_to_Wuhan'] = df3.apply(lambda countries: compute_distance(countries), axis=1)
    df3.sort_values(by='distance_to_Wuhan', inplace=True)
    #################################################

    log("QUESTION 3", output_df=df3[['distance_to_Wuhan']], other=df3.shape)
    return df3


def question_4(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: df4
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    # Find same countries: {'country from Countries-Continents.csv': 'country from exposure.csv'}
    same_countries = {'"Korea, North"': 'North Korea',
                      '"Korea, South"': 'South Korea',
                      'US': 'United States',
                      'Russian Federation': 'Russia',
                      'Congo': 'Republic of the Congo',
                      '"Congo, Democratic Republic of"': 'Democratic Republic of the Congo'}

    # read and preprocess Countries-Continents.csv
    continents = pd.read_csv(continents)
    continents.replace(same_countries, inplace=True)

    # get series 'Covid_19_Economic_exposure_index' and convert to dataframe
    exposure_index = df2['Covid_19_Economic_exposure_index']
    exposure_index = exposure_index.reset_index()

    # remove rows with value x, and convert data x,y into float x.y
    exposure_index.drop(exposure_index[exposure_index.Covid_19_Economic_exposure_index == 'x'].index, inplace=True)
    exposure_index = exposure_index.applymap(lambda x: x.replace(',', '.'))
    exposure_index['Covid_19_Economic_exposure_index'] = pd.to_numeric(exposure_index['Covid_19_Economic_exposure_index'])

    # compute average_covid_19_Economic_exposure_index
    merge_df = pd.merge(exposure_index, continents, on='Country')
    series4 = merge_df.groupby('Continent')["Covid_19_Economic_exposure_index"].mean()

    # convert series into dataframe, set index and sort
    dict_series4 = {'Continent': series4.index, 'Covid_19_Economic_exposure_index': series4.values}
    df4 = pd.DataFrame(dict_series4)
    df4.set_index('Continent', drop=False, inplace=True)
    df4.sort_values(by='Covid_19_Economic_exposure_index', inplace=True)
    #################################################

    log("QUESTION 4", output_df=df4, other=df4.shape)
    return df4


def question_5(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df5
            Data Type: dataframe
            Please read the assignment specs to know how to create the output dataframe
    """
    #################################################
    # Your code goes here ...
    # compute the average "Foreign direct investment" for each income class
    foreign_di = df2[['Income classification according to WB', 'Foreign direct investment']]
    foreign_di.drop(foreign_di[foreign_di['Foreign direct investment'] == 'x'].index, inplace=True)
    foreign_di = foreign_di.applymap(lambda x: x.replace(',', '.'))
    foreign_di['Foreign direct investment'] = pd.to_numeric(foreign_di['Foreign direct investment'])
    series1 = foreign_di.groupby('Income classification according to WB')['Foreign direct investment'].mean()

    # the average "Net_ODA_received_perc_of_GNI" for each income class
    net_orpog = df2[['Income classification according to WB', 'Net_ODA_received_perc_of_GNI']]
    net_orpog.drop(net_orpog[net_orpog['Net_ODA_received_perc_of_GNI'] == "No data"].index, inplace=True)
    net_orpog = net_orpog.applymap(lambda x: x.replace(',', '.'))
    net_orpog['Net_ODA_received_perc_of_GNI'] = pd.to_numeric(net_orpog['Net_ODA_received_perc_of_GNI'])
    series2 = net_orpog.groupby('Income classification according to WB')['Net_ODA_received_perc_of_GNI'].mean()

    # combine and convert series into dataframe, set index
    dict_series = {'Income Class': series1.index, 'Avg Foreign direct investment': series1.values, 'Avg_ Net_ODA_received_perc_of_GNI': series2.values}
    df5 = pd.DataFrame(dict_series)
    df5.set_index('Income Class', drop=False, inplace=True)
    #################################################

    log("QUESTION 5", output_df=df5, other=df5.shape)
    return df5


def question_6(df2):
    """
    :param df2: the dataframe created in question 2
    :return: cities_lst
            Data Type: list
            Please read the assignment specs to know how to create the output dataframe
    """
    cities_lst = []
    #################################################
    # Your code goes here ...
    def get_df(cities):
        for city in cities.split('|||'):
            a_lst.append(json.loads(city))

    # change content in 'Cities' to dataframe, and drop duplicate rows and rows without population
    a_lst = []
    df2['Cities'].apply(get_df)
    df_cities = pd.DataFrame(a_lst)
    df_cities.drop_duplicates(subset=['Country', 'Latitude', 'Longitude'], inplace=True)
    df_cities.drop(['Latitude', 'Longitude'], inplace=True, axis=1)
    df_cities = df_cities[df_cities['Population'].notnull()]

    # get series and convert to dataframe(2 columns)
    df_class = df2['Income classification according to WB']
    df_class.reset_index()

    # find top 5 most populous cities located in LIC
    merge_df = pd.merge(df_cities, df_class, on='Country')
    merge_df = merge_df.loc[merge_df['Income classification according to WB'].isin(['LIC'])]
    merge_df.sort_values(by='Population', ascending=False, inplace=True)
    cities_lst = merge_df['City'].head(5).tolist()
    #################################################

    log("QUESTION 6", output_df=None, other=cities_lst)
    return cities_lst


def question_7(df2):
    """
    :param df2: the dataframe created in question 2
    :return: df7
            Data Type: Dataframe
            Please read the assignment specs to know how to create the output dataframe
    """

    #################################################
    # Your code goes here ...
    def get_df(cities):
        for city in cities.split('|||'):
            a_lst.append(json.loads(city))

    def get_combine(df):
        str_df = ','.join(df.values)
        if len(str_df.split(',')) > 1:
            return str_df.split(',')

    # change content in 'Cities' to dataframe, and drop duplicate rows
    a_lst = []
    df2['Cities'].apply(get_df)
    df7 = pd.DataFrame(a_lst)
    df7.drop_duplicates(subset=['Country', 'Latitude', 'Longitude'], inplace=True)
    df7.drop_duplicates(subset=['Country', 'City'], inplace=True)

    # find cities with same name but located in different countries
    df7 = df7.groupby(['City'])['Country'].apply(get_combine)
    df7 = df7.reset_index()
    df7.dropna(inplace=True)

    # rename and set index
    df7.rename(columns={'City': 'city', 'Country': 'countries'}, inplace=True)
    df7.set_index('city', drop=False, inplace=True)
    #################################################

    log("QUESTION 7", output_df=df7, other=df7.shape)
    return df7


def question_8(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :param continents: the path for the Countries-Continents.csv file
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    same_countries = {'"Korea, North"': 'North Korea',
                      '"Korea, South"': 'South Korea',
                      'US': 'United States',
                      'Russian Federation': 'Russia',
                      'Congo': 'Republic of the Congo',
                      '"Congo, Democratic Republic of"': 'Democratic Republic of the Congo'}

    def get_population(cities):
        lst = []
        for city in cities.split('|||'):
            lst.append(json.loads(city))
        df = pd.DataFrame(lst)
        df.drop_duplicates(subset=['Country', 'Latitude', 'Longitude'], inplace=True)
        return df['Population'].sum()

    # compute percentage of the world population in each country
    df2['country_population'] = df2['Cities'].apply(get_population)
    total_population = df2['country_population'].sum()
    df2['country_population'] = df2['country_population'].apply(lambda x: x/total_population)
    country_population = df2['country_population']
    country_population = country_population.reset_index()

    # find country in South America
    continents = pd.read_csv(continents)
    continents.replace(same_countries, inplace=True)
    continents = continents.loc[continents['Continent'].isin(['South America'])]

    # contains: continents, country, population
    merge_df = pd.merge(country_population, continents, on='Country')
    merge_df.set_index('Country', inplace=True)

    # add a row to represent other continents (except countries in South America) and draw pie chart
    merge_df.loc['Other continents'] = [1 - merge_df['country_population'].sum(), 'South America']
    merge_df['country_population'].plot.pie(autopct='%.2f', fontsize=7, figsize=(10, 10))
    #################################################

    plt.savefig("{}-Q11.png".format(studentid))


def question_9(df2):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    """

    #################################################
    # Your code goes here ...
    def drop_x(c):
        df.drop(df[c == 'x'].index, inplace=True)

    df = df2[['Income classification according to WB', 'Covid_19_Economic_exposure_index_Ex_aid_and_FDI',
              'Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import',
              'Foreign direct investment, net inflows percent of GDP', 'Foreign direct investment']]

    # preprocess data: remove x and convert data x,y into float x.y
    drop_x(df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'])
    drop_x(df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'])
    drop_x(df['Foreign direct investment, net inflows percent of GDP'])
    drop_x(df['Foreign direct investment'])
    df = df.applymap(lambda x: x.replace(',', '.'))
    df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'] = pd.to_numeric(
        df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI'])
    df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'] = pd.to_numeric(
        df['Covid_19_Economic_exposure_index_Ex_aid_and_FDI_and_food_import'])
    df['Foreign direct investment, net inflows percent of GDP'] = pd.to_numeric(
        df['Foreign direct investment, net inflows percent of GDP'])
    df['Foreign direct investment'] = pd.to_numeric(df['Foreign direct investment'])

    # draw bar chart
    df = df.groupby('Income classification according to WB').mean()
    df.plot.bar(figsize=(15, 10))
    #################################################

    plt.savefig("{}-Q12.png".format(studentid))


def question_10(df2, continents):
    """
    :param df2: the dataframe created in question 2
    :return: nothing, but saves the figure on the disk
    :param continents: the path for the Countries-Continents.csv file
    """

    #################################################
    # Your code goes here ...
    same_countries = {'"Korea, North"': 'North Korea',
                      '"Korea, South"': 'South Korea',
                      'US': 'United States',
                      'Russian Federation': 'Russia',
                      'Congo': 'Republic of the Congo',
                      '"Congo, Democratic Republic of"': 'Democratic Republic of the Congo'}

    def get_population(cities):
        lst = []
        for city in cities.split('|||'):
            lst.append(json.loads(city))
        df = pd.DataFrame(lst)
        df.drop_duplicates(subset=['Country', 'Latitude', 'Longitude'], inplace=True)
        return df['Population'].sum()

    def scatter_plot(continent):
        x = continent['avg_longitude'].tolist()
        y = continent['avg_latitude'].tolist()
        area = continent['country_population'].apply(lambda x: math.sqrt(x)/ 80).tolist()
        return x, y, area

    # merge_df contains: Country, avg_latitude, avg_longitude, continent, population
    continents = pd.read_csv(continents)
    continents.replace(same_countries, inplace=True)
    df2['country_population'] = df2['Cities'].apply(get_population)
    location_population = df2[['avg_latitude', 'avg_longitude', 'country_population']]
    location_population = location_population.reset_index()
    merge_df = pd.merge(location_population, continents, on='Country')

    # get content based on different continents
    # used to draw figure with legend and different color
    africa = merge_df[merge_df['Continent'] == 'Africa']
    asia = merge_df[merge_df['Continent'] == 'Asia']
    europe = merge_df[merge_df['Continent'] == 'Europe']
    north_america = merge_df[merge_df['Continent'] == 'North America']
    oceania = merge_df[merge_df['Continent'] == 'Oceania']
    south_america = merge_df[merge_df['Continent'] == 'South America']

    # set the size of figure
    plt.figure(figsize=(15, 10))

    # scatter
    x, y, area = scatter_plot(africa)
    p1 = plt.scatter(x, y, s=area, c='r')
    x, y, area = scatter_plot(asia)
    p2 = plt.scatter(x, y, s=area, c='g')
    x, y, area = scatter_plot(europe)
    p3 = plt.scatter(x, y, s=area, c='y')
    x, y, area = scatter_plot(north_america)
    p4 = plt.scatter(x, y, s=area, c='b')
    x, y, area = scatter_plot(oceania)
    p5 = plt.scatter(x, y, s=area, c='orange')
    x, y, area = scatter_plot(south_america)
    p6 = plt.scatter(x, y, s=area, c='m')

    # set legend and labels
    plt.legend([p1, p2, p3, p4, p5, p6], ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'])
    plt.xlabel('avg_longitude')
    plt.ylabel('avg_latitude')
    #################################################

    plt.savefig("{}-Q13.png".format(studentid))


if __name__ == "__main__":
    df1 = question_1("exposure.csv", "Countries.csv")
    df2 = question_2(df1.copy(True))
    df3 = question_3(df2.copy(True))
    df4 = question_4(df2.copy(True), "Countries-Continents.csv")
    df5 = question_5(df2.copy(True))
    lst = question_6(df2.copy(True))
    df7 = question_7(df2.copy(True))
    question_8(df2.copy(True), "Countries-Continents.csv")
    question_9(df2.copy(True))
    question_10(df2.copy(True), "Countries-Continents.csv")