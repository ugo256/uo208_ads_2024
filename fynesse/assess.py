from .config import *

from . import access
import matplotlib.pyplot as plt
import osmnx as ox

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def plot_city(place_name,latitude,longitude):

    bbox = access.get_bounding_box(latitude,longitude)
    pois = access.get_pois(bbox,{"building":True})

    north=bbox["north"]
    east=bbox["east"]
    south=bbox["south"]
    west=bbox["west"]

    keys = ["addr:housenumber", "addr:street", "addr:postcode","geometry","longitude","latitude"]

    present_keys = [key for key in keys if key in pois.columns]

    all_pois = pois[present_keys]
    all_pois["area"]=all_pois["geometry"].area

    valid_pois = all_pois.dropna(how="any")
    invalid_pois = all_pois[all_pois.isna().any(axis=1)]

    graph = ox.graph_from_bbox(north, south, east, west)

    # Retrieve nodes and edges
    nodes, edges = ox.graph_to_gdfs(graph)

    # Get place boundary related to the place name as a geodataframe
    area = ox.geocode_to_gdf(place_name)

    fig, ax = plt.subplots()

    # Plot the footprint
    area.plot(ax=ax, facecolor="white")

    # Plot street edges
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot all POIs
    valid_pois.plot(ax=ax, color="blue", alpha=0.7, markersize=10)
    invalid_pois.plot(ax=ax, color="red", alpha=0.7, markersize=10)
    plt.tight_layout()

def plot_correlation(merged_df,method='pearson',log=False):
    correlation = (merged_df["area"].corr(merged_df["price"],method=method))
    print(f"The correlation ({method}) is {correlation}")
    print("plotting graph:")

    plt.scatter(merged_df["area"], merged_df["price"], color="blue", alpha=0.7)

    plt.xlabel("Area (sq km)")
    plt.ylabel("Price (£)")
    plt.title("Scatter Plot of Area vs. Price")
    if log:
        plt.yscale("log")
        plt.xscale("log")

    plt.show()

def create_boxplot_subplots(data_list, rows, cols, titles=None, figsize=(10, 6)):

    if len(data_list) > rows * cols:
        raise ValueError("Not enough subplots for the number of datasets.")

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    for i, data in enumerate(data_list):
        axes[i].boxplot(data)
        if titles and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].set_ylabel("Values")

    for j in range(len(data_list), rows * cols):
        axes[j].axis("off")
    
    plt.show()

def create_histogram_subplots(data, columns, bins=10, figsize=(15, 10)):

    num_columns = len(columns)
    rows = (num_columns + 2) // 3
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    axes = axes.flatten()

    for i, column in enumerate(columns):
        axes[i].hist(data[column].dropna(), bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f"Histogram of {column}")
        axes[i].set_xlabel(column)
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()



def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError
