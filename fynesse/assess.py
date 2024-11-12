from .config import *

from . import access
import matplotlib.pyplot as plt

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
