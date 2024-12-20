from fynesse import utils
from .config import *

from . import access
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
import pandas as pd

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

def get_counts_around(lat,lon,dist,tags):
    bbox = fynesse.utils.get_bounding_box(lat,lon,dist)
    matches = db.query(f"""select tag_key from england_osm_geo
    inner join england_osm_tags on england_osm_tags.id = england_osm_geo.id
    where lon between {bbox['west']} and {bbox['east']} and lat between {bbox['south']} and {bbox['north']};""")
    if tags:
        matches = matches[matches["tag_key"].isin(tags)]
    return matches.value_counts()

def get_all_counts_around(locations,dist,tags):
    ans = []
    for i,row in locations.iterrows():
        lat = row["lat"]
        lon = row["lon"]
        counts = get_counts_around(lat,lon,dist,tags)
        for tag in tags:
            if tag not in counts:
                counts[tag]=0
        counts.rename(i,inplace=True)
        ans.append(counts)
    return pd.concat(ans,axis=1).transpose()

def plot_surroundings(lat,lon,dist,colormap={},title="OSM Visualization Near {lat}N, {lon}E"):
    bbox = fynesse.utils.get_bounding_box(lat,lon,dist)
    matches = db.query(f"""select england_osm_geo.id, ST_AsText(geometry) as geometry, tag_key from england_osm_geo
    inner join england_osm_tags on england_osm_tags.id = england_osm_geo.id
    where lon between {bbox['west']} and {bbox['east']} and lat between {bbox['south']} and {bbox['north']};""")

    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    pois = matches[["id","geometry"]].drop_duplicates()

    tags = {}
    for i,row in matches.iterrows():
        if not row["id"] in tags:
            tags[row["id"]]=[]
        tags[row["id"]].append(row["tag_key"])

    node_points = []
    way_lines = []

    for i,row in pois.iterrows():
        geometry_wkt = row["geometry"]
        shp = load_wkt(geometry_wkt)
        if isinstance(shp, Point):
            node_points.append((row["id"],shp))
        if isinstance(shp, LineString):
            way_lines.append((row["id"],shp))

    for id,point in node_points:
        color = "#0000ff"
        for tag_key in tags[id]:
            if tag_key in colormap:
                color = colormap[tag_key]
                break
        ax.plot(point.x, point.y, 'o', color=color, markersize=1)

    for id,line in way_lines:
        color = "#000000"
        for tag_key in tags[id]:
            if tag_key in colormap:
                color = colormap[tag_key]
                break
        if line.is_ring:
            # draw as polygon if the way is closed
            polygon = Polygon(line)
            x, y = polygon.exterior.xy
            ax.fill(x, y, color=color, alpha=0.5)
        else:
            x, y = line.xy
            ax.plot(x, y, '-', color=color, linewidth=0.5)

    plt.title(title.format(lat=lat,lon=lon))
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.xlim(bbox["west"],bbox["east"])
    plt.ylim(bbox["south"],bbox["north"])
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

def plot_correlation(x,y,method='pearson',xlabel = "x",ylabel = "y",xlog=False,ylog=False):
    correlation = (x.corr(y,method=method))
    print(f"The correlation ({method}) is {correlation}")
    print("plotting graph:")

    plt.scatter(x, y, color="blue", alpha=0.7)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"Scatter Plot of {ylabel} against {xlabel}")
    if ylog:
        plt.yscale("log")
    if xlog:
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

def create_histogram_subplots(data, labels,bins=10, figsize=(15, 10),cols=3):

    rows = (len(data) + cols-1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)
    if cols!=1:
        axes = axes.flatten()

    for i, column in enumerate(data):
        axes[i].hist(column.dropna(), bins=bins, color='skyblue', edgecolor='black')
        axes[i].set_title(f"Histogram of {labels[i]}")
        axes[i].set_xlabel(labels[i])
        axes[i].set_ylabel("Frequency")
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.show()

def perform_pca(features):
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)
    pca = PCA(n_components=None)
    principal_components = pca.fit_transform(standardized_features)
    pc_df = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(principal_components.shape[1])])
    plt.figure(figsize=(8, 5))
    x_values = range(1, len(pca.explained_variance_ratio_) + 1)
    y_values = pca.explained_variance_ratio_.cumsum()
    # add (0, 0)
    x_values = [0] + list(x_values)
    y_values = [0] + list(y_values)
    plt.plot(x_values,y_values, marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by Principal Components")
    plt.grid()
    plt.show()
    loadings = pd.DataFrame(
    pca.components_,
    columns=features.columns,
    index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(loadings.T, annot=True, cmap='coolwarm', fmt='.2f', center=0)

    plt.title('Heatmap of PCA Loadings')
    plt.xlabel('Principal Components')
    plt.ylabel('Variables')

    plt.show()
    return pc_df

def hierarchical_clustering(df,num_clusters,method='ward',title="Hierarchical Clustering Dendrogram"):

    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df)

    linkage_matrix = linkage(normalized_data, method=method)

    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=df.index, leaf_rotation=90)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    cluster_counts = pd.Series(cluster_labels).value_counts()
    print(cluster_counts)

    return cluster_labels

def count_duplicates_sql(db,table_name,cols):
    df = db.query(f"""
                    select
                        {" ".join(cols)}
                    from
                        {table_name}
                    """)
    return count_duplicates(df,cols)

def count_duplicates(df,cols):
    ans = df.groupby(cols).size().reset_index(name='count')
    return ans[ans['count']>1]

def group_columns(df: pd.DataFrame,columns: dict[str,list[str]]) -> pd.DataFrame:
    lookup_dict = {
        col: {
            colval: pd.Series(df[colval].values, index=df[col]).to_dict() for colval in columns[col]
        }
        for col in columns
    }

    group_nodes = [set(df[col]) for col in columns]
    G = nx.Graph()
    G.add_nodes_from([(col, val) for i, col in enumerate(columns) for val in group_nodes[i]])
    for index, row in df.iterrows():
        for a, b in zip(list(columns.keys()), list(columns.keys())[1:]):
            G.add_edge((a,row[a]),(b,row[b]))

    Gcc = nx.connected_components(G)

    aggregated_data = []
    for component in list(Gcc):
        aggregated_keys = {
            col: ' '.join(sorted([node[1] for node in component if node[0]==col])) for col in columns
        }
        aggregated_vals = {
            colval: sum(lookup_dict[col][colval][node[1]] for node in component if node[0] == col)
            for col, colvals in columns.items()
            for colval in colvals
        }
        aggregated_row = aggregated_keys | aggregated_vals
        aggregated_data.append(aggregated_row)

    return pd.DataFrame(aggregated_data)



def clustered_scatter_plot(df,x,y,alpha,xlabel,ylabel,title):

  plt.figure(figsize=(10, 6))
  sns.scatterplot(x=x, y=y, hue='cluster', data=df, palette='Set2',alpha=alpha)

  plt.title(title, fontsize=14)
  plt.xlabel(xlabel, fontsize=12)
  plt.ylabel(ylabel, fontsize=12)
  plt.legend(title='Cluster', loc='best')
  plt.grid(True)

  plt.show()

clustered_scatter_plot(grouped_occupancy,"11_oci","1121_change_oci",0.8,"x","y","title")