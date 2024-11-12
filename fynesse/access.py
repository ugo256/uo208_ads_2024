from .config import *

import requests
import pymysql
import csv
import math
import pandas as pd


"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError

def hello_world():
  print("Hello from the data science library!")

def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)

def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn

def housing_upload_join_data(conn, year):
    start_date = str(year) + "-01-01"
    end_date = str(year) + "-12-31"

    cur = conn.cursor()
    print('Selecting data for year: ' + str(year))
    cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
    rows = cur.fetchall()

    csv_file_path = 'output_file.csv'

    # Write the rows to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write the data rows
        csv_writer.writerows(rows)
    print('Storing data for year: ' + str(year))
    cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
    conn.commit()
    print('Data stored for year: ' + str(year))


def get_bounding_box(latitude: float, longitude: float, distance_km: float = 1.0) -> dict[str,float]:
    box_width = distance_km/(40075*math.cos(math.radians(latitude)))*360
    box_height = distance_km/(40075/360)
    north = latitude + box_height/2
    south = latitude - box_height/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    return {"north":north,
            "east":east,
            "south":south,
            "west":west}

def get_pois(bbox,tags):
  return ox.geometries_from_bbox(bbox["north"], bbox["south"], bbox["east"], bbox["west"], tags)

def get_pois_df(bbox:dict,tags:dict) -> pd.DataFrame:
  pois = get_pois(bbox,tags)
  pois_df = pd.DataFrame(pois)
  pois_df['latitude'] = pois_df.apply(lambda row: row.geometry.centroid.y, axis=1)
  pois_df['longitude'] = pois_df.apply(lambda row: row.geometry.centroid.x, axis=1)
  return pois_df

def get_houses_with_transactions(conn,place_name,latitude,longitude):
    cur = conn.cursor()
    cur.execute(f"select * from `pp_data` where town_city = '{place_name.upper()}'")
    data = cur.fetchall()
    columns = [desc[0] for desc in cur.description]
    pp_df = pd.DataFrame(data, columns=columns)

    
    pp_df["streetname"]=pp_df["street"].str.lower()

    bbox = get_bounding_box(latitude,longitude)
    pois = get_pois(bbox,{"building":True})

    north=bbox["north"]
    east=bbox["east"]
    south=bbox["south"]
    west=bbox["west"]

    keys = ["addr:housenumber", "addr:street", "addr:postcode","geometry","longitude","latitude"]

    present_keys = [key for key in keys if key in pois.columns]
    all_pois=pois[present_keys]
    all_pois["streetname"]=all_pois["addr:street"].str.lower()
