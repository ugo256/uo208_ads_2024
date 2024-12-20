from .config import *

import requests
import pymysql
import csv
import math
import pandas as pd
import yaml
from ipywidgets import interact_manual, Text, Password

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


@interact_manual(username=Text(description="Username:"),
                password=Password(description="Password:"),
                url=Text(description="URL:"),
                port=Text(description="Port:"))
def write_credentials(username, password, url, port):
    with open("credentials.yaml", "w") as file:
        credentials_dict = {'username': username,
                           'password': password,
                           'url': url,
                           'port': port}
        yaml.dump(credentials_dict, file)

legal_information = """
Legal Information:

[Census Data and Lookups]
    Source: Office for National Statistics licensed under the Open Government Licence v.3.0

[OpenStreetMap (OSM) Data]
    © OpenStreetMap contributors
    License: www.openstreetmap.org/copyright

"""

class DatabaseConnection:
    def __init__(self):
        self.conn=None
        username,password,url,port = self._read_credentials()
        self._connect(username,password,url,'ads_2024',port)
    
        
    def _read_credentials(self):
        with open("credentials.yaml") as file:
            credentials = yaml.safe_load(file)
        username = credentials["username"]
        password = credentials["password"]
        url = credentials["url"]
        port = credentials["port"]
        return username,password,url,int(port)
    

    def _connect(self, user, password, host, database, port=3306):
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
        self.conn = conn

    def query(self,query,as_df=True):
        cur = self.conn.cursor()
        cur.execute(query)
        if as_df:
            return pd.DataFrame(cur.fetchall(),columns = [desc[0] for desc in cur.description])
        


def count_nulls(db,table_name):
    cur = db.conn.cursor()
    cur.execute(f"select * from {table_name} limit 0;")
    return db.query("select "+ ", ".join([f"sum(case when {desc[0]} is null then 1 else 0 end) as nullcnt_{desc[0]}" for desc in cur.description])+f" from {table_name};")

def load_into_table(db,filename,tablename,ignorelines,fields,delimiter=',',enclosure='"',terminator='\n'):
    db.query(f"""LOAD DATA LOCAL INFILE '{filename}'
    INTO TABLE {tablename}
    FIELDS TERMINATED BY '{delimiter}'
    OPTIONALLY ENCLOSED BY '{enclosure}'
    LINES TERMINATED BY '{terminator}'
    IGNORE {ignorelines} LINES
    {fields};""",False)

osm_tag_colnames = ['id', 'type', 'tagkey', 'tagvalue']
osm_geo_colnames = ['id', 'type','lat','lon','geometry']

save_limit=5000000

def is_interesting(poi):
    if len(poi.tags) >= 4:
        return True
    for tag in poi.tags:
        if tag == "addr:street":
            return True
    return False

class OSMHandler():
    def __init__(self):
        osmium.SimpleHandler.__init__(self)
        self.osm_tag = []
        self.cnt_tag=0
        self.osm_geo = []
        self.cnt_geo=0

    def save(self,final=False):

        update=False

        if len(self.osm_tag) % 100000 <= 10:
            print(len(self.osm_tag))

        if final or len(self.osm_tag) > save_limit:
            pd.DataFrame(self.osm_tag, columns=osm_tag_colnames).to_csv(f"england_tag_{self.cnt_tag}.csv")
            self.cnt_tag+=1
            self.osm_tag = []
            update=True

        if final or len(self.osm_geo) > save_limit:
            pd.DataFrame(self.osm_geo, columns=osm_geo_colnames).to_csv(f"england_geo_{self.cnt_geo}.csv")
            self.cnt_geo+=1
            self.osm_geo = []
            update=True

        if update:
            print(f"tag: {self.cnt_tag} geo: {self.cnt_geo}")




    def node(self, n):
        if not is_interesting(n):
            return
        if not n.location:
            return
        wkt = f"POINT({n.location.lon} {n.location.lat})"
        c_lat = n.location.lat
        c_lon = n.location.lon
        self.osm_geo.append([n.id, 'node', c_lat, c_lon, wkt])
        self.osm_tag.extend([[n.id, 'node', tag.k, tag.v] for tag in n.tags])
        self.save()

    def way(self, w):
        if not is_interesting(w):
            return
        nodes = [(node.lon,node.lat) for node in w.nodes if node.location.valid()]
        if len(nodes)<2:
            return

        c_lat = 0
        c_lon = 0

        if nodes[0] == nodes[-1]:
            area = 0
            for i in range(len(nodes) - 1):  # Skip the last duplicate node
                x1, y1 = nodes[i][0], nodes[i][1]  # lon, lat
                x2, y2 = nodes[i + 1][0], nodes[i + 1][1]
                step = x1 * y2 - x2 * y1
                area += step
                c_lon += (x1 + x2) * step
                c_lat += (y1 + y2) * step

            area *= 0.5
            if area == 0:  # degenerate case (e.g., straight line)
                c_lon = nodes[0][0]
                c_lat = nodes[0][1]

            c_lon /= (6 * area)
            c_lat /= (6 * area)
        else:  # Open way (LineString)
            c_lon = sum(lon for lon, lat in nodes) / len(nodes)
            c_lat = sum(lat for lon, lat in nodes) / len(nodes)



        wkt = "LINESTRING(" + ", ".join(f"{n[0]} {n[1]}" for n in nodes) + ")"
        self.osm_geo.append([w.id, 'way', c_lat, c_lon, wkt])
        self.osm_tag.extend([[w.id, 'way', tag.k, tag.v] for tag in w.tags])

        self.save()

    def relation(self, r):
        if len(r.tags) <= 1:
          return
        self.osm_tag.extend([[r.id, 'relation', tag.k, tag.v] for tag in r.tags])
        self.save()

def upload_tags():
    db.query("""
    CREATE TABLE IF NOT EXISTS `england_osm_tags` (
        `id` bigint unsigned not null,
        `type` enum('node','way','relation') NOT NULL,
        `tag_key` varchar(255),
        `tag_value` varchar(255)
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
    """,False)

    for i in range(0,11):
        print(f"starting {i}")
        db.query(f"""LOAD DATA LOCAL INFILE 'england_tag_{i}.csv'
        INTO TABLE england_osm_tags
        FIELDS TERMINATED BY ','
        OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\n'
        IGNORE 1 LINES
        (@dummy,id,type,tag_key,tag_value);""",False)
        print(f"finshed {i}")

    db.conn.commit()

def upload_geo_osm():
    db.query("drop table if exists `england_osm_geo`",False)

    db.query("""
    CREATE TABLE IF NOT EXISTS `england_osm_geo` (
    `id` bigint unsigned NOT NULL,
    `type` enum('node','way','relation') NOT NULL,
    `lat` DECIMAL(9, 6),
    `lon` DECIMAL(9, 6),
    `geometry` geometry
    ) DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
    """,False)

    for i in range(0,2):
        print(f"starting {i}")
        db.query(f"""LOAD DATA LOCAL INFILE 'england_geo_{i}.csv'
        INTO TABLE england_osm_geo
        FIELDS TERMINATED BY ','
        OPTIONALLY ENCLOSED BY '"'
        LINES TERMINATED BY '\n'
        IGNORE 1 LINES
        (@dummy,id,type,lat,lon,@geometry)
        SET geometry = ST_GeomFromText(@geometry);""",False)
        print(f"finshed {i}")