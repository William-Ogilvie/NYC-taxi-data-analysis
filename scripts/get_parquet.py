"""
get_parquet
=============

This script will scrape the NYC Taxi and Limousine Commission (TLC) trip record data page
to get the links for all the yellow taxi parquet files from 2011 to 2025, the taxi zone
lookup file and a zip file containing the shapefiles for the taxi zones.

You will then need to run the bash script download_and_extract.sh to download and extract
all the files. Please read the .sh file for instructions on how to run it.

Note this script should be run after setup.py to ensure all directories are created.
"""

# --- Imports ---
import requests
from bs4 import BeautifulSoup
import re

# Fetch page
url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
resp = requests.get(url)
html = resp.text

# Parse HTML
soup = BeautifulSoup(html, "html.parser")

# Extract all <a> tags
links = [a['href'] for a in soup.find_all("a", href = True)]

# Filter with regex for yellow taxi parquet files from 2011 to 2025
pattern = re.compile(
    r"yellow[_a-z]*?20(1[1-9]|2[0-5])-\d{2}\.parquet"
)
yellow_links = [link for link in links if pattern.search(link)]

# Filter with regex for taxi zone lookup file
pattern = re.compile(
    r"taxi_zone_lookup.csv"
)
zone_lookup_link = [link for link in links if pattern.search(link)]

# Filter with regex for shapefile.zip
pattern = re.compile(
    r"taxi_zones.zip"
)
shapefile_link = [link for link in links if pattern.search(link)]

# Save the links to a txt, then use bash to download
with open("parquet_files.txt", "w") as f:
    for link in yellow_links:
        f.write(link + "\n")
    for link in zone_lookup_link:
        f.write(link + "\n")
    for link in shapefile_link:
        f.write(link + "\n")

# Read download_and_extract.sh and make it executable (instructions in the file)
