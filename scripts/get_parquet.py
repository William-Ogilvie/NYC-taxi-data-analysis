import requests
from bs4 import BeautifulSoup
import re
from pathlib import Path

"""
    This script will create the folder data/raw if it doesn't exist already
    It will then crawl the nyc.gov website using beatiful soup to find the download links to all of the parquet files we need
    It will save these links to parquet_files.txt which can be downloaded using the bash command below (i.e. in WSL if on windows)
"""


# TODO Add data dictionary download


# scripts/ is location of current file so we go one above to get project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Save dir
SAVE_DIR_LOC = PROJECT_ROOT / "data" / "raw"

# Create the directory if it doesn't already exist
SAVE_DIR_LOC.mkdir(parents= True, exist_ok= True)

# Fetch page
url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
resp = requests.get(url)
html = resp.text

# Parse HTML
soup = BeautifulSoup(html, "html.parser")

# Extract all <a> tags
links = [a['href'] for a in soup.find_all("a", href = True)]

# Filter with regex
pattern = re.compile(
    r"yellow[_a-z]*?20(1[1-9]|2[0-5])-\d{2}\.parquet"
)
yellow_links = [link for link in links if pattern.search(link)]

# Save the links to a txt, then use bash to download
with open("parquet_files.txt", "w") as f:
    for link in yellow_links:
        f.write(link + "\n")

# In bash:

# If it doesn't exist already make the data/raw directory
# mkdir -p ../data/raw

# wget -c -i parquet_files.txt -P ../data/raw
# -c: resume interupted downloads
# -i: input file
# -P: directory prefix

# I wasn't able to find any checksums to confirm successfull download 
