import requests
from bs4 import BeautifulSoup
import re
import os


# Fetch page
url = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
resp = requests.get(url)
html = resp.text

# Parse HTML
soup = BeautifulSoup(html, "html.parser")

# Extract all <a> tags
links = [a['href'] for a in soup.find_all("a", href = True)]

# Filter with regext
pattern = re.compile(
    r"yellow[_a-z]*?20\d{2}-\d{2}\.parquet"
)
yellow_links = [link for link in links if pattern.search(link)]

# Save the links to a txt, then use bash to download
with open("parquet_files.txt", "w") as f:
    for link in yellow_links:
        f.write(link + "\n")

# In bash:

# If it doesn't exist already make the data/raw directory
# mkdir -p ../data/raw

# wget -i urls.txt -P ../data/raw


