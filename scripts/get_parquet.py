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
print(yellow_links)
# # Download files and save into data dir
# SAVE_DIR = "../data/raw"
# os.makedirs(SAVE_DIR, exist_ok=True) 

# for link in yellow_links:
#     filename = link.split("/")[-1]
#     dest = os.path.join(SAVE_DIR, filename)
#     print(f"Downloading {filename}...")

#     r = requests.get(link)
#     with open(dest, "wb") as f:
#         f.write(r.content)


