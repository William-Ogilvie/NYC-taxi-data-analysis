# NYC Taxi Data Analysis
Exploring and visualising New York City Taxi trip data (2024). 

## Objective
Explain what the data is and why it's interesting.
Mention the timeframe (2024) and key goals (visulaizations, forecasting, etc)

## Data Sources

Data source: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page), 
via [NYC Open Data](https://opendata.cityofnewyork.us/).  
© City of New York. Data made available under the NYC Open Data Terms of Use.

## Usage

Clone the repository:
```bash
git clone https://github.com/William-Ogilvie/NYC-taxi-data-analysis.git
cd NYC-taxi-data-analysis
```

- The data you will need are the Taxi Zone Shapefile (PARQUET), Taxi Zone Lookup Table (CSV) and the Yellow Taxi Trip Records (PARQUET) for Januaray 2025. (change once add more data)
- All of which can be downloaded from [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).
- Extract any zip files and place all data in a data/raw directory.

To run the notebook you will need to install the dependencies in requirements.txt. Create a virtual enviroment and install all dependencies:

```bash
pyton -m venv venv
source venv/bin/activate   # on MAC/Linux
venv/Scripts/activate      # On Windows
```
```bash
pip install -r requirements.txt
```

You will now be able to run the Notebooks. Note any output will be stored in a reports directory. 

```bash
jupyter notebook
```

## Reults / Key Findings
