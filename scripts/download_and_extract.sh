#!/bin/bash
wget -c -i parquet_files.txt -P ../data/raw # download all files in parquet_files.txt to data/raw
for f in ../data/raw/*.zip; do # loop through all zip files in the directory
    [ -e "$f" ] || continue # if no zip files, skip to end of loop
    unzip -o "$f" -d ../data/raw # unzip the file to the same directory
    rm "$f"  # Remove the zip file after extraction
done

# -c resume interrupted downloads
# -i read URLs from a file
# -P specify download directory
# -e exists check if file exists
# -o overwrite existing files when extracting
# -d specify extraction directory
# [] are just for test expressions

# To make this script executable in wsl (or if on mac/linux just terminal):
# chmod +x download_and_extract.sh
# Then run it with:
# ./download_and_extract.sh

# chmod is change mode to change permissions of the file
# +x is to add execute permission
# ./ is to run a script in the current directory