"""
map_generator.py

This module provides functions to:
  - Generate and save a map image using Cartopy and Google Tiles.
  - Export the coordinates of the land borders (i.e. coastlines outlining land areas)
    as a list into a text file.
  
When the Python process exits, the saved map image file ("maps.png" by default)
is automatically removed.
"""

import os
import atexit
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles

# Default filenames for the saved map image and border coordinates.

_DEFAULT_MAP_FILENAME = "map.png"
_DEFAULT_BORDERS_FILENAME = "land_borders.txt"


def generate_map(lat_min, lat_max, lon_min, lon_max, zoom=12, filename=_DEFAULT_MAP_FILENAME):

    # Create a GoogleTiles instance (this defines the tiling service and its projection)
    tiler = GoogleTiles()

    # Create the figure and axes using the tiler's coordinate reference system.
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection=tiler.crs)

    # Set the extent. Note that set_extent expects [lon_min, lon_max, lat_min, lat_max],
    # and the provided coordinates are assumed to be in degrees (PlateCarree).
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add the Google tile imagery. The zoom level controls the resolution of the tiles.
    ax.add_image(tiler, zoom)

    # Save the figure to the specified file.
    plt.savefig(filename, bbox_inches="tight")
    plt.close(fig)  # Close the figure to free up memory
    print(f"Map image saved to '{filename}'.")


if __name__ == "__main__":

    test_lat_min = 1.13
    test_lat_max = 1.22
    test_lon_min = 103.75
    test_lon_max = 103.85

    generate_map(test_lat_min, test_lat_max, test_lon_min, test_lon_max)

# singapore lat lon = 1.0 - 1.5, 103.4 - 104.1



