import json
import os

def convert_latlon_to_xy(lon, lat,
                         lon_min=103.60, lon_max=103.66,
                         lat_min=1.16, lat_max=1.22,
                         width=100, height=100):
    """
    Convert a longitude/latitude point (or numpy arrays of points)
    to x/y coordinates in a defined output range.

    Returns:
      x, y: the converted coordinates such that:
            lon == lon_min maps to x == 0 and lon == lon_max maps to x == width;
            lat == lat_min maps to y == 0 and lat == lat_max maps to y == height.
    """
    x = ((lon - lon_min) / (lon_max - lon_min)) * width
    y = ((lat - lat_min) / (lat_max - lat_min)) * height
    return x, y

def process_vessel_files(folder_path="filtered_vessels"):
    """
    Process all JSON files in the specified folder to add x,y coordinates
    for each vessel based on their longitude and latitude.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found")
        return

    # Process each JSON file in the folder
    for filename in os.listdir(folder_path):
        if not filename.endswith('.json'):
            continue
            
        file_path = os.path.join(folder_path, filename)
        
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process each vessel in the data
        for vessel in data:
            if 'longitudeDegrees' in vessel and 'latitudeDegrees' in vessel:
                x, y = convert_latlon_to_xy(
                    vessel['longitudeDegrees'],
                    vessel['latitudeDegrees']
                )
                # Add the new coordinates to the vessel data
                vessel['x_coord'] = float(x)  # Convert to float for JSON serialization
                vessel['y_coord'] = float(y)
        
        # Save the updated data back to the file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

# Example usage:
if __name__ == "__main__":

    process_vessel_files()


