import os
import json
import shutil

def filter_vessel_files(source_folder, destination_folder,
                          lon_min, lon_max, lat_min, lat_max,
                          min_length=None, max_length=None,
                          start_time=None, end_time=None):

    # Delete destination folder if it exists, then create a fresh one
    if os.path.exists(destination_folder):
        shutil.rmtree(destination_folder)
    os.makedirs(destination_folder)

    saved_files = []

    # Process each file in the source folder.
    for filename in os.listdir(source_folder):
        if not filename.lower().endswith('.json'):
            continue  # Skip non-json files

        file_path = os.path.join(source_folder, filename)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        filtered_records = []

        if isinstance(data, list):
            for vessel in data:
                if not isinstance(vessel, dict):
                    continue

                # Get all required fields
                lon = vessel.get("longitudeDegrees")
                lat = vessel.get("latitudeDegrees")
                vessel_length = vessel.get("vesselLength")
                timestamp = vessel.get("timeStamp")

                # Debug prints
                print(f"\nChecking vessel in {filename}:")
                print(f"Location: ({lon}, {lat})")
                print(f"Length: {vessel_length}")
                print(f"Timestamp: {timestamp}")

                # Skip if missing required data
                if lon is None or lat is None:
                    print("Skipping: Missing coordinates")
                    continue

                # Check coordinates
                if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                    print("Skipping: Outside geographic bounds")
                    continue

                # Check vessel length
                if vessel_length is not None:
                    if min_length is not None and vessel_length < min_length:
                        print(f"Skipping: Length {vessel_length} below minimum {min_length}")
                        continue
                    if max_length is not None and vessel_length > max_length:
                        print(f"Skipping: Length {vessel_length} above maximum {max_length}")
                        continue

                # Check timestamp
                if timestamp is not None:
                    if start_time is not None and timestamp < start_time:
                        print(f"Skipping: Timestamp {timestamp} before start time {start_time}")
                        continue
                    if end_time is not None and timestamp > end_time:
                        print(f"Skipping: Timestamp {timestamp} after end time {end_time}")
                        continue

                print("Vessel passed all filters!")
                filtered_records.append(vessel)

        elif isinstance(data, dict):
            # If the file contains a single vessel record (as a dict)
            lon = data.get("longitudeDegrees")
            lat = data.get("latitudeDegrees")
            VL = data.get("vesselLength")
            if lon is not None and lat is not None and VL is not None:
                if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max and VL >= 100.0:
                    filtered_records.append(data)

        else:
            print(f"File {filename} has unexpected JSON structure.")
            continue

        # If we found at least one matching record, write them to the destination folder.
        if filtered_records:
            dest_file_path = os.path.join(destination_folder, filename)
            try:
                with open(dest_file_path, 'w') as f:
                    # Save as a list of matching vessel records.
                    json.dump(filtered_records, f, indent=2)
                print(f"Saved {filename} with {len(filtered_records)} vessel record(s) to destination.")
                saved_files.append(filename)
            except Exception as e:
                print(f"Error writing {filename} to destination: {e}")
        else:
            print(f"{filename} has no vessel records in the region.")

    return saved_files

# Example usage (for testing purposes):
if __name__ == "__main__":
    # Change these paths as needed.
    source_folder = "vessel_data"
    destination_folder = "filtered_vessels"
    # Define the region boundaries.
    # For example, for a region of interest in the Singapore Straits:
    lon_min, lon_max = 103.75, 103.85
    lat_min, lat_max = 1.13, 1.22
    min_length = 180.0

    saved = filter_vessel_files(source_folder, destination_folder,
                                lon_min, lon_max, lat_min, lat_max, min_length)
    print("Files saved:", saved)
