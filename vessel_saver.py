import os
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime

# Function to load all JSON files from a directory
def load_vessel_data(folder_path):
    vessel_data = []
    # Assuming files are named by timestamp e.g. "2025-01-17_17_18_34.json"
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".json"):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data = json.load(file)
                timestamp_list = []
                x_coords = []
                y_coords = []
                speed = []
                for entry in data:
                    timestamp = entry['timeStamp']
                    x_coord = entry['x_coord']
                    y_coord = entry['y_coord']
                    timestamp_list.append(timestamp)
                    speeds = entry['speed']
                    x_coords.append(x_coord)
                    y_coords.append(y_coord)
                    speed.append(speeds)
                vessel_data.append((timestamp_list, x_coords, y_coords,speed))

    # print(vessel_data)
    return vessel_data

# # Function to create the animation
# def animate(i, vessel_data, scat):
#     # Unpack the data (timestamp_list, x_coords, y_coords)
#     timestamp_list, x_coords, y_coords = vessel_data[i]
#     # Update the scatter plot with new vessel positions
#     scat.set_offsets(list(zip(x_coords, y_coords)))

# # Load vessel data from the folder
# folder_path = 'filtered_vessels'  # Replace with your folder path
# vessel_data = load_vessel_data(folder_path)

# # Set up the figure and axis for the plot
# fig, ax = plt.subplots()
# ax.set_xlim(0, 100)  # Adjust x-axis limit as needed
# ax.set_ylim(0, 100)  # Adjust y-axis limit as needed
# scat = ax.scatter([], [], s=10)  # Create an empty scatter plot

# # Create the animation
# ani = animation.FuncAnimation(fig, animate, fargs=(vessel_data, scat), frames=len(vessel_data), interval=100, repeat=False)

def get_obstacles(vessel_data):
    ob = []
    for i in vessel_data:
        for j in range(len(i[1])):
            ob.append([i[1][j],i[2][j]])

    return ob


# print(ob)
# Show the plot
plt.show()
