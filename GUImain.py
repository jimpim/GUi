import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QComboBox, QHBoxLayout, QVBoxLayout, QLineEdit
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import os
import a_star
import dynamic_window_approach_paper as dwa_mod
import map_generator
import vessel_filter
import lonlat_converter
import vessel_saver
import imazudata
from datetime import datetime

def get_first_coord(coord):
    # If coord is a sequence (list, tuple, or numpy array), return its first element.
    # Otherwise, return the coord itself.
    if isinstance(coord, (list, tuple, np.ndarray)):
        return coord[0]
    return coord

class MyGui(QWidget):
    def __init__(self):
        super().__init__()

        self.setStyleSheet("""
        /* Global styles */
        QWidget {
            background-color: white;
            font-family: "Segoe UI", Arial, sans-serif;
            color: #333333;}/* QComboBox styling */
        QComboBox {
            border: 1px solid #C0D6E4;
            border-radius: 4px;
            padding: 5px 30px 5px 10px; /* Increased right padding for arrow */
            background-color: #FFFFFF;
            selection-background-color: #E6F2F9;
            min-height: 28px;
            color: #1A5276;
        }QComboBox:hover {
            border: 1px solid #3498DB;}
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 20px;
            border-left: 1px solid #C0D6E4;
            border-top-right-radius: 3px;
            border-bottom-right-radius: 3px;}QComboBox:hover::drop-down {
            border-left: 1px solid #3498DB;}QComboBox::down-arrow {image: none;width: 10px;height: 10px;background: qconicalgradient(cx:0.5, cy:0.8,angle:225,stop:0 #3498DB, stop:0.5 #3498DB, stop:0.51 rgba(0,0,0,0));}/* Button styling */
        QPushButton {
            border: 1px solid #3498DB;
            border-radius: 4px;
            padding: 6px 15px;
            background-color: #3498DB;
            color: white;
            min-height: 28px;
            font-weight: 500;
        }QPushButton:hover {
            background-color: #2E86C1;
            border: 1px solid #2874A6;
        }QPushButton:pressed {
            background-color: #1F618D;
        }/* Special numbered buttons (for Imazu) */
        [objectName^="imazu_btn_"] {background-color: #F0F8FF;color: #1A5276;border: 1px solid #AED6F1;font-weight: bold;
            min-height: 24px;min-width: 24px;padding: 3px;}[objectName^="imazu_btn_"]:hover {background-color: #D6EAF8;
            border: 1px solid #3498DB;}/* Line edit styling */QLineEdit {border: 1px solid #C0D6E4;
            border-radius: 4px;
            padding: 6px;background-color: #FFFFFF;selection-background-color: #AED6F1;min-height: 28px;}QLineEdit:focus {border: 1px solid #3498DB;}/* Label styling */QLabel {color: #2C3E50;
            font-weight: 500;}/* Canvas/map area styling */FigureCanvas {border: 1px solid #E8F1F5;background-color: #F8FBFC;}""")

        self.selected = None
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        new_width = self.width()
        new_height = self.height()

        self.original_width = 1200
        self.original_height = 700

        # Calculate scale factors
        self.scale_x = new_width / self.original_width
        self.scale_y = new_height / self.original_height

        self.min_lat = 1.13
        self.max_lat = 1.22
        self.min_lon = 103.75
        self.max_lon = 103.85

        self.original_width = 1200
        self.original_height = 700

        # Set initial window size and title
        self.setGeometry(100, 100, self.original_width, self.original_height)
        self.setWindowTitle("Welcome to CDCA testing platform")

        # Initialize scaling factors
        self.scale_x = 1.0
        self.scale_y = 1.0

        # Initialize widgets
        self.init_ui()

    def init_ui(self):
        # -- Matplotlib Figure and Canvas --
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)
        self.previous_robot_pos = None

        self.canvas.setParent(self)  # Manually manage it

        # Initialize the arrow with empty data
        self.direction_arrows = self.ax.quiver(
        [0],
        [0],
        [0],
        [0],
        angles='uv',
        scale_units='inches', #change to inches.
        scale=5, #increase scale.
        color= '#FF0000', #change color.
        headwidth=5, #increase head size.
        headlength=4, #increase head size.
        headaxislength=3, #increase head size. 
        )

        # -- Manually Positioned Right-Side Widgets --
        self.start_btn = QPushButton("Start", self)
        self.pause_btn = QPushButton("Pause/Resume", self)

        self.map_label = QLabel("Map Selection By Region Or Lon/Lat", self)
        self.map_fixed_label = QLabel("Select Map By Region", self)
        self.map_select_btn = QComboBox(self)
        self.map_select_btn.addItems(["", "Singapore Straits", "Imazu Map"])
        self.lon_label = QLabel("Enter Longitude Range", self)
        self.lon_min_input = QLineEdit(self)
        self.lon_min_input.setPlaceholderText("Min Lon")
        self.lon_max_input = QLineEdit(self)
        self.lon_max_input.setPlaceholderText("Max Lon")

        self.latlon_label = QLabel("Select Map By Lon/Lat", self)
        self.lat_label = QLabel("Enter Latitude Range", self)
        self.lat_min_input = QLineEdit(self)
        self.lat_min_input.setPlaceholderText("Min Lat")
        self.lat_max_input = QLineEdit(self)
        self.lat_max_input.setPlaceholderText("Max Lat")

        self.mode_label = QLabel("Select Mode", self)
        self.mode_select_btn = QComboBox(self)
        self.mode_select_btn.addItems(["", "AIS", "Imazu 1"])
        self.status_label = QLabel("Select map and click 'Start DWA' to begin.", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        for i in range(1, 23):
            button = QPushButton(f"{i}", self)  # Create button
            setattr(self, f"imazu_btn_{i}", button)  # Store button as attribute
            button.hide()

        # Connect signals
        self.map_select_btn.currentIndexChanged.connect(self.map_changed)
        self.lon_min_input.returnPressed.connect(self.lon_max_input.setFocus)
        self.lon_max_input.returnPressed.connect(self.lat_min_input.setFocus)
        self.lat_min_input.returnPressed.connect(self.lat_max_input.setFocus)
        self.lat_max_input.returnPressed.connect(self.map_changed)
        self.map_select_btn.currentIndexChanged.connect(self.map_changed)
        self.mode_select_btn.currentIndexChanged.connect(self.mode_changed)

        # Update widget positions and sizes
        self.update_widget_geometry()

    def update_widget_geometry(self):
        # Update canvas position and size
        self.canvas.setGeometry(int(275 * self.scale_x), int(20 * self.scale_y), int(650 * self.scale_x), int(650 * self.scale_y))
        # Update button positions and sizes
        self.start_btn.setGeometry(int(950 * self.scale_x), int(50 * self.scale_y), int(180 * self.scale_x), int(40 * self.scale_y))
        self.pause_btn.setGeometry(int(950 * self.scale_x), int(100 * self.scale_y), int(180 * self.scale_x), int(40 * self.scale_y))
        self.map_label.setGeometry(int(50 * self.scale_x), int(55 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.map_fixed_label.setGeometry(int(50 * self.scale_x), int(100 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.map_select_btn.setGeometry(int(50 * self.scale_x), int(130 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.latlon_label.setGeometry(int(50 * self.scale_x), int(190 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.lon_label.setGeometry(int(50 * self.scale_x), int(220 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.lon_min_input.setGeometry(int(50 * self.scale_x), int(250 * self.scale_y), int(90 * self.scale_x), int(35 * self.scale_y))
        self.lon_max_input.setGeometry(int(150 * self.scale_x), int(250 * self.scale_y), int(90 * self.scale_x), int(35 * self.scale_y))
        self.lat_label.setGeometry(int(50 * self.scale_x), int(290 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.lat_min_input.setGeometry(int(50 * self.scale_x), int(320 * self.scale_y), int(90 * self.scale_x), int(35 * self.scale_y))
        self.lat_max_input.setGeometry(int(150 * self.scale_x), int(320 * self.scale_y), int(90 * self.scale_x), int(35 * self.scale_y))
        self.mode_label.setGeometry(int(50 * self.scale_x), int(390 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.mode_select_btn.setGeometry(int(50 * self.scale_x), int(430 * self.scale_y), int(180 * self.scale_x), int(35 * self.scale_y))
        self.status_label.setGeometry(int(950 * self.scale_x), int(280 * self.scale_y), int(280 * self.scale_x), int(40 * self.scale_y))
        rows = 4  # Number of rows
        cols = 6  # Number of columns
        button_width = 30
        button_height = 30
        start_x = 25  # X position of the first button
        start_y = 480  # Y position of the first button
        spacing_x = 10  # Horizontal spacing
        spacing_y = 10  # Vertical spacing
        for i in range(1, 23):
            row = (i - 1) // cols  # Calculate row index
            col = (i - 1) % cols  # Calculate column index

            x = start_x + col * (button_width + spacing_x)
            y = start_y + row * (button_height + spacing_y)
            button = getattr(self, f"imazu_btn_{i}")
            button.setGeometry(int(x * self.scale_x), int(y * self.scale_y), int(button_width * self.scale_x), int(button_height * self.scale_y))  # Set position & size

    def resizeEvent(self, event):
        # Recalculate scaling factors
        new_width = self.width()
        new_height = self.height()
        self.scale_x = new_width / self.original_width
        self.scale_y = new_height / self.original_height

        # Update widget positions and sizes
        self.update_widget_geometry()

        # Call the parent class's resizeEvent
        super().resizeEvent(event)

        # Timer (updates every 100 ms).
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_step)

        # DWA variables.
        self.dwa_handler = None
        self.is_running = False

        # Vessel data will be initialized when a mode is selected.
        self.raw_vessel_data = None
        self.vessel_groups = {}  # Dictionary to store vessels grouped by timestamp
        self.active_vessels = {}  # Dictionary to track currently active vessels (accumulated)
        self.current_time = 0  # Track simulation time
        self.next_spawn_time = 10  # Time until next vessel group (in seconds)

        # Matplotlib artists.
        self.background_image = None  # for the background image
        self.vessel_plot = self.ax.scatter([], [], c='k', s=20)  # vessel obstacles
        self.robot_plot = self.ax.scatter([], [], c='r', s=20, marker='s')  # robot position
        self.path_line, = self.ax.plot([], [], color='b', linewidth=2)  # robot path
        self.start_marker = self.ax.scatter([], [], c='g', s=60, marker='o')
        self.goal_marker = self.ax.scatter([], [], c='b', s=60, marker='x')

        # Connect button clicks.
        self.start_btn.clicked.connect(self.on_start_dwa)
        self.pause_btn.clicked.connect(self.on_pause_resume)

        # Load the background image (if available).

        # self.update_graph()

    def reset_plot(self):
        # Resets the plot to its initial state.
        self.raw_vessel_data = None
        self.ax.clear()  # Clear all axes content
        
        # Re-initialize the plot elements after clearing the axes
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)

        # Recreate all plot elements from scratch
        self.background_image = None
        self.vessel_plot = self.ax.scatter([], [], c='k', s=20)  # vessel obstacles
        self.robot_plot = self.ax.scatter([], [], c='r', s=20, marker='s')  # robot position
        self.path_line, = self.ax.plot([], [], color='b', linewidth=2)  # robot path
        self.start_marker = self.ax.scatter([], [], c='g', s=60, marker='o')
        self.goal_marker = self.ax.scatter([], [], c='b', s=60, marker='x')
        
        # Reinitialize the direction arrows
        self.direction_arrows = self.ax.quiver(
            [0],
            [0],
            [0],
            [0],
            angles='uv',
            scale_units='inches',
            scale=5,
            color='#FF0000',
            headwidth=5,
            headlength=4,
            headaxislength=3,
        )
        self.direction_arrows.set_visible(False)  # Hide the arrows initially
        
        # Reset tracking variables
        self.previous_robot_pos = None
        self.active_vessels = {}  # Clear any active vessels
        
        self.canvas.draw()  # Redraw the canvas

    def map_changed(self):
        self.selected_map = self.map_select_btn.currentText()
        self.lat_min = self.lat_min_input.text()
        self.lat_max = self.lat_max_input.text()
        self.lon_min = self.lon_min_input.text()
        self.lon_max = self.lon_max_input.text()
        if self.selected_map == "Singapore Straits":
            lat_min = 1.13
            lat_max = 1.22
            lon_min = 103.75
            lon_max = 103.85
            map_generator.generate_map(lat_min, lat_max, lon_min, lon_max)
            self.load_background_image("map.png")
        elif self.selected_map == "Imazu Map":
            lat_min = 1.13
            lat_max = 1.22
            lon_min = 103.51
            lon_max = 103.6
            map_generator.generate_map(lat_min, lat_max, lon_min, lon_max)
            self.load_background_image("map.png")
        elif self.selected_map == "":
            map_generator.generate_map(self.lat_min, self.lat_max, self.lon_min, self.lon_max)
            self.load_background_image("map.png")

    def on_button_clicked(self):
        # Get the button that was clicked
        self.reset_plot()
        button = self.sender()
        self.button_selected = 0
        if button:
            self.reset_plot()
            # Extract the button number from its text or object name
            self.button_number = int(button.text().split()[-1])
            print(f"Button {self.button_number} was clicked")
        if hasattr(self, 'button_number'):
            # print(f"Button {self.button_number} was clicked")
            self.raw_vessel_data = imazudata.get_imazu(self.button_number)
            self.start_pos = imazudata.get_start(self.button_number)
            self.goal_pos = (50.0, 100.0)
            # self.selected = "imazu"+self.button_number
        
        self.load_background_image("map.png")
        self.initialize_vessel_data()
        self.update_graph()

    def mode_changed(self):
        self.reset_plot()
        self.mode_selected = self.mode_select_btn.currentText()

        # Load and process vessel data based on the selected mode.
        if self.mode_selected == "AIS":
            self.raw_vessel_data = vessel_saver.load_vessel_data('filtered_vessels')
            self.start_pos = (75.0, 90.0)
            self.goal_pos = (90.0, 55.0)
            print("AIS selected")
            self.x_label = np.linspace(self.min_lon, self.max_lon, 6)  # Custom x-labels range
            self.y_label = np.linspace(self.min_lat, self.max_lat, 5)  # Custom y-labels range

            # Set x and y ticks on the axes, not on plt directly
            self.ax.set_xticks(np.linspace(0, 100, len(self.x_label)))
            self.ax.set_xticklabels([f"{v:.2f}" for v in self.x_label])

            self.ax.set_yticks(np.linspace(0, 100, len(self.y_label)))
            self.ax.set_yticklabels([f"{v:.2f}" for v in self.y_label])
            self.ax.set_xlabel("Longitude Degrees")
            self.ax.set_ylabel("Latitude Degrees")
            self.selected = "ais"
        elif self.mode_selected == "Imazu 1":
            for i in range(1, 23):
                button = getattr(self, f"imazu_btn_{i}")  # Store button as attribute
                button.show()
                button.clicked.connect(self.on_button_clicked)

        self.load_background_image("map.png")
        self.initialize_vessel_data()
        self.update_graph()

    def initialize_vessel_data(self):
        # Initialize vessel data based on the selected mode.
        if self.raw_vessel_data is None:
            return  # No data to initialize

        # Create a dictionary to count IMO occurrences
        imo_count = {}
        for vessel in self.raw_vessel_data:
            imo = str(vessel[0])  # Convert IMO to string for consistency
            imo_count[imo] = imo_count.get(imo, 0) + 1

        # Initialize a set to track seen IMOs and a list for unique vessels
        seen_imos = set()
        unique_vessels = []

        for record in self.raw_vessel_data:
            # Number of vessels in this record
            num_vessels = len(record[0])

            # Temporary lists for the current record's unique vessels
            imolst = []
            timestamp_list = []
            x_coords = []
            y_coords = []
            speed = []
            direction = []
            file_no = []

            for i in range(num_vessels):
                current_imo = str(record[0][i])
                # If we haven't seen this IMO before, add its data
                if current_imo not in seen_imos:
                    seen_imos.add(current_imo)
                    imolst.append(current_imo)
                    timestamp_list.append(record[1][i])
                    x_coords.append(record[2][i])
                    y_coords.append(record[3][i])
                    speed.append(record[4][i])
                    direction.append(record[5][i])
                    file_no.append(record[6][i])

            # Only append non-empty records
            if imolst:
                unique_vessels.append((imolst, timestamp_list, x_coords, y_coords, speed, direction, file_no))

        self.vessel_data = unique_vessels

        # Group vessels by their timestamps.
        self.group_vessels_by_file()
        # Create a sorted list of timestamps (group keys).
        self.sorted_file = sorted(self.vessel_groups.keys())
        # Keep track of which group index we are up to.
        self.current_group_index = 0

        # Introduce the first group of vessels.
        self.introduce_initial_vessels()

    ########################################################################
    # Load Background Image
    ########################################################################
    def load_background_image(self, img_path="map.png"):
        # Load and display the background image using matplotlib imshow.
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            full_path = os.path.join(script_dir, img_path)

            qimg = QImage(full_path)
            if qimg.isNull():
                raise FileNotFoundError(f"Image not found at {full_path}")

            width = qimg.width()
            height = qimg.height()
            ptr = qimg.constBits()
            ptr.setsize(height * width * 4)  # 4 bytes per pixel (ARGB32)
            img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            img_array = np.flipud(img_array)

            # Convert ARGB to RGBA (matplotlib expects RGBA).
            rgba_img = np.zeros_like(img_array)
            rgba_img[..., 0] = img_array[..., 2]  # R
            rgba_img[..., 1] = img_array[..., 1]  # G
            rgba_img[..., 2] = img_array[..., 0]  # B
            rgba_img[..., 3] = img_array[..., 3]  # A

            self.background_image = self.ax.imshow(
                rgba_img,
                extent=(0, 100, 0, 100),
                alpha=1.0,
                origin='lower'
            )
            self.figure.patch.set_facecolor('none')
            self.ax.set_facecolor('none')
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.canvas.draw()

        except FileNotFoundError as e:
            print(f"Error: {e}")

    ########################################################################
    # Update Graph Elements
    ########################################################################
    def update_graph(self):
        # Update the vessel obstacles and start/goal markers.
        if self.active_vessels:  # Only proceed if there are active vessels
            current_positions = np.array([
                (get_first_coord(vessel[2]), get_first_coord(vessel[3]))
                for vessel in self.active_vessels.values()
            ])

            current_positions = np.array(current_positions)
            print(f"Plotting {len(current_positions)} vessels")

            self.vessel_plot.set_offsets(current_positions)
            self.start_marker.set_offsets([self.start_pos])
            self.goal_marker.set_offsets([self.goal_pos])
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.canvas.draw()

    ########################################################################
    # Timer Callback (Simulation Step)
    ########################################################################
    def update_step(self):
        base_dir = "traj_log"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        file_path = os.path.join("traj_log" ,f"{self.selected}_{self.timestamp}.txt")
        with open(file_path, "a") as f:
        
            if not self.is_running or self.dwa_handler is None:
                return

            dt = self.timer.interval() / 1000.0
            self.current_time += dt

            if self.current_time >= self.next_spawn_time:
                print(f"Current time: {self.current_time}, spawning new vessels")
                self.introduce_new_vessels()
                self.next_spawn_time += 10

            self.move_obstacles()

            current_obstacles = np.array([
                (get_first_coord(vessel[2]), get_first_coord(vessel[3]))
                for vessel in self.active_vessels.values()
            ])

            if len(current_obstacles) > 0:
                self.dwa_handler.ob = current_obstacles

                x, reached = self.dwa_handler.step()
                f.write(np.array2string(x, separator=",")) 

                if x is not None:
                    self.robot_plot.set_offsets([[x[0], x[1]]])
                    traj_arr = np.array(self.dwa_handler.trajectory)
                    if traj_arr.size > 0:
                        self.path_line.set_data(traj_arr[:, 0], traj_arr[:, 1])

                    current_robot_pos = (x[0], x[1])
                    f.write(",".join(map(str, current_robot_pos)))
                    if self.previous_robot_pos is not None:
                        dx = current_robot_pos[0] - self.previous_robot_pos[0]
                        dy = current_robot_pos[1] - self.previous_robot_pos[1]

                        if dx != 0 or dy != 0:
                            robot_angle_rad = math.atan2(dy, dx)
                            u = math.cos(robot_angle_rad)
                            v = math.sin(robot_angle_rad)

                            # Set the arrow's origin and direction vector
                            self.direction_arrows.set_offsets([current_robot_pos])
                            self.direction_arrows.set_UVC([u], [v])  # UVC should be lists
                            self.direction_arrows.set_visible(True)
                            self.canvas.draw()
                    else:
                        # On the first step, use the angle from DWA
                        u = math.cos(x[2])
                        v = math.sin(x[2])

                        # Set the arrow's origin and direction vector
                        self.direction_arrows.set_offsets([current_robot_pos])
                        self.direction_arrows.set_UVC([u], [v])  # UVC should be lists
                        self.direction_arrows.set_visible(True)

                        self.canvas.draw()

                    self.previous_robot_pos = current_robot_pos  # Update previous position

                else:
                    print("DWA step returned None")

                self.vessel_plot.set_offsets(current_obstacles)
                self.canvas.draw()

                if reached:
                    self.status_label.setText("Goal reached!")
                    self.is_running = False
                    self.timer.stop()


    ########################################################################
    # Button Callbacks
    ########################################################################
    def on_start_dwa(self):


        # Use the initial positions (first coordinates) for each vessel.
        current_obstacles = np.array([
            (get_first_coord(vessel[2]), get_first_coord(vessel[3]))
            for vessel in self.active_vessels.values()
        ])
        print("Initial vessel obstacles:", current_obstacles)

        # Define start and goal positions.
        sx, sy = self.start_pos
        gx, gy = self.goal_pos

        # Run A* using the current vessel positions.
        rx, ry = a_star.run_a_star(
            ob=current_obstacles, sx=sx, sy=sy, gx=gx, gy=gy,
            resolution=1.0, robot_radius=3
        )

        # Initialize DWA with the current obstacles.
        x_init = [sx, sy, math.radians(15), 0.0, 0.0]
        goal = [gx, gy]
        self.dwa_handler = dwa_mod.DWAHandler(
            x_init=x_init, goal=goal, ob=current_obstacles
        )

        self.is_running = True
        self.status_label.setText("DWA running...")
        self.direction_arrows.set_visible(True)  # Show arrow when DWA starts

        if not self.timer.isActive():
            self.timer.start()

    def on_pause_resume(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Paused.")
            self.direction_arrows.set_visible(False)
        else:
            if self.is_running:
                self.timer.start()
                self.status_label.setText("DWA running...")
                self.direction_arrows.set_visible(True)

    ########################################################################
    # Vessel Grouping and Introduction
    ########################################################################

    def group_vessels_by_file(self):
        # Group vessels by their actual timestamps,
        # saving each vessel individually under its own IMO.
        # Assumes each element in self.raw_vessel_data is a tuple of lists,
        # where each list holds one attribute for multiple vessels.
        
        for record in self.vessel_data:
            # Determine the group key.
            # Here we assume all entries in record[6] are the same for this record,
            # so we use get_first_coord on the whole list.
            group_key = get_first_coord(record[6])
            if group_key not in self.vessel_groups:
                self.vessel_groups[group_key] = {}

            # Determine how many vessels are in this record.
            # (We assume all attribute lists in the tuple are of the same length.)
            num_vessels = len(record[0])

            # For each vessel in this record, create a vessel-specific data tuple
            # and save it using its IMO as the key.
            for i in range(num_vessels):
                imo = str(record[0][i])
                # Construct a tuple where each element is the i-th entry from the corresponding attribute list.
                vessel_data = tuple(attribute[i] for attribute in record)
                self.vessel_groups[group_key][imo] = vessel_data

        # Debug print: iterate over groups and print the number of vessels per group.
        for group, vessels in self.vessel_groups.items():
            print(f"Group {group}: {len(vessels)} vessels")

    def introduce_initial_vessels(self):
        # Introduce the first group of vessels.
        if self.sorted_file:
            first_file = self.sorted_file[0]
            first_group = self.vessel_groups[first_file]
            # Accumulate the first group.
            self.active_vessels.update({imo: vessel for imo, vessel in first_group.items()})
            print(f"Initialized with {len(self.active_vessels)} vessels from file {first_file}")

    def introduce_new_vessels(self):
        # Introduce new vessels from the next group and accumulate them.
        self.current_group_index += 1
        if self.current_group_index < len(self.sorted_file):
            next_file = self.sorted_file[self.current_group_index]
            new_vessels = self.vessel_groups[next_file]
            print(f"Introducing vessels from timestamp {next_file}")
            print(f"Number of vessels in new group: {len(new_vessels)}")

            # Accumulate the new vessels with the previously active ones.
            self.active_vessels.update(new_vessels)
            print(f"Total active vessels after accumulation: {len(self.active_vessels)}")
        else:
            print("No more vessel groups to display.")
            # Optionally, you could reset the index to loop again.
            # self.current_group_index = 0

    ########################################################################
    # Update Vessel Positions
    ########################################################################
    def move_obstacles(self):
        # Update positions of active vessels.
        dt = self.timer.interval() / 1000.0

        for imo, vessel in list(self.active_vessels.items()):
            imo, timestamp, x, y, speed, direction, file_no = vessel
            x_scalar = get_first_coord(x)
            y_scalar = get_first_coord(y)
            speed_scalar = get_first_coord(speed)
            direction_scalar = get_first_coord(direction)
            file = get_first_coord(file_no)

            new_x = x_scalar + speed_scalar * 0.1 * math.cos(direction_scalar) * dt
            new_y = y_scalar + speed_scalar * 0.1 * math.sin(direction_scalar) * dt

            # Update the vessel's position in active_vessels.
            self.active_vessels[str(imo)] = [imo, timestamp, new_x, new_y, speed_scalar, direction_scalar, file]

def main():
    app = QApplication(sys.argv)
    gui = MyGui()
    gui.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
