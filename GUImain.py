import sys
import math
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QComboBox, QHBoxLayout, QVBoxLayout
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage
# Matplotlib imports for embedding in PyQt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

import os
import a_star
import dynamic_window_approach_paper as dwa_mod


class MyGui(QWidget):
    def __init__(self):
        super().__init__()
 
        self.setGeometry(100, 100, 900, 600)
        self.setWindowTitle("DWA + A* (Matplotlib)")

        # Create the main layout
        self.main_layout = QHBoxLayout(self)

        # -- Matplotlib Figure and Canvas --
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)

        # Add the Matplotlib canvas to the layout
        self.main_layout.addWidget(self.canvas)

        # -- Right panel for buttons/controls --
        self.right_panel = QVBoxLayout()

        self.start_btn = QPushButton("Start DWA", self)
        self.pause_btn = QPushButton("Pause/Resume", self)
        self.map_select_btn = QComboBox(self)
        self.map_select_btn.addItems(["Tuas", "Singapore Straits"])
        self.selected_map = self.map_select_btn.currentText()
        self.status_label = QLabel("Select map and click 'Start DWA' to begin.", self)
        self.status_label.setAlignment(Qt.AlignCenter)

        self.right_panel.addWidget(self.start_btn)
        self.right_panel.addWidget(self.pause_btn)
        self.right_panel.addWidget(self.map_select_btn)
        self.right_panel.addWidget(self.status_label)
        self.right_panel.addStretch(1)  # Push everything up

        self.main_layout.addLayout(self.right_panel)
        self.setLayout(self.main_layout)

        # Timer
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_step)

        # DWA Variables
        self.dwa_handler = None
        self.is_running = False

        # Obstacles
        self.obstacles = np.array([
            [0, 3], [0, 2], [4, 2], [5, 4], [5, 5], [5, 6],
            [5, 9], [8, 9], [7, 9], [8, 10], [9, 11], [12, 12]
        ]).astype(float)

        # Give them random small velocities
        np.random.seed(0)
        self.obstacle_vels = 0.02 * (np.random.rand(*self.obstacles.shape) - 0.5)

        # Start & Goal Positions
        self.start_pos = (2.0, 2.0)
        self.goal_pos = (80.0, 80.0)

        # Matplotlib artists (initialize them empty; we'll set data later)
        # Background image will be shown via imshow
        self.background_image = None

        # Obstacles
        self.obstacle_plot = self.ax.scatter([], [], c='k', s=6)

        # Robot
        self.robot_plot = self.ax.scatter([], [], c='r', s=10)

        # Path line
        self.path_line, = self.ax.plot([], [], color='r', linewidth=2)

        # Start/Goal markers
        self.start_marker = self.ax.scatter([], [], c='g', s=60, marker='o')
        self.goal_marker = self.ax.scatter([], [], c='b', s=60, marker='x')

        # Connect Button Clicks
        self.start_btn.clicked.connect(self.on_start_dwa)
        self.pause_btn.clicked.connect(self.on_pause_resume)

        # Load background image (optional, adjust path and alpha as needed)
        self.load_background_image("map.png")

        # Initialize graph with obstacles & markers
        self.update_graph()

    ########################################################################
    # Load Background Image
    ########################################################################
    def load_background_image(self, img_path="map.png"):
        """Load and display the background image using matplotlib imshow."""
        try:
            # Get the directory of the current script
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct full path to map.png
            full_path = os.path.join(script_dir, img_path)
            
            qimg = QImage(full_path)
            if qimg.isNull():
                raise FileNotFoundError(f"Image not found at {full_path}")

            # Convert QImage to numpy array more safely
            width = qimg.width()
            height = qimg.height()
            ptr = qimg.constBits()
            ptr.setsize(height * width * 4)  # 4 bytes per pixel for ARGB32
            img_array = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))

            # Flip image vertically to match conventional axes
            img_array = np.flipud(img_array)

            # Convert from ARGB to RGBA if needed (matplotlib expects RGBA)
            # You may need to reorder the channels depending on your QImage
            # but in many cases, ARGB -> RGBA is just a shift of channels.
            # Check your specific QImage format if the colors look off.
            # For simplicity, let's do a direct copy of the first 3 channels (RGB)
            # and the 4th channel (alpha). If you see color mismatch, revisit channel order.
            rgba_img = np.zeros_like(img_array)
            rgba_img[..., 0] = img_array[..., 2]  # R
            rgba_img[..., 1] = img_array[..., 1]  # G
            rgba_img[..., 2] = img_array[..., 0]  # B
            rgba_img[..., 3] = img_array[..., 3]  # A

            # Plot it with imshow, setting an extent that matches your domain
            self.background_image = self.ax.imshow(rgba_img,
                                                   extent=(0, 100, 0, 100),
                                                   alpha=1.0,
                                                   origin='lower')

            # Set the figure and axes background to be transparent
            self.figure.patch.set_facecolor('none')
            self.ax.set_facecolor('none')
            
            self.ax.set_xlim(0, 100)
            self.ax.set_ylim(0, 100)
            self.canvas.draw()

        except FileNotFoundError as e:
            print(f"Error: {e}")

    ########################################################################
    # Move Obstacles
    ########################################################################
    def move_obstacles(self):
        """Moves obstacles randomly within the defined range, bouncing if out of bounds."""
        self.obstacles += self.obstacle_vels

        # Bounce obstacles back if they hit boundaries
        for i in range(len(self.obstacles)):
            x, y = self.obstacles[i]
            if x < 0 or x > 100:
                self.obstacle_vels[i, 0] *= -1
            if y < 0 or y > 100:
                self.obstacle_vels[i, 1] *= -1

    ########################################################################
    # Update Graph Elements
    ########################################################################
    def update_graph(self):
        """Updates all graph elements (obstacles, start, goal, etc.) in Matplotlib."""
        # Obstacles
        self.obstacle_plot.set_offsets(self.obstacles)

        # Start/Goal
        self.start_marker.set_offsets([self.start_pos])
        self.goal_marker.set_offsets([self.goal_pos])

        # Set domain
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 100)

        # Redraw
        self.canvas.draw()

    ########################################################################
    # Button Callbacks
    ########################################################################
    def on_start_dwa(self):
        sx, sy = self.start_pos
        gx, gy = self.goal_pos

        rx, ry = a_star.run_a_star(
            ob=self.obstacles, sx=sx, sy=sy, gx=gx, gy=gy,
            resolution=1.0, robot_radius=0.1
        )

        x_init = [sx, sy, math.radians(15), 0.0, 0.0]
        goal = [gx, gy]

        self.dwa_handler = dwa_mod.DWAHandler(
            x_init=x_init, goal=goal, ob=self.obstacles
        )

        self.is_running = True
        self.status_label.setText("DWA running...")

        if not self.timer.isActive():
            self.timer.start()

    def on_pause_resume(self):
        if self.timer.isActive():
            self.timer.stop()
            self.status_label.setText("Paused.")
        else:
            if self.is_running:
                self.timer.start()
                self.status_label.setText("DWA running...")

    ########################################################################
    # Timer Callback (Simulation Step)
    ########################################################################
    def update_step(self):
        if not self.is_running or self.dwa_handler is None:
            return

        # Move obstacles first
        self.move_obstacles()

        # Update the obstacle array in DWA
        self.dwa_handler.ob = self.obstacles

        # Perform one DWA step
        x, reached = self.dwa_handler.step()

        # Update Robot position
        self.robot_plot.set_offsets([[x[0], x[1]]])

        # Update path line
        traj_arr = np.array(self.dwa_handler.trajectory)
        if len(traj_arr) > 0:
            self.path_line.set_data(traj_arr[:, 0], traj_arr[:, 1])

        # Update the obstacles
        self.obstacle_plot.set_offsets(self.obstacles)

        # Redraw the canvas
        self.canvas.draw()

        # Check if goal is reached
        if reached:
            self.status_label.setText("Goal reached!")
            self.is_running = False
            self.timer.stop()


def main():
    app = QApplication(sys.argv)
    gui = MyGui()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
