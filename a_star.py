"""
A* grid planning (modified to include a run_a_star function).

Based on original work by:
Atsushi Sakai(@Atsushi_twi), Nikos Kanargias (nkana@tee.gr)
"""

import math
import matplotlib.pyplot as plt
import numpy as np

show_animation = False  # Set True if you want a separate matplotlib animation

class AStarPlanner:
    def __init__(
        self, ob, resolution, rr, 
        min_x=None, min_y=None, max_x=None, max_y=None
    ):
        """
        Initialize the A* planner.

        ob: Nx2 array of obstacle points (ox, oy)
        resolution: float, grid resolution [m]
        rr: float, robot radius [m]
        min_x, min_y, max_x, max_y: optional bounding values for the grid
        """

        self.ob = ob
        self.resolution = resolution
        self.rr = rr
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y

        self.obstacle_map = None
        self.x_width = 0
        self.y_width = 0

        # 8 possible moves
        self.motion = self.get_motion_model()

        # Build an obstacle map (grid)
        self.calc_obstacle_map()

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

    def planning(self, sx, sy, gx, gy):
        """
        A* path search.

        sx, sy: start coordinates in continuous space
        gx, gy: goal coordinates in continuous space

        returns: (rx, ry), the path from start to goal (list of x and y)
                 If no path found, returns (None, None)
        """
        start_node = self.Node(
            self.calc_xy_index(sx, self.min_x),
            self.calc_xy_index(sy, self.min_y),
            0.0,
            -1
        )
        goal_node = self.Node(
            self.calc_xy_index(gx, self.min_x),
            self.calc_xy_index(gy, self.min_y),
            0.0,
            -1
        )

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if not open_set:
                print("Open set is empty. No path found.")
                return None, None

            # pick node with lowest cost + heuristic
            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o])
            )
            current = open_set[c_id]

            # for debugging or optional animation
            if show_animation:
                plt.plot(
                    self.calc_grid_position(current.x, self.min_x),
                    self.calc_grid_position(current.y, self.min_y), 
                    "xc"
                )
                if len(closed_set) % 10 == 0:
                    plt.pause(0.001)

            # goal check
            if current.x == goal_node.x and current.y == goal_node.y:
                print("A*: Goal found!")
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                rx, ry = self.calc_final_path(goal_node, closed_set)
                return rx, ry

            # move from open to closed
            del open_set[c_id]
            closed_set[c_id] = current

            # explore neighbors
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0],
                    current.y + self.motion[i][1],
                    current.cost + self.motion[i][2],
                    c_id
                )
                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

    def calc_final_path(self, goal_node, closed_set):
        """
        Backtrack from goal_node to get the path.
        """
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        rx.reverse()
        ry.reverse()
        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        # Euclidean distance
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def calc_grid_position(self, index, min_position):
        return index * self.resolution + min_position

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self):
        ox, oy = self.ob[:, 0], self.ob[:, 1]

        if self.min_x is None:
            self.min_x = round(min(ox))
        if self.min_y is None:
            self.min_y = round(min(oy))
        if self.max_x is None:
            self.max_x = round(max(ox))
        if self.max_y is None:
            self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [
            [False for _ in range(self.y_width)] 
            for _ in range(self.x_width)
        ]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in self.ob:
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        return [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1,  math.sqrt(2)],
            [1, -1,  math.sqrt(2)],
            [1, 1,   math.sqrt(2)]
        ]


def run_a_star(ob, sx, sy, gx, gy, resolution=2.0, robot_radius=1.0):
    """
    A convenience function that creates AStarPlanner and returns (rx, ry).
    """
    planner = AStarPlanner(ob, resolution, robot_radius)
    rx, ry = planner.planning(sx, sy, gx, gy)
    return rx, ry


def main():
    print("Running A* standalone test")

    # Example obstacles
    ox, oy = [], []
    for i in range(-10, 40):
        ox.append(20.0)
        oy.append(i)
    for i in range(0, 40):
        ox.append(40.0)
        oy.append(60.0 - i)
    ob = np.array([ox, oy]).T

    sx, sy = 10.0, 10.0
    gx, gy = 50.0, 50.0

    rx, ry = run_a_star(ob, sx, sy, gx, gy, resolution=2.0, robot_radius=1.0)
    print("A* output path length:", None if rx is None else len(rx))

    if show_animation:
        import matplotlib.pyplot as plt
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")
        if rx:
            plt.plot(rx, ry, "-r")
        plt.axis("equal")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    main()
