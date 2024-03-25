import math
import numpy as np
from queue import PriorityQueue
import matplotlib.path as mplPath
import cv2
import matplotlib.pyplot as plt

# Node class to represent a node in the search tree
class Node:
    def __init__(self, pts, ort, parent_node, cost, h):
        self.parent_node = parent_node
        self.pts = pts
        self.cost = cost
        self.heu = h
        self.ort = ort
        self.final_cost = cost + h

    def fetchPoints(self):
        return self.pts

    def fetchOrentation(self):
        return self.ort

    def fetchParent_nodeNode(self):
        return self.parent_node

    def __eq__(self, other):
        return self.pts[0] == other.pts[0] and self.pts[1] == other.pts[1]

    def __hash__(self):
        return hash((self.pts[0], self.pts[1]))

    def __lt__(self, other):
        return self.final_cost < other.final_cost

# Global variables
y = 500
robot_clearance = 0
color_new = (255, 0, 0)
robot_step_size = 0
frame = np.zeros((500, 1200, 3), dtype=np.uint8)
width, height = 1200, 500
video_writer = cv2.VideoWriter('a_star_gowtham_dinesh.avi', cv2.VideoWriter_fourcc(*'XVID'), 900, (width, height))

# Helper functions for hexagonal structure
def hex_corner_help(rad):
    x_c = int(650 + 150 * math.cos(rad))
    y_c = int(250 + 150 * math.sin(rad))
    return (x_c, y_c)

def hex_corners():
    hex_corners = []
    for i in range(6):
        rad = math.radians(30 + 60 * i)
        hex_corners.append(hex_corner_help(rad))
    return hex_corners

# Function to check if coordinates can be moved to
def couldMove(cordinates):
    if cordinates[0] <= 5 or cordinates[0] >= 1196 or cordinates[1] <= 5 or cordinates[1] >= 496:
        return False
    cordinates_color = frame[cordinates[1], cordinates[0]]
    if cordinates_color[0] == 255 and cordinates_color[1] == 0 and cordinates_color[2] == 0:
        return True
    if cordinates_color[0] == 0 and cordinates_color[1] == 0 and cordinates_color[2] == 0:
        return True
    return False

# Function to set up robot clearance and step size
def setup():
    global robot_clearance
    global robot_step_size
    robot_clearance, robot_step_size = int(input("Enter the robot_clearance in: ")), int(input("Enter the step size between 1 and 10: "))

# Initialize hexagonal and rectangular structures
hex_pt = np.array(hex_corners())
um_structure = np.array([[900, y-(450)], [1100, y-450], [1100, y-50], [900, y-(50)], [900, y-125], [1020, y-125], [1020, y-(375)], [900, y-375]])

# Draw hexagonal and rectangular structures on the frame
cv2.polylines(frame, [hex_pt], True, (255, 0, 255), 5)
cv2.fillPoly(frame, [hex_pt], (0, 0, 255))

cv2.polylines(frame, [um_structure], True, (255, 0, 255), 5)
cv2.fillPoly(frame, [um_structure], (0, 0, 255))

cv2.rectangle(frame, (275, y-400), (350, y), (255, 0, 255), 5)
cv2.rectangle(frame, (275, y-400), (350, y), (0, 0, 255), -1)

cv2.rectangle(frame, (100, y-500), (175, y-100), (255, 0, 255), 5)
cv2.rectangle(frame, (100, y-500), (175, y-100), (0, 0, 255), -1)

# Function to get start and goal points from user
def start_goal_inputs():
    x, y = map(int, input("Enter the x and y coordinates of the initial point, separated by a space: ").split())
    first_point = (x, y)

    intial_or = int(input("robot initial orentation: "))

    x, y = map(int, input("Enter the x and y coordinates of the goal point, separated by a space: ").split())
    last_point = (x, y)

    final_or = int(input("robot at goal orentation: "))

    if couldMove(first_point) and couldMove(last_point):
        return first_point, last_point, intial_or, final_or
    else:
        print("wrong input")
        return start_goal_inputs()

# Function to calculate new coordinates based on current position, orientation, and step size
def get_new_coordinates(current, orry, ss, theta):
    cord_x = current[0] + int(round((math.cos(math.radians(orry + theta)) * ss)))
    cord_y = current[1] + int(round((math.sin(math.radians(orry + theta)) * ss)))
    return (cord_x, cord_y)

# Function to calculate new move based on current node, step size, and theta
def calculateNewMove(node, ss, theta):
    current = node.fetchPoints()
    orry = node.fetchOrentation()

    robot_clearance_cord = get_new_coordinates(current, orry, robot_clearance, theta)
    new_cordinates = get_new_coordinates(current, orry, ss, theta)

    if not couldMove(new_cordinates) or not couldMove(robot_clearance_cord):
        return False, None, None, ss
    return True, new_cordinates, orry, ss

# Functions for different actions (straight, left30, left60, right30, right60)
def straight(node, step_size):
    return calculateNewMove(node, step_size, 0)

def left30(node, step_size):
    return calculateNewMove(node, step_size, -30)

def left60(node, step_size):
    return calculateNewMove(node, step_size, -60)

def right30(node, step_size):
    return calculateNewMove(node, step_size, 30)

def right60(node, step_size):
    return calculateNewMove(node, step_size, 60)

# Function to check if current point is at the goal
def at_goal(pt, fpt, gow):
    x, y = pt[0], pt[1]
    ccx, ccy, = fpt[0], fpt[1]
    return math.sqrt((x - ccx) ** 2 + (y - ccy) ** 2) <= gow

# Heuristic function to calculate Euclidean distance between two points
def heu(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to back track the path from goal to start
def back_track_path(node):
    path = []
    while node is not None:
        path.append(node.fetchPoints())
        node = node.fetchParent_nodeNode()

    for i in range(len(path) - 1):
        inital_point = path[i]
        final_ = path[i + 1]
        cv2.arrowedLine(frame, inital_point, final_, (255, 255, 0), 2, tipLength=0.2)
        video_writer.write(frame)
    return path[::-1]

# A* algorithm implementation
def a_star(start_cord, final_cord, intial_or):
    open_que = PriorityQueue()
    dead_nodes = set()
    explored = {}

    start_node = Node(pts=start_cord, ort=intial_or, parent_node=None, cost=0, h=heu(start_cord, final_cord))

    open_que.put((start_node.final_cost, start_node))
    explored[start_cord] = start_node
    dead_nodes.add(start_cord)

    while True:
        curr = open_que.get()[1]
        curr_pt = curr.fetchPoints()

        if at_goal(curr_pt, final_cord, 1.5):
            return back_track_path(curr)

        actions = [straight, left30, left60, right30, right60]

        for action in actions:
            could_move, new_cord, new_pose, cost = action(curr, robot_step_size)
            if could_move:
                possible_nd = Node(pts=new_cord, ort=new_pose, parent_node=curr, cost=curr.cost + cost, h=heu(new_cord, final_cord))
                if new_cord not in dead_nodes:
                    if new_cord in explored:
                        if explored[new_cord].final_cost > possible_nd.final_cost:
                            explored[new_cord] = possible_nd
                            open_que.put((possible_nd.final_cost, possible_nd))
                    else:
                        open_que.put((possible_nd.final_cost, possible_nd))
                        explored[new_cord] = possible_nd
                        dead_nodes.add(new_cord)
                        if possible_nd.fetchParent_nodeNode() is not None:
                            cv2.line(frame, possible_nd.fetchPoints(), possible_nd.fetchParent_nodeNode().fetchPoints(), color_new)
                            video_writer.write(frame)
            else:
                continue

if __name__ == "__main__":
    intial, final, intial_or, final_or = start_goal_inputs()
    setup()
    shortest_path = a_star(intial, final, intial_or)
    video_writer.release()
    plt.imshow(frame)
    plt.show()
