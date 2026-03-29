import numpy as np
import collections
import heapq
import time
import random
import pandas as pd

class GridCell:
    
    def __init__(self, x, y, terrain_type='.', cost=1, is_obstacle=False, is_dynamic_obstacle=False):
        self.x = x
        self.y = y
        self.terrain_type = terrain_type 
        self.cost = cost             
        self.is_obstacle = is_obstacle 
        self.is_dynamic_obstacle = is_dynamic_obstacle 

    def __repr__(self):
        return f"GridCell({self.x},{self.y}, type='{self.terrain_type}', cost={self.cost}, obs={self.is_obstacle}, dyn_obs={self.is_dynamic_obstacle})"

class DynamicObstacle:
    
    def __init__(self, id, initial_x, initial_y, movement_pattern=None):
        self.id = id
        self.current_x = initial_x
        self.current_y = initial_y
        self.movement_pattern = movement_pattern 
        self.path_index = 0

    def move(self, grid_width, grid_height):
        if self.movement_pattern and self.path_index < len(self.movement_pattern):
            dx, dy = self.movement_pattern[self.path_index]
            new_x = self.current_x + dx
            new_y = self.current_y + dy

          
            self.current_x = max(0, min(new_x, grid_width - 1))
            self.current_y = max(0, min(new_y, grid_height - 1))

            self.path_index = (self.path_index + 1) % len(self.movement_pattern)
            return (self.current_x, self.current_y)
        return (self.current_x, self.current_y)

    def __repr__(self):
        return f"DynamicObstacle(id={self.id}, pos=({self.current_x},{self.current_y}))"

class Grid:
    
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid_cells = [[GridCell(x, y) for y in range(height)] for x in range(width)]
        self.dynamic_obstacles = []
        self.start_pos = None
        self.goal_pos = None

        
        self.terrain_costs = {
            '.': 1,  
            '1': 1,
            '2': 2,
            '3': 3,
            '#': float('inf'),
            'S': 1,  
            'G': 1, 
            'D': 1
        }

    def get_cell(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid_cells[x][y]
        return None

    def set_cell_property(self, x, y, terrain_type=None, cost=None, is_obstacle=None, is_dynamic_obstacle=None):
        cell = self.get_cell(x, y)
        if cell:
            if terrain_type is not None:
                cell.terrain_type = terrain_type
                cell.cost = self.terrain_costs.get(terrain_type, cell.cost)
                cell.is_obstacle = (terrain_type == '#')
            if cost is not None:
                cell.cost = cost
            if is_obstacle is not None:
                cell.is_obstacle = is_obstacle
            if is_dynamic_obstacle is not None:
                cell.is_dynamic_obstacle = is_dynamic_obstacle
            return True
        return False

    def add_dynamic_obstacle(self, obstacle):
        self.dynamic_obstacles.append(obstacle)
        
        self.set_cell_property(obstacle.current_x, obstacle.current_y, is_dynamic_obstacle=True)

    @classmethod
    def from_map_string(cls, map_string):
        lines = map_string.strip().split('\n')
        height = len(lines)
        width = len(lines[0]) if height > 0 else 0

        grid = cls(width, height)
        dynamic_obstacle_id_counter = 0

        for y in range(height):
            for x in range(width):
                char = lines[y][x]
                grid.set_cell_property(x, y, terrain_type=char)
                if char == 'S':
                    grid.start_pos = (x, y)
                elif char == 'G':
                    grid.goal_pos = (x, y)
                elif char == 'D':
                    dynamic_obstacle = DynamicObstacle(dynamic_obstacle_id_counter, x, y, movement_pattern=[(0,1), (0,-1), (1,0), (-1,0)]) # Default pattern
                    grid.add_dynamic_obstacle(dynamic_obstacle)
                    dynamic_obstacle_id_counter += 1
                    grid.set_cell_property(x, y, is_dynamic_obstacle=True)

        
        if grid.start_pos: grid.get_cell(grid.start_pos[0], grid.start_pos[1]).cost = grid.terrain_costs.get('S', 1)
        if grid.goal_pos: grid.get_cell(grid.goal_pos[0], grid.goal_pos[1]).cost = grid.terrain_costs.get('G', 1)
        for obs in grid.dynamic_obstacles:
            grid.get_cell(obs.current_x, obs.current_y).cost = grid.terrain_costs.get('D', 1)

        return grid

    def is_position_blocked(self, x, y):
        
        cell = self.get_cell(x, y)
        if cell:
            return cell.is_obstacle or cell.is_dynamic_obstacle
        return True 

    def get_neighbors(self, x, y):
        
        neighbors = []
        cardinal_directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        for dx, dy in cardinal_directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and not self.is_position_blocked(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def get_movement_cost(self, from_pos, to_pos):
        
        to_cell = self.get_cell(to_pos[0], to_pos[1])
        if to_cell:
            return to_cell.cost
        return float('inf')

    def update_dynamic_obstacles_positions(self):
        
        for obstacle in self.dynamic_obstacles:
            
            self.set_cell_property(obstacle.current_x, obstacle.current_y, is_dynamic_obstacle=False)

            
            obstacle.move(self.width, self.height)

            
            self.set_cell_property(obstacle.current_x, obstacle.current_y, is_dynamic_obstacle=True)

    def __repr__(self):
        return f"Grid(width={self.width}, height={self.height}, start={self.start_pos}, goal={self.goal_pos}, num_dyn_obs={len(self.dynamic_obstacles)})"



class Agent:
    
    def __init__(self, start_pos):
        self.position = start_pos 
        self.current_cost = 0
        self.current_time = 0

    def __repr__(self):
        return f"Agent(pos={self.position}, cost={self.current_cost}, time={self.current_time})"



class Node:
    
    def __init__(self, position, cost, parent=None):
        self.position = position
        self.cost = cost
        self.parent = parent

    def __lt__(self, other):
        
        return self.cost < other.cost

    def __eq__(self, other):
        
        return isinstance(other, Node) and self.position == other.position

    def __hash__(self):
        
        return hash(self.position)

    def __repr__(self):
        return f"Node(pos={self.position}, cost={self.cost:.2f})"

def reconstruct_path(goal_node):
    
    path = []
    current = goal_node
    while current:
        path.append(current.position)
        current = current.parent
    return path[::-1] 

def bfs_search(grid, start_pos, goal_pos):
    
    queue = collections.deque()
    start_node = Node(start_pos, 0)
    queue.append(start_node)

    visited = {start_pos: start_node}
    expanded_nodes_count = 0

    while queue:
        current_node = queue.popleft()
        expanded_nodes_count += 1 

        if current_node.position == goal_pos:
            return reconstruct_path(current_node), current_node.cost, expanded_nodes_count

        for neighbor_pos in grid.get_neighbors(current_node.position[0], current_node.position[1]):
            if neighbor_pos not in visited:
                move_cost = grid.get_movement_cost(current_node.position, neighbor_pos)
                new_cost = current_node.cost + move_cost

                neighbor_node = Node(neighbor_pos, new_cost, current_node)
                visited[neighbor_pos] = neighbor_node
                queue.append(neighbor_node)

    return None, float('inf'), expanded_nodes_count

def ucs_search(grid, start_pos, goal_pos):
    
    priority_queue = [] 
    start_node = Node(start_pos, 0)
    heapq.heappush(priority_queue, (0, start_node))

    cost_so_far = {start_pos: 0}
    came_from = {start_pos: start_node}
    expanded_nodes_count = 0

    while priority_queue:
        current_cost, current_node = heapq.heappop(priority_queue)

        
        if current_cost > cost_so_far.get(current_node.position, float('inf')):
            continue

        expanded_nodes_count += 1

        if current_node.position == goal_pos:
            return reconstruct_path(current_node), current_node.cost, expanded_nodes_count

        for neighbor_pos in grid.get_neighbors(current_node.position[0], current_node.position[1]):
            move_cost = grid.get_movement_cost(current_node.position, neighbor_pos)
            new_cost = current_node.cost + move_cost

            if neighbor_pos not in cost_so_far or new_cost < cost_so_far[neighbor_pos]:
                cost_so_far[neighbor_pos] = new_cost
                neighbor_node = Node(neighbor_pos, new_cost, current_node)
                heapq.heappush(priority_queue, (new_cost, neighbor_node))
                came_from[neighbor_pos] = neighbor_node

    return None, float('inf'), expanded_nodes_count 


def manhattan_distance_heuristic(pos1, pos2):
    
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

def euclidean_distance_heuristic(pos1, pos2):
    
    return ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5

def chebyshev_distance_heuristic(pos1, pos2):
    
    return max(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))

def astar_search(grid, start_pos, goal_pos, heuristic_func=manhattan_distance_heuristic):
    
    priority_queue = [] 
    start_node = Node(start_pos, 0)
    heapq.heappush(priority_queue, (heuristic_func(start_pos, goal_pos), start_node))

    g_cost = {start_pos: 0}
    came_from = {start_pos: start_node}
    expanded_nodes_count = 0

    while priority_queue:
        current_f_cost, current_node = heapq.heappop(priority_queue)

        
        if current_node.cost > g_cost.get(current_node.position, float('inf')):
            continue

        expanded_nodes_count += 1 

        if current_node.position == goal_pos:
            return reconstruct_path(current_node), current_node.cost, expanded_nodes_count

        for neighbor_pos in grid.get_neighbors(current_node.position[0], current_node.position[1]):
            move_cost = grid.get_movement_cost(current_node.position, neighbor_pos)
            tentative_g_cost = current_node.cost + move_cost

            if tentative_g_cost < g_cost.get(neighbor_pos, float('inf')):
                g_cost[neighbor_pos] = tentative_g_cost
                f_cost = tentative_g_cost + heuristic_func(neighbor_pos, goal_pos)
                neighbor_node = Node(neighbor_pos, tentative_g_cost, current_node)
                heapq.heappush(priority_queue, (f_cost, neighbor_node))
                came_from[neighbor_pos] = neighbor_node

    return None, float('inf'), expanded_nodes_count



class SimulationController:
    
    def __init__(self, grid: Grid, agent: Agent, search_algorithm, heuristic_func=None):
        self.grid = grid
        self.agent = agent
        self.search_algorithm = search_algorithm
        self.heuristic_func = heuristic_func
        self.path = [] 
        self.path_index = 0
        self.goal_reached = False

    def plan_initial_path(self):
        
        if self.heuristic_func:
            path, cost, _ = self.search_algorithm(self.grid, self.agent.position, self.grid.goal_pos, self.heuristic_func) # Unpack all 3 values
        else:
            path, cost, _ = self.search_algorithm(self.grid, self.agent.position, self.grid.goal_pos) # Unpack all 3 values

        if path and path[0] == self.agent.position:
            self.path = path
            self.path_index = 0
            print(f"Initial path planned: {path} with cost {cost:.2f}")
        else:
            self.path = []
            self.path_index = 0
            print("Failed to plan initial path.")

    def run_step(self):
        
        if self.goal_reached:
            print("Agent already reached the goal.")
            return False

        
        print("\n--- Updating Dynamic Obstacles ---")
        self.grid.update_dynamic_obstacles_positions()
        # Optional: print dynamic obstacle positions
        for obs in self.grid.dynamic_obstacles:
            print(f"  Obstacle {obs.id} moved to ({obs.current_x}, {obs.current_y})")

        
        replan_needed = False
        next_step_pos = None
        if self.path and self.path_index + 1 < len(self.path):
            next_step_pos = self.path[self.path_index + 1]
            if self.grid.is_position_blocked(next_step_pos[0], next_step_pos[1]):
                print(f"!!! Collision detected at next step {next_step_pos}! Replanning needed.")
                replan_needed = True

        
        if replan_needed or not self.path or self.path_index >= len(self.path):
            print("--- Replanning Path ---")
            if self.agent.position == self.grid.goal_pos:
                self.goal_reached = True
                print("Agent is at goal, no replanning needed.")
                return False

            if self.heuristic_func:
                new_path, new_cost, _ = self.search_algorithm(self.grid, self.agent.position, self.grid.goal_pos, self.heuristic_func)
            else:
                new_path, new_cost, _ = self.search_algorithm(self.grid, self.agent.position, self.grid.goal_pos)

            if new_path and new_path[0] == self.agent.position:
                self.path = new_path
                self.path_index = 0
                print(f"New path planned: {new_path} with cost {new_cost:.2f}")
            else:
                self.path = []
                self.path_index = 0
                print("Failed to replan path. Agent is stuck or goal unreachable.")
                return False

        
        if self.path and self.path_index + 1 < len(self.path):
            next_pos = self.path[self.path_index + 1]
            cost_to_move = self.grid.get_movement_cost(self.agent.position, next_pos)

            
            if self.grid.is_position_blocked(next_pos[0], next_pos[1]):
                print(f"Agent cannot move to {next_pos} as it's now blocked, forcing replan in next step.")
                
                return True

            self.agent.position = next_pos
            self.agent.current_cost += cost_to_move
            self.agent.current_time += 1 
            self.path_index += 1
            print(f"Agent moved to {self.agent.position}. Total cost: {self.agent.current_cost:.2f}, Time: {self.agent.current_time}")

            if self.agent.position == self.grid.goal_pos:
                self.goal_reached = True
                print("Agent reached the goal!")
                return False 
            return True 
        elif self.path and self.path_index == len(self.path) - 1 and self.agent.position == self.grid.goal_pos:
            self.goal_reached = True
            print("Agent reached the goal!")
            return False 
        else:
            print("Agent has no path or path exhausted, and not at goal. Simulation halted.")
            return False



def generate_map_string(width, height, obstacle_density=0.2, start_pos=None, goal_pos=None, dynamic_obstacles_pos=None):
    
    grid_chars = [['.' for _ in range(width)] for _ in range(height)]

    all_possible_positions = [(x, y) for x in range(width) for y in range(height)]
    used_positions = set()

    
    num_obstacles = int(width * height * obstacle_density)
    obstacle_positions = random.sample(all_possible_positions, num_obstacles)
    for ox, oy in obstacle_positions:
        grid_chars[oy][ox] = '#'
        used_positions.add((ox, oy))

    
    if start_pos and start_pos not in used_positions:
        sx, sy = start_pos
        if 0 <= sx < width and 0 <= sy < height:
            grid_chars[sy][sx] = 'S'
            used_positions.add((sx, sy))
        else:
            print(f"Warning: start_pos {start_pos} is out of bounds.")
    elif not start_pos:
        available_positions = list(set(all_possible_positions) - used_positions)
        if available_positions:
            sx, sy = random.choice(available_positions)
            grid_chars[sy][sx] = 'S'
            used_positions.add((sx, sy))
        else:
            print("Warning: No available position for Start.")

    
    if goal_pos and goal_pos not in used_positions:
        gx, gy = goal_pos
        if 0 <= gx < width and 0 <= gy < height:
            grid_chars[gy][gx] = 'G'
            used_positions.add((gx, gy))
        else:
            print(f"Warning: goal_pos {goal_pos} is out of bounds.")
    elif not goal_pos:
        available_positions = list(set(all_possible_positions) - used_positions)
        if available_positions:
            gx, gy = random.choice(available_positions)
            grid_chars[gy][gx] = 'G'
            used_positions.add((gx, gy))
        else:
            print("Warning: No available position for Goal.")

    
    if dynamic_obstacles_pos:
        for dox, doy in dynamic_obstacles_pos:
            if (dox, doy) not in used_positions and 0 <= dox < width and 0 <= doy < height:
                grid_chars[doy][dox] = 'D'
                used_positions.add((dox, doy))
            else:
                print(f"Warning: Dynamic obstacle position {dox, doy} is already occupied or out of bounds.")

    
    map_string = "\n".join(["".join(row) for row in grid_chars])
    return map_string



def run_experiment(grid, start_pos, goal_pos, search_algorithm, algorithm_name, heuristic_func=None):
    
    start_time = time.perf_counter()
    if heuristic_func:
        path, cost, expanded_nodes = search_algorithm(grid, start_pos, goal_pos, heuristic_func)
    else:
        path, cost, expanded_nodes = search_algorithm(grid, start_pos, goal_pos)
    end_time = time.perf_counter()

    execution_time = (end_time - start_time) * 1000 

    return {
        'algorithm': algorithm_name,
        'start_pos': start_pos,
        'goal_pos': goal_pos,
        'path_found': path is not None,
        'path_length': len(path) if path else 0,
        'path_cost': cost,
        'execution_time_ms': execution_time,
        'nodes_expanded': expanded_nodes
    }

def conduct_full_experiment(map_configs, algorithms):
    
    all_results = []

    print("\n--- Conducting Full Experiment --- ")
    for map_config in map_configs:
        map_name = map_config['name']
        map_string = map_config['map_string']
        start_pos = map_config['start_pos']
        goal_pos = map_config['goal_pos']

        print(f"\nProcessing Map: {map_name}")
        grid = Grid.from_map_string(map_string) 

        
        if start_pos:
            grid.start_pos = start_pos
        if goal_pos:
            grid.goal_pos = goal_pos

        
        if not grid.start_pos or not grid.goal_pos:
            print(f"Skipping {map_name}: Start or Goal not found/set.")
            continue

        for algo_config in algorithms:
            algo_name = algo_config['name']
            algo_func = algo_config['function']
            heuristic = algo_config['heuristic']

            print(f"  Running {algo_name} on {map_name}...")
            result = run_experiment(grid, grid.start_pos, grid.goal_pos, algo_func, algo_name, heuristic)
            result['map_name'] = map_name
            result['map_width'] = grid.width
            result['map_height'] = grid.height
            all_results.append(result)

    return pd.DataFrame(all_results)
