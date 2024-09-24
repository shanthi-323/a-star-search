#start
import heapq
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# Define the grid dimensions
rows, cols = 6, 10

# Define the colors for each cell type
colors = {
    'obstacle': 'gray',
    'treasure': 'orange',
    'reward_R1': 'springgreen',
    'reward_R2': 'springgreen',
    'trap_T1': 'mediumorchid',
    'trap_T2': 'mediumorchid',
    'trap_T3': 'mediumorchid',
    'trap_T4': 'mediumorchid',
    'entry': 'blue',
    'empty': 'white'
}

# Define the size of each hexagon
hex_size = 1.0

# Define the coordinates for each type
coordinates = {
    'obstacle': [(3, 0), (2, 2), (3, 3), (2, 4), (4, 4), (3, 6), (4, 6), (4, 7), (1, 8)],
    'treasure': [(4, 3), (1, 4), (3, 7), (3, 9)],
    'reward_R1': [(3, 1), (0, 4)],
    'reward_R2': [(5, 5), (2, 7)],
    'trap_T1': [(2, 8)],
    'trap_T2': [(1, 1), (4, 2)],
    'trap_T3': [(3, 5), (1, 6)],
    'trap_T4': [(1, 3)],
    'entry': [(0, 0)]
}

# Combine all coordinates into one grid representation
virtual_world_hex = [['.' for _ in range(cols)] for _ in range(rows)]
for key, value in coordinates.items():
    for (row, col) in value:
        virtual_world_hex[row][col] = key.split('_')[-1].upper() if '_' in key else key[0].upper()

# Define traps effects 
trap_effects = {
    "T1": {"energy_multiplier": 2},
    "T2": {"step_multiplier": 2},
    "T3": {"teleport": 2},
    "T4": {"remove_treasures": True}
}

# Define rewards effects
reward_effects = {
    "R1": {"energy_multiplier": 0.5},
    "R2": {"step_multiplier": 0.5}
}

start = (0, 0)
treasures = [(4, 3), (1, 4), (3, 7), (3, 9)]
rewards = coordinates['reward_R1'] + coordinates['reward_R2']

# Get neighbors in a hexagonal grid based on the  logic
def get_neighbors(cell):
    row, col = cell
    neighbors = []
    
    if col % 2 == 0:  # Lower column (even index)
        directions = [
            (0, 1), (1, 1),  # Right
            (0, -1), (1, -1),  # Left
            (-1, 0), (1, 0)  # Top and Bottom
        ]
    else:  # Higher column (odd index)
        directions = [
            (-1, 1), (0, 1),  # Right
            (-1, -1), (0, -1),  # Left
            (-1, 0), (1, 0)  # Top and Bottom
        ]
    
    for dr, dc in directions:
        r, c = row + dr, col + dc
        if 0 <= r < rows and 0 <= c < cols:
            neighbors.append((r, c))

    return neighbors

# Heuristic function for hexagonal grid
def heuristic(cell, uncollected_treasures, uncollected_rewards):
    if not uncollected_treasures and not uncollected_rewards:
        return 0
    combined_targets = uncollected_treasures | uncollected_rewards
    return min(abs(cell[0] - t[0]) + abs(cell[1] - t[1]) for t in combined_targets)

# A* Search Algorithm for hexagonal grid with treasure collection
def a_star_search(world, start, treasures, rewards):
    rows, cols = len(world), len(world[0])
    open_list = []
    heapq.heappush(open_list, (0, 0, start, [], frozenset(), 100, 0, 1, 1, None))
    closed_list = set()
    g_scores = {(start, frozenset()): 0}
    
    while open_list:
        _, current_cost, current_cell, path, collected_treasures, remaining_energy, total_steps, energy_multiplier, step_multiplier, last_direction = heapq.heappop(open_list)
        
        if (current_cell, collected_treasures) in closed_list:
            continue
        
        path = path + [current_cell]
        
        # Check if current cell is a treasure and add it to collected_treasures
        if current_cell in treasures:
            collected_treasures = frozenset(set(collected_treasures) | {current_cell})
            print(f"Treasure collected at {current_cell}")
        
        #If all treasures collected return the result
        if set(collected_treasures) == set(treasures):
            return path, set(collected_treasures), remaining_energy, total_steps
        
        closed_list.add((current_cell, collected_treasures))
        
        for neighbor in get_neighbors(current_cell):
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                cell_type = world[neighbor[0]][neighbor[1]]
                
                # Calculate step cost based on the type of node
                if cell_type == 'O':
                    step_cost = 2000
                elif cell_type == 'T1':
                    step_cost = 200
                elif cell_type == 'T2':
                    step_cost = 300
                elif cell_type == 'T3':
                    step_cost = 400
                elif cell_type == 'T4':
                    step_cost = 500
                else:
                    step_cost = 1 * step_multiplier
                
                energy_cost = 1 * energy_multiplier
                new_collected_treasures = set(collected_treasures)
                new_last_direction = (neighbor[0] - current_cell[0], neighbor[1] - current_cell[1])
                new_path = path.copy()

                # Apply reward effects
                if cell_type in reward_effects:
                    reward = reward_effects[cell_type]
                    if "energy_multiplier" in reward:
                        new_energy_multiplier = energy_multiplier * reward["energy_multiplier"]
                    if "step_multiplier" in reward:
                        new_step_multiplier = step_multiplier * reward["step_multiplier"]
                else:
                    new_energy_multiplier = energy_multiplier
                    new_step_multiplier = step_multiplier
                
                # Check if neighbor cell is a treasure
                if neighbor in treasures:
                    new_collected_treasures.add(neighbor)
                
                # Implement trap effects
                if cell_type == 'T1':
                    new_energy_multiplier *= trap_effects["T1"]["energy_multiplier"]
                elif cell_type == 'T2':
                    new_step_multiplier *= trap_effects["T2"]["step_multiplier"]
                elif cell_type == 'T3':
                    if len(new_path) >= 2:
                        neighbor = new_path[-2]
                        new_path = new_path[:-1]
                elif cell_type == 'T4':
                    new_collected_treasures = frozenset()

                new_remaining_energy = remaining_energy - energy_cost
                if new_remaining_energy <= 0:
                    continue

                new_total_steps = total_steps + step_cost

                tentative_g_score = current_cost + step_cost
                
                neighbor_state = (neighbor, frozenset(new_collected_treasures))
                if neighbor_state not in g_scores or tentative_g_score < g_scores[neighbor_state]:
                    g_scores[neighbor_state] = tentative_g_score
                    f_score = tentative_g_score + heuristic(neighbor, set(treasures) - new_collected_treasures, set(rewards) - new_collected_treasures)
                    heapq.heappush(open_list, (f_score, tentative_g_score, neighbor, new_path, frozenset(new_collected_treasures), new_remaining_energy, new_total_steps, new_energy_multiplier, new_step_multiplier, new_last_direction))
                    if neighbor in treasures or cell_type in reward_effects:
                        print(f"Moving from {current_cell} to {neighbor}, energy cost: {energy_cost}, step cost: {step_cost}, remaining energy: {new_remaining_energy}, total steps: {new_total_steps}")
    
    return None, set(), 0, 0

# Visualization function for path
def visualize_path(world, path):
    fig, ax = plt.subplots(figsize=(14, 10))
    hex_size = 1.0
    for row in range(rows):
        for col in range(cols):
            x = col * hex_size * 3 / 2
            y = (rows - 1 - row) * hex_size * np.sqrt(3) + (col % 2) * (hex_size * np.sqrt(3) / 2)
            cell_type = 'empty'
            for key, value in coordinates.items():
                if (row, col) in value:
                    cell_type = key
                    break
            color = colors.get(cell_type, 'white')
            hex = patches.RegularPolygon(
                (x, y),
                numVertices=6,
                radius=hex_size,
                orientation=np.radians(30),
                facecolor=color,
                edgecolor='black'
            )
            ax.add_patch(hex)
            if (row, col) in path:
                hex.set_edgecolor('red')
                hex.set_linewidth(3.5)
            ax.text(x, y, f"({row},{col})", ha='center', va='center', size=12)

    ax.set_xlim(-1, cols * hex_size * 3 / 2 + 1)
    ax.set_ylim(-1, rows * hex_size * np.sqrt(3) + 1)
    ax.set_aspect('equal')
    plt.show()

# Run the A* search algorithm
path, collected_treasures, remaining_energy, total_steps = a_star_search(virtual_world_hex, start, treasures, rewards)
print("Path found:", path)
print("Collected treasures:", collected_treasures)
print("Remaining energy:", remaining_energy)
print("Total steps:", total_steps)

# Visualize the path
visualize_path(virtual_world_hex, path)






