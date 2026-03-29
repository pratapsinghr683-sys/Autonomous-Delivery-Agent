import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Setup Maps ---
print("--- Setting Up Maps ---")

# 1. Small Map
small_map_string = generate_map_string(width=5, height=5, obstacle_density=0.1, start_pos=(0,0), goal_pos=(4,4))
small_grid = Grid.from_map_string(small_map_string)
print(f"\nSmall Map:\n{small_map_string}")

# 2. Medium Map
medium_map_string = generate_map_string(width=15, height=15, obstacle_density=0.2)
medium_grid = Grid.from_map_string(medium_map_string)
print(f"\nMedium Map:\n(Details hidden due to size, but loaded)")

# 3. Large Map
large_map_string = generate_map_string(width=25, height=25, obstacle_density=0.25)
large_grid = Grid.from_map_string(large_map_string)
print(f"\nLarge Map:\n(Details hidden due to size, but loaded)")

# 4. Dynamic Obstacle Challenge Map (modified for effective replanning demo)
dynamic_replan_map_string = """
S.X
.D.
.G.
"""
dynamic_grid = Grid.from_map_string(dynamic_replan_map_string)
if dynamic_grid.dynamic_obstacles:
    do = dynamic_grid.dynamic_obstacles[0]
    do.movement_pattern = [(-1,0)] 
print(f"\nDynamic Replanning Demo Map:\n{dynamic_replan_map_string}")


map_configs = [
    {
        'name': 'Small Map',
        'map_string': small_map_string,
        'start_pos': small_grid.start_pos,
        'goal_pos': small_grid.goal_pos
    },
    {
        'name': 'Medium Map',
        'map_string': medium_map_string,
        'start_pos': medium_grid.start_pos,
        'goal_pos': medium_grid.goal_pos
    },
    {
        'name': 'Large Map',
        'map_string': large_map_string,
        'start_pos': large_grid.start_pos,
        'goal_pos': large_grid.goal_pos
    },
    {
        'name': 'Dynamic Obstacle Challenge Map',
        'map_string': dynamic_replan_map_string,
        'start_pos': dynamic_grid.start_pos,
        'goal_pos': dynamic_grid.goal_pos
    }
]

algorithms = [
    {'name': 'BFS', 'function': bfs_search, 'heuristic': None},
    {'name': 'UCS', 'function': ucs_search, 'heuristic': None},
    {'name': 'A* Manhattan', 'function': astar_search, 'heuristic': manhattan_distance_heuristic}
]

results_df = conduct_full_experiment(map_configs, algorithms)
print("\n--- Experiment Results ---")
print(results_df.to_markdown(index=False))


plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='map_name', y='execution_time_ms', hue='algorithm', palette='viridis')
plt.title('Algorithm Execution Time (ms) by Map Type')
plt.xlabel('Map Type')
plt.ylabel('Execution Time (ms)')
plt.yscale('log')
plt.legend(title='Algorithm')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=results_df, x='map_name', y='nodes_expanded', hue='algorithm', palette='viridis')
plt.title('Nodes Expanded by Algorithm and Map Type')
plt.xlabel('Map Type')
plt.ylabel('Nodes Expanded')
plt.yscale('log')
plt.legend(title='Algorithm')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


print("\n--- Running Dynamic Replanning Simulation (Proof-of-Concept) ---")


grid_replan_demo = Grid.from_map_string(dynamic_replan_map_string)
if grid_replan_demo.dynamic_obstacles:
    do_demo = grid_replan_demo.dynamic_obstacles[0]
    do_demo.movement_pattern = [(-1,0)]

agent_replan_demo = Agent(grid_replan_demo.start_pos)
sim_controller_replan_demo = SimulationController(grid_replan_demo, agent_replan_demo, astar_search, manhattan_distance_heuristic)

sim_controller_replan_demo.plan_initial_path()

max_sim_steps = 10
sim_steps_run = 0

while not sim_controller_replan_demo.goal_reached and sim_steps_run < max_sim_steps:
    print(f"\n>>> Simulation Step {sim_steps_run + 1} <<<")
    continue_simulation = sim_controller_replan_demo.run_step()

    if continue_simulation or sim_controller_replan_demo.goal_reached:
        print(f"Agent Status: Position={sim_controller_replan_demo.agent.position}, Cost={sim_controller_replan_demo.agent.current_cost:.2f}, Time={sim_controller_replan_demo.agent.current_time}")

    if not continue_simulation and not sim_controller_replan_demo.goal_reached:
        print("Simulation halted unexpectedly (e.g., agent stuck or no path).")
        break

    sim_steps_run += 1

print("\n--- Dynamic Replanning Simulation Summary ---")
print(f"Final Agent Position: {sim_controller_replan_demo.agent.position}")
print(f"Total Cost: {sim_controller_replan_demo.agent.current_cost:.2f}")
print(f"Total Time Steps: {sim_controller_replan_demo.agent.current_time}")
print(f"Goal Reached: {sim_controller_replan_demo.goal_reached}")

if not sim_controller_replan_demo.goal_reached:
    print(f"Note: Agent did not reach goal within {max_sim_steps} steps.")

print("\nAll simulations and experiments completed.")
