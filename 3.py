import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random
import math
import textwrap
import numpy as np


def forceatlas2_layout(G, iterations=50, gravity=1.0, scaling_ratio=2.0, 
                       strong_gravity=False, linlog_mode=False, seed=None):
    """
    Custom ForceAtlas2 implementation for graph layout.
    
    Parameters:
    - G: NetworkX graph
    - iterations: Number of iterations to run
    - gravity: Gravity constant pulling nodes to center
    - scaling_ratio: Repulsion scaling (higher = more spread out)
    - strong_gravity: If True, gravity increases with distance
    - linlog_mode: If True, use log attraction (tighter clusters)
    - seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}
    
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Initialize random positions
    pos = np.random.rand(n, 2) * 100
    
    # Calculate node degrees for mass
    degrees = np.array([G.degree(node) + 1 for node in nodes], dtype=float)
    
    # ForceAtlas2 iterations
    speed = 1.0
    speed_efficiency = 1.0
    
    for iteration in range(iterations):
        forces = np.zeros((n, 2))
        
        # Repulsion between all node pairs
        for i in range(n):
            for j in range(i + 1, n):
                diff = pos[i] - pos[j]
                dist = np.linalg.norm(diff)
                if dist < 0.01:
                    dist = 0.01
                
                # Repulsion force (degree-based)
                repulsion = scaling_ratio * degrees[i] * degrees[j] / dist
                
                direction = diff / dist
                forces[i] += direction * repulsion
                forces[j] -= direction * repulsion
        
        # Attraction along edges
        for edge in G.edges():
            i = node_to_idx[edge[0]]
            j = node_to_idx[edge[1]]
            
            diff = pos[j] - pos[i]
            dist = np.linalg.norm(diff)
            if dist < 0.01:
                dist = 0.01
            
            if linlog_mode:
                # LinLog mode: log attraction
                attraction = math.log(1 + dist)
            else:
                # Standard attraction
                attraction = dist
            
            direction = diff / dist
            forces[i] += direction * attraction
            forces[j] -= direction * attraction
        
        # Gravity (pull towards center)
        center = np.mean(pos, axis=0)
        for i in range(n):
            diff = center - pos[i]
            dist = np.linalg.norm(diff)
            if dist > 0.01:
                if strong_gravity:
                    grav_force = gravity * degrees[i]
                else:
                    grav_force = gravity * degrees[i] / dist
                forces[i] += (diff / dist) * grav_force
        
        # Apply forces with adaptive speed
        max_force = np.max(np.linalg.norm(forces, axis=1))
        if max_force > 0:
            adaptive_speed = speed * speed_efficiency / max_force
            pos += forces * min(adaptive_speed, 10.0)
        
        # Decay speed over time for stability
        speed *= 0.99
    
    # Convert to dictionary format
    return {node: (pos[i][0], pos[i][1]) for i, node in enumerate(nodes)}

# ==========================================
#        USER CONFIGURATION SECTION
# ==========================================
# For the standard Barabási–Albert (BA) model, you need just two conceptual
# parameters: how many nodes the final graph should have and how many edges
# each new node brings in when it is added
# --- 1. Graph Structure (Barabási-Albert) ---
NUM_NODES = 100
EDGES_TO_ATTACH = 2  # Low number = Tree-like structure
SEED = 70  # Random seed for reproducible graph generation. Set to None for random graph each run.

# --- 2. Hub Definition & Infection Rules ---
# Since this is a random graph, we define a "Hub" by how many connections it ends up with.
HUB_THRESHOLD = 6  # Any node with > 3 neighbors is treated as a Hub

PROB_INFECTION_REGULAR = 1  # Chance a small node infects a neighbor
PROB_INFECTION_HUB = 0  # Chance a Hub infects a neighbor

START_NODE_STRATEGY = 40  # 'max_degree' (start at biggest hub) or 'random' or specific ID (e.g. 0)

# --- 3. Visual Layout ---
# 'spring' = Organic (uses LAYOUT_K)
# 'kamada' = Optimized spacing (needs scipy library)
# 'forceatlas2' = ForceAtlas2 algorithm (needs fa2 library: pip install fa2)
LAYOUT_ALGORITHM = 'kamada'
LAYOUT_K = 2  # Soreness for spring layout

# ForceAtlas2 Parameters
FORCEATLAS2_ITERATIONS = 2000         # Keep high so it has time to expand
FORCEATLAS2_GRAVITY = 0.1             # DECREASED: Weak gravity lets nodes float to the edges
FORCEATLAS2_STRONG_GRAVITY_MODE = False # KEEP FALSE: Strong gravity crushes the graph
FORCEATLAS2_OUTBOUND_ATTRACTION_DISTRIBUTION = False # Standard attraction
FORCEATLAS2_LINLOG_MODE = False       # Keep False to avoid the "kite" shape
FORCEATLAS2_ADJUST_SIZES = True       # Essential for readability
FORCEATLAS2_EDGE_WEIGHT_INFLUENCE = 0.0 # KEEP 0: Prevents connected nodes from sticking too close
FORCEATLAS2_JITTER_TOLERANCE = 1.0
FORCEATLAS2_BARNES_HUT_OPTIMIZE = False
FORCEATLAS2_SCALING_RATIO = 80.0      # INCREASED MASSIVELY: This forces the graph to expand
# --- 4. Color Settings ---
BASE_COLOR_NAME = 'Red'  # Options: 'Blue', 'Green', 'Purple', 'Orange', 'Red'

# Escalation Toggle:
# True  = Gradient (Dark -> Bright of the BASE_COLOR)
# False = Rainbow (Generations: Red -> Yellow -> Orange...)
USE_COLOR_ESCALATION = True

# Gradient Math (If Escalation is True)
COLOR_INTENSITY_OFFSET = 1  # Starting intensity (dark, 0.0-1.0)
COLOR_INTENSITY_MULTIPLIER = 3  # How much to decrease (brightness range)

# Rainbow Palette (If Escalation is False)
# Cycles through rainbow colors with varying tones: Cycle 1 (standard), Cycle 2 (light), Cycle 3 (dark), Cycle 4 (bright)
COLOR_PALETTE = [
    # Cycle 1: Standard rainbow tones
    "red", "orange", "yellow", "green", "blue", "indigo", "violet",
    # Cycle 2: Light/pastel tones
    "lightcoral", "lightsalmon", "lightyellow", "lightgreen", "lightblue", "lightsteelblue", "lavender",
    # Cycle 3: Dark/deep tones
    "darkred", "darkorange", "gold", "darkgreen", "darkblue", "navy", "darkviolet",
    # Cycle 4: Bright/vivid tones
    "crimson", "coral", "gold", "limegreen", "skyblue", "mediumslateblue", "mediumorchid"
]

# --- 5. Node Appearance ---
# True = Show neighbor count, False = Show ID
SHOW_NEIGHBOR_COUNT = True

SIZE_REGULAR_NODE = 300
SIZE_HUB_NODE = 350

# --- 6. Animation Settings ---
FRAME_INTERVAL_MS = 2000
INITIAL_DELAY_SECONDS = 1  # Delay before animation starts (frame 0 duration)
MAX_FRAMES = 100

# ==========================================
#      END OF CONFIGURATION
# ==========================================

# --- 1. Setup Graph (Barabási-Albert) ---
G = nx.barabasi_albert_graph(n=NUM_NODES, m=EDGES_TO_ATTACH, seed=SEED)

# --- 2. Layout Logic ---
print(f"Applying layout algorithm: {LAYOUT_ALGORITHM}...")
if LAYOUT_ALGORITHM == 'kamada':
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        print("Scipy not found. Using Spring.")
        pos = nx.spring_layout(G, seed=SEED, k=LAYOUT_K, iterations=50)
elif LAYOUT_ALGORITHM == 'forceatlas2':
    print(f"Running ForceAtlas2 with {FORCEATLAS2_ITERATIONS} iterations...")
    pos = forceatlas2_layout(
        G,
        iterations=FORCEATLAS2_ITERATIONS,
        gravity=FORCEATLAS2_GRAVITY,
        scaling_ratio=FORCEATLAS2_SCALING_RATIO,
        strong_gravity=FORCEATLAS2_STRONG_GRAVITY_MODE,
        linlog_mode=FORCEATLAS2_LINLOG_MODE,
        seed=SEED
    )
    print(f"ForceAtlas2 layout completed successfully!")
elif LAYOUT_ALGORITHM == 'spring':
    pos = nx.spring_layout(G, seed=SEED, k=LAYOUT_K, iterations=50)
else:
    print(f"Warning: Unknown layout algorithm '{LAYOUT_ALGORITHM}'. Using Spring layout.")
    pos = nx.spring_layout(G, seed=SEED, k=LAYOUT_K, iterations=50)

# --- 3. Prepare Visuals ---
# Determine Sizes based on Hub Threshold
node_sizes = []
for n in G.nodes():
    if G.degree[n] > HUB_THRESHOLD:
        node_sizes.append(SIZE_HUB_NODE)
    else:
        node_sizes.append(SIZE_REGULAR_NODE)

# Determine Labels
labels = {n: G.degree[n] if SHOW_NEIGHBOR_COUNT else n for n in G.nodes()}

# Prepare Colors
try:
    base_cmap = plt.get_cmap(BASE_COLOR_NAME.capitalize() + 's')
except ValueError:
    base_cmap = plt.get_cmap('Reds')

# Initialize State
node_status = {n: 0 for n in G.nodes()}  # 0=Susceptible, 1=Infected
node_colors = {n: '#e6f2ff' for n in G.nodes()}  # Default background color

# --- 4. Set Start Node ---
if START_NODE_STRATEGY == 'max_degree':
    # Find node with most edges
    start_node = max(dict(G.degree).items(), key=lambda x: x[1])[0]
elif isinstance(START_NODE_STRATEGY, int) and START_NODE_STRATEGY in G.nodes():
    start_node = START_NODE_STRATEGY
else:
    start_node = 0

node_status[start_node] = 1
# Set initial color
if USE_COLOR_ESCALATION:
    # Use same formula as newly infected nodes with progress=0 (darkest)
    progress = 0.0
    intensity = COLOR_INTENSITY_OFFSET - (COLOR_INTENSITY_MULTIPLIER * progress)
    intensity = max(0.0, min(intensity, 1.0))
    node_colors[start_node] = base_cmap(intensity)
else:
    node_colors[start_node] = COLOR_PALETTE[0]

# --- 5. Animation Setup ---
fig, ax = plt.subplots(figsize=(14, 10))

# Reserve space for sidebar (Right 30%)
plt.subplots_adjust(left=0.05, bottom=0.05, top=0.90, right=0.70)

# Animation object container so update() can access it
ani_container = {"ani": None}


def update(frame):
    # For frame 0, just display the initial state without spreading
    if frame == 0:
        newly_infected = []
    else:
        infected_nodes = [n for n, s in node_status.items() if s == 1]
        newly_infected = []

        # --- Spreading Logic ---
        for node in infected_nodes:
            # Determine chance based on if the SPREADER is a Hub or Regular
            # (Using the Threshold logic from your code)
            if G.degree[node] > HUB_THRESHOLD:
                current_chance = PROB_INFECTION_HUB
            else:
                current_chance = PROB_INFECTION_REGULAR

            for neighbor in G.neighbors(node):
                if node_status[neighbor] == 0:
                    if random.random() < current_chance:
                        newly_infected.append(neighbor)

        # --- Update State & Colors ---
        for n in newly_infected:
            node_status[n] = 1

            if USE_COLOR_ESCALATION:
                # Gradient Mode
                progress = min(frame, 20) / 20.0
                intensity = COLOR_INTENSITY_OFFSET - (COLOR_INTENSITY_MULTIPLIER * progress)
                intensity = max(0.0, min(intensity, 1.0))
                node_colors[n] = base_cmap(intensity)
            else:
                # Rainbow/Generations Mode
                color_index = frame % len(COLOR_PALETTE)
                node_colors[n] = COLOR_PALETTE[color_index]

        # Check if all nodes are infected and stop animation
        if sum(node_status.values()) == NUM_NODES:
            if ani_container["ani"] is not None:
                ani_container["ani"].event_source.stop()

    ax.clear()

    # --- Draw Graph ---
    nx.draw(G, pos, ax=ax,
            node_color=[node_colors[n] for n in G.nodes()],
            labels=labels,
            with_labels=True,
            font_size=9, font_weight='bold', font_color='black',
            node_size=node_sizes,
            edge_color='#bababa', width=0.8, alpha=1.0)

    # --- Draw Legend ---
    # Legend settings
    LEGEND_X = 1.05  # X-coordinate for both boxes
    LEGEND_WIDTH_CHARS = 18  # Character width to match top and bottom sizes

    num_hubs_no_pass = sum(1 for n in G.nodes() if G.degree[n] > HUB_THRESHOLD and PROB_INFECTION_HUB == 0)

    # Wrap text tightly to match the width of the top legend
    hub_label_raw = f"Number of hubs (don't pass the infection): {num_hubs_no_pass}"
    wrapped_hub_label = textwrap.fill(hub_label_raw, width=LEGEND_WIDTH_CHARS)
    hub_text_patch = mpatches.Patch(color='none', label=wrapped_hub_label)

    # --- Top Legend (Gradient or Generations) ---
    if not USE_COLOR_ESCALATION:
        legend_patches = []
        if frame == 0:
            generations_to_show = 1
        else:
            generations_to_show = min(frame + 2, len(COLOR_PALETTE))
        for i in range(generations_to_show):
            lbl = f"Gen {i}" + (" (Patient Zero)" if i == 0 else "")
            legend_patches.append(mpatches.Patch(color=COLOR_PALETTE[i], label=lbl))

        legend1 = ax.legend(handles=legend_patches, title="Infection Stages",
                            loc='upper left', bbox_to_anchor=(LEGEND_X, 1.0), fontsize='small',
                            frameon=True, fancybox=True, shadow=False)
    else:
        early_progress = 0.0
        early_intensity = COLOR_INTENSITY_OFFSET - (COLOR_INTENSITY_MULTIPLIER * early_progress)
        early_intensity = max(0.0, min(early_intensity, 1.0))

        late_progress = 0.2
        late_intensity = COLOR_INTENSITY_OFFSET - (COLOR_INTENSITY_MULTIPLIER * late_progress)
        late_intensity = max(0.0, min(late_intensity, 1.0))

        start_p = mpatches.Patch(color=base_cmap(early_intensity), label="Early Infection")
        end_p = mpatches.Patch(color=base_cmap(late_intensity), label="Late Infection")

        legend1 = ax.legend(handles=[start_p, end_p], title=f"{BASE_COLOR_NAME} Gradient",
                            loc='upper left', bbox_to_anchor=(LEGEND_X, 1.0), fontsize='small',
                            frameon=True, fancybox=True, shadow=False)

    ax.add_artist(legend1)

    # --- Left Legend (Stats Box) ---
    # handlelength=0 and handletextpad=0 remove the invisible icon space,
    # forcing text to start at the immediate left edge.
    ax.legend(handles=[hub_text_patch],
              loc='upper left',
              bbox_to_anchor=(LEGEND_X - 0.15, 1.0),
              frameon=True, fancybox=True, shadow=False, fontsize='small',
              handlelength=0, handletextpad=0)  # <--- This fixes the "start from beginning" issue

    inf_count = sum(node_status.values())
    # For frame 0, show "Starting..." message
    if frame == 0:
        ax.set_title(
            f"The infection begins in a regular vertex\nvertices with degree above {HUB_THRESHOLD} pass the virus in a probability of {PROB_INFECTION_HUB}\nStarting... | Infected: {inf_count}/{NUM_NODES}",
            fontsize=14)
    elif inf_count == NUM_NODES:
        ax.set_title(
            f"The infection begins in a regular vertex\nvertices with degree above {HUB_THRESHOLD} pass the virus in a probability of {PROB_INFECTION_HUB}\nFrame {frame} | Complete! All {NUM_NODES} nodes infected",
            fontsize=14)
    else:
        ax.set_title(
            f"The infection begins in a regular vertex\nvertices with degree above {HUB_THRESHOLD} pass the virus in a probability of {PROB_INFECTION_HUB}\nFrame {frame} | Infected: {inf_count}/{NUM_NODES}",
            fontsize=14)
    ax.set_axis_off()


# Create frame sequence: show frame 0 once, then delay before frame 1 starts
initial_delay_frames = math.ceil(INITIAL_DELAY_SECONDS * 1000 / FRAME_INTERVAL_MS)
frame_sequence = [0] + [0] * initial_delay_frames + list(range(1, MAX_FRAMES + 1))

ani = animation.FuncAnimation(fig, update, frames=frame_sequence, interval=FRAME_INTERVAL_MS, repeat=False)
ani_container["ani"] = ani

plt.show()