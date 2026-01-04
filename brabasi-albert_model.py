import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random

# ==========================================
#        USER CONFIGURATION SECTION
# ==========================================

# --- 1. Graph Structure (Barabási-Albert) ---
NUM_NODES = 100
EDGES_TO_ATTACH = 1  # Low number = Tree-like structure
SEED = 4  # Fixed seed for consistent graph shape

# --- 2. Hub Definition & Infection Rules ---
# Since this is a random graph, we define a "Hub" by how many connections it ends up with.
HUB_THRESHOLD = 3  # Any node with > 3 neighbors is treated as a Hub

PROB_INFECTION_REGULAR = 1  # Chance a small node infects a neighbor
PROB_INFECTION_HUB = 1  # Chance a Hub infects a neighbor

START_NODE_STRATEGY = 'max_degree'  # 'max_degree' (start at biggest hub) or 'random' or specific ID (e.g. 0)

# --- 3. Visual Layout ---
# 'spring' = Organic (uses LAYOUT_K)
# 'kamada' = Optimized spacing (needs scipy library)
LAYOUT_ALGORITHM = 'kamada'
LAYOUT_K = 0.8  # Soreness for spring layout

# --- 4. Color Settings ---
BASE_COLOR_NAME = 'Red'  # Options: 'Blue', 'Green', 'Purple', 'Orange', 'Red'

# Escalation Toggle:
# True  = Gradient (Light -> Dark of the BASE_COLOR)
# False = Rainbow (Generations: Red -> Yellow -> Orange...)
USE_COLOR_ESCALATION = False

# Gradient Math (If Escalation is True)
COLOR_INTENSITY_OFFSET = 0.3  # Starting brightness
COLOR_INTENSITY_MULTIPLIER = 0.7  # How much darker it gets

# Rainbow Palette (If Escalation is False)
COLOR_PALETTE = ['red', 'gold', 'orange', 'black', 'cyan', 'dodgerblue', 'purple', 'magenta']

# --- 5. Node Appearance ---
# True = Show neighbor count, False = Show ID
SHOW_NEIGHBOR_COUNT = True

SIZE_REGULAR_NODE = 50
SIZE_HUB_NODE = 75

# --- 6. Animation Settings ---
FRAME_INTERVAL_MS = 1200
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
else:
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
    node_colors[start_node] = base_cmap(COLOR_INTENSITY_OFFSET)
else:
    node_colors[start_node] = COLOR_PALETTE[0]

# --- 5. Animation ---
# Increased figure width to fit Legend
fig, ax = plt.subplots(figsize=(14, 10))


def update(frame):
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
            intensity = COLOR_INTENSITY_OFFSET + (COLOR_INTENSITY_MULTIPLIER * progress)
            intensity = min(intensity, 1.0)
            node_colors[n] = base_cmap(intensity)
        else:
            # Rainbow/Generations Mode
            color_index = (frame + 1) % len(COLOR_PALETTE)
            node_colors[n] = COLOR_PALETTE[color_index]

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
    if not USE_COLOR_ESCALATION:
        legend_patches = []
        generations_to_show = min(frame + 2, len(COLOR_PALETTE))
        for i in range(generations_to_show):
            lbl = f"Gen {i}" + (" (Patient Zero)" if i == 0 else "")
            legend_patches.append(mpatches.Patch(color=COLOR_PALETTE[i], label=lbl))

        ax.legend(handles=legend_patches, title="Infection Stages",
                  loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
    else:
        start_p = mpatches.Patch(color=base_cmap(COLOR_INTENSITY_OFFSET), label="Early Infection")
        end_p = mpatches.Patch(color=base_cmap(1.0), label="Late Infection")
        ax.legend(handles=[start_p, end_p], title=f"{BASE_COLOR_NAME} Gradient",
                  loc='upper left', bbox_to_anchor=(1, 1))

    # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)

    inf_count = sum(node_status.values())
    ax.set_title(f"Hubs (>{HUB_THRESHOLD} neighbors) spread faster\nFrame {frame} | Infected: {inf_count}/{NUM_NODES}",
                 fontsize=14)
    ax.set_axis_off()


ani = animation.FuncAnimation(fig, update, frames=MAX_FRAMES, interval=FRAME_INTERVAL_MS, repeat=False)

plt.show()