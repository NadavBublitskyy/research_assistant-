import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random

# ==========================================
#        USER CONFIGURATION SECTION
# ==========================================

# --- 1. Graph Structure ---
NUM_HUB_NODES = 5
HUB_EDGE_RANGE = (8, 10)

NUM_REGULAR_NODES = 100
REGULAR_EDGE_RANGE = (1, 3)

# --- 2. Visual Layout ---
# 'spring' = Organic (uses LAYOUT_K)
# 'kamada' = Optimized spacing (needs scipy)
LAYOUT_ALGORITHM = 'kamada'
LAYOUT_K = 0.7

# --- 3. Color Settings ---
BASE_COLOR_NAME = 'Red'

# True  = Gradient (Light -> Dark)
# False = Rainbow (Generations: Red -> Yellow -> Orange...)
USE_COLOR_ESCALATION = False

COLOR_INTENSITY_OFFSET = 0.3
COLOR_INTENSITY_MULTIPLIER = 2

# Rainbow Palette (Used if Escalation is False)
COLOR_PALETTE = [
    "red",
    "black",
    "orange",
    "pink",
    "grey",
    "black"
]
# --- 4. Node Appearance ---
SHOW_NEIGHBOR_COUNT = True
SIZE_REGULAR_NODE = 100
SIZE_HUB_NODE = 150

# --- 5. Infection Rules ---
PROB_INFECTION_REGULAR = 1
PROB_INFECTION_HUB = 1
START_NODE_ID = 0

# --- 6. Animation Settings ---
FRAME_INTERVAL_MS = 2000
MAX_FRAMES = 100
START_DELAY_FRAMES = 1   # 1 frame = 1 * FRAME_INTERVAL_MS pause



# ==========================================
#      END OF CONFIGURATION
# ==========================================

def create_custom_graph():
    degrees = []
    node_types = {}

    for i in range(NUM_HUB_NODES):
        deg = random.randint(*HUB_EDGE_RANGE)
        degrees.append(deg)
        node_types[i] = "Hub"

    for i in range(NUM_HUB_NODES, NUM_HUB_NODES + NUM_REGULAR_NODES):
        deg = random.randint(*REGULAR_EDGE_RANGE)
        degrees.append(deg)
        node_types[i] = "Regular"

    if sum(degrees) % 2 != 0:
        degrees[0] += 1

    G_multi = nx.configuration_model(degrees, seed=42)
    G = nx.Graph(G_multi)
    G.remove_edges_from(nx.selfloop_edges(G))

    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            u = random.choice(list(components[i]))
            v = random.choice(list(components[i + 1]))
            G.add_edge(u, v)

    return G, node_types


# --- Setup Graph ---
G, node_types = create_custom_graph()

print(f"Applying layout algorithm: {LAYOUT_ALGORITHM}...")
if LAYOUT_ALGORITHM == 'kamada':
    try:
        pos = nx.kamada_kawai_layout(G)
    except:
        print("Scipy not found. Using Spring.")
        pos = nx.spring_layout(G, seed=42, k=LAYOUT_K, iterations=50)
else:
    pos = nx.spring_layout(G, seed=42, k=LAYOUT_K, iterations=50)

node_sizes = [SIZE_HUB_NODE if node_types[n] == "Hub" else SIZE_REGULAR_NODE for n in G.nodes()]
labels = {n: G.degree[n] if SHOW_NEIGHBOR_COUNT else n for n in G.nodes()}

node_status = {n: 0 for n in G.nodes()}
node_colors = {n: '#e6f2ff' for n in G.nodes()}
#d
try:
    base_cmap = plt.get_cmap(BASE_COLOR_NAME.capitalize() + 's')
except ValueError:
    base_cmap = plt.get_cmap('Reds')

# Start Node
if START_NODE_ID in G.nodes():
    node_status[START_NODE_ID] = 1
    if USE_COLOR_ESCALATION:
        node_colors[START_NODE_ID] = base_cmap(COLOR_INTENSITY_OFFSET)
    else:
        node_colors[START_NODE_ID] = COLOR_PALETTE[0]
else:
    node_status[0] = 1
    node_colors[0] = 'red'

# --- Animation ---
# Increased width (14) to make room for the legend
fig, ax = plt.subplots(figsize=(14, 10))


last_advanced_frame = {"v": None}

def update(frame):
    # only ADVANCE simulation if this is a new frame
    if last_advanced_frame["v"] != frame:
        last_advanced_frame["v"] = frame

        infected_nodes = [n for n, s in node_status.items() if s == 1]
        newly_infected = set()  # also fixes duplicates

        for node in infected_nodes:
            chance = PROB_INFECTION_HUB if node_types[node] == "Hub" else PROB_INFECTION_REGULAR
            for neighbor in G.neighbors(node):
                if node_status[neighbor] == 0 and random.random() < chance:
                    newly_infected.add(neighbor)

        print("frame", frame, "new:", len(newly_infected))

        for n in newly_infected:
            node_status[n] = 1
            if USE_COLOR_ESCALATION:
                progress = min(frame, 20) / 20.0
                intensity = COLOR_INTENSITY_OFFSET + (COLOR_INTENSITY_MULTIPLIER * progress)
                node_colors[n] = base_cmap(min(intensity, 1.0))
            else:
                color_index = (frame + 1) % len(COLOR_PALETTE)
                node_colors[n] = COLOR_PALETTE[color_index]

    # --- ALWAYS DRAW ---
    ax.clear()
    nx.draw(G, pos, ax=ax,
            node_color=[node_colors[n] for n in G.nodes()],
            labels=labels,
            with_labels=True,
            font_size=9, font_weight='bold', font_color='black',
            node_size=node_sizes,
            edge_color='#bababa', width=0.8, alpha=1.0)

    inf_count = sum(node_status.values())
    ax.set_title(f"Frame {frame} | Infected: {inf_count}/{len(G.nodes())}", fontsize=14)
    ax.set_axis_off()



ani = animation.FuncAnimation(fig, update, frames=MAX_FRAMES, interval=FRAME_INTERVAL_MS, repeat=False)

plt.show()