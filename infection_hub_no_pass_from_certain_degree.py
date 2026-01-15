import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import random
import math
import textwrap

# ==========================================
#        USER CONFIGURATION SECTION
# ==========================================
# --- 1. Graph Structure (Barabási-Albert) ---
NUM_NODES = 100
EDGES_TO_ATTACH = 2
SEED = 70

# --- 2. Hub Definition & Infection Rules ---
HUB_THRESHOLD = 20
PROB_INFECTION_REGULAR = 1
PROB_INFECTION_HUB = 0
START_NODE_STRATEGY = 13

# --- 3. Visual Layout ---
LAYOUT_ALGORITHM = 'kamada'
LAYOUT_K = 2

# --- 4. Color Settings ---
BASE_COLOR_NAME = 'Green'
USE_COLOR_ESCALATION = True
COLOR_INTENSITY_OFFSET = 1
COLOR_INTENSITY_MULTIPLIER = 1.5

COLOR_PALETTE = [
    "red", "orange", "yellow", "green", "blue", "indigo", "violet",
    "lightcoral", "lightsalmon", "lightyellow", "lightgreen", "lightblue", "lightsteelblue", "lavender",
    "darkred", "darkorange", "gold", "darkgreen", "darkblue", "navy", "darkviolet",
    "crimson", "coral", "gold", "limegreen", "skyblue", "mediumslateblue", "mediumorchid"
]

# --- 5. Node Appearance ---
SHOW_NEIGHBOR_COUNT = True
SIZE_REGULAR_NODE = 300
SIZE_HUB_NODE = 350

# --- 6. Animation Settings ---
FRAME_INTERVAL_MS = 2000
INITIAL_DELAY_SECONDS = 1
MAX_FRAMES = 100

# ==========================================
#      END OF CONFIGURATION
# ==========================================

# --- 1. Setup Graph ---
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
node_sizes = []
for n in G.nodes():
    if G.degree[n] > HUB_THRESHOLD:
        node_sizes.append(SIZE_HUB_NODE)
    else:
        node_sizes.append(SIZE_REGULAR_NODE)

labels = {n: G.degree[n] if SHOW_NEIGHBOR_COUNT else n for n in G.nodes()}

try:
    base_cmap = plt.get_cmap(BASE_COLOR_NAME.capitalize() + 's')
except ValueError:
    base_cmap = plt.get_cmap('Reds')

node_status = {n: 0 for n in G.nodes()}
node_colors = {n: '#e6f2ff' for n in G.nodes()}

# --- 4. Set Start Node ---
if START_NODE_STRATEGY == 'max_degree':
    start_node = max(dict(G.degree).items(), key=lambda x: x[1])[0]
elif isinstance(START_NODE_STRATEGY, int) and START_NODE_STRATEGY in G.nodes():
    start_node = START_NODE_STRATEGY
else:
    start_node = 0

node_status[start_node] = 1
if USE_COLOR_ESCALATION:
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

ani_container = {"ani": None}


def update(frame):
    if frame == 0:
        newly_infected = []
    else:
        infected_nodes = [n for n, s in node_status.items() if s == 1]
        newly_infected = []

        # --- Spreading Logic ---
        for node in infected_nodes:
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
                progress = min(frame, 20) / 20.0
                intensity = COLOR_INTENSITY_OFFSET - (COLOR_INTENSITY_MULTIPLIER * progress)
                intensity = max(0.0, min(intensity, 1.0))
                node_colors[n] = base_cmap(intensity)
            else:
                color_index = frame % len(COLOR_PALETTE)
                node_colors[n] = COLOR_PALETTE[color_index]

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

        late_progress = 0.5
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
    title_text = f"The infection begins in a hub vertex\nVertices with degree > {HUB_THRESHOLD} don't pass the virus\n"
    if frame == 0:
        title_text += f"Starting... | Infected: {inf_count}/{NUM_NODES}"
    elif inf_count == NUM_NODES:
        title_text += f"Frame {frame} | Complete! All {NUM_NODES} nodes infected"
    else:
        title_text += f"Frame {frame} | Infected: {inf_count}/{NUM_NODES}"

    ax.set_title(title_text, fontsize=14)
    ax.set_axis_off()


initial_delay_frames = math.ceil(INITIAL_DELAY_SECONDS * 1000 / FRAME_INTERVAL_MS)
frame_sequence = [0] + [0] * initial_delay_frames + list(range(1, MAX_FRAMES + 1))

ani = animation.FuncAnimation(fig, update, frames=frame_sequence, interval=FRAME_INTERVAL_MS, repeat=False)
ani_container["ani"] = ani

plt.show()