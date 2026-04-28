import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arc
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/Shot Chart Details Raw"
OUT_DIR = PROJECT_ROOT / "data/Shot Feature"
# Assuming your data is loaded into a DataFrame named 'df'
# Filter for made shots
df = pd.read_csv(DATA_DIR / "raw_shotchart_S1_to_S4_main.csv")  # Adjust the filename as needed
made_shots = df[df['SHOT_MADE_FLAG'] == 1] 
curry_data = df[
    (df['PLAYER_NAME'].str.lower() == 'stephen curry') & 
    (df['SHOT_MADE_FLAG'] == 1)
]

print(f"Total made shots found for Stephen Curry: {len(curry_data)}")


def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    """Draws an NBA half-court on a matplotlib axis."""
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court
    # Create the basketball hoop
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint (outer box)
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color, fill=False)
    # The paint (inner box)
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color, fill=False)

    # Free Throw Top Arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180, 
                         linewidth=lw, color=color, fill=False)
    # Free Throw Bottom Arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0, 
                            linewidth=lw, color=color, linestyle='dashed')

    # Restricted Zone (Arc around the hoop)
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw, color=color)

    # 3 Point Line
    # Corner 3 point lines (straight lines)
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw, color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3 Point Arc
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw, color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0, 
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0, 
                           linewidth=lw, color=color)

    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc, center_inner_arc]

    if outer_lines:
        # Draw the half court boundaries
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw, color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax
# Assuming 'lebron_data' is already filtered from the previous step

plt.figure(figsize=(10, 9.5))
ax = plt.gca()

# 1. Plot the Density Heatmap first
sns.kdeplot(
    x=curry_data['LOC_X'],
    y=curry_data['LOC_Y'],
    fill=True,
    cmap='Reds',      # Switched to Reds so black lines are visible
    thresh=0.05,
    levels=50,
    zorder=1          # Ensures this plots underneath the lines
)

# 2. Draw the court lines on top
draw_court(ax, color='black', lw=2, outer_lines=True)

# 3. Adjust the axes so the court is centered and scaled properly
ax.set_xlim(-250, 250)
ax.set_ylim(-47.5, 422.5) # Sets the baseline and top of half court
ax.set_aspect('equal')    # CRITICAL: Ensures the court isn't stretched out

plt.title("Stephen Curry: Made Shot Density", fontsize=16)
plt.axis('off') # Hides the default matplotlib X/Y coordinate ticks

plt.show()

plt.figure(figsize=(10, 9.5))
ax = plt.gca()

# Plot individual shots
plt.scatter(
    x=curry_data['LOC_X'],
    y=curry_data['LOC_Y'],
    color='red',
    alpha=0.3,    # Transparency: 0 is invisible, 1 is solid
    s=15,         # Size of the dot
    zorder=1
)

# Draw the court lines on top
draw_court(ax, color='black', lw=2, outer_lines=True)

ax.set_xlim(-250, 250)
ax.set_ylim(-47.5, 422.5)
ax.set_aspect('equal') 

plt.title("Stephen Curry: Made Shots Scatter Plot", fontsize=16)
plt.axis('off') 

plt.show()

plt.figure(figsize=(10, 9.5))
ax = plt.gca()

# Create the Hexbin plot
hb = ax.hexbin(
    x=curry_data['LOC_X'],
    y=curry_data['LOC_Y'],
    gridsize=35,      # Controls how large the hexagons are (lower = bigger hexes)
    cmap='Reds',      
    mincnt=1,         # Only draw a hexagon if there is at least 1 shot in it
    edgecolors='white', # Optional: adds a faint border to the hexes
    alpha=0.8,
    zorder=1
)

# Optional: Add a colorbar to show what the colors mean
# cb = plt.colorbar(hb, ax=ax, orientation='horizontal', pad=0.05)
# cb.set_label('Number of Made Shots')

draw_court(ax, color='black', lw=2, outer_lines=True)

ax.set_xlim(-250, 250)
ax.set_ylim(-47.5, 422.5)
ax.set_aspect('equal') 

plt.title("Stephen Curry: Made Shots Hexbin Density", fontsize=16)
plt.axis('off') 

plt.show()