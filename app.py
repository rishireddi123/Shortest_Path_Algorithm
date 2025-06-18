import heapq
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import sys

class CapitalGraph:
    def __init__(self):
        self.graph = defaultdict(dict)
        self.capitals = set()
        self.coordinates = {}  

    def add_edge(self, city1, state1, city2, state2, distance):
        node1 = (city1, state1)
        node2 = (city2, state2)
        self.graph[node1][node2] = distance
        self.graph[node2][node1] = distance
        self.capitals.add(node1)
        self.capitals.add(node2)

    def set_coordinates(self, city, state, lat, lon):
        self.coordinates[(city, state)] = (float(lat), float(lon))

    def get_neighbors(self, node):
        return self.graph[node].items()

    def get_all_capitals(self):
        return sorted(self.capitals, key=lambda x: (x[1], x[0]))

def load_data():
    graph = CapitalGraph()

    with open("capitals.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  
        for parts in reader:
            parts = [part.strip() for part in parts]
            if len(parts) == 5:
                city1, state1, city2, state2, distance = parts
                distance = float(distance)
                graph.add_edge(city1, state1, city2, state2, distance)

    with open("state_capitals_coordinates.csv", newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        state_info = {}
        for row in reader:
            if len(row) < 6:
                continue
            abbr = row[0].strip()
            city = row[2].strip()
            lat = row[3].strip()
            lon = row[4].strip()
            neighbors = set(row[5].strip().split())
            state_info[abbr] = {
                "city": city,
                "lat": lat,
                "lon": lon,
                "neighbors": neighbors
            }

        for abbr, info in state_info.items():
            city1 = info["city"]
            state1 = abbr
            lat1 = info["lat"]
            lon1 = info["lon"]
            for neighbor_abbr in info["neighbors"]:
                if neighbor_abbr == abbr or neighbor_abbr not in state_info:
                    continue
                city2 = state_info[neighbor_abbr]["city"]
                state2 = neighbor_abbr
                lat2 = state_info[neighbor_abbr]["lat"]
                lon2 = state_info[neighbor_abbr]["lon"]
                node1 = (city1, state1)
                node2 = (city2, state2)
                # Only add edge if not already present
                if node2 not in graph.graph[node1]:
                    # Calculate haversine distance
                    from math import radians, sin, cos, sqrt, atan2
                    R = 3958.8  # miles
                    dlat = radians(lat2 - lat1)
                    dlon = radians(lon2 - lon1)
                    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                    c = 2 * atan2(sqrt(a), sqrt(1 - a))
                    distance = R * c
                    graph.add_edge(city1, state1, city2, state2, distance)

    if hasattr(graph, "set_coordinates"):
        with open("state_capitals_coordinates.csv", newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                if len(row) < 5:
                    continue
                city = row[2].strip()
                state = row[0].strip()
                lat = row[3].strip()
                lon = row[4].strip()
                graph.set_coordinates(city, state, lat, lon)

    return graph

def dijkstra_shortest_path(graph, start, end):
    heap = []
    heapq.heappush(heap, (0, start, [start]))
    visited = set()

    while heap:
        total_distance, current_node, path = heapq.heappop(heap)

        if current_node == end:
            return total_distance, path

        if current_node in visited:
            continue

        visited.add(current_node)

        for neighbor, distance in graph.get_neighbors(current_node):
            if neighbor not in visited:
                heapq.heappush(heap, (total_distance + distance, neighbor, path + [neighbor]))

    return float('inf'), []

def plot_graph(graph, shortest_path, total_distance=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    positions = {}

    # Use real coordinates for plotting
    for (city, state) in sorted(graph.capitals):
        if (city, state) in graph.coordinates:
            lat, lon = graph.coordinates[(city, state)]
            x, y = lon, lat
            positions[(city, state)] = (x, y)
            # Determine color for start/end/other
            if shortest_path and (city, state) == shortest_path[0]:
                ax.plot(x, y, 'o', color='black', markersize=10)
                ax.text(x, y + 1.2, "Start", fontsize=15, color='deeppink', ha='center', fontweight='bold')
            elif shortest_path and (city, state) == shortest_path[-1]:
                ax.plot(x, y, 'o', markerfacecolor='white', markeredgecolor='black', markersize=10)
                ax.text(x, y + 1.2, "End", fontsize=15, color='orange', ha='center', fontweight='bold')
            else:
                ax.plot(x, y, 'ro')
            ax.text(x, y + 0.5, f"{city}, {state}", fontsize=6, ha='center', rotation=30, color='white')

    # Draw all edges in red
    drawn_edges = set()
    for city1 in graph.graph:
        for city2, dist in graph.graph[city1].items():
            if (city2, city1) not in drawn_edges and city1 in positions and city2 in positions:
                x1, y1 = positions[city1]
                x2, y2 = positions[city2]
                ax.plot([x1, x2], [y1, y2], 'r-', linewidth=1)
                drawn_edges.add((city1, city2))

    path_x = []
    path_y = []
    for i in range(len(shortest_path) - 1):
        c1 = shortest_path[i]
        c2 = shortest_path[i + 1]
        if c1 in positions and c2 in positions:
            x1, y1 = positions[c1]
            x2, y2 = positions[c2]
            ax.plot([x1, x2], [y1, y2], 'b-', linewidth=2.5)
            path_x.extend([x1, x2])
            path_y.extend([y1, y2])

    # Show total distance above the path
    if total_distance is not None and path_x and path_y:
        mid_x = sum(path_x) / len(path_x)
        mid_y = sum(path_y) / len(path_y)
        ax.text(mid_x, mid_y + 1.5, f"Total Distance: {total_distance:.2f} miles", color='green', fontsize=18, ha='center', fontweight='bold')

    # Set all axes, title, and tick labels to white
    ax.set_title("US State Capitals Graph - Shortest Path", fontsize=14, color='white')
    ax.set_xlabel("Longitude", color='white')
    ax.set_ylabel("Latitude", color='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

    plt.tight_layout()
    plt.show()

def get_city_state_input(prompt, capitals):
    while True:
        user_input = input(prompt).strip()
        if user_input.lower() == "quit":
            print("Exiting...")
            sys.exit(0)
        if ',' in user_input:
            city, state = [x.strip() for x in user_input.split(',', 1)]
            city = city.title() 
            state = state.upper()
            if (city, state) in capitals:
                return (city, state)
            else:
                print("Not found. Please enter as valid US city and state abbreviation")
        else:
            print("Invalid format. Please enter as valid US city and state abbreviation")

if __name__ == "__main__":
    graph = load_data()
    capitals = set(graph.capitals)
    while True:
        print("\n Please enter the starting and destination cities and state abbreviations (e.g. Atlanta, GA)(or type 'quit' to exit):")
        start = get_city_state_input("Please enter the starting city and state abbreviation: ", capitals)
        end = get_city_state_input("Please enter the destination city and state abbreviation: ", capitals)
        if start == end:
            print("Starting and destination cities cannot be the same.")
            continue
        distance, path = dijkstra_shortest_path(graph, start, end)
        if not path:
            print("No path found between the selected capitals.")
        else:
            print(f"Total Distance from {start[0]}, {start[1]} to {end[0]}, {end[1]} is: {distance:.2f} miles")
            plot_graph(graph, path, total_distance=distance)
            print("\nYou can search for another shortest path or type 'quit' to exit.")
