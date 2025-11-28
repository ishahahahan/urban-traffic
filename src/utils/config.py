"""
Configuration constants for the multi-layer transportation network.
"""

# Layer definitions
LAYER_DEFINITIONS = {
    0: {
        'name': 'road',
        'mode': 'car',
        'avg_speed_kmh': 40,
        'avg_speed_mps': 11.11,
        'color': '#FF6B6B',
        'description': 'Vehicular road network'
    },
    1: {
        'name': 'metro',
        'mode': 'metro',
        'avg_speed_kmh': 60,
        'avg_speed_mps': 16.67,
        'color': '#4ECDC4',
        'description': 'Metro/subway rail network'
    },
    2: {
        'name': 'walk',
        'mode': 'walk',
        'avg_speed_kmh': 5,
        'avg_speed_mps': 1.39,
        'color': '#95E1D3',
        'description': 'Pedestrian pathways'
    },
    3: {
        'name': 'auto',
        'mode': 'auto',
        'avg_speed_kmh': 25,
        'avg_speed_mps': 6.94,
        'color': '#F38181',
        'description': 'Auto-rickshaw network'
    }
}

# Transfer time penalties (seconds)
TRANSFER_TIMES = {
    'metro_entry': 120,  # 2 minutes (stairs, security, ticketing)
    'metro_exit': 60,    # 1 minute (stairs/elevator)
    'mode_switch': 30,   # 30 seconds (parking, getting vehicle)
    'station_transfer': 180  # 3 minutes (platform to platform)
}

# Maximum distances for transfers (meters)
MAX_TRANSFER_DISTANCES = {
    'metro_entry': 300,  # Max walk to metro station
    'metro_exit': 300,   # Max walk from metro station
    'walk_edge': 1000,   # Max length for pedestrian edges
}

# Traffic generation parameters
TRAFFIC_CONFIG = {
    'steps_per_hour': 12,  # 5-minute intervals
    'hours_per_day': 24,
    'chunk_size': 10000,   # For memory-efficient generation
}

# Time-of-day congestion multipliers
TOD_MULTIPLIERS = {
    'night': (0, 5, 0.9),      # 12am-5am: low traffic
    'morning': (5, 8, 1.2),    # 5am-8am: building up
    'peak_am': (8, 10, 1.6),   # 8am-10am: morning rush
    'midday': (10, 16, 1.1),   # 10am-4pm: moderate
    'peak_pm': (16, 19, 1.7),  # 4pm-7pm: evening rush
    'evening': (19, 22, 1.2),  # 7pm-10pm: winding down
    'late': (22, 24, 0.95),    # 10pm-12am: low traffic
}

# Delhi Metro Lines (for synthetic generation)
DELHI_METRO_STATIONS = [
    {'name': 'Rajiv Chowk', 'x': 77.2088, 'y': 28.6329, 'line': 'Blue', 'interchange': True},
    {'name': 'Barakhamba Road', 'x': 77.2183, 'y': 28.6304, 'line': 'Blue', 'interchange': False},
    {'name': 'Mandi House', 'x': 77.2346, 'y': 28.6261, 'line': 'Blue', 'interchange': True},
    {'name': 'Patel Chowk', 'x': 77.2079, 'y': 28.6217, 'line': 'Yellow', 'interchange': False},
    {'name': 'Central Secretariat', 'x': 77.2095, 'y': 28.6157, 'line': 'Yellow', 'interchange': True},
    {'name': 'Shivaji Stadium', 'x': 77.2025, 'y': 28.6433, 'line': 'Orange', 'interchange': False},
    {'name': 'Ramakrishna Ashram Marg', 'x': 77.1959, 'y': 28.6405, 'line': 'Blue', 'interchange': False},
]

# File paths
PATHS = {
    'raw_nodes': 'data/raw/nodes.csv',
    'raw_edges': 'data/raw/edges.csv',
    'multilayer_dir': 'data/multilayer',
    'final_dir': 'data/final',
}
