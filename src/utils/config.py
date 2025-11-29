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

# Delhi Metro Lines Configuration (Real Data)
# Based on actual DMRC (Delhi Metro Rail Corporation) network
DELHI_METRO_LINES = {
    'Red': {
        'code': 'Line 1',
        'color': '#EE1C25',
        'frequency_peak_min': 5,
        'frequency_offpeak_min': 8,
        'operating_hours': (5, 23),  # 5 AM to 11 PM
        'avg_speed_kmh': 32,
        'stations_count': 29
    },
    'Yellow': {
        'code': 'Line 2',
        'color': '#FFCC00',
        'frequency_peak_min': 4,
        'frequency_offpeak_min': 6,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 34,
        'stations_count': 37
    },
    'Blue': {
        'code': 'Line 3/4',
        'color': '#0066B3',
        'frequency_peak_min': 3,
        'frequency_offpeak_min': 5,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 35,
        'stations_count': 50  # Main line + Branch
    },
    'Green': {
        'code': 'Line 5',
        'color': '#008D36',
        'frequency_peak_min': 6,
        'frequency_offpeak_min': 10,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 30,
        'stations_count': 21
    },
    'Violet': {
        'code': 'Line 6',
        'color': '#8B4789',
        'frequency_peak_min': 5,
        'frequency_offpeak_min': 8,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 33,
        'stations_count': 34
    },
    'Pink': {
        'code': 'Line 7',
        'color': '#EC008C',
        'frequency_peak_min': 6,
        'frequency_offpeak_min': 9,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 32,
        'stations_count': 38
    },
    'Magenta': {
        'code': 'Line 8',
        'color': '#8E0F56',
        'frequency_peak_min': 6,
        'frequency_offpeak_min': 9,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 34,
        'stations_count': 25
    },
    'Orange': {
        'code': 'Airport Express',
        'color': '#F58220',
        'frequency_peak_min': 15,
        'frequency_offpeak_min': 15,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 100,  # Express service
        'stations_count': 6
    },
    'Grey': {
        'code': 'Line 9',
        'color': '#808080',
        'frequency_peak_min': 8,
        'frequency_offpeak_min': 12,
        'operating_hours': (5, 23),
        'avg_speed_kmh': 30,
        'stations_count': 4
    },
    'Aqua': {
        'code': 'Aqua Line',
        'color': '#00BFFF',
        'frequency_peak_min': 8,
        'frequency_offpeak_min': 12,
        'operating_hours': (6, 22),
        'avg_speed_kmh': 32,
        'stations_count': 21
    }
}

# Known interchange stations
DELHI_METRO_INTERCHANGES = {
    'Rajiv Chowk': ['Blue', 'Yellow'],
    'Kashmere Gate': ['Red', 'Yellow', 'Violet'],
    'Central Secretariat': ['Yellow', 'Violet'],
    'Mandi House': ['Blue', 'Violet'],
    'Hauz Khas': ['Yellow', 'Magenta'],
    'Kalkaji Mandir': ['Magenta', 'Violet'],
    'Janakpuri West': ['Blue', 'Magenta'],
    'Botanical Garden': ['Blue', 'Magenta'],
    'Azadpur': ['Pink', 'Yellow'],
    'Kirti Nagar': ['Blue', 'Green'],
    'Ashok Park Main': ['Green', 'Blue'],
    'INA': ['Yellow', 'Pink'],
    'Lajpat Nagar': ['Violet', 'Pink'],
    'Mayur Vihar Phase 1': ['Blue', 'Pink'],
    'Anand Vihar': ['Blue', 'Pink'],
    'Netaji Subhash Place': ['Red', 'Pink'],
    'Inderlok': ['Red', 'Green'],
    'Yamuna Bank': ['Blue', 'Blue Extension'],
    'New Delhi': ['Yellow', 'Airport Express'],
    'Dwarka Sector 21': ['Blue', 'Airport Express'],
    'Welcome': ['Pink', 'Red'],
    'Majlis Park': ['Pink', 'Red'],
}

# Delhi Metro synthetic stations (for testing when OSM data unavailable)
DELHI_METRO_STATIONS = [
    {'name': 'Rajiv Chowk', 'x': 77.2088, 'y': 28.6329, 'line': 'Blue', 'interchange': True},
    {'name': 'Barakhamba Road', 'x': 77.2183, 'y': 28.6304, 'line': 'Blue', 'interchange': False},
    {'name': 'Mandi House', 'x': 77.2346, 'y': 28.6261, 'line': 'Blue', 'interchange': True},
    {'name': 'Patel Chowk', 'x': 77.2079, 'y': 28.6217, 'line': 'Yellow', 'interchange': False},
    {'name': 'Central Secretariat', 'x': 77.2095, 'y': 28.6157, 'line': 'Yellow', 'interchange': True},
    {'name': 'Shivaji Stadium', 'x': 77.2025, 'y': 28.6433, 'line': 'Orange', 'interchange': False},
    {'name': 'Ramakrishna Ashram Marg', 'x': 77.1959, 'y': 28.6405, 'line': 'Blue', 'interchange': False},
    {'name': 'Kashmere Gate', 'x': 77.2289, 'y': 28.6673, 'line': 'Red', 'interchange': True},
    {'name': 'Chandni Chowk', 'x': 77.2301, 'y': 28.6560, 'line': 'Yellow', 'interchange': False},
    {'name': 'New Delhi', 'x': 77.2090, 'y': 28.6425, 'line': 'Yellow', 'interchange': True},
]

# HERE Traffic API Configuration
HERE_CONFIG = {
    # Delhi bounding box for traffic queries
    'bbox': {
        'west': 76.8,
        'south': 28.4,
        'east': 77.5,
        'north': 28.9
    },
    
    # Matching parameters (OSM edges to HERE segments)
    'max_match_distance_m': 150,  # Maximum distance for edge matching
    'max_bearing_diff_deg': 45,   # Maximum bearing difference for matching
    
    # Traffic data settings
    'cache_ttl_hours': 1,        # Cache TTL for real-time data
    'historical_slots': 168,      # 7 days × 24 hours
    
    # Fallback behavior when HERE data unavailable
    'fallback_to_osm': True,     # Use OSM speeds when no HERE match
    
    # Speed adjustment (HERE speeds tend to be optimistic)
    'speed_adjustment_factor': 0.9,  # Reduce HERE speeds by 10%
}

# Traffic data source configuration
TRAFFIC_DATA_SOURCE = {
    'mode': 'here',  # 'here', 'synthetic', or 'hybrid'
    'here_patterns_file': 'data/here/historical_patterns.parquet',
    'here_mapping_file': 'data/here/edge_here_mapping.csv',
    'synthetic_timeseries': 'data/final/multimodal_timeseries.parquet'
}

# File paths
PATHS = {
    'raw_nodes': 'data/raw/nodes.csv',
    'raw_edges': 'data/raw/edges.csv',
    'multilayer_dir': 'data/multilayer',
    'final_dir': 'data/final',
    'output_dir': 'output',
    'here_dir': 'data/here',
}

# OSM extraction settings
OSM_SETTINGS = {
    'place_name': 'Delhi, India',
    'network_type': 'drive',  # For road network
    'metro_tags': {
        'railway': 'station',
        'station': 'subway'
    },
    'route_tags': {
        'railway': ['subway', 'light_rail']
    }
}
