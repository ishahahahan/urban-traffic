"""
Fix Metro Network Connectivity Issues
=====================================
This script addresses the following issues:
1. Many metro stations not connected properly
2. Interchange stations not connected across lines
3. Sparse graph due to incorrect line assignments

The solution:
- Use actual Delhi Metro line data with proper station ordering
- Create proper sequential connections between stations
- Add interchange connections at junction stations
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from math import radians, sin, cos, sqrt, atan2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


# Complete Delhi Metro Network Definition
# Stations listed in order along each line
DELHI_METRO_LINES = {
    'Red': {
        'stations': [
            'Rithala', 'Rohini West', 'Rohini East', 'Pitam Pura', 
            'Kohat Enclave', 'Netaji Subhash Place', 'Keshav Puram', 
            'Kanhaiya Nagar', 'Shastri Nagar', 'Inderlok', 'Pratap Nagar',
            'Pul Bangash', 'Tis Hazari', 'Kashmere Gate', 'Shastri Park',
            'Seelampur', 'Welcome', 'Shahdara', 'Mansarovar Park', 
            'Jhilmil', 'Dilshad Garden', 'New Bus Adda'
        ],
        'frequency_min': 5,
        'speed_kmh': 60
    },
    'Yellow': {
        'stations': [
            'Samaypur Badli', 'Rohini Sector 18, 19', 'Haiderpur Badli Mor',
            'Jahangirpuri', 'Adarsh Nagar', 'Azadpur', 'Model Town',
            'GTB Nagar', 'Vishwavidyalaya', 'Vidhan Sabha', 'Civil Lines',
            'Kashmere Gate', 'Chandni Chowk', 'Chawri Bazar', 'New Delhi',
            'Rajiv Chowk', 'Patel Chowk', 'Central Secretariat', 'Udyog Bhawan',
            'Lok Kalyan Marg', 'Jor Bagh', 'INA', 'AIIMS', 'Green Park',
            'Hauz Khas', 'Malviya Nagar', 'Saket', 'Qutab Minar',
            'Chhattarpur', 'Sultanpur', 'Ghitorni', 'Arjangarh',
            'Guru Dronacharya', 'Sikanderpur', 'MG Road', 'IFFCO Chowk',
            'HUDA City Centre'
        ],
        'frequency_min': 4,
        'speed_kmh': 60
    },
    'Blue': {
        'stations': [
            'Dwarka Sector 21', 'Dwarka Sector 8', 'Dwarka Sector 9',
            'Dwarka Sector 10', 'Dwarka Sector 11', 'Dwarka Sector 12',
            'Dwarka Sector 13', 'Dwarka Sector 14', 'Dwarka',
            'Dwarka Mor', 'Nawada', 'Uttam Nagar West', 'Uttam Nagar East',
            'Janakpuri West', 'Janakpuri East', 'Tilak Nagar', 'Subhash Nagar',
            'Tagore Garden', 'Rajouri Garden', 'Ramesh Nagar', 'Moti Nagar',
            'Kirti Nagar', 'Shadipur', 'Patel Nagar', 'Rajendra Place',
            'Karol Bagh', 'Jhandewalan', 'Ramakrishna Ashram Marg',
            'Rajiv Chowk', 'Barakhamba Road', 'Mandi House', 'Supreme Court',
            'Indraprastha', 'Yamuna Bank', 'Akshardham', 'Mayur Vihar Phase 1',
            'Mayur Vihar Extension', 'New Ashok Nagar', 'Noida Sector 15',
            'Noida Sector 16', 'Noida Sector 18', 'Botanical Garden',
            'Golf Course', 'Noida City Centre', 'Noida Sector 34',
            'Noida Sector 52', 'Noida Sector 61', 'Noida Sector 59',
            'Noida Sector 62', 'Noida Electronic City'
        ],
        'frequency_min': 3,
        'speed_kmh': 60
    },
    'Blue_Branch': {
        'stations': [
            'Yamuna Bank', 'Laxmi Nagar', 'Nirman Vihar', 'Preet Vihar',
            'Karkarduma', 'Anand Vihar', 'Kaushambi', 'Vaishali'
        ],
        'frequency_min': 4,
        'speed_kmh': 60
    },
    'Green': {
        'stations': [
            'Inderlok', 'Ashok Park Main', 'Punjabi Bagh', 'Shivaji Park',
            'Madipur', 'Paschim Vihar East', 'Paschim Vihar West',
            'Peeragarhi', 'Udyog Nagar', 'Surajmal Stadium', 'Nangloi',
            'Nangloi Railway Station', 'Rajdhani Park', 'Mundka',
            'Mundka Industrial Area', 'Ghevra', 'Tikri Kalan', 'Tikri Border',
            'Pandit Shree Ram Sharma', 'Bahadurgarh City', 'Brigadier Hoshiar Singh'
        ],
        'frequency_min': 6,
        'speed_kmh': 60
    },
    'Green_Branch': {
        'stations': [
            'Ashok Park Main', 'Satguru Ram Singh Marg', 'Kirti Nagar'
        ],
        'frequency_min': 6,
        'speed_kmh': 60
    },
    'Violet': {
        'stations': [
            'Kashmere Gate', 'Lal Quila', 'Jama Masjid', 'Delhi Gate',
            'ITO', 'Mandi House', 'Janpath', 'Central Secretariat',
            'Khan Market', 'Jawaharlal Nehru Stadium', 'Jangpura',
            'Lajpat Nagar', 'Moolchand', 'Kailash Colony', 'Nehru Place',
            'Kalkaji Mandir', 'Govind Puri', 'Okhla', 'Jasola Apollo',
            'Sarita Vihar', 'Mohan Estate', 'Tughlakabad', 'Badarpur Border',
            'Sarai', 'NHPC Chowk', 'Mewala Maharajpur', 'Sector 28',
            'Badkal Mor', 'Old Faridabad', 'Neelam Chowk Ajronda',
            'Bata Chowk', 'Escorts Mujesar', 'Sant Surdas Sihi',
            'Raja Nahar Singh Ballabhgarh'
        ],
        'frequency_min': 5,
        'speed_kmh': 60
    },
    'Pink': {
        'stations': [
            'Majlis Park', 'Azadpur', 'Shalimar Bagh', 'Netaji Subhash Place',
            'Shakurpur', 'Punjabi Bagh West', 'ESI Hospital', 'Rajouri Garden',
            'Maya Puri', 'Naraina Vihar', 'Delhi Cantt', 
            'Durgabai Deshmukh South Campus', 'Sir Vishweshwaraiah Moti Bagh',
            'Bhikaji Cama Place', 'Sarojini Nagar', 'INA', 'South Extension',
            'Lajpat Nagar', 'Vinobapuri', 'Ashram', 'Hazrat Nizamuddin',
            'Mayur Vihar Phase 1', 'Mayur Vihar Pocket 1', 'Trilokpuri Sanjay Lake',
            'Vinod Nagar East', 'Mandawali West Vinod Nagar', 'IP Extension',
            'Anand Vihar', 'Karkarduma', 'Karkarduma Court', 'Krishna Nagar',
            'East Azad Nagar', 'Welcome', 'Jaffrabad', 'Maujpur Babarpur',
            'Gokulpuri', 'Johri Enclave', 'Shiv Vihar'
        ],
        'frequency_min': 6,
        'speed_kmh': 60
    },
    'Magenta': {
        'stations': [
            'Janakpuri West', 'Dabri Mor', 'Dashrath Puri', 'Palam',
            'Sadar Bazar Cantonment', 'Terminal 1 IGI Airport', 'Shankar Vihar',
            'Vasant Vihar', 'Munirka', 'R K Puram', 'IIT Delhi', 'Hauz Khas',
            'Panchsheel Park', 'Chirag Delhi', 'Greater Kailash',
            'Nehru Enclave', 'Kalkaji Mandir', 'Okhla NSIC', 'Sukhdev Vihar',
            'Jamia Millia Islamia', 'Okhla Vihar', 'Jasola Vihar Shaheen Bagh',
            'Kalindi Kunj', 'Okhla Bird Sanctuary', 'Botanical Garden'
        ],
        'frequency_min': 6,
        'speed_kmh': 60
    },
    'Airport_Express': {
        'stations': [
            'New Delhi', 'Shivaji Stadium', 'Dhaula Kuan', 'Delhi Aerocity',
            'Airport Terminal 3', 'Dwarka Sector 21'
        ],
        'frequency_min': 15,
        'speed_kmh': 80
    },
    'Aqua': {
        'stations': [
            'Noida Sector 51', 'Noida Sector 50', 'Noida Sector 76',
            'Noida Sector 101', 'Noida Sector 81', 'NSEZ', 'Noida Sector 83',
            'Noida Sector 137', 'Noida Sector 142', 'Noida Sector 143',
            'Noida Sector 144', 'Noida Sector 145', 'Noida Sector 146',
            'Noida Sector 147', 'Noida Sector 148', 'Knowledge Park II',
            'Pari Chowk', 'Alpha 1', 'Alpha 2', 'Delta 1', 'Depot'
        ],
        'frequency_min': 8,
        'speed_kmh': 60
    },
    'Grey': {
        'stations': [
            'Dwarka', 'Nangli', 'Najafgarh', 'Dhansa Bus Stand'
        ],
        'frequency_min': 8,
        'speed_kmh': 60
    }
}

# Interchange stations - these connect multiple lines
INTERCHANGE_STATIONS = {
    'Kashmere Gate': ['Red', 'Yellow', 'Violet'],
    'Rajiv Chowk': ['Blue', 'Yellow'],
    'Central Secretariat': ['Yellow', 'Violet'],
    'Mandi House': ['Blue', 'Violet'],
    'New Delhi': ['Yellow', 'Airport_Express'],
    'Hauz Khas': ['Yellow', 'Magenta'],
    'Kalkaji Mandir': ['Violet', 'Magenta'],
    'Janakpuri West': ['Blue', 'Magenta'],
    'Botanical Garden': ['Blue', 'Magenta'],
    'Azadpur': ['Yellow', 'Pink'],
    'Netaji Subhash Place': ['Red', 'Pink'],
    'INA': ['Yellow', 'Pink'],
    'Lajpat Nagar': ['Violet', 'Pink'],
    'Inderlok': ['Red', 'Green'],
    'Kirti Nagar': ['Blue', 'Green_Branch'],
    'Ashok Park Main': ['Green', 'Green_Branch'],
    'Dwarka Sector 21': ['Blue', 'Airport_Express'],
    'Yamuna Bank': ['Blue', 'Blue_Branch'],
    'Mayur Vihar Phase 1': ['Blue', 'Pink'],
    'Anand Vihar': ['Blue_Branch', 'Pink'],
    'Karkarduma': ['Blue_Branch', 'Pink'],
    'Welcome': ['Red', 'Pink'],
    'Dwarka': ['Blue', 'Grey']
}


def find_station_match(station_name, stations_df, threshold=0.7):
    """
    Find the best matching station from the dataframe.
    Uses fuzzy string matching.
    """
    from difflib import SequenceMatcher
    
    best_match = None
    best_score = 0
    
    station_name_lower = station_name.lower().strip()
    
    for idx, row in stations_df.iterrows():
        row_name = str(row['name']).lower().strip()
        
        # Exact match
        if row_name == station_name_lower:
            return idx, 1.0
        
        # Check if one contains the other
        if station_name_lower in row_name or row_name in station_name_lower:
            score = 0.9
            if score > best_score:
                best_score = score
                best_match = idx
        
        # Fuzzy match
        score = SequenceMatcher(None, station_name_lower, row_name).ratio()
        if score > best_score:
            best_score = score
            best_match = idx
    
    if best_score >= threshold:
        return best_match, best_score
    return None, 0


def fix_metro_network(input_dir='data/multilayer', output_dir='data/multilayer'):
    """
    Fix metro network connectivity issues.
    
    Parameters:
    -----------
    input_dir : str
        Directory with existing metro_nodes.csv
    output_dir : str
        Directory for output files
        
    Returns:
    --------
    tuple : (fixed_nodes_df, fixed_edges_df)
    """
    
    print("="*70)
    print("FIXING METRO NETWORK CONNECTIVITY")
    print("="*70)
    
    # Load existing metro stations
    metro_nodes_file = f'{input_dir}/metro_nodes.csv'
    
    if os.path.exists(metro_nodes_file):
        stations_df = pd.read_csv(metro_nodes_file)
        print(f"Loaded {len(stations_df)} existing metro stations")
    else:
        print(f"ERROR: Metro nodes file not found at {metro_nodes_file}")
        return None, None
    
    # Create mapping from station names to node_ids
    station_mapping = {}
    unmatched_stations = []
    
    print("\n--- Matching Stations to Lines ---")
    
    for line_name, line_data in DELHI_METRO_LINES.items():
        print(f"\n{line_name} Line:")
        matched = 0
        
        for station_name in line_data['stations']:
            match_idx, score = find_station_match(station_name, stations_df)
            
            if match_idx is not None:
                node_id = stations_df.loc[match_idx, 'node_id']
                actual_name = stations_df.loc[match_idx, 'name']
                
                if station_name not in station_mapping:
                    station_mapping[station_name] = {
                        'node_id': node_id,
                        'actual_name': actual_name,
                        'lines': [line_name],
                        'x': stations_df.loc[match_idx, 'x'],
                        'y': stations_df.loc[match_idx, 'y']
                    }
                else:
                    station_mapping[station_name]['lines'].append(line_name)
                
                matched += 1
            else:
                unmatched_stations.append((line_name, station_name))
        
        print(f"  Matched: {matched}/{len(line_data['stations'])} stations")
    
    if unmatched_stations:
        print(f"\n⚠ Unmatched stations ({len(unmatched_stations)}):")
        for line, station in unmatched_stations[:20]:
            print(f"    {line}: {station}")
        if len(unmatched_stations) > 20:
            print(f"    ... and {len(unmatched_stations) - 20} more")
    
    # Update stations with correct line assignments
    print("\n--- Updating Station Line Assignments ---")
    
    lines_list_updates = {}
    for station_name, info in station_mapping.items():
        node_id = info['node_id']
        lines_list_updates[node_id] = info['lines']
    
    def update_lines_list(row):
        node_id = row['node_id']
        if node_id in lines_list_updates:
            return str(lines_list_updates[node_id])
        return row['lines_list'] if pd.notna(row.get('lines_list')) else "['Unknown']"
    
    stations_df['lines_list'] = stations_df.apply(update_lines_list, axis=1)
    
    # Create properly connected edges
    print("\n--- Creating Metro Edges ---")
    
    metro_edges = []
    edge_id = 0
    
    for line_name, line_data in DELHI_METRO_LINES.items():
        stations = line_data['stations']
        speed_mps = line_data['speed_kmh'] * 1000 / 3600
        frequency = line_data['frequency_min']
        
        print(f"\n{line_name} Line (speed: {line_data['speed_kmh']} km/h):")
        edges_created = 0
        
        for i in range(len(stations) - 1):
            station1 = stations[i]
            station2 = stations[i + 1]
            
            # Get node info
            if station1 not in station_mapping or station2 not in station_mapping:
                continue
            
            info1 = station_mapping[station1]
            info2 = station_mapping[station2]
            
            # Calculate distance
            distance = haversine_distance(
                info1['y'], info1['x'],
                info2['y'], info2['x']
            )
            
            # Calculate travel time
            travel_time = distance / speed_mps
            
            # Create bidirectional edges
            for u_name, v_name, u_info, v_info in [
                (station1, station2, info1, info2),
                (station2, station1, info2, info1)
            ]:
                metro_edges.append({
                    'edge_id': f'metro_{edge_id}',
                    'layer': 1,
                    'u': u_info['node_id'],
                    'v': v_info['node_id'],
                    'key': 0,
                    'length_m': distance,
                    'mode': 'metro',
                    'speed_mps': speed_mps,
                    'travel_time_s': travel_time,
                    'edge_type': 'metro_segment',
                    'properties': json.dumps({
                        'line': line_name,
                        'frequency_min': frequency,
                        'from_station': u_name,
                        'to_station': v_name,
                        'real_data': True
                    })
                })
                edge_id += 1
                edges_created += 1
        
        print(f"  Created {edges_created // 2} bidirectional edges")
    
    # Create interchange connections
    print("\n--- Creating Interchange Connections ---")
    
    interchange_edges = 0
    
    for station_name, lines in INTERCHANGE_STATIONS.items():
        if station_name not in station_mapping:
            print(f"  ⚠ Interchange station '{station_name}' not found")
            continue
        
        info = station_mapping[station_name]
        node_id = info['node_id']
        
        # For each pair of lines at this interchange
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i], lines[j]
                
                # Interchange transfer time (walking + waiting)
                transfer_time = 180  # 3 minutes
                
                # Create interchange edge (self-loop with weight)
                metro_edges.append({
                    'edge_id': f'metro_{edge_id}',
                    'layer': 1,
                    'u': node_id,
                    'v': node_id,
                    'key': edge_id,
                    'length_m': 50,
                    'mode': 'metro',
                    'speed_mps': 0.5,
                    'travel_time_s': transfer_time,
                    'edge_type': 'interchange',
                    'properties': json.dumps({
                        'from_line': line1,
                        'to_line': line2,
                        'station': station_name,
                        'interchange': True
                    })
                })
                edge_id += 1
                interchange_edges += 1
    
    print(f"  Created {interchange_edges} interchange connections")
    
    # Create final DataFrames
    edges_df = pd.DataFrame(metro_edges)
    
    print("\n--- Summary ---")
    print(f"Total stations: {len(stations_df)}")
    print(f"Total edges: {len(edges_df)}")
    print(f"  - Segment edges: {len(edges_df[edges_df['edge_type'] == 'metro_segment'])}")
    print(f"  - Interchange edges: {len(edges_df[edges_df['edge_type'] == 'interchange'])}")
    
    # Verify connectivity
    import networkx as nx
    G = nx.Graph()
    for _, edge in edges_df.iterrows():
        if edge['edge_type'] == 'metro_segment':
            G.add_edge(edge['u'], edge['v'])
    
    components = list(nx.connected_components(G))
    print(f"\nConnected components: {len(components)}")
    
    if len(components) > 1:
        print("  Component sizes:", sorted([len(c) for c in components], reverse=True)[:5])
        largest = max(components, key=len)
        print(f"  Largest component has {len(largest)} stations")
    else:
        print("  ✓ Network is fully connected!")
    
    # Save files
    os.makedirs(output_dir, exist_ok=True)
    stations_df.to_csv(f'{output_dir}/metro_nodes.csv', index=False)
    edges_df.to_csv(f'{output_dir}/metro_edges.csv', index=False)
    
    print(f"\n✓ Saved to {output_dir}")
    
    return stations_df, edges_df


if __name__ == "__main__":
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    fix_metro_network(
        input_dir=os.path.join(project_root, 'data/multilayer'),
        output_dir=os.path.join(project_root, 'data/multilayer')
    )
