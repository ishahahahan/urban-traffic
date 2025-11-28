"""
extract_real_metro_data.py
Extract actual Delhi Metro stations and routes from OpenStreetMap.
"""

import osmnx as ox
import pandas as pd
import numpy as np
import json
import os
import sys
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance using Haversine formula"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    return R * c


def extract_delhi_metro_stations(place_name, output_dir):
    """
    Extract Delhi Metro stations from OpenStreetMap.
    
    Parameters:
    -----------
    place_name : str
        Place name or bounding box
    output_dir : str
        Output directory
        
    Returns:
    --------
    DataFrame : stations dataframe
    """
    
    print(f"Extracting Delhi Metro data for: {place_name}")
    
    try:
        print("\n--- Extracting Metro Stations (Points) ---")
        
        # Define tags for metro stations
        tags = {
            'railway': 'station',
            'station': 'subway'
        }
        
        # Get metro stations as GeoDataFrame
        stations_gdf = ox.features_from_place(place_name, tags=tags)
        
        print(f"Found {len(stations_gdf)} metro station features")
        
        # Process stations
        metro_stations = []
        station_id = 0
        
        for idx, station in stations_gdf.iterrows():
            # Get coordinates
            if hasattr(station.geometry, 'centroid'):
                x = station.geometry.centroid.x
                y = station.geometry.centroid.y
            else:
                x = station.geometry.x
                y = station.geometry.y
            
            # Extract station name
            name = station.get('name', f'Station_{station_id}')
            if pd.isna(name):
                name = f'Station_{station_id}'
            
            # Extract line information
            line = station.get('line', 'Unknown')
            if pd.isna(line):
                line = 'Unknown'
            if isinstance(line, list):
                line = ', '.join(str(l) for l in line)
            
            # Extract other attributes
            network = station.get('network', 'Delhi Metro')
            if pd.isna(network):
                network = 'Delhi Metro'
            operator = station.get('operator', 'DMRC')
            if pd.isna(operator):
                operator = 'DMRC'
            
            metro_stations.append({
                'node_id': f'metro_{station_id}',
                'layer': 1,
                'osmid': idx[1] if isinstance(idx, tuple) else idx,
                'x': x,
                'y': y,
                'node_type': 'metro_station',
                'name': name,
                'properties': json.dumps({
                    'line': str(line),
                    'network': str(network),
                    'operator': str(operator),
                    'wheelchair': str(station.get('wheelchair', 'unknown')),
                    'elevator': str(station.get('elevator', 'unknown')),
                    'real_data': True
                })
            })
            station_id += 1
        
        stations_df = pd.DataFrame(metro_stations)
        
        # Remove duplicates by name
        stations_df = stations_df.drop_duplicates(subset=['name'], keep='first')
        
        # Re-index node_ids after deduplication
        stations_df = stations_df.reset_index(drop=True)
        stations_df['node_id'] = [f'metro_{i}' for i in range(len(stations_df))]
        
        print(f"\nProcessed {len(stations_df)} unique metro stations")
        print(f"Station names sample: {stations_df['name'].head(10).tolist()}")
        
        # Save stations
        os.makedirs(output_dir, exist_ok=True)
        stations_df.to_csv(f'{output_dir}/metro_stations_raw.csv', index=False)
        
        return stations_df
        
    except Exception as e:
        print(f"Error extracting stations: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_delhi_metro_routes(place_name, output_dir):
    """
    Extract Delhi Metro routes (rail lines) from OpenStreetMap.
    
    Parameters:
    -----------
    place_name : str
        Place name or bounding box
    output_dir : str
        Output directory
        
    Returns:
    --------
    GeoDataFrame : metro routes
    """
    
    print("\n--- Extracting Metro Routes (Lines) ---")
    
    try:
        # Define tags for metro routes
        tags = {
            'railway': ['subway', 'light_rail']
        }
        
        # Get metro routes
        routes_gdf = ox.features_from_place(place_name, tags=tags)
        
        print(f"Found {len(routes_gdf)} metro route features")
        
        # Filter to keep only LineStrings (actual routes)
        line_mask = routes_gdf.geometry.type.isin(['LineString', 'MultiLineString'])
        routes_gdf = routes_gdf[line_mask]
        
        print(f"Filtered to {len(routes_gdf)} line segments")
        
        # Save raw routes
        os.makedirs(output_dir, exist_ok=True)
        if len(routes_gdf) > 0:
            routes_gdf.to_file(f'{output_dir}/metro_routes_raw.geojson', driver='GeoJSON')
        
        return routes_gdf
        
    except Exception as e:
        print(f"Error extracting routes: {e}")
        return None


def get_line_frequency(line_name):
    """
    Get typical frequency for Delhi Metro lines.
    Based on actual DMRC schedules.
    
    Returns:
    --------
    int : Frequency in minutes
    """
    
    # Actual Delhi Metro frequencies (approximate)
    frequencies = {
        'Blue': 3,
        'Blue Line': 3,
        'Yellow': 4,
        'Yellow Line': 4,
        'Red': 5,
        'Red Line': 5,
        'Green': 6,
        'Green Line': 6,
        'Violet': 5,
        'Violet Line': 5,
        'Orange': 7,
        'Orange Line': 7,
        'Magenta': 6,
        'Magenta Line': 6,
        'Pink': 6,
        'Pink Line': 6,
        'Airport Express': 15,
        'Airport Express Line': 15,
        'Grey': 8,
        'Grey Line': 8,
        'Rapid': 10,
        'Aqua': 8,
        'Aqua Line': 8
    }
    
    return frequencies.get(line_name, 5)  # Default 5 min


def infer_line_from_station_name(station_name):
    """
    Infer metro line from station name based on known Delhi Metro stations.
    """
    
    # Known stations and their lines (partial list for major stations)
    known_stations = {
        # Blue Line
        'Rajiv Chowk': 'Blue',
        'Barakhamba Road': 'Blue',
        'Mandi House': 'Blue',
        'Pragati Maidan': 'Blue',
        'Indraprastha': 'Blue',
        'Yamuna Bank': 'Blue',
        'Dwarka': 'Blue',
        'Janakpuri West': 'Blue',
        'Kirti Nagar': 'Blue',
        'Karol Bagh': 'Blue',
        'Rajendra Place': 'Blue',
        'Ramakrishna Ashram Marg': 'Blue',
        
        # Yellow Line
        'Chandni Chowk': 'Yellow',
        'Chawri Bazar': 'Yellow',
        'New Delhi': 'Yellow',
        'Patel Chowk': 'Yellow',
        'Central Secretariat': 'Yellow',
        'Udyog Bhawan': 'Yellow',
        'Race Course': 'Yellow',
        'Jor Bagh': 'Yellow',
        'INA': 'Yellow',
        'AIIMS': 'Yellow',
        'Hauz Khas': 'Yellow',
        'Malviya Nagar': 'Yellow',
        'Saket': 'Yellow',
        'Qutab Minar': 'Yellow',
        'Chattarpur': 'Yellow',
        'Sultanpur': 'Yellow',
        'Ghitorni': 'Yellow',
        'Arjangarh': 'Yellow',
        'Guru Dronacharya': 'Yellow',
        'Sikanderpur': 'Yellow',
        'MG Road': 'Yellow',
        'IFFCO Chowk': 'Yellow',
        'HUDA City Centre': 'Yellow',
        
        # Red Line
        'Kashmere Gate': 'Red',
        'Tis Hazari': 'Red',
        'Pul Bangash': 'Red',
        'Pratap Nagar': 'Red',
        'Shastri Nagar': 'Red',
        'Inderlok': 'Red',
        'Kanhaiya Nagar': 'Red',
        'Keshav Puram': 'Red',
        'Netaji Subhash Place': 'Red',
        'Kohat Enclave': 'Red',
        'Pitam Pura': 'Red',
        'Rohini East': 'Red',
        'Rohini West': 'Red',
        'Rithala': 'Red',
        
        # Green Line
        'Inderlok': 'Green',
        'Ashok Park Main': 'Green',
        'Punjabi Bagh': 'Green',
        'Shivaji Park': 'Green',
        'Madipur': 'Green',
        'Paschim Vihar East': 'Green',
        'Paschim Vihar West': 'Green',
        'Peera Garhi': 'Green',
        'Udyog Nagar': 'Green',
        'Maharaja Surajmal Stadium': 'Green',
        'Nangloi': 'Green',
        'Nangloi Railway Station': 'Green',
        'Rajdhani Park': 'Green',
        'Mundka': 'Green',
        'Mundka Industrial Area': 'Green',
        'Ghevra': 'Green',
        
        # Violet Line
        'Kashmere Gate': 'Violet',
        'Lal Quila': 'Violet',
        'Jama Masjid': 'Violet',
        'Delhi Gate': 'Violet',
        'ITO': 'Violet',
        'Mandi House': 'Violet',
        'Janpath': 'Violet',
        'Central Secretariat': 'Violet',
        'Khan Market': 'Violet',
        'Jawaharlal Nehru Stadium': 'Violet',
        'Jangpura': 'Violet',
        'Lajpat Nagar': 'Violet',
        'Moolchand': 'Violet',
        'Kailash Colony': 'Violet',
        'Nehru Place': 'Violet',
        'Kalkaji Mandir': 'Violet',
        'Govind Puri': 'Violet',
        'Okhla': 'Violet',
        'Jasola Apollo': 'Violet',
        'Sarita Vihar': 'Violet',
        'Mohan Estate': 'Violet',
        'Tughlakabad': 'Violet',
        'Badarpur Border': 'Violet',
        'Raja Nahar Singh': 'Violet',
        
        # Magenta Line
        'Botanical Garden': 'Magenta',
        'Okhla Bird Sanctuary': 'Magenta',
        'Kalindi Kunj': 'Magenta',
        'Jasola Vihar Shaheen Bagh': 'Magenta',
        'Okhla NSIC': 'Magenta',
        'Sukhdev Vihar': 'Magenta',
        'Jamia Millia Islamia': 'Magenta',
        'Okhla Vihar': 'Magenta',
        'Jasola Apollo': 'Magenta',
        'Sarita Vihar': 'Magenta',
        'Kalkaji Mandir': 'Magenta',
        'Nehru Enclave': 'Magenta',
        'Greater Kailash': 'Magenta',
        'Chirag Delhi': 'Magenta',
        'Panchsheel Park': 'Magenta',
        'Hauz Khas': 'Magenta',
        'IIT Delhi': 'Magenta',
        'R K Puram': 'Magenta',
        'Munirka': 'Magenta',
        'Vasant Vihar': 'Magenta',
        'Shankar Vihar': 'Magenta',
        'Terminal 1 IGI Airport': 'Magenta',
        'Sadar Bazar Cantonment': 'Magenta',
        'Palam': 'Magenta',
        'Dashrath Puri': 'Magenta',
        'Dabri Mor': 'Magenta',
        'Janakpuri West': 'Magenta',
        
        # Pink Line
        'Majlis Park': 'Pink',
        'Azadpur': 'Pink',
        'Shalimar Bagh': 'Pink',
        'Netaji Subhash Place': 'Pink',
        'Shakurpur': 'Pink',
        'Punjabi Bagh West': 'Pink',
        'ESI Hospital': 'Pink',
        'Rajouri Garden': 'Pink',
        'Maya Puri': 'Pink',
        'Naraina Vihar': 'Pink',
        'Delhi Cantt': 'Pink',
        'Durgabai Deshmukh South Campus': 'Pink',
        'Sir Vishweshwaraiah Moti Bagh': 'Pink',
        'Bhikaji Cama Place': 'Pink',
        'Sarojini Nagar': 'Pink',
        'INA': 'Pink',
        'South Extension': 'Pink',
        'Lajpat Nagar': 'Pink',
        'Vinobapuri': 'Pink',
        'Ashram': 'Pink',
        'Hazrat Nizamuddin': 'Pink',
        'Mayur Vihar Phase 1': 'Pink',
        'Mayur Vihar Pocket 1': 'Pink',
        'Trilokpuri Sanjay Lake': 'Pink',
        'Vinod Nagar East': 'Pink',
        'Mandawali West Vinod Nagar': 'Pink',
        'IP Extension': 'Pink',
        'Anand Vihar': 'Pink',
        'Karkarduma': 'Pink',
        'Karkarduma Court': 'Pink',
        'Krishna Nagar': 'Pink',
        'East Azad Nagar': 'Pink',
        'Welcome': 'Pink',
        'Jaffrabad': 'Pink',
        'Maujpur Babarpur': 'Pink',
        'Gokulpuri': 'Pink',
        'Johri Enclave': 'Pink',
        'Shiv Vihar': 'Pink',
        
        # Airport Express
        'New Delhi': 'Airport Express',
        'Shivaji Stadium': 'Airport Express',
        'Dhaula Kuan': 'Airport Express',
        'Delhi Aerocity': 'Airport Express',
        'Airport Terminal 3': 'Airport Express',
        'Dwarka Sector 21': 'Airport Express',
    }
    
    return known_stations.get(station_name, 'Unknown')


def create_metro_network_from_stations(stations_df, output_dir):
    """
    Create metro network by connecting nearby stations on same line.
    
    Parameters:
    -----------
    stations_df : DataFrame
        Metro stations with line information
    output_dir : str
        Output directory
        
    Returns:
    --------
    DataFrame : metro_edges
    """
    
    print("\n--- Creating Metro Network Connections ---")
    
    # Parse line information for each station
    def get_lines(row):
        try:
            props = json.loads(row['properties'])
            line = props.get('line', 'Unknown')
            if line == 'Unknown':
                # Try to infer from station name
                line = infer_line_from_station_name(row['name'])
            return [l.strip() for l in str(line).split(',')]
        except:
            return ['Unknown']
    
    stations_df['lines_list'] = stations_df.apply(get_lines, axis=1)
    
    # Group stations by line
    lines = defaultdict(list)
    
    for idx, station in stations_df.iterrows():
        for line in station['lines_list']:
            if line and line != 'Unknown':
                lines[line].append(idx)
    
    print(f"Found {len(lines)} metro lines:")
    for line, stations in lines.items():
        print(f"  {line}: {len(stations)} stations")
    
    # Create edges by connecting stations on same line
    metro_edges = []
    edge_id = 0
    
    for line, station_indices in lines.items():
        if len(station_indices) < 2:
            continue
        
        # Get coordinates of all stations on this line
        line_stations = stations_df.loc[station_indices].copy()
        
        # Sort stations geographically (simple heuristic)
        line_stations = line_stations.sort_values(['y', 'x'])
        
        # Connect consecutive stations
        prev_station = None
        for idx, station in line_stations.iterrows():
            if prev_station is not None:
                s1 = prev_station
                s2 = station
                
                # Calculate distance
                distance = haversine_distance(s1['y'], s1['x'], s2['y'], s2['x'])
                
                # Skip if distance is unreasonably large (likely not connected)
                if distance > 5000:  # 5km threshold
                    print(f"  Skipping {s1['name']} → {s2['name']} (distance: {distance:.0f}m)")
                    prev_station = station
                    continue
                
                # Metro speed: 60 km/h = 16.67 m/s
                speed_mps = 16.67
                travel_time = distance / speed_mps
                
                # Create bidirectional edges
                for u, v in [(s1['node_id'], s2['node_id']), 
                            (s2['node_id'], s1['node_id'])]:
                    metro_edges.append({
                        'edge_id': f'metro_{edge_id}',
                        'layer': 1,
                        'u': u,
                        'v': v,
                        'key': 0,
                        'length_m': distance,
                        'mode': 'metro',
                        'speed_mps': speed_mps,
                        'travel_time_s': travel_time,
                        'edge_type': 'metro_segment',
                        'properties': json.dumps({
                            'line': line,
                            'frequency_min': get_line_frequency(line),
                            'from_station': s1['name'],
                            'to_station': s2['name'],
                            'real_data': True
                        })
                    })
                    edge_id += 1
            
            prev_station = station
    
    edges_df = pd.DataFrame(metro_edges)
    
    print(f"\nCreated {len(edges_df)} metro connections")
    
    # Save
    edges_df.to_csv(f'{output_dir}/metro_edges_inferred.csv', index=False)
    
    return edges_df


def manual_metro_network_correction(stations_df, edges_df, output_dir):
    """
    Manual corrections for Delhi Metro network.
    Add known interchange connections and fix topology issues.
    
    Parameters:
    -----------
    stations_df : DataFrame
        Metro stations
    edges_df : DataFrame
        Metro edges
    output_dir : str
        Output directory
        
    Returns:
    --------
    tuple : (corrected_stations, corrected_edges)
    """
    
    print("\n--- Applying Manual Corrections ---")
    
    # Known interchange stations in Delhi Metro
    interchanges = {
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
    }
    
    # Add interchange transfers
    additional_edges = []
    edge_id = len(edges_df) if len(edges_df) > 0 else 0
    
    for station_name, lines in interchanges.items():
        # Find this station in our data
        station = stations_df[stations_df['name'] == station_name]
        
        if len(station) == 0:
            # Try partial match
            station = stations_df[stations_df['name'].str.contains(station_name, case=False, na=False)]
        
        if len(station) == 0:
            print(f"  Warning: Interchange station '{station_name}' not found")
            continue
        
        station_id = station.iloc[0]['node_id']
        
        # Add self-loop for line changes (represents walking within station)
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                # Line change time: typically 3-5 minutes
                transfer_time = 180  # 3 minutes
                
                additional_edges.append({
                    'edge_id': f'metro_{edge_id}',
                    'layer': 1,
                    'u': station_id,
                    'v': station_id,
                    'key': edge_id,
                    'length_m': 100,  # Nominal
                    'mode': 'metro',
                    'speed_mps': 0.33,  # Slow (represents waiting/walking)
                    'travel_time_s': transfer_time,
                    'edge_type': 'line_change',
                    'properties': json.dumps({
                        'from_line': lines[i],
                        'to_line': lines[j],
                        'interchange': True,
                        'station': station_name,
                        'real_data': True
                    })
                })
                edge_id += 1
    
    # Combine original and additional edges
    if additional_edges:
        additional_df = pd.DataFrame(additional_edges)
        if len(edges_df) > 0:
            edges_df = pd.concat([edges_df, additional_df], ignore_index=True)
        else:
            edges_df = additional_df
        print(f"Added {len(additional_edges)} interchange connections")
    
    # Save corrected network
    stations_df.to_csv(f'{output_dir}/metro_nodes.csv', index=False)
    edges_df.to_csv(f'{output_dir}/metro_edges.csv', index=False)
    
    return stations_df, edges_df


def extract_real_metro(place_name='Delhi, India', output_dir='data/multilayer'):
    """
    Main function to extract real Delhi Metro network.
    
    Parameters:
    -----------
    place_name : str
        Place name for extraction
    output_dir : str
        Output directory
        
    Returns:
    --------
    tuple : (stations_df, edges_df)
    """
    
    print("="*70)
    print("EXTRACTING REAL DELHI METRO NETWORK FROM OPENSTREETMAP")
    print("="*70)
    
    # Step 1: Extract stations
    stations_df = extract_delhi_metro_stations(place_name, output_dir)
    
    if stations_df is None or len(stations_df) == 0:
        print("\nERROR: Could not extract metro stations")
        print("Possible reasons:")
        print("  1. Place name too specific (try 'Delhi, India')")
        print("  2. Network connection issues")
        print("  3. OSM data incomplete for this area")
        return None, None
    
    # Step 2: Extract routes (optional, for validation)
    routes_gdf = extract_delhi_metro_routes(place_name, output_dir)
    
    # Step 3: Create network from stations
    edges_df = create_metro_network_from_stations(stations_df, output_dir)
    
    # Step 4: Apply manual corrections
    stations_df, edges_df = manual_metro_network_correction(
        stations_df, edges_df, output_dir
    )
    
    print("\n" + "="*70)
    print("METRO NETWORK EXTRACTION COMPLETE")
    print("="*70)
    print(f"Total stations: {len(stations_df)}")
    print(f"Total connections: {len(edges_df)}")
    print(f"\nOutput files:")
    print(f"  {output_dir}/metro_nodes.csv")
    print(f"  {output_dir}/metro_edges.csv")
    
    # Print statistics by line
    print("\nStations by line:")
    line_counts = {}
    for _, station in stations_df.iterrows():
        try:
            lines = json.loads(station['properties']).get('line', 'Unknown').split(', ')
        except:
            lines = ['Unknown']
        for line in lines:
            line_counts[line] = line_counts.get(line, 0) + 1
    
    for line, count in sorted(line_counts.items()):
        print(f"  {line}: {count} stations")
    
    return stations_df, edges_df


if __name__ == "__main__":
    # Extract Delhi Metro network
    stations, edges = extract_real_metro(
        place_name='Delhi, India',
        output_dir='data/multilayer'
    )
