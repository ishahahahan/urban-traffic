"""
Google Maps Traffic Provider

Fetches real-time traffic data from Google Maps Distance Matrix API
for accurate travel time estimation on Delhi road network.
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import time


class GoogleMapsTrafficProvider:
    """
    Provides real-time and historical traffic data using Google Maps API.
    """
    
    DISTANCE_MATRIX_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialize Google Maps traffic provider.
        
        Parameters:
        -----------
        api_key : str, optional
            Google Maps API key. Reads from GOOGLE_MAPS_API_KEY env var if not provided.
        cache_dir : str, optional
            Directory for caching API responses.
        """
        self.api_key = api_key or os.environ.get('GOOGLE_MAPS_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Google Maps API key required. Set GOOGLE_MAPS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Setup cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / 'cache'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
        print(f"Google Maps Traffic Provider initialized")
        print(f"  Cache: {self.cache_dir}")
    
    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()
    
    def _get_cache_key(self, origins: list, destinations: list, mode: str) -> str:
        """Generate cache key."""
        data = f"{origins}:{destinations}:{mode}:{datetime.now().strftime('%Y%m%d%H')}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str, max_age_minutes: int = 15) -> Optional[dict]:
        """Get cached response if valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - file_time
        
        if age > timedelta(minutes=max_age_minutes):
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_cache(self, cache_key: str, data: dict):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def get_travel_time(self, origin: Tuple[float, float], destination: Tuple[float, float],
                        mode: str = 'driving', departure_time: str = 'now',
                        use_cache: bool = True) -> dict:
        """
        Get travel time between two points.
        
        Parameters:
        -----------
        origin : tuple
            (latitude, longitude) of origin
        destination : tuple
            (latitude, longitude) of destination
        mode : str
            Travel mode: 'driving', 'walking', 'transit'
        departure_time : str
            'now' or Unix timestamp
        use_cache : bool
            Whether to use cached responses
            
        Returns:
        --------
        dict : Travel time info including duration, distance, traffic duration
        """
        origin_str = f"{origin[0]},{origin[1]}"
        dest_str = f"{destination[0]},{destination[1]}"
        
        # Check cache
        if use_cache:
            cache_key = self._get_cache_key([origin_str], [dest_str], mode)
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        
        # Rate limit
        self._rate_limit()
        
        # Build request
        params = {
            'origins': origin_str,
            'destinations': dest_str,
            'mode': mode,
            'key': self.api_key
        }
        
        if mode == 'driving':
            params['departure_time'] = departure_time
            params['traffic_model'] = 'best_guess'
        
        try:
            response = requests.get(self.DISTANCE_MATRIX_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'OK':
                return {'error': data['status'], 'message': data.get('error_message', '')}
            
            element = data['rows'][0]['elements'][0]
            
            if element['status'] != 'OK':
                return {'error': element['status']}
            
            result = {
                'distance_m': element['distance']['value'],
                'distance_text': element['distance']['text'],
                'duration_s': element['duration']['value'],
                'duration_text': element['duration']['text'],
                'origin': origin,
                'destination': destination,
                'mode': mode
            }
            
            # Add traffic duration for driving
            if 'duration_in_traffic' in element:
                result['duration_traffic_s'] = element['duration_in_traffic']['value']
                result['duration_traffic_text'] = element['duration_in_traffic']['text']
                result['congestion_ratio'] = (
                    element['duration_in_traffic']['value'] / element['duration']['value'] - 1
                )
            
            # Cache result
            if use_cache:
                self._save_cache(cache_key, result)
            
            return result
            
        except requests.exceptions.RequestException as e:
            return {'error': 'request_failed', 'message': str(e)}
    
    def get_batch_travel_times(self, origins: List[Tuple[float, float]], 
                               destinations: List[Tuple[float, float]],
                               mode: str = 'driving') -> List[List[dict]]:
        """
        Get travel times for multiple origin-destination pairs.
        Google allows up to 25 origins × 25 destinations per request.
        
        Parameters:
        -----------
        origins : list
            List of (lat, lon) tuples
        destinations : list
            List of (lat, lon) tuples
        mode : str
            Travel mode
            
        Returns:
        --------
        list : Matrix of travel time results [origin_idx][dest_idx]
        """
        # Convert to strings
        origin_strs = [f"{o[0]},{o[1]}" for o in origins]
        dest_strs = [f"{d[0]},{d[1]}" for d in destinations]
        
        # Rate limit
        self._rate_limit()
        
        params = {
            'origins': '|'.join(origin_strs),
            'destinations': '|'.join(dest_strs),
            'mode': mode,
            'key': self.api_key,
            'departure_time': 'now'
        }
        
        if mode == 'driving':
            params['traffic_model'] = 'best_guess'
        
        try:
            response = requests.get(self.DISTANCE_MATRIX_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] != 'OK':
                return [[{'error': data['status']}]]
            
            results = []
            for i, row in enumerate(data['rows']):
                row_results = []
                for j, element in enumerate(row['elements']):
                    if element['status'] != 'OK':
                        row_results.append({'error': element['status']})
                    else:
                        result = {
                            'distance_m': element['distance']['value'],
                            'duration_s': element['duration']['value'],
                            'origin_idx': i,
                            'dest_idx': j
                        }
                        if 'duration_in_traffic' in element:
                            result['duration_traffic_s'] = element['duration_in_traffic']['value']
                        row_results.append(result)
                results.append(row_results)
            
            return results
            
        except requests.exceptions.RequestException as e:
            return [[{'error': str(e)}]]
    
    def get_route_travel_time(self, waypoints: List[Tuple[float, float]], 
                              mode: str = 'driving') -> dict:
        """
        Get total travel time for a route with multiple waypoints.
        
        Parameters:
        -----------
        waypoints : list
            List of (lat, lon) tuples representing the route
        mode : str
            Travel mode
            
        Returns:
        --------
        dict : Total route info
        """
        if len(waypoints) < 2:
            return {'error': 'Need at least 2 waypoints'}
        
        total_duration = 0
        total_duration_traffic = 0
        total_distance = 0
        segments = []
        
        for i in range(len(waypoints) - 1):
            result = self.get_travel_time(
                waypoints[i], 
                waypoints[i + 1], 
                mode=mode
            )
            
            if 'error' in result:
                segments.append(result)
                continue
            
            total_duration += result['duration_s']
            total_distance += result['distance_m']
            
            if 'duration_traffic_s' in result:
                total_duration_traffic += result['duration_traffic_s']
            else:
                total_duration_traffic += result['duration_s']
            
            segments.append(result)
        
        return {
            'total_distance_m': total_distance,
            'total_duration_s': total_duration,
            'total_duration_traffic_s': total_duration_traffic,
            'segments': segments,
            'waypoints_count': len(waypoints)
        }


def test_google_maps():
    """Test Google Maps API integration."""
    print("=" * 60)
    print("Testing Google Maps Traffic API")
    print("=" * 60)
    
    try:
        provider = GoogleMapsTrafficProvider()
        
        # Test locations in Delhi
        connaught_place = (28.6315, 77.2167)
        dwarka = (28.5921, 77.0460)
        rajiv_chowk = (28.6328, 77.2197)
        
        # Test 1: Single route
        print("\n1. Connaught Place → Dwarka (Driving)")
        result = provider.get_travel_time(connaught_place, dwarka, mode='driving')
        
        if 'error' in result:
            print(f"   ❌ Error: {result}")
        else:
            print(f"   Distance: {result['distance_text']}")
            print(f"   Duration (no traffic): {result['duration_text']}")
            if 'duration_traffic_text' in result:
                print(f"   Duration (with traffic): {result['duration_traffic_text']}")
                print(f"   Congestion: +{result['congestion_ratio']*100:.0f}%")
        
        # Test 2: Walking
        print("\n2. Rajiv Chowk → Connaught Place (Walking)")
        result = provider.get_travel_time(rajiv_chowk, connaught_place, mode='walking')
        
        if 'error' not in result:
            print(f"   Distance: {result['distance_text']}")
            print(f"   Duration: {result['duration_text']}")
        
        # Test 3: Transit
        print("\n3. Connaught Place → Dwarka (Transit)")
        result = provider.get_travel_time(connaught_place, dwarka, mode='transit')
        
        if 'error' not in result:
            print(f"   Distance: {result['distance_text']}")
            print(f"   Duration: {result['duration_text']}")
        
        print("\n" + "=" * 60)
        print("✅ Google Maps API is working!")
        print("=" * 60)
        
        return True
        
    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_google_maps()
