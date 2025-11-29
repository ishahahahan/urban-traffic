"""
HERE Traffic API Client for fetching real-time and historical traffic data.

This module provides:
- Real-time traffic flow data for Delhi road network
- Historical traffic patterns by day of week and time
- Caching to minimize API calls
"""

import os
import json
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class HERETrafficClient:
    """
    Client for HERE Traffic API.
    Supports both real-time flow and historical traffic patterns.
    """
    
    # HERE API endpoints
    FLOW_URL = "https://data.traffic.hereapi.com/v7/flow"
    
    # Delhi bounding box (SW corner to NE corner)
    DELHI_BBOX = {
        'west': 76.8,
        'south': 28.4,
        'east': 77.5,
        'north': 28.9
    }
    
    def __init__(self, api_key: str = None, cache_dir: str = None):
        """
        Initialize HERE Traffic client.
        
        Parameters:
        -----------
        api_key : str, optional
            HERE API key. If not provided, reads from HERE_API_KEY env variable.
        cache_dir : str, optional
            Directory for caching API responses.
        """
        self.api_key = api_key or os.environ.get('HERE_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "HERE API key required. Set HERE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Setup cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(__file__).parent.parent.parent / 'data' / 'here' / 'cache'
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"HERE Traffic Client initialized")
        print(f"  Cache directory: {self.cache_dir}")
    
    def _get_cache_key(self, endpoint: str, params: dict) -> str:
        """Generate cache key from request parameters."""
        param_str = json.dumps(params, sort_keys=True)
        hash_input = f"{endpoint}:{param_str}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _get_cached(self, cache_key: str, max_age_hours: int = 24) -> Optional[dict]:
        """Get cached response if valid."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if not cache_file.exists():
            return None
        
        # Check age
        file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
        age = datetime.now() - file_time
        
        if age > timedelta(hours=max_age_hours):
            return None
        
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    def _save_cache(self, cache_key: str, data: dict):
        """Save response to cache."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    def get_traffic_flow(self, bbox: dict = None, use_cache: bool = True) -> dict:
        """
        Get real-time traffic flow for a bounding box.
        
        Parameters:
        -----------
        bbox : dict, optional
            Bounding box with keys: west, south, east, north
            Defaults to Delhi bbox.
        use_cache : bool
            Whether to use cached data (default: True, 1 hour cache)
            
        Returns:
        --------
        dict : HERE Traffic Flow API response
        """
        bbox = bbox or self.DELHI_BBOX
        
        params = {
            'apiKey': self.api_key,
            'in': f'bbox:{bbox["west"]},{bbox["south"]},{bbox["east"]},{bbox["north"]}',
            'locationReferencing': 'shape'  # Include road geometry
        }
        
        # Check cache (short TTL for real-time data)
        if use_cache:
            cache_key = self._get_cache_key('flow', {k: v for k, v in params.items() if k != 'apiKey'})
            cached = self._get_cached(cache_key, max_age_hours=1)
            if cached:
                print("Using cached traffic flow data")
                return cached
        
        print(f"Fetching real-time traffic flow from HERE API...")
        print(f"  Bounding box: {bbox}")
        
        try:
            response = requests.get(self.FLOW_URL, params=params, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Cache the response
            if use_cache:
                self._save_cache(cache_key, data)
            
            # Log statistics
            if 'results' in data:
                print(f"  Received {len(data['results'])} road segments")
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching traffic flow: {e}")
            return {'results': [], 'error': str(e)}
    
    def parse_flow_results(self, flow_data: dict) -> List[dict]:
        """
        Parse HERE flow response into usable format.
        
        Parameters:
        -----------
        flow_data : dict
            Raw HERE API response
            
        Returns:
        --------
        list : List of parsed road segments with traffic data
        """
        segments = []
        
        for result in flow_data.get('results', []):
            location = result.get('location', {})
            current_flow = result.get('currentFlow', {})
            
            # Extract coordinates from shape
            shape = location.get('shape', {})
            links = shape.get('links', [])
            
            for link in links:
                points = link.get('points', [])
                
                if len(points) < 2:
                    continue
                
                # Calculate segment center for matching
                lats = [p.get('lat', 0) for p in points]
                lngs = [p.get('lng', 0) for p in points]
                center_lat = sum(lats) / len(lats)
                center_lng = sum(lngs) / len(lngs)
                
                segment = {
                    # Location info
                    'center_lat': center_lat,
                    'center_lng': center_lng,
                    'start_lat': points[0].get('lat'),
                    'start_lng': points[0].get('lng'),
                    'end_lat': points[-1].get('lat'),
                    'end_lng': points[-1].get('lng'),
                    'points': points,
                    
                    # Road info
                    'length_m': location.get('length', 0),
                    'description': location.get('description', ''),
                    
                    # Traffic data
                    'speed_kmh': current_flow.get('speed', 0),
                    'speed_uncapped_kmh': current_flow.get('speedUncapped', 0),
                    'free_flow_kmh': current_flow.get('freeFlow', 0),
                    'jam_factor': current_flow.get('jamFactor', 0),
                    'confidence': current_flow.get('confidence', 0),
                    'traversability': current_flow.get('traversability', 'open'),
                }
                
                # Calculate derived values
                if segment['free_flow_kmh'] > 0:
                    segment['congestion_ratio'] = 1 - (segment['speed_kmh'] / segment['free_flow_kmh'])
                else:
                    segment['congestion_ratio'] = 0
                
                # Convert to m/s
                segment['speed_mps'] = segment['speed_kmh'] / 3.6
                segment['free_flow_mps'] = segment['free_flow_kmh'] / 3.6
                
                segments.append(segment)
        
        return segments
    
    def get_traffic_for_time(self, day_of_week: str, hour: int, minute: int = 0) -> dict:
        """
        Get typical traffic patterns for a specific day and time.
        
        Note: HERE's free tier doesn't include historical API.
        This method simulates historical patterns based on typical traffic models.
        
        Parameters:
        -----------
        day_of_week : str
            Day name: 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN'
        hour : int
            Hour of day (0-23)
        minute : int
            Minute (0-59), rounded to nearest 15-min slot
            
        Returns:
        --------
        dict : Traffic pattern data
        """
        # Round minute to 15-min slot
        slot = (minute // 15) * 15
        time_str = f"{hour:02d}:{slot:02d}"
        
        # Check if we have cached historical data
        cache_key = f"historical_{day_of_week}_{time_str}"
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # If no cached historical data, use current flow as base
        # and apply time-based multipliers
        print(f"No historical data for {day_of_week} {time_str}")
        print("Using current flow with time-based adjustment...")
        
        # Get current flow
        flow_data = self.get_traffic_flow(use_cache=True)
        segments = self.parse_flow_results(flow_data)
        
        # Apply time-based adjustment factor
        adjustment = self._get_time_adjustment(day_of_week, hour)
        
        for segment in segments:
            # Adjust jam factor based on typical patterns
            base_jam = segment['jam_factor']
            adjusted_jam = min(10, base_jam * adjustment)
            segment['jam_factor'] = adjusted_jam
            
            # Recalculate speed based on adjusted jam factor
            if segment['free_flow_kmh'] > 0:
                segment['speed_kmh'] = segment['free_flow_kmh'] * (1 - adjusted_jam * 0.08)
                segment['speed_mps'] = segment['speed_kmh'] / 3.6
        
        return {
            'day_of_week': day_of_week,
            'time': time_str,
            'segments': segments,
            'adjustment_factor': adjustment
        }
    
    def _get_time_adjustment(self, day_of_week: str, hour: int) -> float:
        """
        Get traffic adjustment factor based on typical Delhi patterns.
        
        Returns multiplier for jam_factor:
        - 1.0 = average traffic
        - >1.0 = higher than average (rush hours)
        - <1.0 = lower than average (night, weekends)
        """
        # Weekend adjustment
        is_weekend = day_of_week in ['SAT', 'SUN']
        
        # Hour-based patterns (Delhi specific)
        if is_weekend:
            # Weekend patterns - generally lighter
            if 0 <= hour < 6:
                return 0.3  # Very light
            elif 6 <= hour < 10:
                return 0.6  # Light morning
            elif 10 <= hour < 14:
                return 0.9  # Moderate midday
            elif 14 <= hour < 18:
                return 0.8  # Moderate afternoon
            elif 18 <= hour < 22:
                return 1.0  # Evening activity
            else:
                return 0.5  # Night
        else:
            # Weekday patterns
            if 0 <= hour < 6:
                return 0.3  # Very light
            elif 6 <= hour < 8:
                return 0.8  # Building up
            elif 8 <= hour < 10:
                return 1.5  # Morning rush
            elif 10 <= hour < 12:
                return 1.0  # Post-rush
            elif 12 <= hour < 14:
                return 0.9  # Lunch
            elif 14 <= hour < 17:
                return 1.0  # Afternoon
            elif 17 <= hour < 20:
                return 1.6  # Evening rush (worst)
            elif 20 <= hour < 22:
                return 1.0  # Evening
            else:
                return 0.5  # Night
    
    def fetch_and_cache_patterns(self, days: List[str] = None, hours: List[int] = None):
        """
        Pre-fetch and cache traffic patterns for multiple time slots.
        Useful for building historical pattern dataset.
        
        Parameters:
        -----------
        days : list, optional
            Days to fetch. Default: all days.
        hours : list, optional
            Hours to fetch. Default: every hour.
        """
        days = days or ['MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN']
        hours = hours or list(range(24))
        
        print(f"Pre-fetching traffic patterns for {len(days)} days x {len(hours)} hours")
        
        # First get base flow data
        base_flow = self.get_traffic_flow(use_cache=False)
        base_segments = self.parse_flow_results(base_flow)
        
        total = len(days) * len(hours)
        count = 0
        
        for day in days:
            for hour in hours:
                count += 1
                print(f"  [{count}/{total}] {day} {hour:02d}:00")
                
                # Calculate adjustment
                adjustment = self._get_time_adjustment(day, hour)
                
                # Adjust segments
                adjusted_segments = []
                for seg in base_segments:
                    adjusted = seg.copy()
                    base_jam = seg['jam_factor']
                    adjusted_jam = min(10, base_jam * adjustment)
                    adjusted['jam_factor'] = adjusted_jam
                    
                    if seg['free_flow_kmh'] > 0:
                        adjusted['speed_kmh'] = seg['free_flow_kmh'] * (1 - adjusted_jam * 0.08)
                        adjusted['speed_mps'] = adjusted['speed_kmh'] / 3.6
                    
                    adjusted_segments.append(adjusted)
                
                # Save to cache
                pattern_data = {
                    'day_of_week': day,
                    'hour': hour,
                    'time': f"{hour:02d}:00",
                    'segments': adjusted_segments,
                    'adjustment_factor': adjustment,
                    'base_timestamp': datetime.now().isoformat()
                }
                
                cache_key = f"historical_{day}_{hour:02d}:00"
                cache_file = self.cache_dir / f"{cache_key}.json"
                
                with open(cache_file, 'w') as f:
                    json.dump(pattern_data, f)
        
        print(f"\nCached {total} traffic patterns to {self.cache_dir}")


def test_here_client():
    """Test HERE Traffic client."""
    print("=" * 60)
    print("Testing HERE Traffic Client")
    print("=" * 60)
    
    try:
        client = HERETrafficClient()
        
        # Test real-time flow
        print("\n1. Fetching real-time traffic flow...")
        flow_data = client.get_traffic_flow()
        
        if 'error' in flow_data:
            print(f"   Error: {flow_data['error']}")
            return False
        
        # Parse results
        segments = client.parse_flow_results(flow_data)
        print(f"   Parsed {len(segments)} road segments")
        
        if segments:
            # Show sample
            sample = segments[0]
            print(f"\n   Sample segment:")
            print(f"     Location: ({sample['center_lat']:.4f}, {sample['center_lng']:.4f})")
            print(f"     Speed: {sample['speed_kmh']:.1f} km/h ({sample['speed_mps']:.1f} m/s)")
            print(f"     Free flow: {sample['free_flow_kmh']:.1f} km/h")
            print(f"     Jam factor: {sample['jam_factor']:.1f}/10")
            print(f"     Congestion: {sample['congestion_ratio']*100:.1f}%")
        
        # Test time-based patterns
        print("\n2. Testing time-based patterns...")
        pattern = client.get_traffic_for_time('MON', 8, 30)
        print(f"   Monday 8:30 AM adjustment factor: {pattern.get('adjustment_factor', 'N/A')}")
        
        pattern = client.get_traffic_for_time('SAT', 14, 0)
        print(f"   Saturday 2:00 PM adjustment factor: {pattern.get('adjustment_factor', 'N/A')}")
        
        print("\n" + "=" * 60)
        print("HERE Traffic Client test completed successfully!")
        print("=" * 60)
        
        return True
        
    except ValueError as e:
        print(f"\nConfiguration error: {e}")
        print("\nTo fix:")
        print("  1. Get HERE API key from https://developer.here.com/")
        print("  2. Set environment variable: set HERE_API_KEY=your_key")
        return False
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_here_client()
