"""
Haversine distance calculation utility.
Calculate distances between geographic coordinates.
"""

from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two points on Earth using Haversine formula.
    
    Parameters:
    -----------
    lat1, lon1 : float
        Latitude and longitude of first point (decimal degrees)
    lat2, lon2 : float
        Latitude and longitude of second point (decimal degrees)
    
    Returns:
    --------
    float : Distance in meters
    
    Examples:
    ---------
    >>> # Distance between two points in Delhi
    >>> d = haversine_distance(28.6139, 77.2090, 28.6329, 77.2088)
    >>> print(f"{d:.1f} meters")
    2113.2 meters
    """
    R = 6371000  # Earth radius in meters
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance = R * c
    
    return distance


def haversine_distance_vectorized(lats1, lons1, lats2, lons2):
    """
    Vectorized version for numpy arrays.
    
    Parameters:
    -----------
    lats1, lons1 : array-like
        Arrays of latitudes and longitudes for first points
    lats2, lons2 : array-like
        Arrays of latitudes and longitudes for second points
    
    Returns:
    --------
    array : Distances in meters
    """
    import numpy as np
    
    R = 6371000
    
    lats1, lons1, lats2, lons2 = map(np.radians, [lats1, lons1, lats2, lons2])
    
    dlat = lats2 - lats1
    dlon = lons2 - lons1
    
    a = np.sin(dlat/2)**2 + np.cos(lats1) * np.cos(lats2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c
