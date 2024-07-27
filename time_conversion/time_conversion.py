import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import List
import rasters as rt
from rasters import Raster, RasterGeometry

def UTC_to_solar(time_UTC: datetime, lon: float) -> datetime:
    return time_UTC + timedelta(hours=(np.radians(lon) / np.pi * 12))

def calculate_UTC_offset_hours(geometry: RasterGeometry) -> Raster:
    return Raster(np.radians(geometry.lon) / np.pi * 12, geometry=geometry)

def calculate_day_of_year(time_UTC: datetime, geometry: RasterGeometry) -> Raster:
    doy_UTC = time_UTC.timetuple().tm_yday
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = UTC_offset_hours(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    doy = doy_UTC
    doy = rt.where(hour_of_day < 0, doy - 1, doy)
    doy = rt.where(hour_of_day > 24, doy + 1, doy)

    return doy

def calculate_hour_of_day(time_UTC: datetime, geometry: RasterGeometry) -> Raster:
    hour_UTC = time_UTC.hour + time_UTC.minute / 60 + time_UTC.second / 3600
    UTC_offset_hours = calculate_UTC_offset_hours(geometry=geometry)
    hour_of_day = hour_UTC + UTC_offset_hours
    hour_of_day = rt.where(hour_of_day < 0, hour_of_day + 24, hour_of_day)
    hour_of_day = rt.where(hour_of_day > 24, hour_of_day - 24, hour_of_day)

    return hour_of_day




