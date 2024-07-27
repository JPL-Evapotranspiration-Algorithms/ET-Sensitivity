from typing import Union
import warnings
import numpy as np
from rasters import Raster
import rasters as rt


def daylight_from_SHA(SHA_deg: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    This function calculates daylight hours from sunrise hour angle in degrees.
    """
    return (2.0 / 15.0) * SHA_deg


def sunrise_from_SHA(SHA_deg: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    This function calculates sunrise hour from sunrise hour angle in degrees.
    """
    return 12.0 - (SHA_deg / 15.0)


def solar_dec_deg_from_day_angle_rad(day_angle_rad: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    This function calculates solar declination in degrees from day angle in radians.
    """
    return (0.006918 - 0.399912 * np.cos(day_angle_rad) + 0.070257 * np.sin(day_angle_rad) - 0.006758 * np.cos(
        2 * day_angle_rad) + 0.000907 * np.sin(2 * day_angle_rad) - 0.002697 * np.cos(
        3 * day_angle_rad) + 0.00148 * np.sin(
        3 * day_angle_rad)) * (180 / np.pi)


def day_angle_rad_from_doy(doy: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    This function calculates day angle in radians from day of year between 1 and 365.
    """
    return (2 * np.pi * (doy - 1)) / 365


def SHA_deg_from_doy_lat(doy: Union[Raster, np.ndarray, int], latitude: np.ndarray) -> Raster:
    """
    This function calculates sunrise hour angle in degrees from latitude in degrees and day of year between 1 and 365.
    """
    # calculate day angle in radians
    day_angle_rad = day_angle_rad_from_doy(doy)

    # calculate solar declination in degrees
    solar_dec_deg = solar_dec_deg_from_day_angle_rad(day_angle_rad)

    # convert latitude to radians
    latitude_rad = np.radians(latitude)

    # convert solar declination to radians
    solar_dec_rad = np.radians(solar_dec_deg)

    # calculate cosine of sunrise angle at latitude and solar declination
    # need to keep the cosine for polar correction
    sunrise_cos = -np.tan(latitude_rad) * np.tan(solar_dec_rad)

    # calculate sunrise angle in radians from cosine
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        sunrise_rad = np.arccos(sunrise_cos)

    # convert to degrees
    sunrise_deg = np.degrees(sunrise_rad)

    # apply polar correction
    sunrise_deg = rt.where(sunrise_cos >= 1, 0, sunrise_deg)
    sunrise_deg = rt.where(sunrise_cos <= -1, 180, sunrise_deg)

    return sunrise_deg