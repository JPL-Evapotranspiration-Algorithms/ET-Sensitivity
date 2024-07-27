from typing import Union
from daylight.daylight import SHA_deg_from_doy_lat, daylight_from_SHA, sunrise_from_SHA
from meteorology_conversion.meteorology_conversion import celcius_to_kelvin
import rasters as rt
from rasters import Raster
import numpy as np
import pandas as pd

from verma_net_radiation.verma_net_radiation import daily_Rn_integration_verma

# latent heat of vaporization for water at 20 Celsius in Joules per kilogram
LAMBDA_JKG_WATER_20C = 2450000.0

def lambda_Jkg_from_Ta_K(Ta_K: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    # Calculate the latent heat of vaporization (J kg-1)
    return (2.501 - 0.002361 * (Ta_K - 273.15)) * 1e6

def lambda_Jkg_from_Ta_C(Ta_C: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    Ta_K = celcius_to_kelvin(Ta_C)
    lambda_Jkg = lambda_Jkg_from_Ta_K(Ta_K)

    return lambda_Jkg

def daily_ET_from_daily_LE(
        LE_daylight: Union[Raster, np.ndarray], 
        daylight_hours: Union[Raster, np.ndarray],
        lambda_Jkg: float = LAMBDA_JKG_WATER_20C) -> Union[Raster, np.ndarray]:
    """
    Calculate daily evapotranspiration (ET) from daily latent heat flux (LE).

    Parameters:
        LE_daily (Union[Raster, np.ndarray]): Daily latent heat flux.
        daylight_hours (Union[Raster, np.ndarray]): Length of day in hours.
        latent_vaporization (float, optional): Latent heat of vaporization. Defaults to LATENT_VAPORIZATION.

    Returns:
        Union[Raster, np.ndarray]: Daily evapotranspiration in kilograms.
    """
    # convert length of day in hours to seconds
    daylight_seconds = daylight_hours * 3600.0

    # factor seconds out of watts to get joules and divide by latent heat of vaporization to get kilograms
    ET_daily_kg = rt.clip(LE_daylight * daylight_seconds / LAMBDA_JKG_WATER_20C, 0.0, None)

    return ET_daily_kg

def process_daily_ET_table(input_df: pd.DataFrame) -> pd.DataFrame:
    hour_of_day = input_df.hour_of_day
    doy = input_df.doy
    lat = input_df.lat
    LE = input_df.LE
    Rn = input_df.Rn
    EF = LE / Rn

    SHA_deg = SHA_deg_from_doy_lat(doy, lat)
    sunrise_hour = sunrise_from_SHA(SHA_deg)
    daylight_hours = daylight_from_SHA(SHA_deg)

    Rn_daylight = daily_Rn_integration_verma(
        Rn=Rn,
        hour_of_day=hour_of_day,
        doy=doy,
        lat=lat,
        sunrise_hour=sunrise_hour,
        daylight_hours=daylight_hours
    )

    LE_daylight = EF * Rn_daylight
    ET = daily_ET_from_daily_LE(LE_daylight, daylight_hours)

    output_df = input_df.copy()
    output_df["EF"] = EF
    output_df["sunrise_hour"] = sunrise_hour
    output_df["daylight_hours"] = daylight_hours
    output_df["Rn_daylight"] = Rn_daylight
    output_df["ET"] = ET

    return output_df
    