from typing import Union, Dict
import warnings
import numpy as np
from daylight.daylight import daylight_from_SHA
from daylight.daylight import sunrise_from_SHA
from daylight.daylight import SHA_deg_from_doy_lat
from rasters import Raster

STEFAN_BOLTZMAN_CONSTANT = 5.67036713e-8  # SI units watts per square meter per kelvin to the fourth


def process_verma_net_radiation(
        SWin: np.ndarray,
        albedo: np.ndarray,
        ST_C: np.ndarray,
        emissivity: np.ndarray,
        Ta_C: np.ndarray,
        RH: np.ndarray,
        cloud_mask: np.ndarray = None) -> Dict:
    results = {}

    # Convert surface temperature from Celsius to Kelvin
    ST_K = ST_C + 273.15

    # Convert air temperature from Celsius to Kelvin
    Ta_K = Ta_C + 273.15

    # Calculate water vapor pressure in Pascals using air temperature and relative humidity
    Ea_Pa = (RH * 0.6113 * (10 ** (7.5 * (Ta_K - 273.15) / (Ta_K - 35.85)))) * 1000
    
    # constrain albedo between 0 and 1
    albedo = np.clip(albedo, 0, 1)

    # calculate outgoing shortwave from incoming shortwave and albedo
    SWout = np.clip(SWin * albedo, 0, None)
    results["SWout"] = SWout

    # calculate instantaneous net radiation from components
    SWnet = np.clip(SWin - SWout, 0, None)

    # calculate atmospheric emissivity
    eta1 = 0.465 * Ea_Pa / Ta_K
    # atmospheric_emissivity = (1 - (1 + eta1) * np.exp(-(1.2 + 3 * eta1) ** 0.5))
    eta2 = -(1.2 + 3 * eta1) ** 0.5
    eta2 = eta2.astype(float)
    eta3 = np.exp(eta2)
    atmospheric_emissivity = np.where(eta2 != 0, (1 - (1 + eta1) * eta3), np.nan)

    if cloud_mask is None:
        # calculate incoming longwave for clear sky
        LWin = atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
    else:
        # calculate incoming longwave for clear sky and cloudy
        LWin = np.where(
            ~cloud_mask,
            atmospheric_emissivity * STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4,
            STEFAN_BOLTZMAN_CONSTANT * Ta_K ** 4
        )
    
    results["LWin"] = LWin

    # constrain emissivity between 0 and 1
    emissivity = np.clip(emissivity, 0, 1)

    # calculate outgoing longwave from land surface temperature and emissivity
    LWout = emissivity * STEFAN_BOLTZMAN_CONSTANT * ST_K ** 4
    results["LWout"] = LWout

    # LWnet = np.clip(LWin - LWout, 0, None)
    LWnet = np.clip(LWin - LWout, 0, None)

    # constrain negative values of instantaneous net radiation
    Rn = np.clip(SWnet + LWnet, 0, None)
    results["Rn"] = Rn

    return results

def daily_Rn_integration_verma(
        Rn: Union[Raster, np.ndarray],
        hour_of_day: Union[Raster, np.ndarray],
        doy: Union[Raster, np.ndarray] = None,
        lat: Union[Raster, np.ndarray] = None,
        sunrise_hour: Union[Raster, np.ndarray] = None,
        daylight_hours: Union[Raster, np.ndarray] = None) -> Raster:
    """
    calculate daily net radiation using solar parameters
    this is the average rate of energy transfer from sunrise to sunset
    in watts per square meter
    watts are joules per second
    to get the total amount of energy transferred, factor seconds out of joules
    the number of seconds for which this average is representative is (daylight_hours * 3600)
    documented in verma et al, bisht et al, and lagouARDe et al
    :param Rn:
    :param hour_of_day:
    :param sunrise_hour:
    :param daylight_hours:
    :return:
    """
    if daylight_hours is None or sunrise_hour is None and doy is not None and lat is not None:
        sha_deg = SHA_deg_from_doy_lat(doy, lat)
        daylight_hours = daylight_from_SHA(sha_deg)
        sunrise_hour = sunrise_from_SHA(sha_deg)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        Rn_daily = 1.6 * Rn / (np.pi * np.sin(np.pi * (hour_of_day - sunrise_hour) / (daylight_hours)))
    
    return Rn_daily
