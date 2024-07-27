from typing import Union
import numpy as np
import rasters as rt
from rasters import Raster

def calculate_tmin_factor(
        Tmin: Union[Raster, np.ndarray],
        tmin_open: Union[Raster, np.ndarray],
        tmin_close: Union[Raster, np.ndarray]) -> Union[Raster, np.ndarray]:
    """
    Calculates the minimum temperature factor for MOD16 equations.

    Args:
        Tmin (Union[Raster, np.ndarray]): Minimum temperature in Celsius.
        tmin_open (Union[Raster, np.ndarray]): Open minimum temperatures.
        tmin_close (Union[Raster, np.ndarray]): Closed minimum temperatures.

    Returns:
        Union[Raster, np.ndarray]: Minimum temperature factor (mTmin).

    Raises:
        ValueError: If the input arrays have different shapes.

    Notes:
        The minimum temperature factor (mTmin) is calculated based on the following conditions:
        - If Tmin is greater than or equal to tmin_open, mTmin is set to 1.0.
        - If tmin_close is less than Tmin and Tmin is less than tmin_open, mTmin is calculated as
          (Tmin - tmin_close) / (tmin_open - tmin_close).
        - If Tmin is less than or equal to tmin_close, mTmin is set to 0.0.
    """
    # Calculate minimum temperature factor using queried open and closed minimum temperatures
    mTmin = rt.where(Tmin >= tmin_open, 1.0, np.nan)

    mTmin = rt.where(
        rt.logical_and(
            tmin_close < Tmin,
            Tmin < tmin_open
        ),
        (Tmin - tmin_close) / (tmin_open - tmin_close),
        mTmin
    )

    mTmin = rt.where(Tmin <= tmin_close, 0.0, mTmin)

    return mTmin
