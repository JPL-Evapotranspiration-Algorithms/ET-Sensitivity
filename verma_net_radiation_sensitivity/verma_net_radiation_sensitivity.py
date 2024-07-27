import numpy as np
import pandas as pd
from pandas import DataFrame
from verma_net_radiation import process_verma_net_radiation

from sun_angles import calculate_SZA_from_datetime
from sentinel_tiles import sentinel_tiles
from koppengeiger import load_koppen_geiger


def process_verma_net_radiation_table(verma_net_radiation_inputs_df: DataFrame) -> DataFrame:
    SWin = np.array(verma_net_radiation_inputs_df.Rg)
    albedo = np.array(verma_net_radiation_inputs_df.albedo)
    ST_C = np.array(verma_net_radiation_inputs_df.LST - 273.15)
    emissivity = np.array(verma_net_radiation_inputs_df.EmisWB)
    Ta_C = np.array(verma_net_radiation_inputs_df.Ta_C)
    RH = np.array(verma_net_radiation_inputs_df.RH)

    results = process_verma_net_radiation(
        SWin=SWin,
        albedo=albedo,
        ST_C=ST_C,
        emissivity=emissivity,
        Ta_C=Ta_C,
        RH=RH,
    )

    verma_net_radiation_outputs_df = verma_net_radiation_inputs_df.copy()

    for key, value in results.items():
        verma_net_radiation_outputs_df[key] = value

    return verma_net_radiation_outputs_df
