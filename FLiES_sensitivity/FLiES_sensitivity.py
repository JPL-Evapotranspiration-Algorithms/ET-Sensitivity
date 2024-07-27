import logging

import numpy as np
import pandas as pd
from dateutil import parser
from pandas import DataFrame
from BESS.FLiESANN import process_FLiES

import rasters as rt
from geos5fp import GEOS5FP
from sun_angles import calculate_SZA_from_datetime
from sentinel_tiles import sentinel_tiles
from koppengeiger import load_koppen_geiger

logger = logging.getLogger(__name__)


def generate_FLiES_inputs(
        FLiES_inputs_from_calval_df: DataFrame,
        GEOS5FP_connection: GEOS5FP) -> DataFrame:
    """
    FLiES_inputs_from_claval_df:
        Pandas DataFrame containing the columns: tower, lat, lon, time_UTC, albedo, elevation_km
    return:
        Pandas DataFrame containing the columns: tower, lat, lon, time_UTC, doy, albedo, elevation_km, AOT, COT, vapor_gccm, ozone_cm, SZA, KG
    """
    # output_rows = []
    FLiES_inputs_df = FLiES_inputs_from_calval_df.copy()

    doy = []
    AOT = []
    COT = []
    vapor_gccm = []
    ozone_cm = []
    SWin = []
    Tmin_K = []
    SZA = []
    KG = []

    for i, input_row in FLiES_inputs_from_calval_df.iterrows():
        tower = input_row.tower
        lat = input_row.lat
        lon = input_row.lon
        time_UTC = input_row.time_UTC
        albedo = input_row.albedo
        elevation_km = input_row.elevation_km
        logger.info(f"collecting FLiES inputs for tower {tower} lat {lat} lon {lon} time {time_UTC} UTC")
        time_UTC = parser.parse(str(time_UTC))
        doy.append(time_UTC.timetuple().tm_yday)
        date_UTC = time_UTC.date()
        tile = sentinel_tiles.toMGRS(lat, lon)[:5]

        try:
            tile_grid = sentinel_tiles.grid(tile=tile, cell_size=70)
        except Exception as e:
            logger.error(e)
            logger.warning(f"unable to process tile {tile}")
            AOT.append(np.nan)
            COT.append(np.nan)
            vapor_gccm.append(np.nan)
            ozone_cm.append(np.nan)
            SZA.append(np.nan)
            KG.append(np.nan)
            continue

        rows, cols = tile_grid.shape
        row, col = tile_grid.index_point(rt.Point(lon, lat))
        geometry = tile_grid[max(0, row - 1):min(row + 2, rows - 1),
                             max(0, col - 1):min(col + 2, cols - 1)]

        if not "AOT" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("retrieving GEOS-5 FP aerosol optical thickness raster")
                AOT.append(np.nanmedian(GEOS5FP_connection.AOT(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                AOT.append(np.nan)
                logger.exception(e)

        if not "COT" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("generating GEOS-5 FP cloud optical thickness raster")
                COT.append(np.nanmedian(GEOS5FP_connection.COT(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                COT.append(np.nan)
                logger.exception(e)
        
        if not "vapor_gccm" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("generating GEOS5-FP water vapor raster in grams per square centimeter")
                vapor_gccm.append(np.nanmedian(GEOS5FP_connection.vapor_gccm(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                vapor_gccm.append(np.nan)
                logger.exception(e)

        if not "ozone_cm" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("generating GEOS5-FP ozone raster in grams per square centimeter")
                ozone_cm.append(np.nanmedian(GEOS5FP_connection.ozone_cm(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                ozone_cm.append(np.nan)
                logger.exception(e)
        
        if not "SWin" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("generating GEOS5-FP incoming solar radiation raster in watts per square meter")
                SWin.append(np.nanmedian(GEOS5FP_connection.SWin(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                SWin.append(np.nan)
                logger.exception(e)

        if not "Tmin_K" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("generating GEOS5-FP minimum temperature in Kelvin")
                Tmin_K.append(np.nanmedian(GEOS5FP_connection.Tmin_K(time_UTC=time_UTC, geometry=geometry)))
            except Exception as e:
                Tmin_K.append(np.nan)
                logger.exception(e)

        if not "SZA" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("calculating solar zenith angle")
                SZA.append(calculate_SZA_from_datetime(time_UTC, lat, lon))
            except Exception as e:
                SZA.append(np.nan)
                logger.exception(e)

        if not "KG" in FLiES_inputs_from_calval_df.columns:
            try:
                logger.info("selecting Koppen Geiger climate classification")
                KG.append(load_koppen_geiger(geometry=geometry)[1, 1][0][0])
            except Exception as e:
                KG.append(np.nan)
                logger.exception(e)

    if not "doy" in FLiES_inputs_df.columns:
        FLiES_inputs_df["doy"] = doy

    if not "AOT" in FLiES_inputs_df.columns:
        FLiES_inputs_df["AOT"] = AOT
    
    if not "COT" in FLiES_inputs_df.columns:
        FLiES_inputs_df["COT"] = COT
    
    if not "vapor_gccm" in FLiES_inputs_df.columns:
        FLiES_inputs_df["vapor_gccm"] = vapor_gccm

    if not "ozone_cm" in FLiES_inputs_df.columns:
        FLiES_inputs_df["ozone_cm"] = ozone_cm
    
    FLiES_inputs_df["SWin"] = SWin
    FLiES_inputs_df["Tmin_K"] = Tmin_K 
    
    if not "SZA" in FLiES_inputs_df.columns:
        FLiES_inputs_df["SZA"] = SZA
    
    if not "KG" in FLiES_inputs_df.columns:
        FLiES_inputs_df["KG"] = KG
    
    if "Ta" in FLiES_inputs_df and "Ta_C" not in FLiES_inputs_df:
        FLiES_inputs_df.rename({"Ta": "Ta_C"}, inplace=True)
    
    return FLiES_inputs_df

def process_FLiES_table(FLiES_inputs_df: DataFrame) -> DataFrame:
    Ra, Rg, UV, VIS, NIR, VISdiff, NIRdiff, VISdir, NIRdir, tm, puv, pvis, pnir, fduv, fdvis, fdnir = process_FLiES(
        doy=FLiES_inputs_df.doy,
        albedo=FLiES_inputs_df.albedo,
        COT=FLiES_inputs_df.COT,
        AOT=FLiES_inputs_df.AOT,
        vapor_gccm=FLiES_inputs_df.vapor_gccm,
        ozone_cm=FLiES_inputs_df.ozone_cm,
        elevation_km=FLiES_inputs_df.elevation_km,
        SZA=FLiES_inputs_df.SZA,
        KG_climate=FLiES_inputs_df.KG
    )

    FLiES_outputs_df = FLiES_inputs_df.copy()
    FLiES_outputs_df["Ra"] = Ra
    FLiES_outputs_df["Rg"] = Rg
    FLiES_outputs_df["UV"] = UV
    FLiES_outputs_df["VIS"] = VIS
    FLiES_outputs_df["NIR"] = NIR
    FLiES_outputs_df["VISdiff"] = VISdiff
    FLiES_outputs_df["NIRdiff"] = NIRdiff
    FLiES_outputs_df["VISdir"] = VISdir
    FLiES_outputs_df["NIRdir"] = NIRdir
    FLiES_outputs_df["tm"] = tm
    FLiES_outputs_df["puv"] = puv
    FLiES_outputs_df["pvis"] = pvis
    FLiES_outputs_df["pnir"] = pnir
    FLiES_outputs_df["fduv"] = fduv
    FLiES_outputs_df["fdvis"] = fdvis
    FLiES_outputs_df["fdnir"] = fdnir

    return FLiES_outputs_df
