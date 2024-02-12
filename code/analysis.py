"""Analysis functions

A collection of functions that are used for the analysis notebooks
of both study areas.

This script requires the pandas and geopandas packages to be installed 
within the Python environment you are running this script in.

This file contains the following functions:

    * load_thor - load the thor data and perform key processing steps
    * process_thor - convert the data into different categories
"""


import pandas as pd
import geopandas as gpd


def load_thor(path):
    """
    Reads THOR (Theater History of Operations Reports) data from a CSV file, processes it,
    and returns a GeoDataFrame.

    Parameters:
    - path (str): The file path to the THOR CSV file.

    Returns:
    geopandas.GeoDataFrame: A GeoDataFrame containing processed THOR data.
    """
    df = pd.read_csv(path, low_memory=False)

    # set column names to lowercase
    df.columns = df.columns.str.lower()

    # wrong date as 1970 not a leap year so does not have 29th February
    # adjust it to the first of March instead
    df["msndate"] = df["msndate"].replace("19700229", "1970-03-01")

    # convert date to datetime
    df["date"] = pd.to_datetime(df["msndate"], format="mixed")
    df["year"] = pd.DatetimeIndex(df["date"]).year

    # make missing weapontypes explicit
    df["weapontype"] = df["weapontype"].fillna("N/A")

    # fix wrong mfunc code
    df.loc[df["mfunc_desc"] == "COMBT CARGO AIR  DROP", "mfunc_desc_class"] = "KINETIC"
    df.loc[df["mfunc_desc"] == "COMBT CARGO AIR  DROP", "mfunc_desc"] = "HEAVY BOMBARD"

    # add second weapontypeweight based on the weaponsloadedweight and the
    # number of weapons loaded, this is not perfect as it includes not only
    # the amount of explosives but it is still a good approximation
    df["weapontypeweight_est"] = df["weapontypeweight"].astype("float64")
    cond = (df["numweaponsdelivered"] > 0) & (df["weaponsloadedweight"] > 0)
    df.loc[cond, "weapontypeweight_est"] = (
        df.loc[cond, "weaponsloadedweight"] / df.loc[cond, "numweaponsdelivered"] / 10
    )

    # impute weapontype for SEADAB data based on the estimated
    # weight of each weapon type
    df["weapontype_imp"] = df["weapontype"].copy()

    wt_mapping = {
        260: "MK81 GP BOMB (250)",
        531: "MK 82 GP BOMB (500) LD",
        571: "MK82 GP BOMB (500) HD",
        1100: "MK83 GP BOMB (1000)",
    }

    for wt_weight, wt in wt_mapping.items():
        df.loc[df["weapontypeweight_est"] == wt_weight, "weapontype_imp"] = wt

    # for weapontype 820 weight split it up by type of plane
    aircraft_m117 = ["A-1", "A-37", "B-52", "B-57", "F-100", "F-105", "F-5"]
    df.loc[
        (df["weapontypeweight_est"] == 820)
        & (df["valid_aircraft_root"].isin(aircraft_m117)),
        "weapontype_imp",
    ] = "M117 GP BOMB (750) LD"
    df.loc[
        (df["weapontypeweight_est"] == 820)
        & (~df["valid_aircraft_root"].isin(aircraft_m117)),
        "weapontype_imp",
    ] = "CBU49 AN PR MINE"

    # calculate number of weapons by plane
    df["weapons_per_plane"] = df["numweaponsdelivered"] / df["numofacft"]

    # convert to geo dataframe
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.tgtlonddd_ddd_wgs84, df.tgtlatdd_ddd_wgs84),
        crs="EPSG:4131",
    )

    gdf["tgtlonddd_ddd_wgs84"] = gdf.to_crs("EPSG:4326").geometry.x
    gdf["tgtlatdd_ddd_wgs84"] = gdf.to_crs("EPSG:4326").geometry.y
    return gdf


def process_thor(df, reference_date):
    """
    Processes THOR data by applying various filters
    and calculating counts based on specified criteria.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame containing THOR data.
    - reference_date (str): The data of KH-9 image acquisition in the format 'YYYY-MM-DD'.

    Returns:
    pandas.DataFrame: The processed DataFrame with added columns representing counts
    based on different filters and criteria.
    """
    # Convert the reference date to a pd.Timestamp object
    reference_timestamp = pd.Timestamp(reference_date)

    # Define the filter criteria
    one_year_before = reference_timestamp - pd.DateOffset(years=1)
    three_years_before = reference_timestamp - pd.DateOffset(years=3)

    valid_wt_cacta = [
        "500LB GP MK-82",
        "750LB GP M-117",
        "250LB MK-81",
        "500LB GP M-64",
        "250LB M-57",
        "200/260 M81/88",
        "1000LB MK-83",
        "100LB GP M-30",
        "1000LB GP M-65",
        "2000LB MK-84",
        "2000LB M-66",
        "3000LB M-118",
    ]

    valid_wt_saccoact = ["MK82 B", "750 GP", "M64A1", "MK83 B"]

    valid_wt_seadab = [
        "MK 82 GP BOMB (500) LD",
        "M117 GP BOMB (750) LD",
        "MK81 GP BOMB (250)",
        "MK82 GP BOMB (500) HD",
        "MK 82 GP BOMB (500)",
        "MK83 GP BOMB (1000)",
    ]

    valid_wts = [*valid_wt_cacta, *valid_wt_saccoact, *valid_wt_seadab]

    # Processing filters
    df["flag_no_duplicate"] = 1 - (df["sourcerecord"] == "SACCOACT") * (
        df["msndate"] >= "1971-04-01"
    )
    df["flag_within_capacity"] = df["weapons_per_plane"] <= 108
    df["flag_valid_weapontype"] = df["weapontype_imp"].isin(valid_wts)

    df["flag_filtered"] = (
        df["flag_no_duplicate"]
        * df["flag_within_capacity"]
        * df["flag_valid_weapontype"]
    )

    # Temporal analysis
    df["flag_last_year"] = (df["date"] < reference_timestamp) * (
        df["date"] >= one_year_before
    )
    df["flag_last_3_years"] = (df["date"] < reference_timestamp) * (
        df["date"] >= three_years_before
    )

    # Kinetic/ Nonkinetic
    df["flag_kinetic"] = df["mfunc_desc_class"] == "KINETIC"
    df["flag_nonkinetic"] = df["mfunc_desc_class"] == "NONKINETIC"

    # Weapon type and weapon type weight
    df["flag_weapontype_unknown"] = df["weapontype"] == "N/A"
    df["flag_under_200lbs"] = df["weapontypeweight_est"] < 200
    df["flag_over_200lbs"] = df["weapontypeweight_est"] >= 200

    # Type of Aircraft
    df["flag_B52"] = df["valid_aircraft_root"] == "B-52"
    df["flag_F4"] = df["valid_aircraft_root"] == "F-4"

    # Default counts with filtered dataset
    df["ct_all_years"] = df["numweaponsdelivered"] * df["flag_filtered"]
    df["ct_last_year"] = df["ct_all_years"] * df["flag_last_year"]
    df["ct_last_3_years"] = df["ct_all_years"] * df["flag_last_3_years"]
    df["ct_not_last_year"] = df["ct_all_years"] * (~df["flag_last_year"])

    # Kinetic weapons with weapontypeweight over 200lbs
    df["ct_all_years_kinetic_over200lbs"] = (
        df["numweaponsdelivered"]
        * df["flag_kinetic"]
        * df["flag_over_200lbs"]
        * df["flag_no_duplicate"]
    )
    df["ct_last_year_kinetic_over200lbs"] = (
        df["ct_all_years_kinetic_over200lbs"] * df["flag_last_year"]
    )

    # Kinetic weapons with weapontypeweight under 200lbs
    df["ct_all_years_kinetic_under200lbs"] = (
        df["numweaponsdelivered"]
        * df["flag_kinetic"]
        * df["flag_under_200lbs"]
        * df["flag_no_duplicate"]
    )
    df["ct_last_year_kinetic_under200lbs"] = (
        df["ct_all_years_kinetic_under200lbs"] * df["flag_last_year"]
    )

    # Nonkinetic missions
    df["ct_all_years_nonkinetic"] = df["numweaponsdelivered"] * df["flag_nonkinetic"]
    df["ct_last_year_nonkinetic"] = df["ct_all_years_nonkinetic"] * df["flag_last_year"]

    # Unknown weapontype
    df["ct_all_years_weapontype_unknown"] = (
        df["numweaponsdelivered"] * df["flag_weapontype_unknown"]
    )
    df["ct_last_year_weapontype_unknown"] = (
        df["ct_all_years_weapontype_unknown"] * df["flag_last_year"]
    )

    # Saccoact last year
    df["ct_last_year_B52_saccoact"] = (
        df["numweaponsdelivered"]
        * df["flag_B52"]
        * df["flag_last_year"]
        * (df["sourcerecord"] == "SACCOACT")
        * df["flag_within_capacity"]
        * df["flag_valid_weapontype"]
    )
    df["ct_last_year_B52_seadab"] = (
        df["numweaponsdelivered"]
        * df["flag_B52"]
        * df["flag_last_year"]
        * (df["sourcerecord"] == "SEADAB")
        * df["flag_within_capacity"]
        * df["flag_valid_weapontype"]
    )

    # Saccoact all years
    df["ct_all_years_B52_saccoact"] = (
        df["numweaponsdelivered"]
        * df["flag_B52"]
        * (df["sourcerecord"] == "SACCOACT")
        * df["flag_within_capacity"]
        * df["flag_valid_weapontype"]
    )
    df["ct_all_years_B52_seadab"] = (
        df["numweaponsdelivered"]
        * df["flag_B52"]
        * (df["sourcerecord"] == "SEADAB")
        * df["flag_within_capacity"]
        * df["flag_valid_weapontype"]
    )

    return df
