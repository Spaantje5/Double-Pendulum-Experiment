#IMPORTS

from helper_functions import *
import json
import numpy as np
from pathlib import Path

#LOADING CONSTANTS

FPS = 120
LEN_ERR_MM = 0.5

BASE_DIR = Path(__file__).resolve().parent
GEOM_PATH = BASE_DIR / "geometry.json"
GEOM = json.load(open(GEOM_PATH))

mm, g = 1e-3, 1e-3 #mm -> m, g -> kg

#POSITIONS RELATIVE TO THE PIVOT/JOINT (M)

JOINT1 = GEOM["arm1"]["joint"] * mm
PIVOT1 = JOINT2 = 0.0
BLUE = GEOM["arm1"]["sticker_centre"] * mm
RED = GEOM["arm2"]["sticker_centre"] * mm

TOP1 = -GEOM["arm1"]["pivot_top"] * mm
BOTTOM1 = (GEOM["arm1"]["length"] - GEOM["arm1"]["pivot_top"]) * mm
TOP2 = -GEOM["arm2"]["pivot_top"] * mm
BOTTOM2 = (GEOM["arm2"]["length"] - GEOM["arm2"]["pivot_top"]) * mm

#MASSES (KG)

M1 = (GEOM["arm1"]["mass"] + GEOM["arm1"].get("mass_pivot", 0) + GEOM["arm1"].get("mass_joint", 0)) * g
M2 = GEOM["arm2"]["mass"] * g

#CENTRES OF MASS

COM1 = (GEOM["arm1"]["mass"] * g * (GEOM["arm1"]["length"]/2 - GEOM["arm1"]["pivot_top"])*mm + GEOM["arm1"].get("mass_joint", 0)*g*JOINT1)/M1
COM2 = (GEOM["arm2"]["length"]/2 - GEOM["arm2"]["pivot_top"]) * mm

#MOMENTS OF INERTIA (KG M^2)

I1_O = (
    GEOM["arm1"]["mass"]*g *
        ((GEOM["arm1"]["length"]*mm)**2
         + (GEOM["arm1"]["width"]*mm)**2)/12
    + GEOM["arm1"]["mass"]*g*(
        (GEOM["arm1"]["length"]/2 - GEOM["arm1"]["pivot_top"])*mm
    )**2
    + GEOM["arm1"].get("mass_pivot", 0)*g *
        (GEOM["arm1"].get("size_pivot", 0)*mm)**2/6
    + GEOM["arm1"].get("mass_joint", 0)*g * (
        (GEOM["arm1"].get("diameter_joint", 0)*mm/2)**2
        + JOINT1**2
    )
)

I2_O = (M2 * ((GEOM["arm2"]["length"]*mm)**2 + (GEOM["arm2"]["width"]*mm)**2)/12 + M2 * COM2**2)

#MOMENTS OF INERTIA ABOUT CENTRES OF MASS (KG M^2)

I1_C = I1_O - M1*COM1**2
I2_C = I2_O - M2*COM2**2

#PROCESSING RUN DATA

DATA_DIR = BASE_DIR/"data"
runs = {}
for f in sorted(DATA_DIR.glob("*.csv")):
    data = np.genfromtxt(f, dtype=float, delimiter=",", skip_header=1, missing_values="", filling_values=np.nan)

    parts = f.stem.replace("_data", "").split("_")
    init, run = int(parts[1]), int(parts[2])
    t=data[:, 1]
    x_red_px, y_red_px = 1920 - data[:, 2], 1080 - data[:, 3]
    x_blue_px, y_blue_px = 1920 - data[:, 4], 1080 - data[:, 5]
    x_hinge_px, y_hinge_px = np.nanmedian(1920 - data[:, 6]), np.nanmedian(1080 - data[:, 7])

    pixel_distance = np.hypot(x_blue_px - x_hinge_px, y_blue_px - y_hinge_px)
    ppm = np.nanmean(pixel_distance) / BLUE

    #take the hinge as the origin in the new coordinate system
    hinge = np.zeros(2)

    joint = np.column_stack(((x_blue_px - x_hinge_px) / ppm * JOINT1 / BLUE, (y_blue_px - y_hinge_px) / ppm * JOINT1 / BLUE))

    com1 = np.column_stack(((x_blue_px - x_hinge_px) / ppm * COM1 / BLUE, (y_blue_px - y_hinge_px) / ppm * COM1 / BLUE))

    com2 = joint + (np.column_stack((x_red_px, y_red_px)) / ppm - np.array([x_hinge_px, y_hinge_px]) / ppm - joint) * (COM2 / RED)

    theta1_raw = np.arctan2(joint[:, 1], joint[:, 0])

    red_relative = (np.column_stack((x_red_px, y_red_px)) - np.array([x_hinge_px, y_hinge_px])) / ppm
    theta2_raw = np.arctan2(red_relative[:, 1] - joint[:, 1], red_relative[:, 0] - joint[:, 0])

    theta1_unwrapped = unwrap_with_nans(theta1_raw)
    theta2_unwrapped = unwrap_with_nans(theta2_raw)

    runs[(init, run)] = {
        "ppm": ppm,
        "t": data[:, 1],
        "hinge": np.zeros(2),
        "joint": joint,
        "com1": com1,
        "com2": com2,
        "theta1_raw": theta1_raw,
        "theta2_raw": theta2_raw,
        "theta1_unw": theta1_unwrapped,
        "theta2_unw": theta2_unwrapped
    }

print(len(runs), "runs loaded")