from priestley_taylor import GAMMA_PA

# TODO need to defend picking arbitrary maximum to avoid extreme values
MIN_RESISTANCE = 0.0
MAX_RESISTANCE = 2000.0

# gas constant for dry air in joules per kilogram per kelvin
RD = 286.9

# gas constant for moist air in joules per kilogram per kelvin
RW = 461.5

# specific heat of water vapor in joules per kilogram per kelvin
CPW = 1846.0

# specific heat of dry air in joules per kilogram per kelvin
CPD = 1005.0

# psychrometric constant in Pascal per kelvin
# GAMMA = 67.0

# Stefan Boltzmann constant
SIGMA = 5.678e-8

# cuticular conductance in meters per second
CUTICULAR_CONDUCTANCE = 0.00001

# VPD factor in Pascal for soil moisture constraint
# MOD16 uses 200, but PT-JPL uses 1000
BETA = 200

RH_THRESHOLD = 0.7

MIN_FWET = 0.0001
