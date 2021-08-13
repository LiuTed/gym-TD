from gym.logger import DISABLED


DEBUG = 0
FULL = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50

writer = print

LEVEL = 20

ALLOW_REGIONS = []
ENABLE_ALL_REGION = False

def set_level(level):
    global LEVEL
    LEVEL = level

def set_writer(w):
    global writer
    writer = w

def add_region(region):
    global ALLOW_REGIONS
    ALLOW_REGIONS.append(region)

def enable_all_region():
    global ENABLE_ALL_REGION
    ENABLE_ALL_REGION = True

def __region_allowed(region):
    return ENABLE_ALL_REGION or region in ALLOW_REGIONS

def __output(region, level, prefix, msg, *args, **kwargs):
    global LEVEL
    if __region_allowed(region) and LEVEL <= level:
        writer('[{}] {}: {}'.format(region, prefix, msg.format(*args, **kwargs)))

def debug(region, msg, *args, **kwargs):
    __output(region, DEBUG, 'DEBUG', msg, *args, **kwargs)

def verbose(region, msg, *args, **kwargs):
    __output(region, FULL, 'VERBOSE', msg, *args, **kwargs)

def info(region, msg, *args, **kwargs):
    __output(region, INFO, 'INFO', msg, *args, **kwargs)

def warn(region, msg, *args, **kwargs):
    __output(region, WARN, 'WARN', msg, *args, **kwargs)

def error(region, msg, *args, **kwargs):
    __output(region, ERROR, 'ERROR', msg, *args, **kwargs)
