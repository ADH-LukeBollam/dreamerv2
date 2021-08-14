import json
from pysc2.lib.units import get_unit_embed_lookup
import numpy as np

# all credit to Burny for the raw data: https://raw.githubusercontent.com/BurnySc2/sc2-techtree/develop/data/data_readable.json

def get_embedding_properties(raw_data):
    data = []

    if 'max_health' in raw_data:
        data.append(raw_data['max_health'] / 2500.0)
    else:
        data.append(0.0)

    if 'max_shield' in raw_data:
        data.append(raw_data['max_shield' ] / 1000.0)
    else:
        data.append(0.0)

    if 'max_energy' in raw_data:
        data.append(raw_data['max_energy'] / 200.0)
    else:
        data.append(0.0)

    if 'armor' in raw_data:
        data.append(raw_data['armor'] / 3.0)
    else:
        data.append(0.0)

    if 'radius' in raw_data:
        data.append(raw_data['radius'] / 5.0)
    else:
        data.append(0.0)

    if 'sight' in raw_data:
        data.append(raw_data['sight'] / 12.0)
    else:
        data.append(0.0)

    if 'speed' in raw_data:
        data.append(raw_data['speed'] / 5.0)
    else:
        data.append(0.0)

    if 'minerals' in raw_data:
        data.append(raw_data['minerals'] / 700.0)
    else:
        data.append(0.0)

    if 'gas' in raw_data:
        data.append(raw_data['gas'] / 400.0)
    else:
        data.append(0.0)

    if 'supply' in raw_data:
        data.append(raw_data['supply'] + 10.0 / 20.0)
    else:
        data.append(0.0)

    if 'time' in raw_data:
        data.append(raw_data['time'] / 2560.0)     # longest unit to make is mothership
    else:
        data.append(0.0)

    binary_properties = [0] * 9
    if raw_data['accepts_addon']:
        binary_properties[0] = 1
    if raw_data['is_addon']:
        binary_properties[1] = 1
    if raw_data['is_flying']:
        binary_properties[2] = 1
    if raw_data['is_structure']:
        binary_properties[3] = 1
    if raw_data['is_townhall']:
        binary_properties[4] = 1
    if raw_data['is_worker']:
        binary_properties[5] = 1
    if raw_data['needs_creep']:
        binary_properties[6] = 1
    if raw_data['needs_geyser']:
        binary_properties[7] = 1
    if raw_data['needs_power']:
        binary_properties[8] = 1
    data += binary_properties

    attributes = [0] * 12
    for p in raw_data['attributes']:
        if p == 'Structure':
            attributes[0] = 1
        elif p == 'Armored':
            attributes[1] = 1
        elif p == 'Mechanical':
            attributes[2] = 1
        elif p == 'Massive':
            attributes[3] = 1
        elif p == 'Light':
            attributes[4] = 1
        elif p == 'Biological':
            attributes[5] = 1
        elif p == 'Psionic':
            attributes[6] = 1
        elif p == 'Heroic':
            attributes[7] = 1
        elif p == 'Pathable':
            attributes[8] = 1
        elif p == 'Uncontrollable':
            attributes[9] = 1
        elif p == 'Invulnerable':
            attributes[10] = 1
        elif p == 'Summoned':
            attributes[11] = 1
        else:
            raise AttributeError(f"missing attribute from definitions: {p}")
    data += attributes

    race = [0] * 4
    if raw_data['race'] == 'Neutral':
        race[0] = 1
    elif raw_data['race'] == 'Terran':
        race[1] = 1
    elif raw_data['race'] == 'Protoss':
        race[2] = 1
    else:   # zerg
        race[3] = 1
    data += race

    return np.asarray(data)


with open('unit_data.json') as f:
    raw_data = json.load(f)

    base_attributes = {}
    for d in raw_data['Unit']:
        base_attributes[d['id']] = get_embedding_properties(d)

with open('generic_neutral_data.json') as f:
    neutral_data = json.load(f)

pysc2_units = get_unit_embed_lookup()
