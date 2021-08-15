import json
from pysc2.lib.units import get_unit_embed_lookup
import numpy as np

# all credit to Burny for the raw data: https://raw.githubusercontent.com/BurnySc2/sc2-techtree/develop/data/data_readable.json
from unit2vec.ability_remap import reassign_abilities


def get_embedding_properties(raw_data, num_abilities, ability_map, wep_bonus_map):
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
        data.append((raw_data['supply'] + 10.0) / 20.0)
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

    unit_abilities = [0] * num_abilities
    for a in raw_data['abilities']:
        unit_abilities[ability_map[a['ability']]] = 1
    data += unit_abilities

    def get_wep_attributes(wep):
        wep_attr = [0] * 5
        wep_attr[0] = wep['attacks'] / 6
        wep_attr[1] = wep['cooldown'] / 2.4
        wep_attr[2] = wep['damage_per_hit'] / 75.0
        wep_attr[3] = wep['damage_splash']
        wep_attr[4] = wep['range'] / 75.0

        wep_bonus = [0] * len(wep_bonus_map)
        for b in wep['bonuses']:
            wep_bonus[wep_bonus_map[b['against']]] = b['damage'] / 35.0  # max damage is disruputer with 55 bonus against shields


    for w in raw_data['weapons']:
        if w['target_type'] == "Air":

        if w['target_type'] == "Ground":

        elif:       # can attack any

    return np.asarray(data)


with open('unit_data.json') as f:
    raw_data = json.load(f)

    # create a lookup to map used abilities to an index in an abiity array
    ability_lookup = {r['id']: r for r in raw_data['Ability']}
    all_unit_abilities = [a for a_list in [u['abilities'] for u in raw_data['Unit']] for a in a_list]

    ra_abilities = reassign_abilities(all_unit_abilities, ability_lookup)   # re-assign common abilities to base types
    discrete_ability_out = list(set(ra_abilities.values()))
    ability_embedding_indices = dict(zip(discrete_ability_out, [*range(0, len(discrete_ability_out), 1)]))
    ability_map = {k: ability_embedding_indices[v] for (k, v) in ra_abilities.items()}

    # create a lookup for weapon bonus indexes
    wep_bonuses = {'Light': 0, 'Armored': 1, 'Massive': 2, 'Biological': 3, 'Mechanical': 4, 'Psionic': 5, 'Structure': 6, 'Heroic': 7}

    base_attributes = {}
    for d in raw_data['Unit']:
        base_attributes[d['id']] = get_embedding_properties(d, len(discrete_ability_out), ability_map, wep_bonuses)

with open('generic_neutral_data.json') as f:
    neutral_data = json.load(f)

pysc2_units = get_unit_embed_lookup()
