import json
from pysc2.lib.units import get_unit_embed_lookup
import numpy as np

# all credit to Burny for the raw data: https://raw.githubusercontent.com/BurnySc2/sc2-techtree/develop/data/data_readable.json
from unit2vec.ability_remap import reassign_abilities
from unit2vec.attribute_extractor import get_embedding_properties

with open('unit_data.json') as f:
    raw_data = json.load(f)

with open('generic_neutral_data.json') as f:
    neutral_data = json.load(f)

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
neutral_attributes = {}

for d in raw_data['Unit']:
    base_attributes[d['id']] = get_embedding_properties(d, len(discrete_ability_out), ability_map, wep_bonuses)

for n in neutral_data['Unit']:
    neutral_attributes[n['id']] = get_embedding_properties(n, len(discrete_ability_out), ability_map, wep_bonuses)

pysc2_units = get_unit_embed_lookup()
pysc2_features = {}

for unit_id, embed_id in pysc2_units.items():
    # check if there is a superclass for this unit type
    # all regular units will have features, neutral units are grouped into superclasses because different rocks have pretty much the same properties
    if embed_id in neutral_attributes:
        features = neutral_attributes[embed_id]
    else:
        features = base_attributes[unit_id]
    pysc2_features[embed_id] = features

pass




