
# re-assign abilities to a smaller subset of super-classes
def reassign_abilities(ability_set, ability_lookup):
    new_abilities = {}
    for a in ability_set:
        base_ability = ability_lookup[a['ability']]
        if 'PROTOSSBUILD' in base_ability['name'] or 'TERRANBUILD' in base_ability['name'] or 'ZERGBUILD' in base_ability['name'] or 'BUILD_SHIELDBATTERY' in base_ability['name'] or 'BUILD_LURKERDEN' in base_ability['name']:
            new_abilities[base_ability['id']] = 'build'
        elif 'HALLUCINATION' in base_ability['name']:
            new_abilities[base_ability['id']] = 'hallucinate'
        elif 'HARVEST' in base_ability['name']:
            new_abilities[base_ability['id']] = 'harvest'
        elif 'REPAIR' in base_ability['name']:
            new_abilities[base_ability['id']] = 'repair'
        elif 'RALLY' in base_ability['name']:
            new_abilities[base_ability['id']] = 'rally'
        elif 'LIFT' in base_ability['name'] or 'LAND' in base_ability['name'] and 'GLAND' not in base_ability['name']:
            new_abilities[base_ability['id']] = 'float'
        elif 'REACTOR' in base_ability['name'] or 'TECHLAB' in base_ability['name']:
            new_abilities[base_ability['id']] = 'addon'
        elif 'LEVEL1' in base_ability['name'] or 'LEVEL2' in base_ability['name'] or 'LEVEL3' in base_ability['name']:
            new_abilities[base_ability['id']] = 'upgrade'
        elif 'LOAD' in base_ability['name']:
            new_abilities[base_ability['id']] = 'transport'
        elif 'BURROW' in base_ability['name'] and 'RESEARCH_BURROW' not in base_ability['name']:
            new_abilities[base_ability['id']] = 'burrow'
        else:
            new_abilities[base_ability['id']] = base_ability['id']
    return new_abilities
