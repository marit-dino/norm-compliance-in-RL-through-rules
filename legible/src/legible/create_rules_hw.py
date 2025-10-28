import sys
from rule_learning.util import load_pickle, save_pickle
from shield.create_rules_common import create_new_rules
from shield.rule_classes import string_to_rule, Fact, RuleBody, Rule, RuleHead
from shield.shields import AspShield


def create_similar_conditions(conditions, domain_knowledge):
    groups_of_similar = domain_knowledge["groups_of_similar"]
    cond_indices = []
    largest_index = max([max(group) for group in groups_of_similar])
    for cond in conditions:
        # find condition indices
        for i in range(len(groups_of_similar)):
            for j in range(len(groups_of_similar[i])):
                if groups_of_similar[i][j] == cond.feature:
                    cond_indices.append((i,j))

    sim_rule_bodies = []
    for offset in range(0,len(groups_of_similar[0])):
        # include 0 offset to have the original rule as well
        offset_cond_indices = [(i,j + offset) for (i,j) in cond_indices]
        if any([j >= len(groups_of_similar[i]) for (i,j) in offset_cond_indices]):
            continue
        sim_conditions = []
        for cond,(offset_index_i,offset_index_j) in zip(conditions,offset_cond_indices):
            offset_index = groups_of_similar[offset_index_i][offset_index_j]
            sim_conditions.append(Fact(offset_index,cond.valuation))
        sim_rule_bodies.append(RuleBody(sim_conditions))
    return sim_rule_bodies

def create_similar_conditions_rules(rule, domain_knowledge):
    rule_polarity = rule.polarity
    action = rule.rule_head.action
    conditions = list(rule.rule_body.conditions)
    sim_rule_bodies = create_similar_conditions(conditions, domain_knowledge)
    new_rules = []
    for rule_body in sim_rule_bodies:
        new_rules.append(Rule(rule_polarity, RuleHead(action), rule_body))
    return new_rules


def mirror_interval(interval_index, intervals):
    interval = intervals[interval_index]
    is_negative = interval[0] < 0.0
    exact_mirror = (-interval[1],-interval[0])
    best_match = (0,10e6)
    best_index = -1
    for index,potential_mirror in enumerate(intervals):
        other_is_negative = potential_mirror[0] < 0.0
        # find best match
        if is_negative != other_is_negative:
            # must be inverted
            if exact_mirror[1] < potential_mirror[0] or exact_mirror[0] > potential_mirror[1]:
                # no overlap
                overlap = 0
                min_dist = min(abs(potential_mirror[0]-exact_mirror[1]),abs(exact_mirror[0] - potential_mirror[1]))
                best_overlap,best_min_dist = best_match
                if best_overlap == 0:
                    if min_dist < best_min_dist:
                        best_match = (0,min_dist)
                        best_index = index
            else:
                upper_overlap = min(potential_mirror[1],exact_mirror[1])
                lower_overlap = max(potential_mirror[0],exact_mirror[0])
                overlap = abs(upper_overlap - lower_overlap)

                # upper_non_overlap = max(potential_mirror[1], exact_mirror[1])
                # lower_non_overlap = min(potential_mirror[0], exact_mirror[0])
                # non_overlap = abs(upper_non_overlap - lower_non_overlap)
                if overlap > best_match[0]:
                    best_match = (overlap,0)
                    best_index = index
    print(f"Found: {intervals[best_index]} as mirror of {interval}")
    print(f"Out of: {intervals}")
    return best_index


def mirror_along_y_body(rule_body, x_intervals, vx_intervals, x_features, vx_features):
    mirrored_conds = []
    for cond in rule_body.conditions:
        feat = cond.feature
        val = cond.valuation
        if feat in x_features:
            print(x_intervals)
            mirrored_val = mirror_interval(val,x_intervals)
            mirrored_conds.append(Fact(feat, mirrored_val))
        elif feat in vx_features:
            mirrored_val = mirror_interval(val,vx_intervals)
            mirrored_conds.append(Fact(feat, mirrored_val))
        else:
            mirrored_conds.append(Fact(feat,val))
    return RuleBody(mirrored_conds)

def mirror_along_x_body(rule_body, y_intervals, vy_intervals, y_features, vy_features):
    mirrored_conds = []
    for cond in rule_body.conditions:
        feat = cond.feature
        val = cond.valuation
        if feat in y_features:
            mirrored_val = mirror_interval(val,y_intervals)
            mirrored_conds.append(Fact(feat, mirrored_val))
        elif feat in vy_features:
            mirrored_val = mirror_interval(val,vy_intervals)
            mirrored_conds.append(Fact(feat, mirrored_val))
        else:
            mirrored_conds.append(Fact(feat, val))
    return RuleBody(mirrored_conds)

def mirror_along_y(rule, x_features, vx_features,domain_knowledge):
    x_intervals = domain_knowledge["x_intervals"]
    vx_intervals = domain_knowledge["vx_intervals"]
    mirrored_body = mirror_along_y_body(rule.rule_body,x_intervals,vx_intervals,x_features,vx_features)
    if rule.rule_head.action == 0:  # [0, 2, 1]:  # LANE_LEFT or LANE_RIGHT or IDLE
        mirrored_head = RuleHead(2)
    elif rule.rule_head.action == 2:
        mirrored_head = RuleHead(0)
    else: # rule.rule_head.action == 1:
        mirrored_head = RuleHead(1)
    # include original rule
    return [rule, Rule(rule.polarity,mirrored_head,mirrored_body)]

def mirror_along_x(rule, y_features, vy_features, domain_knowledge):
    y_intervals = domain_knowledge["y_intervals"]
    vy_intervals = domain_knowledge["vy_intervals"]
    mirrored_body = mirror_along_x_body(rule.rule_body, y_intervals, vy_intervals,y_features, vy_features)
    if rule.rule_head.action == 3:  # [3, 4, 1] FASTER or SLOWER or IDLE
        mirrored_head = RuleHead(4)
    elif rule.rule_head.action == 4:
        mirrored_head = RuleHead(3)
    else:  # rule.rule_head.action == 1:
        mirrored_head = RuleHead(1)
    # include original rule
    return [rule, Rule(rule.polarity, mirrored_head, mirrored_body)]


def create_symmetric_rules(rule, domain_knowledge):
    # y direction is from left to right of the road
    # x direction is along the road
    # features ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]
    x_features = list(range(1, 70, 7))
    y_features = list(range(2, 70, 7))
    vx_features = list(range(3, 70, 7))
    vy_features = list(range(4, 70, 7))
    if rule.rule_head.action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
        new_rules = mirror_along_y(rule, x_features, vx_features,domain_knowledge)
    elif rule.rule_head.action in [3, 4]:  # FASTER or SLOWER
        new_rules = mirror_along_x(rule, y_features,vy_features,domain_knowledge)
    elif rule.rule_head.action in [1]:  #  IDLE
        new_rules_x = mirror_along_y(rule, x_features,vx_features,domain_knowledge)
        new_rules_y = mirror_along_x(rule, y_features,vy_features,domain_knowledge)
        new_rules = [new_rules_x[0],new_rules_x[1],new_rules_y[1]]
    else:
        raise Exception("Unknown Action")

    return new_rules

def create_rules_hw(rule_str, domain_knowledge):
    rule = string_to_rule(rule_str)
    new_rules = dict()
    new_rules_sim = create_similar_conditions_rules(rule, domain_knowledge)
    new_rules["similar"] = new_rules_sim

    new_rules_symmetry = create_symmetric_rules(rule,domain_knowledge)
    new_rules["symmetry"] = new_rules_symmetry

    return new_rules


def generalize_rules(shield : AspShield, shield_name,exact_model_number=None):
    domain_knowledge = {}
    groups_of_similar = []
    for i in range(7):
        sim_feat_from_i = list(range(i, 70, 7))
        groups_of_similar.append(sim_feat_from_i)
    domain_knowledge["groups_of_similar"] = groups_of_similar
    domain_knowledge["x_intervals"] = shield.feature_intervals[1]
    domain_knowledge["y_intervals"] = shield.feature_intervals[2]
    domain_knowledge["vx_intervals"] = shield.feature_intervals[3]
    domain_knowledge["vy_intervals"] = shield.feature_intervals[4]

    mutated_shield = create_new_rules(shield, domain_knowledge,
                                      create_rules=lambda rule_str, domain_knowledge: create_rules_hw(rule_str,
                                                                                                       domain_knowledge),
                                      do_remove=False)
    mutated_shield_name = shield_name.replace("uncorr", "improved")
    if exact_model_number is None:
        save_pickle(mutated_shield_name, mutated_shield)
    else:
        mutated_shield_name = f"{mutated_shield_name}"
        save_pickle(mutated_shield_name, mutated_shield, exact_match=True)
    return mutated_shield, mutated_shield_name


if __name__ == "__main__":
    env_name = sys.argv[1]
    steps = int(sys.argv[2])
    nr_features = int(sys.argv[3])
    algo = sys.argv[4]
    corr = sys.argv[5] == "True"
    corr_string = "corr" if corr else "uncorr"

    shield_name = f"pickles/shields/{corr_string}/{algo}_{env_name}_0_feat_{nr_features}_{steps}_shield"

    exact_model_number = None
    for arg in sys.argv:
        if "--exact_mod" in arg:
            exact_model_number = int(arg.replace("--exact_mod",""))
    if exact_model_number is None:
        shield = load_pickle(shield_name)
    else:
        shield_name = f"{shield_name}_{exact_model_number}.pkl"
        shield = load_pickle(shield_name, exact_match=True)

    generalize_rules(shield,shield_name,exact_model_number=exact_model_number)