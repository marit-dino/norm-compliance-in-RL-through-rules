import re
from collections import defaultdict

import random
import numpy as np
import sklearn
import lime
import lime.lime_tabular
import math
import pandas as pd
from scipy.stats import chi2_contingency

from rule_learning.util import predict_act

NEG_INFTY = -1e10
POS_INFTY = 1e10
INTERVAL_EPS = 0.01

def extract_feature_and_interval(disc_name):

    if "=" in disc_name and "<" not in disc_name and ">" not in disc_name:
        return None, None
    if "<=" in disc_name and "< " not in disc_name:
        # upper bounded intervals
        pattern = "(\\d+ )<= (.+)"
        match = re.search(pattern, disc_name)
        if not match:
            raise Exception(f"Could not parse {disc_name}")
        match_g = match.groups()
        feature_name = int(match_g[0].strip())
        upper_bound = float(match_g[1].strip())
        return feature_name, (NEG_INFTY, upper_bound)
    elif "<=" in disc_name and "< " in disc_name:
        # both bounds present
        pattern = "(.+) < (\\d+ )<= (.+)"
        match = re.search(pattern, disc_name)
        if not match:
            raise Exception(f"Could not parse {disc_name}")
        match_g = match.groups()
        lower_bound = float(match_g[0].strip())
        feature_name = int(match_g[1].strip())
        upper_bound = float(match_g[2].strip())
        if upper_bound - lower_bound < INTERVAL_EPS:
            print("Widening interval")
            return feature_name, (lower_bound-INTERVAL_EPS, upper_bound + INTERVAL_EPS)
        return feature_name, (lower_bound, upper_bound)

    if ">" in disc_name:
        # lower bounded intervals
        pattern = "(\\d+ )> (.+)"
        match = re.search(pattern, disc_name)
        if not match:
            raise Exception(f"Could not parse {disc_name}")
        match_g = match.groups()
        feature_name = int(match_g[0].strip())
        lower_bound = float(match_g[1].strip())
        return feature_name, (lower_bound, POS_INFTY)
    else:
        raise Exception(f"Could not parse {disc_name}")


def complete_intervals(intervals):
    last_upper = NEG_INFTY
    # to_be_added = []
    completed_intervals = []
    # first, change intervals so that they don't overlap
    for i in range(len(intervals)):
        (lower, upper) = intervals[i]
        (next_lower, next_upper) = intervals[i+1] if i < len(intervals) - 1 else (POS_INFTY,POS_INFTY)
        if lower < last_upper:
            lower = last_upper
        if next_lower < upper:
            upper = next_lower
        completed_intervals.append((lower,upper))
        last_upper = upper
    # at this point there may be singleton intervals again
    to_be_removed = []
    for i in range(len(completed_intervals)):
        (lower, upper) = completed_intervals[i]
        if upper - lower < 1e-6:
            to_be_removed.append((lower,upper))
    for interval in to_be_removed:
        completed_intervals.remove(interval)

    # fill in holes
    last_upper = NEG_INFTY
    to_be_added = []
    completed_intervals = sorted(completed_intervals)
    for i in range(len(completed_intervals)):
        (lower, upper) = completed_intervals[i]
        if lower > last_upper:
            to_be_added.append((last_upper,lower))
        last_upper = upper
    if last_upper < POS_INFTY:
        to_be_added.append((last_upper, POS_INFTY))
    completed_intervals.extend(to_be_added)
    return sorted(completed_intervals)


def discretized_feature_names_to_intervals(all_discretized_feature_names, categorical_features):
    feature_intervals = defaultdict(list)
    for disc_name in all_discretized_feature_names:
        feature, interval = extract_feature_and_interval(disc_name)
        if feature is None:  # categorical
            continue
        # print(f"Parsed {feature} from {disc_name} into: {interval}")
        feature_intervals[feature].append(interval)
    for feature in feature_intervals.keys():
        print(f"Intervals for {feature}:")
        intervals = sorted(list(set(feature_intervals[feature])))
        intervals = complete_intervals(intervals)
        print(intervals)
        feature_intervals[feature] = intervals
    return feature_intervals

def cramersV(var1, var2):
    '''
    var1 and var2 are pandas series
    '''
    N = len(var1)

    obs = pd.crosstab(var1, var2)

    if len(obs) != 1 and len(obs.columns) != 1:
        # calculation of Chi square
        chi2, _, _, _ = chi2_contingency(obs)
        # calculation of Cramer's V
        m = min(len(obs.columns)-1, len(obs[obs.columns[0]]))
        return (var1.name, var2.name, math.sqrt(chi2/(N*m)))
    else:
        return (var1.name, var2.name, 0)


def get_correlated_categ_vars(X, min_corr=0.9):
    '''
    It returns a list of lists where each list is (var1, var2, Cramer's V)
    '''
    N = len(X)
    unique_vals_col = [[col, X[col].nunique()] for col in X.columns] # TODO: dictionary better

    # remove columns with only one unique value
    cols_to_drop = [X.columns[i] for i in range(len(unique_vals_col)) if unique_vals_col[i][1] == 1]
    X = X.drop(columns=cols_to_drop)

    original_columns = list(X.columns)

    # create new dummy columns with each of the values of each of the features
    X_dummies = pd.get_dummies(X,prefix=original_columns, columns = original_columns, drop_first=False)
    dummy_names = list(X_dummies.columns)
    
    X = pd.concat([X, X_dummies], axis=1)
    
    corr_vars = []

    for i in range(len(original_columns)-1):
        col1 = original_columns[i]
        if i % 10 == 0:
            print(f'Calculating correlations for {col1}')

        # remove the current feature col1 from the dummy names
        dummy_names = list(filter(lambda x: (not x.startswith(f'{col1}_')), dummy_names))

        # remove features that are correlated with the current feature col1
        dummy_names_reduced = dummy_names.copy()
        new_columns = original_columns.copy()
        for group in corr_vars:
            if col1 in group:
                for var in group:
                    dummy_names_reduced = list(filter(lambda x: (not x.startswith(f'{var}_')), dummy_names_reduced))
                    new_columns.remove(var)
                break # the groups should be disjoint pairwise

        # compute table. It is sum() because the dummy features only have 1s or 0s
        X_grouped = X[[col1] + dummy_names_reduced].groupby(col1).sum()

        skipvars = set() # when we find a correlated var, we do not need to analyse the other features that are correlated with this one
        for j in range(i+1, len(new_columns)):
            col2 = new_columns[j]
            if col2 not in skipvars:
                # the names of the dummy features corresponding to col2
                col2_dummy_names = list(filter(lambda x: (x.startswith(f'{col2}_')), dummy_names))
                # compute chi-square test
                chi2, p, _, _ = chi2_contingency(X_grouped[col2_dummy_names])
                # normalize
                mcol1 = list(filter(lambda x: (x[0]==col1), unique_vals_col))[0][1] - 1
                mcol2 = list(filter(lambda x: (x[0]==col2), unique_vals_col))[0][1] - 1
                m = min(mcol1, mcol2)
                if (p >= 0.05) and math.sqrt(chi2/(N*m)) > min_corr:
                    print("ERROR")
                if math.sqrt(chi2/(N*m)) > min_corr:
                    corr_vars.append([col1, col2])
                    corr_vars = join_multiple_corr_vars(corr_vars)
                    for group in corr_vars:
                        if col2 in group:
                            skipvars.update(set([x for x in group if x != col1]))
                            break # the groups should be disjoint pairwise
        
    return [sorted([int(x.replace('f', '')) for x in group]) for group in corr_vars]


# def get_correlated_categ_vars(X, min_corr=0.9):
#     '''
#     It should only be used with categorical variables
#     returns a list of the pairs of correlated variables according to Cramer's V
#     '''

#     # filter by the minimum correlation and remove the 'f' from the variable name
#     return [(int(x[0].replace('f', '')), int(x[1].replace('f', ''))
#              ) for x in get_cramersVs(X)]


def get_correlated_cont_vars(X, min_corr):
    '''
    It should only be used with continuous variables
    returns a list of the pairs of correlated variables according to Pearson's correlation
    '''
    # create a list with the pairs that we will drop from the correlation matrix
    # we just need the triangular matrix
    pairs_to_drop = set()
    cols = X.columns
    for i in range(0, X.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))

    # obtain correlations and filter by minimum correlation
    corr_matrix = X.corr().abs()

    corr_list = corr_matrix.unstack().drop(labels=pairs_to_drop)

    return [[int(x[0].replace('f', '')), int(x[1].replace('f', ''))
             ] for x in list(corr_list[corr_list > 0.9].keys())]


def join_multiple_corr_vars(corr_vars):
    '''
    It joins variables that are correlated with more than one variable.
    It takes a list of pairs of correlated variables, e.g.,
    [(33, 126), (12, 33), (45, 46), (45, 57), (45, 60)]
    and returns a list of n-tuples of correlated variables, e.g.,
    [(33, 126, 12), (45, 46, 57, 60)]
    
    Thanks to https://www.reddit.com/r/learnpython/comments/n84jn4/list_of_lists_merge_sublists_with_common_elements/
    '''

    list1 = corr_vars.copy()

    # List of all values
    values = sum(list1, [])

    # Each value into its own set
    values = list(map(lambda x: {x}, set(values)))

    # Loop sublists
    for item in map(set, list1):

        # All value sets that share at least one value with sublist
        values_in = [x for x in values if x & item]

        # All value sets with no shared values with sublist
        values_out = [x for x in values if not x & item]

        # Merge value sets with shared values into one set
        values_in = set([]).union(*values_in)

        # Re-define the value sets with the new sets, if any were merged
        if values_in:
            values = values_out + [values_in]

    list2 = [sorted(list(y for y in x)) for x in values]

    return list2


def get_correlated_vars(data, categorical_features, most_important_features=list(), min_corr=0.9):
    '''
    It returns a list of highly correlated features.
    If the data contains categorical features, they should be passed as a list of strings
    '''

    categorical_features_names = [f'f{i}' for i in categorical_features if i in most_important_features]

    X = pd.DataFrame(data, columns=[f'f{i}' for i in range(0, len(data[0, :]))])[[f'f{i}' for i in most_important_features]]

    corr_cat_vars = []
    corr_cont_vars = []

    if len(categorical_features_names) > 0:
        print("Calculating correlations for categorical features")
        # some categorical features and potentially some continuous features
        corr_cat_vars = get_correlated_categ_vars(X[categorical_features_names], min_corr)
        print("Done with categorical features")
        # the following list could be empty
        continuous_features = [feature for feature in X.columns if feature not in categorical_features_names]
    else:
        # only continuous features
        continuous_features = X.columns

    if len(continuous_features) > 0:
        print("Calculating correlations for continuous features")        
        corr_cont_vars = get_correlated_cont_vars(X[continuous_features], min_corr)
        print("Done with continuous features")

    return join_multiple_corr_vars(corr_cat_vars + corr_cont_vars)


def get_grouped_values(features, corr_vars):
    '''
    returns a list where the first element of each item is a tuple with the indeces
    of the features. The second element is the sum of values.
    For example,
    [[[78,79], 321], [[28,29], 123], ... ]
    It means features 78 and 79 are correlated and their sum is 321.
    
    features: int dictionary with (feature, value)
    corr_vars: list of the n-tuples of correlated variables (e.g., output from get_correlated_vars())
    '''

    # TODO assert the list of corr_vars cannot have duplicates.

    features_grouped = []

    corr_vars_flattened = []

    for corr_pair in corr_vars:
        count = 0
        for var in corr_pair:
            count += features[var]
            corr_vars_flattened.append(var)
        features_grouped.append([corr_pair, count])

    for feature in features:
        if feature not in corr_vars_flattened:
            features_grouped.append([[feature], features[feature]])

    return sorted(features_grouped, key=lambda x: x[1], reverse=True)


def get_most_important_corr_features(X, target, feature_indices):
    '''
    It returns for every group of correlated features, the most important one
    with respect to its predictive power for the target using Cramer's V
    '''
    # TODO other possibilities: Mutual information

    most_important_features = []
    for corr_vars in feature_indices:
        # look for the feature with the highest correlation with the target
        highest_corr_with_target = 0
        most_important_feature = corr_vars[0]
        for var in corr_vars:
            cramersv_value = cramersV(pd.Series(name="target",data=target),X[f'f{var}'])[2]
            if cramersv_value > highest_corr_with_target:
                highest_corr_with_target = cramersv_value
                most_important_feature = var
        most_important_features.append(most_important_feature)

    return most_important_features


def harmonize_single_group(intervals_for_group):
    all_borders = set()
    for interval_list in intervals_for_group:
        for interval in interval_list:
            all_borders.add(interval[0])
            all_borders.add(interval[1])
    all_borders = sorted(list(all_borders))
    common_int = []
    for i in range(len(all_borders)-1):
        common_int.append((all_borders[i],all_borders[i+1]))
    return common_int


def harmonize_similar_features(feature_intervals,groups_of_similar):
    print(feature_intervals.keys())
    for group_of_similar in groups_of_similar:
        # group_feat_name = [f"f{feat}" for feat in group_of_similar]
        intervals_for_group = [feature_intervals[feat] for feat in group_of_similar]
        common_intervals = harmonize_single_group(intervals_for_group)
        for feat in group_of_similar:
            feature_intervals[feat] = common_intervals
    print("after harmonize")
    print(feature_intervals.keys())
    for feature in sorted(list(feature_intervals.keys())):
        print(f"Intervals for {feature}:")
        print(feature_intervals[feature])

def detect_features(env_name,data, model, nr_features, lime_test_size, action_tensor, nr_features_all, n_actions,algo_name,
                    groups_of_similar,
                    categorical_features=None,
                    sample_reconstruction=None,
                    compute_correlation=True,
                    only_features_for_neg=False,
                    act_probs_for_label=False,
                    use_all_actions=True):
    """
    Returns a list of pairs (index, name), where index is the feature index in the flattened observation of the agent
    name is a readable name
    Categorical_features = None means that all features are categorical
    """
    if sample_reconstruction is None:
        sample_reconstruction = lambda x,y: x
    if act_probs_for_label:
        labels = np.argmax(data[:, :n_actions],axis=1)
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, n_actions:]
    else:
        labels = data[:, 0]
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(labels)
        labels = le.transform(labels)
        data = data[:, 1:]
    if categorical_features is None:
        categorical_features = range(nr_features_all)
    data = data.astype(float)


    # set shuffle to false since we need to match with the latent state
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80,
                                                                                      shuffle=False)
    class_names = [f"a{i}" for i in range(n_actions)]

    feature_names = [str(i) for i in range(train.shape[1])]
    if "highway" in env_name or "intersection" in env_name or "roundabout" in env_name or "merge" in env_name in env_name:
        discretizer = HwDiscretizer(train, categorical_features,feature_names)
    else:
        discretizer = "decile"

    explainer = lime.lime_tabular.LimeTabularExplainer(train, class_names=class_names,  #['left', 'right', 'continue'],
                                                       # feature_names=feature_names,
                                                       categorical_features=categorical_features,
                                                       kernel_width=np.sqrt(nr_features_all-len(categorical_features)) * .5,
                                                       discretizer=discretizer,
                                                       verbose=False)
    feature_importances = dict()
    for action in range(n_actions):
        # negative and positive direction
        feature_importances[(action,True)] = defaultdict(float)
        feature_importances[(action,False)] = defaultdict(float)

    most_significant_features = defaultdict(int)
    most_significant_features_prob_sum = defaultdict(float)
    test_subset = random.choices(range(0, len(test)), k=min(lime_test_size,len(test)))
    cnt = 0
    all_discretized_feature_names = []
    for i in test_subset:
        if cnt % 10 == 0:
            print(f"Explaining index {cnt}")
        cnt += 1
        predict_fn = lambda x: predict_act(model, x, action_tensor, sample_reconstruction,algo_name)
        exp = explainer.explain_instance(test[i], predict_fn, num_features=nr_features, top_labels=
                                                                                    n_actions if use_all_actions else 3)
        all_discretized_feature_names.extend(exp.domain_mapper.discretized_feature_names)

        map = exp.as_map()
        for act,expl_per_act in map.items():
            act_prob = exp.intercept[act]
            for (feat, val) in expl_per_act:
                if only_features_for_neg and val < 0:
                    most_significant_features[feat] += 1
                    most_significant_features_prob_sum[feat] += abs(val)
                    feature_importances[(act,False)][feat] += abs(val) * abs(act_prob)
                elif not only_features_for_neg:
                    most_significant_features[feat] += 1
                    most_significant_features_prob_sum[feat] += abs(val)
                    feature_importances[(act,val >= 0)][feat] += abs(val) * abs(act_prob)

    if compute_correlation:
        corr_vars = get_correlated_vars(data, categorical_features=categorical_features, most_important_features=list(most_significant_features))
    else:
        corr_vars = [[i] for i in range(nr_features_all)]

    print("Correlated features:")
    print(corr_vars)
    # normalization
    feature_importances_normalized = dict()
    for act_polarity in feature_importances.keys():
        importance_dict = feature_importances[act_polarity]
        max_imp = max(importance_dict.values())
        feature_importances_normalized[act_polarity] = {feat : importance_dict[feat] /max_imp for feat in most_significant_features.keys()}

    feature_importances = feature_importances_normalized
    # group features that are highly correlated
    most_signicant_features_grouped = get_grouped_values(
        most_significant_features, corr_vars)

    most_significant_features_prob_sum_grouped = get_grouped_values(
        most_significant_features_prob_sum, corr_vars)

    detected_features = []
    feature_intervals = discretized_feature_names_to_intervals(all_discretized_feature_names, categorical_features)
    harmonize_similar_features(feature_intervals,groups_of_similar)
    for msf, count in most_significant_features_prob_sum_grouped:
        if len(detected_features) < nr_features:
            feature_name = "f" + "__".join([str(idx) for idx in msf])
            print(
                f'{feature_name} : {count} with {list(filter(lambda x: (x[0] == msf), most_significant_features_prob_sum_grouped))[0][1]}')
            detected_features.append((msf, feature_name))

    return detected_features, feature_intervals,feature_importances
