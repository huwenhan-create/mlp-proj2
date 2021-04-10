import numpy as np
import pandas as pd

# Define the function to analysis the association between features and `is_canceled`


def association_analysis(target: str, plot: bool, data, axes=None, log=False):
    """This function is used to do some univ_analysis: If we grouped by the target variables,
    We can calculate the frenquency of the happens of canceled books. Return the prob matrix and
    visualize it
    

    Args:
        target (str): the target variable
        plot (bool): If plot=True, we plot the bars, or we will not plot
        data (DataFrame): The data we used

    Returns:
        prob_taget: The probability of the book will be canceled if in different status of the target
        variable.
    """
    _target = data.loc[:, ['is_canceled', target]]
    group_target = _target.groupby(by=target)
    cancel_target = group_target.sum()
    total_target = group_target.size()
    prob_target = [int(m)/int(n)
                   for m, n in zip(cancel_target.to_numpy(), total_target.to_numpy())]

    if log:
        log_prob_target = [np.log(p) for p in prob_target]
    else:
        log_prob_target = [p for p in prob_target]

    if plot:
        labels = list(group_target.groups.keys())
        ax = axes
        ax.bar(x=labels, width=0.5, height=log_prob_target,
               color=['cornflowerblue'])
    return prob_target


def property_analysis(target: str, df):
    tar = df[target]
    freq = tar.value_counts()/tar.count()
    return freq

def entropyValue(prob):
    if (prob == 0 or prob == 1):
        entropy = 0
    else:
        entropy = -(prob*np.log2(prob)+(1-prob)*np.log2(1-prob))
    return entropy


def cal_Ent(prob: list):
    ent_list = []
    for p in prob:
        ent=entropyValue(p)
        ent = round(ent, 2)
        ent_list.append(ent)
    return ent_list

# Define the function to calculate the information gain
def cal_Gain(Ent, Ent_new, prop):
    Gain = []
    n = len(Ent_new)
    for i in range(n):
        Gain.append(prop[i]*Ent_new[i])
    Gain = Ent-sum(Gain)
    return Gain
