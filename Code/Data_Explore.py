# This file is to do some data explorations
# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
import seaborn as sns

# Load the data
hotel = pd.read_csv('Code\Data\hotel.csv')
print(hotel.shape)  # (119390, 30)

# -----------------------------------------------< section 1>--------------------------------------------------------------
# {1} Fisrt section: Basic preprocessing
# {1.1} missing value
# ----Find the missing values----
missing = hotel.isna().apply(np.sum,axis=0)
# {children:4/0.000034, country:488/0.004087, agent:16340/0.136862, company:112593/0.943069}
print(missing[missing != 0]/119390)

# ----Working with missing data----
# [1]. Children: only 4 missing value (nearly no effect on result after impute them by using mode)
children = hotel.loc[:,'children']
print(children.mode())
children = children.fillna(0)
children.to_csv('Code\Data\children.csv', index=False)


# [2]. agent: we regard the missing value as one of the attributes and take it into account. 
#             Here, we just replace them of 'none':str 
agent = hotel.loc[:, 'agent'].replace(np.nan, 0)

# [3]. country: Same as the children, we choose the mode to impute the data
country = hotel.loc[:,'country']
print(country.mode())
country = country.fillna('PRT')

# [4]. company: Because there are too many missing values, we choose to drop this variable out.
# Fianlly, we get the final data
imputed = pd.concat([children,agent,country],axis=1)
imputed.to_csv('Code\Data\imputed.csv',index=False)
hotel.loc[:,['children','agent','country']] = imputed.values
hotel = hotel.drop(['company'],axis=1)
print(len(hotel.columns))

# -----------------------------------------------< section 2 >----------------------------------------------------------
# {2} Second section: Advanced preprocessing
# := This section, our goal is to primarily reduce the dimensions through association studys, plots, common sence and so on
#    And we also seperate the variables into two groups: categorical and numeric 
#    For those categorical variable --> association analysis, classification, one-hot transformation
#    For those numerical variable --> association analysis, scale.

# {2.1} Seperate the data into groups: (optional: maybe useless)
cat_variates = hotel.select_dtypes(include=['object'])  # categorical variates
num_variates = hotel.select_dtypes(include=['int64','float64'])  # maybe the numeric variates

# {2.2} Univariates analysis

# [1]. Hotel --> naturally one-hot encode (Through the plot, we say there are relationships between hotels
# and the outcome of interest)

"""[--test--]
# _hotel = hotel.loc[:,['is_canceled','hotel']]                                                    #
# group_hotel = _hotel.groupby(by='hotel')                                                         #
# cancel_hotel = group_hotel.sum()                                                                 #
# total_hotel = group_hotel.size()                                                                 #
# prob_hotel = [int(m)/int(n) for m,n in zip(cancel_hotel.to_numpy(), total_hotel.to_numpy())]     #
# labels = list(group_hotel.groups.keys())                                                         #
# fig = plt.figure()                                                                               #
# ax = fig.add_subplot()                                                                           #
# ax.bar(x=labels,width=0.2,height=prob_hotel, color=['cornflowerblue','coral'])                   #
# ax.set_title('The effect of different hotels')                                                   #
# plt.show()                                                                                       #
"""
# We can make this process a function:
def univ_analysis_one(target:str, plot:bool, data:DataFrame, axes, log=False):
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
    _target = data.loc[:,['is_canceled',target]]
    group_target = _target.groupby(by=target)
    cancel_target = group_target.sum()
    total_target = group_target.size()
    prob_target = [int(m)/int(n) for m, n in zip(cancel_target.to_numpy(), total_target.to_numpy())]

    if log:
        log_prob_target = [np.log(p) for p in prob_target]
    else:
        log_prob_target = [p for p in prob_target]
    
    if plot:
        labels = list(group_target.groups.keys())
        ax = axes
        ax.bar(x=labels, width=0.5, height=log_prob_target,
            color=['cornflowerblue'])
        ax.set_title(target)
    return prob_target

# Plot
fig = plt.figure()
ax1 = fig.add_subplot(111)
prob_hotel = univ_analysis_one('hotel',True,hotel,ax1)



 

# [] <time::= arrival_date_year, arrival_date_month, arrival_date_week, day>

"""
# year--> drop out (From the plot, there are little difference between each year, which means that the year has little effect on the outcome)
# The arrival weeks can identify the arrival_data_month, and so we can just drop this variable
"""
fig_time = plt.figure()
ax_lt = fig_time.add_subplot(221)
prob_lead_time = univ_analysis_one('lead_time', True, hotel, ax_lt)

ax2 = fig_time.add_subplot(222)
prob_arrival_year = univ_analysis_one('arrival_date_year', True, hotel, ax2)


ax3 = fig_time.add_subplot(223)
prob_arrival_weeks = univ_analysis_one('arrival_date_week_number', True, hotel,ax3)  # non-linear relationship


ax4 = fig_time.add_subplot(224)
prob_arrival_days = univ_analysis_one('arrival_date_day_of_month', True, hotel,ax4)  # not so obvious

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()
print()

# [4] <Groups:== adults, children, babies>

"""
Contain these three variables
"""

fig_groups = plt.figure()

# ax_adults = fig_groups.add_subplot(211)
# prob_adults = univ_analysis_one('adults',True, hotel,ax_adults)

# ax_children = fig_groups.add_subplot(223)
# prob_children = univ_analysis_one('children', True, hotel, ax_children)

ax_babies = fig_groups.add_subplot(111)
prob_babies = univ_analysis_one('babies', True, hotel, ax_babies)

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()

# [5] meal and country
fig = plt.figure()
ax1 = fig.add_subplot(121)
prob_meal = univ_analysis_one('meal',True,hotel,ax1)
ax2 = fig.add_subplot(122)
prob_country = univ_analysis_one('agent',True,hotel,ax2)
plt.show()

# [6] reserved_room and assigned_roon
res_room, ass_room = hotel.loc[:,'reserved_room_type'], hotel.loc[:,'assigned_room_type']
same_room = []
for i,j in zip(res_room,ass_room):
    if i==j:
        same_room.append(1)
    else:
        same_room.append(0)




# Feature engineering
hotel['same_room'] = same_room
hotel = hotel.drop(['arrival_date_year', 'arrival_date_month','reserved_room_type','assigned_room_type'], axis=1)
hotel = pd.get_dummies(hotel)


# Output the final data:
hotel.to_csv('Code\Data\hotel_pre.csv',index=False)
