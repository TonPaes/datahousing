
import loadData
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit as sss
# from pandas.plotting import scatter_matrix

# loadData.fetch_housing_data()
housing = loadData.load_housing_data()

# creating a new catgory to stratifi data
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# split data in train/test set
test_set, train_set = train_test_split(housing, test_size=0.2, random_state=42)

# split  strafied data in train/test set
split = sss(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(housing["income_cat"].value_counts() / len(housing))
print(train_set["income_cat"].value_counts()/len(train_set))

# to exclude income_cat
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# copy data to vizualize and tweaks
housing2 = strat_train_set.copy()

# correlation matrix
corr_matrix = housing2.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))


# setting attributtes to scater matrix
# att = [
#    "median_house_value",
#    "median_income",
#    "total_rooms",
#    "housing_median_age"]
#
# sct = scatter_matrix(housing[att], figsize=(12, 8))
# plt.show()

housing = strat_train_set.drop("median_house_value", axis=1)
hosing_labels = strat_train_set.copy()
