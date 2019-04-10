import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, _tree
import seaborn as sns
import gc
import math


def get_grouped_data(input_data, feature, target_col, bins, cuts=0, is_train=0):
    """
    :param is_train: trigger to judge whether it's used for train
    :param input_data: data frame contain features and target column
    :param feature: feature column name
    :param target_col: target column
    :param bins: Number bins required
    :param cuts: if buckets of certain specific cuts are required. Used on test data to use cuts from train.
    :return: If cuts are passed only grouped data is returned, else cuts and grouped data is returned
    """
    has_null = pd.isnull(input_data[feature]).sum() > 0
    if has_null == 1:
        data_null = input_data[pd.isnull(input_data[feature])]
        input_data = input_data[~pd.isnull(input_data[feature])]
        input_data.reset_index(inplace=True, drop=True)

    if cuts == 0:
        is_train = 1
        prev_cut = min(input_data[feature]) - 1
        cuts = [prev_cut]
        reduced_cuts = 0
        for i in range(1, bins + 1):
            next_cut = np.percentile(input_data[feature], i * 100 / bins)
            if next_cut != prev_cut:
                cuts.append(next_cut)
            else:
                reduced_cuts = reduced_cuts + 1
            prev_cut = next_cut

        # if reduced_cuts>0:
        #     print('Reduced the number of bins due to less variation in feature')
        cut_series = pd.cut(input_data[feature], cuts)
    else:
        cut_series = pd.cut(input_data[feature], cuts)

    grouped = input_data.groupby([cut_series], as_index=True).agg(
        {target_col: [np.size, np.mean], feature: [np.mean]})
    grouped.columns = ['_'.join(cols).strip() for cols in grouped.columns.values]
    grouped[grouped.index.name] = grouped.index
    grouped.reset_index(inplace=True, drop=True)
    grouped = grouped[[feature] + list(grouped.columns[0:3])]
    grouped = grouped.rename(index=str, columns={target_col + '_size': 'Samples_in_bin'})
    grouped = grouped.reset_index(drop=True)
    corrected_bin_name = '[' + str(min(input_data[feature])) + ', ' + str(grouped.loc[0, feature]).split(',')[1]
    grouped[feature] = grouped[feature].astype('category')
    grouped[feature] = grouped[feature].cat.add_categories(corrected_bin_name)
    grouped.loc[0, feature] = corrected_bin_name

    if has_null == 1:
        grouped_null = grouped.loc[0:0, :].copy()
        grouped_null[feature] = grouped_null[feature].astype('category')
        grouped_null[feature] = grouped_null[feature].cat.add_categories('Nulls')
        grouped_null.loc[0, feature] = 'Nulls'
        grouped_null.loc[0, 'Samples_in_bin'] = len(data_null)
        grouped_null.loc[0, target_col + '_mean'] = data_null[target_col].mean()
        grouped_null.loc[0, feature + '_mean'] = np.nan
        grouped[feature] = grouped[feature].astype('str')
        grouped = pd.concat([grouped_null, grouped], axis=0)
        grouped.reset_index(inplace=True, drop=True)

    grouped[feature] = grouped[feature].astype('str').astype('category')
    if is_train == 1:
        return cuts, grouped
    else:
        return grouped


def draw_plots(input_data, origin_data, feature, target_col, trend_correlation=None):
    """
    :param origin_data: origin data to plot the distribute
    :param input_data: grouped data contained bins of feature and target mean.
    :param feature: feature column name
    :param target_col: target column
    :param trend_correlation: correlation between train and test trends of feature wrt target
    :return: Draws trend plots for feature
    """
    sns.set_style("whitegrid")
    trend_changes = get_trend_changes(grouped_data=input_data, feature=feature,
                                      target_col=target_col)
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121)
    ax1.plot(input_data[target_col + '_mean'], marker='o', color='g')
    ax1.set_xticks(np.arange(len(input_data)))
    ax1.set_xticklabels((input_data[feature]).astype('str'))
    plt.xticks(rotation=50)
    ax1.grid(False)
    ax1.set_xlabel('Bins of ' + feature)
    ax1.set_ylabel('Average of ' + target_col)
    ax1.yaxis.tick_left()
    ax1.yaxis.set_label_position('left')
    comment = "Trend changed " + str(trend_changes) + " times"
    if trend_correlation == 0:
        comment = comment + '\n' + 'Correlation with train trend: NA'
    elif trend_correlation is not None:
        comment = comment + '\n' + 'Correlation with train trend: ' + str(int(trend_correlation * 100)) + '%'

    props = dict(boxstyle='round', facecolor='snow', alpha=0.5)
    ax1.text(0.4, 0.5, comment, fontsize=10, verticalalignment='center', bbox=props, transform=ax1.transAxes)
    plt.title('Average and Samples in bins of ' + target_col + ' on ' + feature)

    ax2 = ax1.twinx()
    ax2.bar(np.arange(len(input_data)), input_data['Samples_in_bin'], alpha=0.4)
    ax2.set_ylabel('Bin-Wise sample size')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    plt.tight_layout()

    ax3 = fig.add_subplot(122)  # This can not be delete
    flag_value = np.array(origin_data[target_col].unique()).astype(int)
    for i in flag_value:
        sns.distplot(origin_data.loc[origin_data[target_col] == i, feature].fillna(-900), hist=True, bins=50, label=i)
    plt.title("Origin distribution of %s" % feature)
    plt.subplots_adjust(wspace=0.25)
    gc.collect()
    plt.show()


def get_trend_changes(grouped_data, feature, target_col, threshold=0.03):
    """
    :param grouped_data: grouped dataset
    :param feature: feature column name
    :param target_col: target column
    :param threshold: minimum % difference required to count as trend change
    :return: number of trend chagnes for the feature
    """
    grouped_data = grouped_data.loc[grouped_data[feature] != 'Nulls', :].reset_index(drop=True)
    target_diffs = grouped_data[target_col + '_mean'].diff()
    target_diffs = target_diffs[~np.isnan(target_diffs)].reset_index(drop=True)
    max_diff = grouped_data[target_col + '_mean'].max() - grouped_data[target_col + '_mean'].min()
    target_diffs_mod = target_diffs.fillna(0).abs()
    low_change = target_diffs_mod < threshold * max_diff
    target_diffs_norm = target_diffs.divide(target_diffs_mod)
    target_diffs_norm[low_change] = 0
    target_diffs_norm = target_diffs_norm[target_diffs_norm != 0]
    target_diffs_lvl2 = target_diffs_norm.diff()
    changes = target_diffs_lvl2.fillna(0).abs() / 2
    tot_trend_changes = int(changes.sum()) if ~np.isnan(changes.sum()) else 0
    return tot_trend_changes


def get_trend_correlation(grouped, grouped_test, feature, target_col):
    """
    :param grouped: train grouped data
    :param grouped_test: test grouped data
    :param feature: feature column name
    :param target_col: target column name
    :return: trend correlation between train and test
    """
    grouped = grouped[grouped[feature] != 'Nulls'].reset_index(drop=True)
    grouped_test = grouped_test[grouped_test[feature] != 'Nulls'].reset_index(drop=True)

    if grouped_test.loc[0, feature] != grouped.loc[0, feature]:
        grouped_test[feature] = grouped_test[feature].cat.add_categories(grouped.loc[0, feature])
        grouped_test.loc[0, feature] = grouped.loc[0, feature]
    grouped_test_train = grouped.merge(grouped_test[[feature, target_col + '_mean']], on=feature, how='left',
                                       suffixes=('', '_test'))
    nan_rows = pd.isnull(grouped_test_train[target_col + '_mean']) | pd.isnull(
        grouped_test_train[target_col + '_mean_test'])
    grouped_test_train = grouped_test_train.loc[~nan_rows, :]
    if len(grouped_test_train) > 1:
        trend_correlation = np.corrcoef(grouped_test_train[target_col + '_mean'],
                                        grouped_test_train[target_col + '_mean_test'])[0, 1]
    else:
        trend_correlation = 0
        print("Only one bin created for " + feature + ". Correlation can't be calculated")

    return trend_correlation


def tree_split_bins(input_data, feature, target_col, bins=10, get_bins_alone=0, min_samples_leaf=0.05,
                    min_samples_split=0.1):
    """
    :param min_samples_leaf: tree split min samples in leaf
    :param min_samples_split: tree split min samples in split
    :param input_data: pandas data frame  containing  feature and target columns
    :param feature: the feature column name
    :param target_col: the target column name
    :param bins: the maximum number of buckets
    :param get_bins_alone: whether to obtain interval values ​​separately for other purposes
    :return: the bins
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_depth=bins,
        max_leaf_nodes=bins)
    x = input_data[feature].fillna(-900).values.reshape(input_data[feature].shape[0], 1)
    y = input_data[target_col]
    clf.fit(x, y)
    count_leaf = 0
    for i in clf.tree_.children_left:
        if i == _tree.TREE_LEAF:
            count_leaf += 1
    threshold = clf.tree_.threshold
    count = 0
    for i in threshold:
        if i == -2: count += 1
    new_threshold = list(filter(lambda x: x != -2, threshold))
    if count > count_leaf: new_threshold += [-2]
    if get_bins_alone == 1:
        new_threshold_2 = np.sort(new_threshold)
    else:
        prev_cut = min(x) - 1  # THE BEGINING PART
        new_threshold.append(prev_cut)
        new_threshold_2 = np.sort(new_threshold)
    return new_threshold_2.tolist()


def variate_plotter(feature, data, target_col, bins=10, data_test=0, tree_split=False):
    print(' {:^100} '.format('Plots for ' + feature))
    if data[feature].dtype == 'O':
        print('Categorical feature not supported')
    elif tree_split:
        cut = tree_split_bins(input_data=data, feature=feature, target_col=target_col, bins=bins)

        cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col,
                                         bins=bins,
                                         cuts=cut, is_train=1)
        has_test = type(data_test) == pd.core.frame.DataFrame
        if has_test:
            grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True),
                                            feature=feature,
                                            target_col=target_col, bins=bins, cuts=cuts)
            trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
            print(' {:^100} '.format('Train data plots'))

            draw_plots(input_data=grouped, origin_data=data, feature=feature, target_col=target_col)
            print(' {:^100} '.format('Test data plots'))

            draw_plots(input_data=grouped_test, origin_data=data, feature=feature, target_col=target_col,
                       trend_correlation=trend_corr)
        else:
            draw_plots(input_data=grouped, origin_data=data, feature=feature, target_col=target_col)
        print(
            '---' * 20)
        print('\n')
        # if has_test:
        #     return grouped, grouped_test
        # else:
        #     return grouped
    else:
        cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col,
                                         bins=bins)
        has_test = type(data_test) == pd.core.frame.DataFrame
        if has_test:
            grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True),
                                            feature=feature,
                                            target_col=target_col, bins=bins, cuts=cuts)
            trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
            print(' {:^100} '.format('Train data plots'))

            draw_plots(input_data=grouped, origin_data=data, feature=feature, target_col=target_col)
            print(' {:^100} '.format('Test data plots'))

            draw_plots(input_data=grouped_test, origin_data=data, feature=feature, target_col=target_col,
                       trend_correlation=trend_corr)
        else:
            draw_plots(input_data=grouped, origin_data=data, feature=feature, target_col=target_col)
        print(
            '---' * 20)
        print('\n')
        # if has_test:
        #     return grouped, grouped_test
        # else:
        #     return grouped


def PSI_cal(grouped, grouped_test, target_col):
    sum = 0
    count = 0
    for i in grouped.index:
        var1 = grouped_test.loc[i, target_col + '_mean'] - grouped.loc[i, target_col + '_mean']
        var2 = grouped_test.loc[i, target_col + '_mean'] / grouped.loc[i, target_col + '_mean']
        if math.isnan(var1) or math.isnan(var2):
            count += 1
        else:
            count += 1
            sum += var1 * math.log(var2)
    if count < grouped.shape[0]:
        print('The input data have nan value! Please use pd.fillna() to process')
    return sum


class FeatureExplore(object):

    def __init__(self, tree_split=False):
        self.tree_split = tree_split

    def feature_trend_stats(self, data, target_col, features_list=0, bins=10, data_test=0):

        if type(features_list) == int:
            features_list = list(data.columns)
            features_list.remove(target_col)

        stats_all = []
        has_test = type(data_test) == pd.core.frame.DataFrame
        ignored = []
        for feature in features_list:
            if data[feature].dtype == 'O' or feature == target_col:
                ignored.append(feature)
            elif self.tree_split:
                cut = tree_split_bins(input_data=data, feature=feature, target_col=target_col, bins=bins)
                cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col,
                                                 bins=bins,
                                                 cuts=cut, is_train=1)
                trend_changes = get_trend_changes(grouped_data=grouped, feature=feature,
                                                  target_col=target_col)
                if has_test:
                    grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True),
                                                    feature=feature,
                                                    target_col=target_col, bins=bins, cuts=cuts)
                    trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
                    trend_changes_test = get_trend_changes(grouped_data=grouped_test, feature=feature,
                                                           target_col=target_col)
                    PSI_value = PSI_cal(grouped, grouped_test, target_col)
                    stats = [feature, trend_changes, trend_changes_test, trend_corr, PSI_value]
                else:
                    stats = [feature, trend_changes]
                stats_all.append(stats)
            else:

                cuts, grouped = get_grouped_data(input_data=data, feature=feature, target_col=target_col,
                                                 bins=bins)
                trend_changes = get_trend_changes(grouped_data=grouped, feature=feature,
                                                  target_col=target_col)
                if has_test:
                    grouped_test = get_grouped_data(input_data=data_test.reset_index(drop=True),
                                                    feature=feature,
                                                    target_col=target_col, bins=bins, cuts=cuts)
                    trend_corr = get_trend_correlation(grouped, grouped_test, feature, target_col)
                    trend_changes_test = get_trend_changes(grouped_data=grouped_test, feature=feature,
                                                           target_col=target_col)
                    PSI_value = PSI_cal(grouped, grouped_test, target_col)
                    stats = [feature, trend_changes, trend_changes_test, trend_corr, PSI_value]
                else:
                    stats = [feature, trend_changes]
                stats_all.append(stats)
        stats_all_df = pd.DataFrame(stats_all)
        stats_all_df.columns = ['Feature', 'Trend_changes'] if has_test == False else ['Feature', 'Trend_changes',
                                                                                       'Trend_changes_test',
                                                                                       'Trend_correlation', 'PSI_value']
        if len(ignored) > 0:
            print('Categorical features ' + str(ignored) + ' ignored. Categorical features not supported yet.')

        print('Returning stats for all numeric features')
        return stats_all_df

    def feature_exp_plots(self, data, target_col, features_list=0, bins=10, data_test=0):

        if type(features_list) == int:
            features_list = list(data.columns)
            features_list.remove(target_col)

        for cols in features_list:
            if cols != target_col and data[cols].dtype == 'O':
                print(cols + ' is categorical. Categorical features not supported.')
            elif cols != target_col and data[cols].dtype != 'O':
                variate_plotter(feature=cols, data=data, target_col=target_col, bins=bins,
                                data_test=data_test,
                                tree_split=self.tree_split)

    def get_tree_bins(self, data, target_col, features_list=0, bins=10, min_samples_leaf=0.05, min_samples_split=0.1):

        if type(features_list) == int:
            features_list = list(data.columns)
            features_list.remove(target_col)
        ops = {}
        if self.tree_split:
            for cols in features_list:
                if cols != target_col and data[cols].dtype == 'O':
                    print(cols + ' is categorical. Categorical features not supported.')
                elif cols != target_col and data[cols].dtype != 'O':
                    tree_bin = tree_split_bins(input_data=data, feature=cols, target_col=target_col, bins=bins,
                                               get_bins_alone=1, min_samples_leaf=min_samples_leaf,
                                               min_samples_split=min_samples_split)
                    # print('This is the bin of feature %s' % cols)
                    ops[cols] = tree_bin
                    # print(tree_bin)
                    # print('---' * 20)
        return ops
