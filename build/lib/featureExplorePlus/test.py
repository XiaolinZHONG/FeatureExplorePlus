import featureExplorePlus as fep
#
data_path = "/Users/xiaolin/Desktop/examples.csv"
pdtt = fep.PandasTools(data_path, sep=",")  # 默认是显示详细内容
df = pdtt.read_data()  # 这里可以设置只读取部分 nrows=1000

data_path2 = "/Users/xiaolin/Desktop/examples_test.csv"
pdtt2 = fep.PandasTools(data_path2, sep=",")  # 默认是显示详细内容
df2 = pdtt2.read_data()
#
fet = fep.FeatureExplore(tree_split=True)  # 默认不使用决策树分BIN
fet.feature_exp_plots(data=df,data_test=df,target_col='label',features_list=['col2','col1'])
# df = fet.feature_trend_stats(data=df.fillna(-1), data_test=df2.fillna(-1), target_col='label', features_list=['col2','col1'])
# print(df.head())
# # s=fet.get_tree_bins(df, target_col='label', features_list=['col2', 'col1'])
#
# # print(s)
