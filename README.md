# FeatureExplorePlus

feature_explore_plus for supervised learning feature explore

### FeatureExplorePlus 简介

基于业内（尤其是金融界业内）经常使用的 BIVAR 变量分析方法衍生而来。既将变量分BIN，分析变量标签在 BIN 之间的走势和基于标签的原始分布等。

该方法的常用于下面场景：

1. 对于传统的 LR 等线性模型由于模型对非线性变量的拟合能力较差，可以通过上述方法优化变量分BIN。
2. 筛选变量，一般较好的变量走势 是线性和 U型 的，这种变量的区分能力较强。通过传入测试集的数据，对比变量的变化趋势是否一致。不一致的变量不适合进入模型。
3. 分析变量走势，根据走势，制定基于该变量的产品策略。例如：绝大部分用户对一个 ITEM 隔 5-8 天后 更倾向于再次点击，那么可以基于此作变量对满足条件的 ITEM 提权。
4. 同上2的情况用于变量的监控，内部通过numpy 的 corrcoef 来计算训练数据和线上数据的走势趋势来作为变量监控。如果变量的走势变化过大，说明变量失效了。添加了变量的 PSI 指标。
5. 可以判断变量是否有特征穿越或者标签泄露的风险。一般变量的单个BIN的 正负样本占比不会超过 0.1，如果超过，说明该变量有可能有时间穿越的问题，要排查逻辑。
6. 原始变量和标签的分布关系分析。

序：
2017年，本人实现了该工具包的基础功能（可以参见作者其他的git项目有当时实现的代码），但是一直有个瑕疵没有解决，直到看到了 Abhay-Spawar 的工具包，并试用来一段时间。但是业内已经非常流行的最优分 BIN 法，Abhay-Spawar 一直不支持，自己做了BI-VAR的变量分析方法，加入基于CART TREE 分 BIN 的方法，加入原始变量基于标签的分布，并集成了基于PANDAS的基本分析方法。
其中该工具包里面基于决策树的最优分BIN 法，可以单独调用，将 BIN 作为输出 用于 TensorFlow 的变量预处理，其中每个叶子节点的样本占比等都可以修改。集成的 Pandas 工具可以减少数据的内存占用，并直接给出缺失率和单一值情况分析。


### 如何使用

#### 1. 安装
```
pip install featureExplorePlus
```

需要注意 Python版本为3.* 以上，依赖 sklearn、numpy、pandas、seaborn、 matplotlib。不要安装0.0.1 的该工具包，有一个文档BUG 会导致安装失败。

#### 2. 引入
```
import featureExplorePlus as fep

pt=fep.PandasTools(your_data_path) # 这里可以传入要读入数据的路径，和分隔符等，默认是逗号分隔。
fp=fep.FeatureExplore() # 这里可以参入参数是否使用决策树分BIN，默认使用等频分BIN。

```
#### 3. 详细的例子

详细的使用案例参见项目中的 [examples](https://github.com/XiaolinZHONG/FeatureExplorePlus/blob/master/examples/Features_Explore_Analysis_Examples.ipynb) 可能需要多次reload 才能打开
