from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import xarray as xr

def data_get(filename):
    with open(filename, 'rb') as f:
        df = pd.read_csv(f, encoding='gbk')
    items = ["close", "high", "low", "open", "trade_status"]
    df = df.set_index(["symbol", "trade_date"])
    df = df.to_xarray().to_array().T
    y = df.sel(variable='close')
    trade_status = df.sel(variable='trade_status')
    df = df.drop(['close'], dim='variable')
    df = df.drop(['trade_status'], dim='variable')

    return df, y, trade_status
    pass

# 获取数据

factor_data, hq_data, trade_status = data_get(filename='.\df.csv')  # 因子原始数据，用于计算衍生因子,行情用于计算收益率
# 数据分割
X_train, X_test, y_train, y_test, trade_status_train, trade_status_test = train_test_split(factor_data, hq_data, trade_status, test_size=0.2)


from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn_xarray import wrap
class MATransformer(TransformerMixin, BaseEstimator):
    def __init__(self, n_sma):
        super(MATransformer, self).__init__()
        self.n_sma = n_sma

    def fit(self, X, y):
        return self

    def transform(self, X):
        return pd.DataFrame(X).rolling(self.n_sma).mean().values
class Strategy(BaseEstimator):
    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def score(self, X, y):
        pass
# 新建无量纲化对象
wrap(StandardScaler())
step1 = ('MinMaxScaler', StandardScaler())
# 自定义因子生成方式
step2 = ('MATransformer', MATransformer(10))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step3 = ('Strategy', Strategy)
#新建流水线处理对象
#参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
pipeline = Pipeline(steps=[step1, step2])
for dd in X_train['variable']:
    pipeline.fit(X_train.sel(variable=dd), y_train)
y_pred = pipeline.predict(X_test)
print(y_pred)


from sklearn.pipeline import FeatureUnion, _fit_transform_one, _transform_one
from sklearn.externals.joblib import Parallel, delayed
from scipy import sparse
import numpy as np

#部分并行处理，继承FeatureUnion
class FeatureUnionExt(FeatureUnion):
    #相比FeatureUnion，多了idx_list参数，其表示每个并行工作需要读取的特征矩阵的列
    def __init__(self, transformer_list, idx_list, n_jobs=1, transformer_weights=None):
        self.idx_list = idx_list
        FeatureUnion.__init__(self, transformer_list=map(lambda trans:(trans[0], trans[1]), transformer_list), n_jobs=n_jobs, transformer_weights=transformer_weights)

    #由于只部分读取特征矩阵，方法fit需要重构
    def fit(self, X, y=None):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        transformers = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit方法
            delayed(_fit_one_transformer)(trans, X[:,idx], y)
            for name, trans, idx in transformer_idx_list)
        self._update_transformer_list(transformers)
        return self

    #由于只部分读取特征矩阵，方法fit_transform需要重构
    def fit_transform(self, X, y=None, **fit_params):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        result = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入fit_transform方法
            delayed(_fit_transform_one)(trans, name, X[:,idx], y,
                                        self.transformer_weights, **fit_params)
            for name, trans, idx in transformer_idx_list)

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs

    #由于只部分读取特征矩阵，方法transform需要重构
    def transform(self, X):
        transformer_idx_list = map(lambda trans, idx:(trans[0], trans[1], idx), self.transformer_list, self.idx_list)
        Xs = Parallel(n_jobs=self.n_jobs)(
            #从特征矩阵中提取部分输入transform方法
            delayed(_transform_one)(trans, name, X[:,idx], self.transformer_weights)
            for name, trans, idx in transformer_idx_list)
        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        else:
            Xs = np.hstack(Xs)
        return Xs


step2 = ('FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))


# 处理计算持仓信息

# 计算每个bar的收益率

# 计算组合收益率







