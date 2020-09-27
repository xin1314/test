from numpy import hstack, vstack, array, median, nan
from numpy.random import choice
from sklearn.datasets import load_iris
from numpy import log1p
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV



# 特征矩阵加工
# 使用vstack增加一行含缺失值的样本(nan, nan, nan, nan)
# 使用hstack增加一列表示花的颜色（0-白、1-黄、2-红），花的颜色是随机的，意味着颜色并不影响花的分类
iris = load_iris()
iris.data = hstack((choice([0, 1, 2], size=iris.data.shape[0]+1).reshape(-1,1), vstack((iris.data, array([nan, nan, nan, nan]).reshape(1,-1)))))
# 目标值向量加工
# 增加一个目标值，对应含缺失值的样本，值为众数
iris.target = hstack((iris.target, array([median(iris.target)])))
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)


#新建计算缺失值的对象
step1 = ('Imputer', SimpleImputer())
#新建将部分特征矩阵进行定性特征编码的对象
# step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
#新建将部分特征矩阵进行对数函数转换的对象
step2_2 = ('ToLog', FunctionTransformer(log1p))
#新建将部分特征矩阵进行二值化类的对象
step2_3 = ('ToBinary', Binarizer())
#参数transformer_list为需要并行处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
step2 = ('FeatureUnion', FeatureUnion(transformer_list=[step2_2, step2_3]))
#新建无量纲化对象
step3 = ('MinMaxScaler', MinMaxScaler())
#新建卡方校验选择特征的对象
step4 = ('SelectKBest', SelectKBest(chi2, k=3))
#新建PCA降维的对象
step5 = ('PCA', PCA(n_components=2))
#新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
#新建流水线处理对象
#参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(y_pred)


classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
    ]
for classifier in classifiers:
    pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])
    pipeline.fit(X_train, y_train)
    print(classifier)
    print("model score: %.3f" % pipeline.score(X_test, y_test))

# 自动化调参
# 新建网格搜索对象
# 第一参数为待训练的模型
# param_grid为待调参数组成的网格，字典格式，键为参数名称（格式“对象名称__子对象名称__参数名称”），值为可取的参数值列表
param_grid = {
    'FeatureUnion__ToBinary__threshold': [1.0, 2.0, 3.0, 4.0],
    'LogisticRegression__C': [0.1, 0.2, 0.4, 0.8]}


CV = GridSearchCV(pipeline, param_grid, n_jobs=1)
CV.fit(X_train, y_train)
print(CV.best_params_)
print(CV.best_score_)
