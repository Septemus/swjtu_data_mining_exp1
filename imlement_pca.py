import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.font_manager import FontProperties
threshold = 0.99


def sklearn_pca(data, threshold: float):
    principal = PCA(n_components=threshold)  # 主成分个数系数
    principal.fit(data)
    x = principal.transform(data)
    return [x, principal.components_]


def manual_pca(data, threshold: float):
    # 计算协方差矩阵
    co_var = np.cov(data, rowvar=False)
    # 求特征根和特征向量
    eigen_values, eigen_vectors = np.linalg.eig(co_var)
    # 根据主成分个数系数对特征向量进行筛选
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sum = sorted_eigenvalue.sum()
    tmp = 0
    count = 0
    for v in sorted_eigenvalue:
        tmp += v
        count += 1
        if (tmp/sum > threshold):
            break
    sorted_index = sorted_index[:count]
    sorted_eigenvalue = sorted_eigenvalue[:count]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # 用特征向量描述原来的矩阵
    X_reduced = np.dot(sorted_eigenvectors.transpose(),
                       data.transpose()).transpose()

    return [X_reduced, sorted_eigenvectors.transpose()]


plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签

if __name__=="__main__":
    
    # construct a dataframe using pandas
    table = pd.read_excel('主成分分析.xlsx', sheet_name='5-3')
    data=table
    data.drop(['Unnamed: 0','厂家编号及指标'],axis=1,inplace=True)
    data = (data - data.mean()) / data.std()  # z-score标准化


    ans1 = sklearn_pca(data, threshold)

    plt.figure()
    for i in range(ans1[1].shape[0]):
        plt.scatter([i for j in ans1[1][i]], ans1[1][i], c="red")

    ans2 = manual_pca(data, threshold)

    for i in range(ans2[1].shape[0]):
        plt.scatter([i for j in ans2[1][i]], ans2[1][i], c="green", alpha=0.5)

    plt.show()
