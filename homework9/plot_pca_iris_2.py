# IPython log file
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.data
pca = PCA()

projected = pca.fit_transform(X)
proj1, proj2 = projected[:, 0], projected[:, 1]