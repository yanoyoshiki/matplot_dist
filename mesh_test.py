"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
 
# 定義域[-1, 1]のx, yを50個区切りで生成
x = np.linspace(-10, 10, 100)
y = np.linspace(-10, 10, 100)

x_z = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
y_z = np.exp(-y**2 / 2) / np.sqrt(2 * np.pi)
# メッシュグリッドを生成
xv, yv = np.meshgrid(x, y)
xv_z, yv_z = np.meshgrid(x_z, y_z)

 
 
# 関数x^2 - y^2の値をzに代入
z = xv_z+yv_z
 
# x, y, zをワイヤフレームで表示
ax = Axes3D(plt.figure())
ax.plot_wireframe(xv, yv, z)
plt.show()
"""



from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from re import S
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import ipdb

#関数に投入するデータを作成
x = np.linspace(-100, 100, 100)
y = np.linspace(-100, 100, 100)
X, Y = np.meshgrid(x, y)

z = np.c_[X.ravel(),Y.ravel()]

#二次元正規分布の確率密度を返す関数
def gaussian(x,x_p,y_p):
    mu = np.array([x_p,y_p])
    #分散共分散行列の行列式
    det = np.linalg.det(sigma)
    print(det)
    #分散共分散行列の逆行列
    inv = np.linalg.inv(sigma)
    n = x.ndim
    print(inv)
    return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))

def insert(x_get_data_as_mean,y_get_data_as_mean):
    print("Start insert")
    
if __name__ == "__main__":
    #2変数の平均値を指定
    x_p1=-50
    y_p1=50

    x_p2=10
    y_p2=20
    #2変数の分散共分散行列を指定
    # sigma=np.cov(x,y)
    sigma = np.array([[100,0],[0,100]])

    # ipdb.set_trace()

    Z1 = gaussian(z,x_p1,y_p1)
    Z2 = gaussian(z,x_p2,y_p2)

    shape = X.shape
    Z1 = Z1.reshape(shape)
    Z2 = Z2.reshape(shape)


    Z=Z1+Z2

    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X, Y, Z)
    # ipdb.set_trace()
    plt.show()