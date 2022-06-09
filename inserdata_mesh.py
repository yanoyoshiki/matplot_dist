from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from re import S
from readline import insert_text
from cv2 import add
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import ipdb


class Data_dist():
    
    def insert(self,x_get_data_as_mean,y_get_data_as_mean):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        Z = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)
        Z = Z.reshape(shape)

        return X,Y,Z
    
    
    #二次元正規分布の確率密度を返す関数
    def gaussian(self,x,x_p,y_p):
        #2変数の分散共分散行列を指定
        # sigma=np.cov(x,y)
        sigma = np.array([[100,0],[0,100]])

        mu = np.array([x_p,y_p])
        #分散共分散行列の行列式
        det = np.linalg.det(sigma)
        print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        print(inv)
        return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    def addition_distribute(self,Z_f,Z_s):
        Z_con=Z_f+Z_s

        return Z_con

if __name__ == "__main__":
    #2変数の平均値を指定
    x_p1=-50
    y_p1=50
    value_p1=[1,2,3]

    x_p2=10
    y_p2=20
    value_p2=[4,5,6]
    
    A=Data_dist()
    X,Y,Z1=A.insert(x_p1,y_p1,value_p1)
    X,Y,Z2=A.insert(x_p2,y_p2,value_p2)
    Z=A.addition_distribute(Z1,Z2)

    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X, Y, Z)
    # ipdb.set_trace()
    plt.show()