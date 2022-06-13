from pyexpat import XML_PARAM_ENTITY_PARSING_ALWAYS
from random import gauss
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
        #二次元正規分布の確率密度を返す関数
    def gaussian(self,x,x_p,y_p):
        #2変数の分散共分散行列を指定
        # sigma=np.cov(x,y)
        sigma = np.array([[100,0],[0,100]])

        mu = np.array([x_p,y_p])
        # mu = np.array([1,1])
        #分散共分散行列の行列式
        det = np.linalg.det(sigma)
        print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        print(inv)
        
        return np.exp(-np.diag((x - mu)@inv@(x - mu).T)/2.0) / (np.sqrt((2 * np.pi) ** n * det))
    
    def baseline(self):
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape
        z = np.c_[X.ravel(),Y.ravel()]
        x=y=0
        #ここでベースを強制的に0にしている
        Z=self.gaussian(z,x,y)*0
        Z=Z.reshape(shape)
        return X,Y,Z
    
    def makeing_dataset(self,data1,data2):
        data=[]
        data.append(data1)
        data.append(data2)
        return data
    
    def insert(self,x_get_data_as_mean,y_get_data_as_mean):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        Z1_1 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)
        Z = Z1_1.reshape(shape)

        return X,Y,Z
    def multi_insert(self,x_get_data_as_mean,y_get_data_as_mean,multi_dim_data):
        #関数に投入するデータを作成
        x = np.linspace(-100, 100, 100)
        y = np.linspace(-100, 100, 100)
        X, Y = np.meshgrid(x, y)
        shape = X.shape

        z = np.c_[X.ravel(),Y.ravel()]
        print("Start insert")
        # print("value_1 is {0} ,value_2 is {1} , value_3 is {2}".format(multi_dim_data[0],multi_dim_data[1],multi_dim_data[2]))
        # ipdb.set_trace()
        
        z_1=z*multi_dim_data[0]
        z_2=z*multi_dim_data[1]
        z_3=z*multi_dim_data[2]
        
        Z_1 = self.gaussian(z_1,x_get_data_as_mean,y_get_data_as_mean)
        Z_2 = self.gaussian(z_2,x_get_data_as_mean,y_get_data_as_mean)
        Z_3 = self.gaussian(z_3,x_get_data_as_mean,y_get_data_as_mean)
        
        Z_1 = Z_1.reshape(shape)
        Z_2 = Z_2.reshape(shape)
        Z_3 = Z_3.reshape(shape)
        
        return X,Y,Z_1,Z_2,Z_3
    
    def addition_distribute(self,Z_b1,Z_b2,Z_b3,Z_1,Z_2,Z_3):
        Z_1=Z_b1+Z_1
        Z_2=Z_b2+Z_2
        Z_2=Z_b3+Z_3
        
        return Z_1,Z_2,Z_3
    
if __name__ == "__main__":
    A=Data_dist()
    
    X_b,Y_b,Z_b=A.baseline()
    
    #generating sample data
    x_p1=-50
    y_p1=50
    value_p1=[1,2,3]
    r1=[x_p1,y_p1]
    r1.extend(value_p1)
    print("posision and value are {}".format(r1))
    # ipdb.set_trace()
    
    x_p2=10
    y_p2=20
    value_p2=[4,5,6]
    r2=[x_p2,y_p2]
    r2.extend(value_p2)
    
    data_n=A.makeing_dataset(r1,r2)
    # ipdb.set_trace()
    
    #継続的に増え続ける
    #ここを繰り返し処理に使う必要がある
    #とりあえず1台の多次元情報を継続的に使用できるようにしよう
    
    #これで一台分の環境情報取得補完が完成する
    X_m,Y_m,Z_1,Z_2,Z_3=A.multi_insert(r1[0],r1[1],r1[2:])
    Z_1,Z_2,Z_3=A.addition_distribute(Z_b,Z_b,Z_b,Z_1,Z_2,Z_3)
    
    
    #複数データをdata_nとして扱っている
    #r1,r2をdataとして扱うこと
    X_m,Y_m,Z_1_1,Z_1_2,Z_1_3=A.multi_insert(data_n[0][0],data_n[0][1],data_n[0][2:])
    X_m,Y_m,Z_2_1,Z_2_2,Z_2_3=A.multi_insert(data_n[1][0],data_n[1][1],data_n[1][2:])
    Z_1_1,Z_1_2,Z_1_3=A.addition_distribute(Z_b,Z_b,Z_b,Z_1_1,Z_1_2,Z_1_3)
    Z_2_1,Z_2_2,Z_2_3=A.addition_distribute(Z_1_1,Z_1_2,Z_1_3,Z_2_1,Z_2_2,Z_2_3)
    #これがほんまならセンサの次元分繰り返すことになっている
    
    #グラフ作成
    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X_b, Y_b, Z_b)
    ipdb.set_trace()
    plt.show()