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
        # print(det)
        #分散共分散行列の逆行列
        inv = np.linalg.inv(sigma)
        n = x.ndim
        # print(inv)
        
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
    
    def makeing_dataset(self,data1,data2,data3):
        data=[]
        data.append(data1)
        data.append(data2)
        data.append(data3)
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
        
        Z_1 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[0]
        Z_2 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[1]
        Z_3 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[2]
        Z_4 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[3]
        Z_5 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[4]
        Z_6 = self.gaussian(z,x_get_data_as_mean,y_get_data_as_mean)*multi_dim_data[5]
        
        Z_1 = Z_1.reshape(shape)
        Z_2 = Z_2.reshape(shape)
        Z_3 = Z_3.reshape(shape)
        Z_4 = Z_4.reshape(shape)
        Z_5 = Z_5.reshape(shape)
        Z_6 = Z_6.reshape(shape)
        
        # ipdb.set_trace()
        return X,Y,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6
    
    def addition_distribute(self,Z_b1,Z_b2,Z_b3,Z_b4,Z_b5,Z_b6,Z_1,Z_2,Z_3,Z_4,Z_5,Z_6):
        Z_1=Z_b1+Z_1
        Z_2=Z_b2+Z_2
        Z_3=Z_b3+Z_3
        Z_4=Z_b4+Z_4
        Z_5=Z_b5+Z_5
        Z_6=Z_b6+Z_6
        
        return Z_1,Z_2,Z_3,Z_4,Z_5,Z_6
    
if __name__ == "__main__":
    A=Data_dist()
    
    X_b,Y_b,Z_b=A.baseline()
    
    #generating sample data-----------
    x_p1=-50
    y_p1=50
    value_p1=[1,2,3,4,5,6]
    r1=[x_p1,y_p1]
    r1.extend(value_p1)
    # ipdb.set_trace()
    
    x_p2=10
    y_p2=20
    value_p2=[2,5,6,3,4,2]
    r2=[x_p2,y_p2]
    r2.extend(value_p2)
    
    x_p3=-10
    y_p3=-50
    value_p3=[6,5,2,3,4,2]
    r3=[x_p3,y_p3]
    r3.extend(value_p3)
    
    data_n=A.makeing_dataset(r1,r2,r3)
    #--------------------------------
    
    #start insert code
    #treating multidata is data_n
    X_m,Y_m,Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6=A.multi_insert(data_n[0][0],data_n[0][1],data_n[0][2:])
    X_m,Y_m,Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6=A.multi_insert(data_n[1][0],data_n[1][1],data_n[1][2:])
    X_m,Y_m,Z_3_1,Z_3_2,Z_3_3,Z_3_4,Z_3_5,Z_3_6=A.multi_insert(data_n[2][0],data_n[2][1],data_n[2][2:])
    
    Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6=A.addition_distribute(Z_b,Z_b,Z_b,Z_b,Z_b,Z_b,Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6)
    Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6=A.addition_distribute(Z_1_1,Z_1_2,Z_1_3,Z_1_4,Z_1_5,Z_1_6,Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6)
    Z_3_1,Z_3_2,Z_3_3,Z_2_4,Z_2_5,Z_2_6=A.addition_distribute(Z_2_1,Z_2_2,Z_2_3,Z_2_4,Z_2_5,Z_2_6,Z_3_1,Z_3_2,Z_3_3,Z_3_4,Z_3_5,Z_3_6)
    Zs1,Zs2,Zs3,Zs4,Zs5,Zs6=Z_3_1,Z_3_2,Z_3_3,Z_2_4,Z_2_5,Z_2_6
    
    #making graph
    ax = Axes3D(plt.figure())
    ax.plot_wireframe(X_b, Y_b, Zs3)
    # ipdb.set_trace()
    plt.show()