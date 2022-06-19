import torch
import torch.nn.functional as F
import numpy as np
import ipdb


class KL_dist():
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
        
    def torch_KL(self,first_dist,second_dist):
        torch_KL_d_value=F.kl_div(first_dist.log(), second_dist, None, None, 'sum')
        return torch_KL_d_value

    def sprit_calcu(self,insert_dist,av_dist):
        #ここで計算領域を碁盤の目状に分割
        #それぞれのセンサー値取得後の区画分布と平均分布の区画分布のKLを算出
        #区画毎にKLを出力できるようにする
        print("now codeing")
    
    def calcu_point():
        #1つの評価分布を獲得した後
        #その評価区画から優先度の高い区画を3つ座標として出力する
        first_point=1
        second_point=1
        third_point=1
        return first_point,second_point,third_point
    
if __name__ == "__main__":
    P = torch.Tensor([[0.36, 0.48, 0.16],[0.36, 0.48, 0.16]])
    Q = torch.Tensor([[0.333, 0.333, 0.333],[0.333, 0.333, 0.333]])
    
    
    K=KL_dist()
    X_b,Y_b,Z_b=K.baseline()
    
    dist1=torch.Tensor((Z_b+0.11)*2)
    dist2=torch.Tensor((Z_b*0.15)*1.4)
    
    KL_value=K.torch_KL(dist1,dist2)
    
    ipdb.set_trace()
    print("{} is KL value".format(KL_value))

