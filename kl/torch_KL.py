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
        #ここで獲得可能な分布の形状は(センサー種，時系列情報，x座標，y座標)になっている。
        #つまり(6,対象時刻分の長さ,100,100)となる
        # in_1_dist=insert_dist[:,0,:,:]
        # in_2_dist=insert_dist[:,1,:,:]
        # in_3_dist=insert_dist[:,2,:,:]
        # in_4_dist=insert_dist[:,3,:,:]
        # in_5_dist=insert_dist[:,4,:,:]
        # in_6_dist=insert_dist[:,5,:,:]
        
        # av_1_dist=av_dist[0,:,:]
        # av_2_dist=av_dist[1,:,:]
        # av_3_dist=av_dist[2,:,:]
        # av_4_dist=av_dist[3,:,:]
        # av_5_dist=av_dist[4,:,:]
        # av_6_dist=av_dist[5,:,:]
        
        KL_list=[]
        ipdb.set_trace()
        print(int(len(av_dist[0,:,:])/10))
        for i in range(int(len(av_dist[0,:,:])/10)):
            for l in range(int(len(av_dist[0,:,:])/10)):
                KL_value=self.torch_KL(insert_dist[:,0,i*10:i*10+10,l*10:l*10+10],av_dist[:2,i*10:i*10+10,l*10:l*10+10])
                print("{}to{} dist shape is{}".format(i*10,i*10+10,insert_dist[:,0,i*10:i*10+10,l*10:l*10+10].shape))
                KL_list.append(KL_value)
        return KL_list
        
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
    
    av_dist=torch.Tensor(np.stack([Z_b,Z_b,Z_b,Z_b,Z_b,Z_b]))
    dist1=torch.Tensor((np.stack([np.stack([Z_b,Z_b]),np.stack([Z_b,Z_b])])+0.11)*4)
    dist2=torch.Tensor((np.stack([np.stack([Z_b,Z_b]),np.stack([Z_b,Z_b])])+0.15)*1.4)
    
    KL_value=K.torch_KL(dist1,dist2)
    
    KL_list=K.sprit_calcu(dist1,av_dist)
    
    ipdb.set_trace()
    print("{} is KL value".format(KL_value))

