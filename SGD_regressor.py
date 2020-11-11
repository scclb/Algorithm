# python 3.6
# Author: Scc_hy
# Create date: 2020-11-10
# Func: 从线性回归到简单全连接层网络示例


# 一、线性回归
# ========================================
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, SGDRegressor
from tqdm import tqdm


def data_xy(sklearn_regdata=False):
    if sklearn_regdata:
        from sklearn.datasets import make_regression
        x, y = make_regression(n_samples=200, n_features=4, n_informative=3, bias=20)
        return x, y

    x = np.linspace(start=1, stop=100, num=100)
    y = 3 * x + 30 + np.random.normal(size=100)*10
    return x.reshape((-1, 1)), y


def plot_xy(x, y, seaborn_plot=False):
    if seaborn_plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].scatter(x, y, alpha=0.7)
        axes[0].set_title('matplotlib plot x & y')
        sns.regplot(x, y, ax=axes[1])
        axes[1].set_title('seaborn plot x & y')
    else:
        plt.figure(figsize=(8, 8))
        plt.scatter(x, y, alpha=0.7)
        plt.title('matplotlib plot x & y')
    plt.show()





class SGDLinear_reg():
    def __init__(self, max_iter=10, eta = 0.1, batch_size=50, bais=True):
        self.eta = eta
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.bais = bais

    def initial_weight(self, x_shape_tuple : tuple):
        self.w = np.random.normal(size=x_shape_tuple[1])
        if self.bais:
            self.w = np.random.normal(size=x_shape_tuple[1]+1)

        
    def fit(self, x, y):
        self.x = x
        self.y = y
        self.n_samples = x.shape[0]
        self.initial_weight(self.x.shape)
        self.batch_bins = self.n_samples // self.batch_size
        self.batch_extra = self.n_samples % self.batch_size
        self.batch_loop = self.batch_bins+2 if self.batch_extra > 0 else self.batch_bins+1
        self.train()


    def predict(self, x):
        if self.bais:
            return np.dot(np.c_[x, np.ones((x.shape[0],1))], self.w)
        else:
            return np.dot(x, self.w)


    def get_bacth(self, n):
        """
        获取第n批数据
        """
        start = (n - 1) * self.batch_size
        if n > self.batch_bins:
            end = None
        else:
            end = n * self.batch_size
        
        if end is None:
            return self.x[start: ], self.y[start: ]
        return self.x[start:end], self.y[start:end]


    def cost(self, y_true, y_pred):
        error_ = y_true - y_pred
        return np.sum(error_**2) * 0.5


    def apply_derivative(self, batch_x, bacth_y):
        pred_y = self.predict(batch_x)
        if self.bais:
            batch_x = np.c_[batch_x, np.ones((batch_x.shape[0],1))]

        error_ =  pred_y - bacth_y 
        gd_w = np.dot(batch_x.T, error_)
        self.w -= (self.eta * gd_w / self.batch_size)
        return self.cost(bacth_y, pred_y)


    def one_loop_train(self, batch_n):
        batch_x, batch_y = self.get_bacth(batch_n)
        return self.apply_derivative(batch_x, batch_y)


    def train(self):
        self.train_loss = []
        for _ in tqdm(range(self.max_iter) ):
            for batch_n in range(1, self.batch_loop):
                error_f = self.one_loop_train(batch_n)

                self.train_loss.append(error_f)


    # def early_stop_message(self, stop_count, max_iter, msg_type='early_stop'):
    #     stop_dict = {'early_stop': '误差小于阈值', 'not_improve':'误差没有减少' }
    #     # print(f'in early_stop_message:{msg_type} - {stop_count}')
    #     if stop_count >= self.early_stop:
    #         print(f'>> {stop_dict[msg_type]}{self.early_stop}次, 提前停止, 迭代 {max_iter+1} 次')
    #         return True
    #     return False


    def plot_train_loss(self):
        plt.figure(figsize=(16, 8))
        plt.plot(self.train_loss)
        plt.title('Train loss')
        plt.ylabel('loss')
        plt.xlabel('gd times')
        plt.show()



def test_sampledata():
    x, y = data_xy()
    # My reg
    lr = SGDLinear_reg(1000, eta=0.0001, batch_size=10)
    lr.fit(x, y)
    lr.plot_train_loss()

    ## sklearn reg
    lrskt = LinearRegression(fit_intercept=True)
    lrskt.fit(x, y)

    ## sklearn sgd
    sgd_reg = SGDRegressor(max_iter = 1000,   # 迭代次数
                        penalty = None, # 正则项为空
                        eta0 = 0.0001,      # 学习率
                        early_stopping=True
                        )
    sgd_reg.fit(x,y)

    x_s = np.arange(1, 100, 1).reshape((-1, 1))
    plt.plot(x_s, lr.predict(x_s) ,label=f'my linear:y={lr.w[0]:.2f}x+{lr.w[1]:.2f}, loss: {lr.cost(y, lr.predict(x)):.2f}')
    plt.plot(x_s, lrskt.predict(x_s), lw=2, label=f'sklearn: y={lrskt.coef_[0]:.2f}x+{lrskt.intercept_:.2f}, loss: {lr.cost(y, lrskt.predict(x)):.2f}')
    plt.plot(x_s, sgd_reg.predict(x_s), linestyle='--', label=f'sklearn sgd: y={sgd_reg.coef_[0]:.2f}x+{sgd_reg.intercept_[0]:.2f}, loss: {lr.cost(y, sgd_reg.predict(x)):.2f}')
    plt.scatter(x, y, alpha=0.6)
    plt.legend()
    plt.show()


def test_sklreandata():
    x, y = data_xy(sklearn_regdata=True)
    # My reg
    lr = SGDLinear_reg(100, eta=0.01, batch_size=10)
    lr.fit(x, y)
    lr.plot_train_loss()

    ## sklearn sgd
    sgd_reg = SGDRegressor(max_iter = 100,   # 迭代次数
                        penalty = None, # 正则项为空
                        eta0 = 0.01,      # 学习率
                        early_stopping=True
                        )
    sgd_reg.fit(x,y)

    ## sklearn reg
    lrskt = LinearRegression(fit_intercept=True)
    lrskt.fit(x, y)

    print(f'my linear loss: {lr.cost(y, lr.predict(x)):.2f}')
    print(f'sklearn loss: {lr.cost(y, lrskt.predict(x)):.2f}')
    print(f'sklearn sgd loss: {lr.cost(y, sgd_reg.predict(x)):.2f}')



if __name__ == '__main__':
    print('=='*60, '\nStart test sample data with My_SGD Sklearn_SGD Sklearn_reg')
    test_sampledata()
    print('=='*60, '\nStart test sklearn data with My_SGD Sklearn_SGD Sklearn_reg')
    test_sklreandata()


