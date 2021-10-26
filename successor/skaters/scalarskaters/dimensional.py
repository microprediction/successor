import numpy as np


def squeeze_out_middle(x):
    sh = np.shape(x)
    if len(sh)==3:
        return np.squeeze(x,axis=1)
    else:
        return x


if __name__=='__main__':
    x1 = np.random.randn(300,1,20)
    y1 = squeeze_out_middle(x1)
    print(np.shape(y1))
    x2 = np.random.randn(300,20)
    y2 = squeeze_out_middle(x2)
    print(np.shape(y2))

