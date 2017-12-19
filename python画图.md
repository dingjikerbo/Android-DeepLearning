[官方tutorial](https://matplotlib.org/users/pyplot_tutorial.html)

第一个例子如下：

```
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()
```

横轴坐标[0,3]，纵轴坐标[1,4]，这里plot指定的范围当做了纵轴坐标，横轴长度和纵轴一致，且默认从0开始，因此横轴是[0,3]。

如果要指定横轴范围，可按如下方式给出两个数组，注意数组长度要一致。

```
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
```

再看如下代码，这里通过plot指定了四个点，通过axis指定viewport为横轴[0,6]，纵轴[0,20]。

```
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro')
plt.axis([0, 6, 0, 20])
plt.show()
```

这里ro的r表示红色，o表示红点，其它的参考
https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot

再看如下：

```
import numpy as np
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)

# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()
```

这里在一个图里画了三条曲线，t，t^2，t^3。横轴范围[0,5]，以0.2为间隔描一个点。

再看如下：

```
import numpy as np
import matplotlib.pyplot as plt

def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()
```

此处分上下两个图，上面的图用两种方式描点，一种按t1较稀疏地描蓝色的原点，一种按黑色实线密集地描点。下图按红色虚线密集地描点。