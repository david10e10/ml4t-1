import pandas as pd
a = pd.Series([1, 2, 4, 7, 11, 16])
b = a-a.shift(1)
b=b[1:]
print a
print b.values[-1]*1.0
print b

import pandas as pd
a = pd.Series([1, 2, 4, 7, 11, 16])
c=pd.Series([])
for i in range(len(a)-1):
    c[i] = a[i+1] - a[i]
print a
print c.values[-1]*1.0
print c

import pandas as pd
a = pd.Series([1, 2, 4, 7, 11, 16])
d = a-a.shift(1)
print a
print d.values[-1]*1.0
print d

import pandas as pd
a = pd.Series([1, 2, 4, 7, 11, 16])
e = a-a.shift(1)
e=e[:-1]
print a
print e.values[-1]*1.0
print e