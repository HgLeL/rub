import numpy as np
file = open('/home/iqx/文档/1.txt').read()
import re

p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
result = re.findall(p2,file)
dele = list(set(result))
for i in range(len(dele)):
    file = file.split(dele[i])
    file = "$".join(file)

set_city = set(file.split('$'))-set([''])
city = list(set_city)
city.append('河南')

import pandas as pd
data = pd.DataFrame(pd.read_csv('/home/iqx/文档/名单.csv', header=None))
data2 = pd.concat((data, pd.DataFrame(np.repeat(0,data.shape[0]))), axis=1)
for i in range(data2.shape[0]):
    for j in range(len(city)):
        if city[j] in str(data2.iloc[i,0]):
            data2.iloc[i,1] = 1

