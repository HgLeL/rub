
import pymysql
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from PIL import Image

db = pymysql.connect(
    host='offlinecentre.cigru3mivzpd.rds.cn-north-1.amazonaws.com.cn',   # 连接你要取出数据库的ip，如果是本机可以不用写
    port = 3306,
    user='t7',     # 你的数据库用户名
    passwd='t7@DCFOffLine#9387%',# 你的数据库密码
    db ='t7',
    charset='utf8',)

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

# SQL 查询语句
sql = "SELECT buyer_name, bill_amount, date(bill_time) AS bill_time FROM m_core_bill_match " \
       "where bill_time IS NOT NULL and bill_amount>0"

try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   chain_name = []
   bill_amount = []
   bill_time = []
   for row in results:
       chain_name.append(hashlib.md5(row[0].encode(encoding='gb2312')).hexdigest())  # 对链属名称进行MD5加密
       bill_amount.append(row[1])
       bill_time.append(row[2])
except:
   print("Error: unable to fecth data")

# 关闭数据库连接
db.close()


data = pd.concat([pd.DataFrame(chain_name), pd.DataFrame(bill_amount),pd.DataFrame(bill_time)], axis=1)
data.columns = ['chain_name', 'bill_amount', 'bill_time']
data['bill_time_2'] = data['bill_time'].astype('str').apply(lambda x: x.split('-')[0] + x.split('-')[1]+ x.split('-')[2])
data['bill_amount'] = data['bill_amount'].astype('float')
dataSorted = data[['chain_name', 'bill_amount', 'bill_time_2']].sort_values(by=['chain_name', 'bill_time_2'],ascending=True)
dataSorted['bill_time_2'] = dataSorted['bill_time_2'].apply(lambda x: datetime.strptime(x, "%Y%m%d"))
dataSorted['bill_time_year_month'] = dataSorted['bill_time_2'].astype('str').apply(lambda x: x.split('-')[0] + x.split('-')[1])
# 整理数据为年月，金额
dataOfMonth = dataSorted.groupby(['chain_name', 'bill_time_year_month'], as_index=None).sum()
dataOfMonth = dataOfMonth.sort_values(by=['chain_name', 'bill_time_year_month'], ascending=True)

# 统计链属的名称
nameOfChain = dataSorted.groupby(by='chain_name').count().index
# 统计日期范围
min_date = dataSorted['bill_time_2'].min()
max_date = dataSorted['bill_time_2'].max()
date_range = pd.DataFrame(pd.date_range(start=min_date,end=max_date,freq='M'))
data_range_month = date_range.astype('str').applymap(lambda x: x.split('-')[0] + x.split('-')[1])
plotData = pd.DataFrame(data_range_month)
plotData.columns = ['date_range']

# 绘制订单金额时序图-纯净版
for i in range(len(nameOfChain)):
    setOfData = dataOfMonth[dataOfMonth['chain_name'] == nameOfChain[i]]
    plotData = pd.merge(plotData[['date_range']], setOfData[['bill_time_year_month', 'bill_amount']],left_on='date_range',
                        right_on='bill_time_year_month', how='left')
    plotData = plotData[['date_range', 'bill_amount']].fillna(0)
    x = plotData.date_range  # x是付款日期
    y = plotData.bill_amount  # y是当月订单金额和
    fig = plt.figure()
    plt.xticks([])  # 不显示x轴的刻度
    plt.yticks([])  # 不显示x轴的刻度
    plt.axis('off') # 不显示坐标轴
    fig.subplots_adjust(bottom=0.005, top=1, right=0.999, left=0.0005)
    plt.bar(left=x, height=y,width=1)
    plt.savefig('/home/iqx/文档/项目/订单时序图_建模2/%d.jpg' % i, pad_inches=0)  # 保存图片到文件夹
    plt.clf()
    plt.close()

# 绘制订单金额图
for i in range(len(nameOfChain)):
    setOfData = dataOfMonth[dataOfMonth['chain_name'] == nameOfChain[i]]
    plotData = pd.merge(plotData[['date_range']], setOfData[['bill_time_year_month', 'bill_amount']],left_on='date_range',
                        right_on='bill_time_year_month', how='left')
    plotData = plotData[['date_range', 'bill_amount']].fillna(0)
    x = plotData.date_range  # x是付款日期
    y = plotData.bill_amount  # y是当月订单金额和
    plt.xticks(np.array([0,6,12,18,24,(len(x)-1)]))
    plt.title(r'%s' % nameOfChain[i])
    plt.bar(left=x, height=y)
    plt.savefig('/home/iqx/文档/项目/订单时序图2/%d.png' % i)  # 保存图片到文件夹
    plt.clf()
    plt.close()

# 将链属名称(密文)和对应的图片号码保存为csv文件
df = pd.DataFrame(nameOfChain)
df['image_number'] = pd.DataFrame(np.arange(0,len(nameOfChain),1))
df.to_csv('/home/iqx/文档/项目/chain_name.csv', index=None, encoding='gbk')



