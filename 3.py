import pandas as pd
import pymysql
import matplotlib.pyplot as plt
import re
import numpy as np

db = pymysql.connect(
    host='**',   # 连接你要取出数据库的ip，如果是本机可以不用写
    port = 3306,
    user='**',     # 你的数据库用户名
    passwd='**',# 你的数据库密码
    db ='t7',
    charset='utf8',)
# 使用cursor()方法获取操作游标
cursor = db.cursor()

# SQL 查询语句
sql = "SELECT buyer_name, bill_amount, date_format(bill_time,'%Y-%m') AS bill_time FROM m_core_bill_match " \
       "where bill_time IS NOT NULL and bill_amount>0"   # 订单金额条形图

try:
   # 执行SQL语句
   cursor.execute(sql)
   # 获取所有记录列表
   results = cursor.fetchall()
   chain_name = []
   bill_amount = []
   bill_time = []
   for row in results:
       chain_name.append(row[0])
       bill_amount.append(row[1])
       bill_time.append(row[2])
except:
   print("Error: unable to fecth data")

# 关闭数据库连接
db.close()

data = pd.concat([pd.DataFrame(chain_name), pd.DataFrame(bill_amount),pd.DataFrame(bill_time)], axis=1)
data.columns = ['chain_name', 'bill_amount', 'bill_time']

#去除掉链属名称后面的字母d和p
pattern = re.compile(r'(.*?)\w$')
for i in range(len(data.chain_name)):
    data.iat[i,0] = re.findall(pattern, data.iat[i,0])[0]

# 将公司名称一致，付款天数在同一天的付款金额合并
data = data.groupby(by=['chain_name','bill_time']).sum()
data = data.reset_index()  # 新数据表

min_date = data.sort_values(by=['bill_time'], ascending=True).bill_time.values[0]   # 最小的付款日期：字符格式
max_date = data.sort_values(by=['bill_time'], ascending=True).bill_time.values[-1]  # 最大的付款日期：字符格式

# 填充数据，月份缺失的补充为0
date = []
year = ['2015','2016','2017']
month = ['01','02','03','04','05','06','07','08','09','10','11','12']
for i in range(len(year)):
    for j in range(len(month)):
        if i == (len(year)-1) and j > int(max_date[5:7])-1:
            break
        else:
            date.append(year[i]+'-'+month[j])

name = data.groupby(by='chain_name').count().iloc[:,0:1].index

data_insert = pd.DataFrame(None) # 建立一个空数据框，保存要填补的数据
for k in range(len(name)):
    for i in range(len(date)):
        if date[i] not in data[data['chain_name']==name[k]].bill_time.values:
            data_insert = pd.concat([data_insert, pd.DataFrame([[name[k], date[i], int(0)]])], ignore_index=True)
data_insert.columns = ['chain_name', 'bill_time','bill_amount']   # 如果两个数据框的列名不一致是无法合并的
data_new = pd.concat([data, data_insert],axis=0)   # 将原数据与要填补的数据合并
data_new = data_new.sort_values(by=['chain_name','bill_time'], ascending=True)  # 将数据进行排序，先按照链属名称，再按照订单月份

# 得到链属名称及其订单条数
unique_chain = data_new.groupby(by='chain_name').count().iloc[:,0:1]

# 产生新列：累计付款金额 bill_amount_cumsum
# data_new['bill_amount_cumsum'] = None
# t = 0
# l = 0
# n = len(date)
# while (t+n) == n * len(unique_chain):
#     if l == 0:
#         data_new.bill_amount_cumsum.values[0:n] = np.cumsum(data_new.bill_amount.values[0:n])
#         l = 1
#     else:
#         t += n
#         data_new.bill_amount_cumsum.values[t:t+n] = np.cumsum(data_new.bill_amount.values[t:t+n])

# 绘制直方图
# for i in range(len(name)):
#     a_chain_data = data_new[data_new['chain_name']==name[i]]
#     x = a_chain_data.bill_time     # x是付款日期
#     y = a_chain_data.bill_amount # y是当月付款金额和
#     fig = plt.figure(figsize=(1, 1))
#     fig.subplots_adjust(bottom=0.01, top=1, right=0.99, left=0.01)  # 只留取方框内的条形图
#     plt.xticks([])  # 不显示x轴的刻度
#     plt.yticks([])  # 不显示x轴的刻度
#     plt.bar(left = x, height=y)
#     plt.savefig(r'C:\Users\admin\Desktop\项目\订单月金额时序图\%d.png' % (i+1), pad_inches=0)  # 保存图片到文件夹
#     plt.clf()
#     plt.close()

for i in range(len(name)):
    a_chain_data = data_new[data_new['chain_name']==name[i]]
    x = a_chain_data.bill_time     # x是付款日期
    y = a_chain_data.bill_amount # y是当月付款金额和
    plt.xticks(np.arange(1,32,5))  # 不显示x轴的刻度
    plt.bar(left = x, height=y)
    plt.savefig(r'C:\Users\admin\Desktop\项目\订单金额时序图_大\%d.png' % (i+1), pad_inches=0)  # 保存图片到文件夹
    plt.clf()
    plt.close()
