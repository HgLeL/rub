import numpy as np
"""以下输出仅有一个键key"""
import re
dict = {}
for i in range(31):
    file = open('/home/iqx/文档/省市/%d.txt' % (i+1), encoding='utf-8', errors='ignore').read()
    p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
    result = re.findall(p2,file)
    dele = list(set(result))
    for k in range(len(dele)):
        file = file.split(dele[k])
        file = "$".join(file)
    file_clear = file.split('$')

    # 删除可能存在的空字符
    while '' in file_clear:
        file_clear.remove('')

    # 为确保匹配的准确性将xx县、xx市、xx区等后缀剔除
    end_str = '省市县区'
    for j in range(len(file_clear)):
        leng = list(file_clear[j])
        if leng[-1] in end_str and len(leng) > 2:
            file_clear[j] = ''.join(leng[:-1])

    # 以第一个省份名称为键值key,剩余的市县为value
    province_name = file_clear[0]
    file_clear.remove(province_name)
    dict[province_name] = file_clear

import pandas as pd
import itertools
data = pd.DataFrame(pd.read_csv('/home/iqx/文档/名单.csv', header=None))
chain_count = data.shape[0]       # 链属个数
zero_df = np.repeat(np.nan,chain_count).reshape((chain_count,1))
data = pd.DataFrame(np.hstack((np.array(data), zero_df, zero_df, zero_df, zero_df)))
province_List = list(dict.keys())
for i in range(chain_count):
    m = 1
    for j in range(len(province_List)):
        p_name = province_List[j]
        c_name = dict[p_name]
        if p_name in data.iloc[i,0]:
            for k in range(len(c_name)):
                if c_name[k] in data.iloc[i,0]:
                    data.iloc[i,m] = p_name + '-' + c_name[k]
                    m += 1
                else:
                    data.iloc[i,m] = p_name
                if m > 4:
                    break
        else:
            for k in range(len(c_name)):
                if c_name[k] in data.iloc[i,0]:
                    data.iloc[i,m] = p_name + '-' + c_name[k]
                    m += 1
                else:
                    pass
                if m > 4:
                    break
        if m > 4:
            break

data.to_csv('/home/iqx/文档/data.csv', index=None, encoding='gbk')

"""以下输出有两个键key"""
def produce_dict(file):
    """file是个包含省市名称的字符串"""
    dict = {}
    for i in range(31):
        p2 = re.compile(u'[^\u4e00-\u9fa5]')  # 中文的编码范围是：\u4e00到\u9fa5
        result = re.findall(p2,file)
        dele = list(set(result))
        for k in range(len(dele)):
            file = file.split(dele[k])
            file = "$".join(file)
        file_clear = file.split('$')

        # 删除可能存在的空字符
        while '' in file_clear:
            file_clear.remove('')

        # 为确保匹配的准确性将xx县、xx市、xx区等后缀剔除
        end_str = '省市县区'
        for j in range(len(file_clear)):
            leng = list(file_clear[j])
            if leng[-1] in end_str and len(leng) > 2:
                file_clear[j] = ''.join(leng[:-1])

        # 以第一个省份名称为键值key,剩余的市县为value
        province_name = file_clear[0]
        file_clear.remove(province_name)
        dict[province_name] = file_clear
    return dict

dict_name = {}
for i in range(31):                          # 31个省份，不包含台港澳
    file2 = open('/home/iqx/文档/省市/%d.txt' % (i+1), encoding='utf-8', errors='ignore')
    line_list = file2.readlines()
    if len(line_list) == 1:                  # 此条件筛选针对于直辖市，在本地保存的txt文件中，直辖市只有一行数据
        dict_temp = produce_dict(line_list[0])
        dict_name.update(dict_temp)
    else:                                    # 非直辖市
        list_temp = []
        # 一个list_temp对应一个省的下属市
        # list_temp内的元素为dict，如陕西省为例，list_temp内的每个dict是以陕西省某个地级市为key,该地级市下的县级市及管辖的区为value
        for i in range(1, len(line_list)):
            list_temp.append(produce_dict(line_list[i]))
        pro_name = re.findall(re.compile(u'[\u4e00-\u9fa5]'), line_list[0])
        # 每个pro_name对应一个省名称
        dict_name["".join(pro_name)] = list(list_temp)

# 将结果导出为json文件
# import json
# js_obj = json.dumps(dict_name)
# file_obj = open('/home/iqx/文档/省市.json', 'w')
# with file_obj:
#     file_obj.write(js_obj)

import pandas as pd
data = pd.DataFrame(pd.read_csv('/home/iqx/文档/名单.csv', header=None))
chain_count = data.shape[0]       # 链属个数
data = pd.concat((data, pd.DataFrame(np.repeat(0,chain_count))), axis=1)
data.columns = ['chain_name', 'place']
# for i in range(data2.shape[0]):
#     for j in range(len(city)):
#         if city[j] in str(data2.iloc[i,0]):
#             data2.iloc[i,1] = 1

"""
筛选逻辑是：首先检测链属名称内是否存在某个省或直辖市，
                如果否，那遍历地级市或直辖市的区，检测是否存在，
                        如果否，则遍历县级市，检测是否存在，
                                如果是，给出此链属所在的省-地级市-县级市，END
                                如果否，给出unknown，END
                        如果是，则遍历此地级市的县级市，判断是否有其信息，
                                如果是，同上，END
                                如果否，给出省-地级市，END
                如果是，步骤类似于上。
"""
Municipality = ['北京', '上海', '重庆', '天津']
pro_list = list(dict_name.keys())    # 存放省名称-province_list
cityName = []
for x in pro_list:
    for j in range(len(dict_name[x])):
        y = dict_name[x][j]
        if type(y) == str:
            cityName.append(y)
        else:
            cityName.append(list(y.keys()))


# for k in range(chain_count):
#     for i in range(len(pro_list)):
#         pro_name = pro_list[i]
#         if pro_name in data.iloc[k, 0] and pro_name in Municipality:
#             for j in range(len(dict_name[pro_name])):
#                 district_name = dict_name[pro_name][j]
#                 if district_name in data.iloc[k, 0]:
#                     data.iloc[k,1] = pro_name + '-' + district_name
#                 else:
#                     data.iloc[k,1] = pro_name
#         elif pro_name in data.iloc[k,0] and pro_name not in Municipality:
#             for j in range(len(dict_name[pro_name])):
#                 city_name = [list(x.keys())[0] for x in dict_name[pro_name]]
#                 if city_name[j] in data.iloc[k,0]:
#                     county_name = dict_name[pro_name][city_name[j]]
#                     for t in range(len(county_name)):
#                         if county_name[i] in data.iloc[k,0]:
#                             data.iloc[k,1] = pro_name +'-'+ city_name[j] + county_name[i]
#                         else:
#                             data.iloc[k, 1] = pro_name + '-' + city_name[j]
#                 else:
#                     data.iloc[k,1] = pro_name
#         elif

