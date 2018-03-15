import requests
import json
from numpy import *

url="https://movie.douban.com/j/search_subjects?type=movie&tag=%E7%83%AD%E9%97%A8&sort=rank&page_limit=50&page_start=0"
r = requests.get(url,verify=False)
content = r.content

result = json.loads(content)
tvs= result['subjects']

title,rat,movie_url=[],[],[]
for i in range(len(tvs)):
    rat.append(str(tvs[i]['rate']))
    title.append(tvs[i]['title'])
    movie_url.append(tvs[i]['url'])
info_mat=[title,rat,movie_url]
# print(info_mat)
# print(info_mat[:,0])

for i in range(len(rat)):
    rat[i]=float(rat[i])  #这里转换时，不能用int，应该用float
