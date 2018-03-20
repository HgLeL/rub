from urllib.request import urlopen
from bs4 import BeautifulSoup  # B和S要大写
url = "http://www.pythonscraping.com/pages/page1.html"
try:
    html = urlopen(url)
except urllib.error.HTTPError as e:
    print(e)     # 当网页在服务器上不存在（或者获取页面时出现错误），会出现HTTPError
else:
    bs_obj = BeautifulSoup(html)  # 不是html.read()
    print(bs_obj.title)

# 完整版
"""urlopen()会出现两种异常，一种是第8行所述，另一种是服务器不存在，即链接打不开或者是URL链接写错了，此时会
返回一个None对象。
另外，如果想调用的标签不存在，BeautifulSoup会返回None对象，如果再调用这个None对象下面的子标签，会发生
AttributeError错误。
因此，需要对上述可能发生的错误进行检查。
"""
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
from bs4 import BeautifulSoup

def getTitle(url):
    try:
        html = urlopen(url)
    except (HTTPError,URLError) as e:
        return None
    try:
        bs_obj = BeautifulSoup(html)
        title = bs_obj.body.h1
    except AttributeError as e:
        return None
    return title
url = "http://www.pythonscraping.com/pages/page1.html"
title = getTitle(url)
if title == None:
    print('title could not be found!')
else:
    print(title)


html = urlopen("http://www.pythonscraping.com/pages/warandpeace.html")
bs_obj = BeautifulSoup(html)
namelist = bs_obj.findAll("span",{"class":"green"})  # 提取只包含在<span class="green">...</span>标签里的文字
    # namelist 是一个列表： [..., <span class="green">The prince</span>, <span class="green">AnnaPavlovna</span>,...]
for name in namelist:          # 调用bs_obj.findAll(tagName, tagAtrributes)可以获取页面中所有指定的标签，bs_obj.find()是只提取第一个标签
    print(name.get_text())     # tagAtrributes 是一个字典,tagName可以是一个集合包含很多标签{...}
                            #  .get_text() 会把你正在处理的HTML文档中所有的标签都清除，然后返回一个只包含文字的字符串


html = urlopen("http://www.pythonscraping.com/pages/page3.html")
bs_obj = BeautifulSoup(html)
# 处理子标签和其他后代标签
for child in bs_obj.find("table",{"id":"giftList"}).children:  # 只找出子标签，如果找出后代标签用.descendants()
    print(child)
# 处理兄弟标签
for sibling in bs_obj.find("table",{"id":"giftList"}).tr.next_siblings:  # tr标签的下一个兄弟标签，不包括它自己
    print(sibling)     # 还有其他类似的next_sibling; previous_siblings; previous_sibling
# 处理父标签，parent; parents


# 2.4正则表达式, BeautifulSoup和正则表达式总是配合使用的，大多数支持字符串参数的函数都可以用正则表达式实现，正则表达式可以作为BeautifulSoup语句的任意一个参数
import re
for img_url in bs_obj.findAll("img",{"src":re.compile("\.\.\/img\/gifts\/img\d\.jpg")}):
    print(img_url["src"])
# 2.5获取属性： 对于一个标签对象，可用myTag.attrs获取它的全部属性，其返回的是一个python字典对象，可以获取和操作这些对象，如myImgTag.attrs["src"]


# 3 开始采集
from urllib.request import urlopen
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import re,datetime,random

pages = set()
random.seed(datetime.datetime.now())

# 获取页面所有内链的列表
def getInternalLinks(bsObj, includeUrl):
    includeUrl = urlparse(includeUrl).scheme+"://"+ urlparse(includeUrl).netloc  #关于urlparse的学习链接http://blog.csdn.net/vip_wangsai/article/details/51997882


import time,datetime
time.time()  # 返回当前的总秒数如1521523828.2364452
datetime.datetime.now()  # 返回如datetime.datetime(2018, 3, 20, 13, 31, 36, 743143)

import json
from urllib.request import urlopen
def getCountry(ipaddress):
    response = urlopen("http://freegeoip.net/json/"+ipaddress).read().decode("utf-8")
    responseJson = json.loads(response)
    return responseJson.get("country_code")
print(getCountry("50.78.253.58"))

html = urlopen("https://en.wikipedia.org/w/index.php?title=Wiki&offset=20110722192552&limit=500&action=history")
bs_obj = BeautifulSoup(html)
for ip in bs_obj.findAll(text=re.compile("^(\d{1,3}\.){3}\d{1,3}$")):
    print(ip)