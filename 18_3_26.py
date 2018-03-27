from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import string
from collections import Counter   # Counter可以找出每个元素出现的频率，其方法.most_common(n)可以找出前n个出现次数最多的

def cleanInput(input):
    input = re.sub('\n+',' ',input) # 把换行符或多个换行符替换成空格，re.sub功能是对于一个输入的字符串，利用正则表达式，来实现字符串替换处理的功能返回处理后的字符串
    input = re.sub(' +',' ',input)  # 把连续的多个空格替换成一个空格
    # re.sub("\s+"," ",input) 可以实现上面两个的同样效果
    input = re.sub('\[\d*\]',' ',input)
    input = bytes(input,"UTF-8")   # 把内容转换成utf-8以消除转义符，把str转成bytes类型
    input = input.decode("ascii",errors='ignore')   # str.decode([encoding[, errors]]),使用encoding指示的编码，对str进行解码，返回一个unicode对象。默认情况下encoding是“字符串默认编码”，比如ascii
    cleanInput = []
    input = input.split(' ')        # 英文字符的ASCII、UTF-8、GBK等编码都是一样的; str是一个字节数组,这个字节数组表示的是对unicode对象编码(可以是utf-8、gbk、cp936、GB2312)后的存储的格式
    for item in input:
        item = item.strip(string.punctuation)    # string.punctuation获得python中的所有标点符号,去除字母两头的符号，不包含位于字母内的符号
        if len(item) > 1 or (item.lower() in ('a','i')):
            cleanInput.append(item)
    return cleanInput

# bytes([source[, encoding[, errors]]])    当source参数为字符串时，encoding参数也必须提供，函数将字符串使用str.encode方法转换成字节数组
def ngrams(input, n):
    input = cleanInput(input)
    result = []                      # unicode才是真正意义上的字符串，对字节串str使用正确的字符编码进行解码后获得
    for i in range(len(input)-n+1):
        result.append(input[i:i+n])
    return result

html = urlopen("https://en.wikipedia.org/wiki/Python_(programming_language)")
bs_obj = BeautifulSoup(html)
content = bs_obj.find("div",id="mw-content-text").get_text()
ngram = ngrams(content,2)
ngram = Counter(sorted(dict(ngram).items(),key = lambda t:t[1],reverse=True))  # list.sort()仅仅作用于列表，sorted()可作用于任何迭代器
print(ngram)                                             # key = lambda t:t[1] 表示按['fda','ere']第二个进行排序

"""穿越网页表单与登陆窗口进行采集"""
# 表单网页是一个网站和访问者开展互动的窗口，表单可以用来在网页中发送数据，特别是经常被用在联系表单-用户输入信息然后发送到Email中
# 实际用在HTML中的标签有form、 input、 textarea、 select和option。表单标签form定义的表单里头，必须有行为属性action，它告诉表单当提交的时候将内容发往何处。
# 可选的方法属性method告诉表单数据将怎样发送，有get(默认的)和post两个值。常用到的是设置post值，它可以隐藏信息(get的信息会暴露在URL中)
import requests
params = {'mobile':'17321207569','mobilepassWd':'123456','Verify':'111'}
r = requests.post('http://reg.cctv.com/register/mobileRegSucceed.action',data=params)
print(r.text)

# 处理登陆和cookie
import requests
params = {'username':'fds',"password":'password'}
r = requests.post('http://pythonscraping.com/pages/cookies/welcome.php',params)  # 向欢迎页面发送一个登陆参数，它的作用就像登陆表单的处理器
print('Cookie is set to:')
print(r.cookies.get_dict())    # 从请求结果中获取cookie,打印登陆状态的验证结果
print("-"*10)
print("Going to profile page...")
r = requests.get('http://pythonscraping.com/pages/cookies/profile.php',cookies=r.cookies) # 再通过cookies参数把cookie发送到简介页面
print(r.text)

# 当面对的网站比较复杂，它经常暗自调整cookie，或者如果它从一开始就完全不想要cookie，处理如下
session = requests.Session()
params = {'username':'fds',"password":'password'}
s = session.post('http://pythonscraping.com/pages/cookies/welcome.php',params)
print('Cookie is set to:')
print(s.cookies.get_dict())    # 从请求结果中获取cookie,打印登陆状态的验证结果
print("-"*10)
print("Going to profile page...")
s = session.get('http://pythonscraping.com/pages/cookies/profile.php')
print(s.text)

# HTTP基本接入认证
import requests
from requests.auth import AuthBase
from requests.auth import HTTPBasicAuth
auth = HTTPBasicAuth('hll','password')
r = requests.post("http://pythonscraping.com/pages/auth/login.php",auth = auth)
print(r.text)

# 用selenium执行JavaScript
from selenium import webdriver
import time
driver = webdriver.PhantomJS(executable_path=r'E:\phantomjs-2.1.1-windows\bin\phantomjs')
driver.get("http://pythonscraping.com/pages/javascript/ajaxDemo.html")
time.sleep(3)
print(driver.find_element_by_id('content').text)
driver.close()
# 下面的程序用id是loadedButton的按钮检查页面是不是已经完全加载
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
driver = webdriver.PhantomJS(executable_path=r'E:\phantomjs-2.1.1-windows\bin\phantomjs')
driver.get("http://pythonscraping.com/pages/javascript/ajaxDemo.html")
try:
    element = WebDriverWait(driver,timeout=10).until(EC.presence_of_element_located(locator=(By.ID,"loadedButton"))) # 隐式等待
finally:
    print(driver.find_element_by_id("content").text)
    driver.close()

# 处理重定向   检测客户端重定向是否完成
from selenium import webdriver
import time
from selenium.webdriver.remote.webelement import WebElement
from selenium.common.exceptions import StaleElementReferenceException

def waitForLoad(driver):
    elem = driver.find_element_by_tag_name('html')
    count = 0
    while True:
        count +=1
        if count> 20:
            print('Timing out after 10 seconds and returning')
            return
        time.sleep(.5)
        try:
            elem== driver.find_element_by_tag_name('html')
        except StaleElementReferenceException:  # 重复调用‘html'，直到抛出异常，说明元素不再页面的DOM里，说明网站已经跳转
            return
driver = webdriver.Chrome(executable_path=r'D:\chromedriver')
driver.get("http://pythonscraping.com/pages/javascript/redirectDemo1.html")
waitForLoad(driver)
print(driver.page_source)