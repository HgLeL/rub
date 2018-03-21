# 5 存储数据
# 5.1 媒体文件
from urllib.request import urlretrieve  # 可以根据文件的URL下载文件
from urllib.request import urlopen
from bs4 import BeautifulSoup
import re

url = "http://www.pythonscraping.com"
html = urlopen(url)
bs_obj = BeautifulSoup(html.read())
imageLocation = bs_obj.find("img",src = re.compile("(.*?)"),alt= "Home")["src"]
urlretrieve(imageLocation, "logo.jpg")

"""
下面把http://www.pythonscraping.com主页上的所有src属性的文件都下载下来
"""
import os
from urllib.request import urlretrieve  # 可以根据文件的URL下载文件
from urllib.request import urlopen
from bs4 import BeautifulSoup

downloadDirectory = "downloaded"
baseUrl = "http://www.pythonscraping.com"

def getAbsoluteURL(baseUrl, source):
    if source.startswith("http://www."):   #startswith() 方法用于检查字符串是否是以指定子字符串开头，如果是则返回 True，
        url = "http://"+source[11:]      #否则返回 False。如果参数 beg 和 end 指定值，则在指定范围内检查。str.startswith(str, beg=0,end=len(string))
    elif source.startswith("http://"):
        url = source
    elif source.startswith("www."):
        url = "http://"+source[4:]
    else:
        url = baseUrl+"/"+source
    if baseUrl not in url:
        return None
    return

def getDownloadPath(baseUrl,absoluteUrl, downloadDirectory):
    path = absoluteUrl.replace("www.","")
    path = path.replace(baseUrl, '')
    path = downloadDirectory + path
    directory = os.path.dirname(path)   # 获取当前运行脚本的绝对路径

    if not os.path.exists(directory):  # os.path.exists() 用于判断变量、文件等是否存在
        os.makedirs(directory)    # os.makedirs() 方法用于递归创建目录

    return path

html = urlopen("http://www.pythonscraping.com")
bs_obj = BeautifulSoup(html)
downloadList = bs_obj.findAll(src = True)

for download in downloadList:
    fileUrl = getAbsoluteURL(baseUrl, download["src"])
    if fileUrl is not None:
        print(fileUrl)
        urlretrieve(fileUrl, getDownloadPath(baseUrl, fileUrl, downloadDirectory))

# 5.2 把数据存储到csv
import csv  # csv库可以非常简单的修改csv文件，甚至从零开始创建一个csv文件

csvFile = open("test.csv", "w+")
try:
    writer = csv.writer(csvFile)
    writer.writerow(('number',"number plus 2",'number times 2'))    # 另一种形式writer.writerows(someiterable)
    for i in range(10):
        writer.writerow((i, i+2,i*2))
finally:   # try下的全部操作如果某项失败的话就终止并执行 finally 下定义的语句。如果全部操作都没有报错，那么最后也执行 finally 下定义的语句
    csvFile.close()
# with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，
    # 比如文件使用后自动关闭、线程中锁的自动获取和释放等。

# 5.3 使用pymysql
import pymysql
connection = pymysql.connect(host = '127.0.0.1',user = 'root',passwd = '******',db = 'mysql')
cur = connection.cursor()
cur.execute("use scraping")
cur.execute("select * from pages")
print(cur.fetchone())
cur.close()
connection.close()

# 6 读取文档
# 读取csv
from urllib.request import urlopen
from io import StringIO
import csv
data = urlopen("http://www.pythonscraping.com/files/MontyPythonAlbums.csv").read().decode('ascii','ignore')
dataFile = StringIO(data)   # 直接把文件读成字符串，然后封装成StringIO对象，让python把它当做文件来处理
# csvReader = csv.reader(dataFile)
# i = 0
# for row in csvReader:
#     i +=1
#     print("Monty的第"+str(i)+"个专辑"+row[0]+"发表于"+row[1]+"年！")
#     print("--------------------------")
dictReader = csv.DictReader(dataFile)   #csv.DictReader会返回把csv文件每一行转换成python的字典对象返回，而不是列表对象，并把字段列表保存在
print(dictReader.fieldnames)            #变量dictReader.fieldnames内，字段列表同时作为字典对象的键
for row in dictReader:
    print(row)


