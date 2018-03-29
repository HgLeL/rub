"""
获取前程无忧的数据分析师招聘页面的公司名称、公司招聘url、月薪、工作地点
"""
def extra_n_pages(n):
    import re,requests,time
    t=time.time()
    place,salary,website_company=[],[],[]
    for i in range(1,n):
        url='http://search.51job.com/list/020000,000000,0000,00,9,99,%25E6%2595%25B0%25E6%258D%25AE%25E5%2588%2586%25E6%259E%2590%25E5%25B8%2588,2,'+str(i)+\
            '.html?lang=c&stype=1&postchannel=0000&workyear=99&cotype=99&degreefrom=99&jobterm=99&companysize=99&lonlat=0%2C0&radius=-1&ord_field=0&confirmdate=9&fromType=&dibiaoid=0&address=&line=&specialarea=00&from=&welfare='
        r = requests.get(url)
        r.encoding = 'GBK'
        place.append(re.findall(r'<span\sclass="t3">(.*?)</span>',r.text,re.S)[2:])
        salary.append(re.findall(r'<span\sclass="t4">(.*?)</span>',r.text,re.S)[2:])
        website_company.append(re.findall(r'<span\sclass="t2"><a\starget="_blank"\stitle=".*?"\s'
                                          r'href="(.*?)">(.*?)</a></span>',r.text,re.S)[1:])

    place=place[0]+place[1]+place[2]
    salary=salary[0]+salary[1]+salary[2]
    website_company=website_company[0]+website_company[1]+website_company[2]

    result = []
    for i in range(len(place)):
        a=list(website_company[i])
        a.append(place[i])
        a.append(salary[i])
        result.append(a)
    print('总用时：'+str(time.time()-t)+'秒')
    return result

result=extra_n_pages(7)
with open('C:\\Users\\Administrator\\Desktop\\qcwy.txt','w',encoding='utf-8') as f: #保存数据
    for i in range(len(result)):
        f.write('{}\n'.format(result[i]))


