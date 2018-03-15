class MachineLearning():
    def __init__(self,name,major):
        self.name = name
        self.major = major

    def yourInterest(self,insterest):
        if insterest == 'ML':
            print('you are compatiable!')
        else:
            while True:
                promp=input("您对ML没兴趣吗?\n")
                if promp=='yes':
                    print("you are compatiable!")
                    break
                elif promp=='no':
                    print("Alright!")
                    break
                else:
                    print("请输入yes或no!")

theOne=MachineLearning('john','math')


import re
li = ["alec", " aric", "Alec", "Tony", "rain"]
lp=[]
for i in range(len(li)):
    li[i]=li[i].strip()
    print(re.findall(r'^[aA].*c$',li[i]))
print(li)
print(lp)