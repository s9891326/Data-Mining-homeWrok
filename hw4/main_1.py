import pandas as pd

df_init = pd.read_excel("rule.xlsx")
a = df_init['選擇']
b = df_init['推薦']

product_list = []
product_output = []

product = input("請輸入(多個值請用 ',' 隔開) = ")
print("input = " , product)
product = product.split(', ')

##抓資料
for i in range(0,len(a)):
    for item in product:
        for j in range(0,len(a[i])):
            if item == a[i]:
                product_list.append(b[i])

##整理
for i in product_list:
    print(i)
    product_output.append(i)

##去重複
product_output = list(set(product_output))
print("product output = ",product_output)

