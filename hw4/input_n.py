a = [['coffe','test'],['milk','qqq'],['test'],['some'],['milk'],['coffe','test']]
b = [['coffe2'],['milk2','123','milk_two'],['test2'],['some2'],['milk_two'],['coffe2','test2']]

product_list = []
product_output = []

product = input("請輸入(多個值請用 ',' 隔開) = ")
print("input = " , product)
product = product.split(',')

##抓資料
for i in range(0,len(a)):
    for item in product:
        print("item = ",item)
        for j in range(0,len(a[i])):
            if item == a[i][j]:
                product_list.append(b[i])

##整理
for i in product_list:
    for j in i:
        product_output.append(j)

##去重複
product_output = list(set(product_output))
print("product output = ",product_output)