import pandas as pd
from apyori import apriori
import pyfpgrowth
import time

df_init = pd.read_excel("交易資料集.xlsx", dtype={'ITEM_NO':str, 'INVOICE_NO':str})

##刪除數量 = 0 及負值
df_init = df_init[df_init['QUANTITY'] > 0]

#抓取不重複的INVOICE_NO值
df_init = df_init.drop_duplicates(['INVOICE_NO','ITEM_NO'],'first')

#把INVOICE_NO重複的值PRODUCT_TYPE相加
df_list = df_init.groupby(by='INVOICE_NO').apply(lambda x:[(','.join(x['ITEM_NO']).split(','))][0])

#計算apriori開始時間
apriori_start_time = time.time()

#依照支持度 信心度 進行apriori的關聯規則探索
rules = list(apriori(df_list[:50]))

#計算apriori結束時間
apriori_end_time = time.time()

for i in range(0,len(rules)):
    for j in range(0,len(rules[i][2])):
        for k in range(0,len(rules[i][2][j])):
            print(rules[i][2][j][k])
        print("-"*20)

print("apriori cost time = %.2f" %(apriori_end_time-apriori_start_time))
print("\n\n")
#計算fg_growth開始時間
fp_growth_start_time = time.time()

#選擇支持度在100以上的
patterns = pyfpgrowth.find_frequent_patterns(df_list[:50], 5)

#選擇信心度在0.7以上的
fp_rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

#計算fg_growth結束時間
fp_growth_end_time = time.time()

for i,k in fp_rules.items():
    print("選擇的話 : ",i)
    print("會被推測為會購買 : ",k)
    print("-"*20)

print("FP-Growth cost time = %.2f" %(fp_growth_end_time-fp_growth_start_time))