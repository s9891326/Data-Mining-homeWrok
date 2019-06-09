import pandas as pd
# from mlxtend.preprocessing import TransactionEncoder
# from efficient_apriori import apriori
from apyori import apriori
import pyfpgrowth
import time

df_list = []
df = pd.read_excel("交易資料集.xlsx", dtype={'ITEM_NO':str, 'INVOICE_NO':str})

##刪除數量 = 0 及負值
df = df[df['QUANTITY'] > 0]

#抓取不重複的INVOICE_NO值
df = df.drop_duplicates(['INVOICE_NO','ITEM_NO'],'first')

#把INVOICE_NO重複的值ITEM_NO相加
# df = df.groupby(by='INVOICE_NO').apply(lambda x: [",".join(map(str,x['ITEM_NO']))])
#
# # print("df =",df)
#
# for i in df:
#     df_list.append(i[0].split(','))

df_list = df.groupby(by='INVOICE_NO').apply(lambda x:[(','.join(x['ITEM_NO']).split(','))][0])

print("df_list = " ,df_list)

#計算apriori開始時間
apriori_start_time = time.time()

#依照支持度 信心度 進行apriori的關聯規則探索
rules = list(apriori(df_list,min_support=0.005,min_confidence=0.5))

#計算apriori結束時間
apriori_end_time = time.time()

print("rules = " ,rules)

# for i in range(0,len(rules)):
#     for j in range(0,len(rules[i][2])):
#         for k in range(0,len(rules[i][2][j])):
#             print(rules[i][2][j][k])
#         print("-"*20)

for item in rules:
   pair = item[0]
   items = [x for x in pair]
   print("Rule: " + items[0] + " -> " + items[1])
   print("Support: " + str(item[1]))
   print("Confidence: " + str(item[2][0][2]))
   print("Lift: " + str(item[2][0][3]))
   print("=====================================")


print("apriori cost time = %.2f\n" %(apriori_end_time-apriori_start_time))

#計算fg_growth開始時間
fp_growth_start_time = time.time()

#選擇支持度在100以上的
patterns = pyfpgrowth.find_frequent_patterns(df_list,25)

print("pattern = " ,patterns)

#選擇信心度在0.7以上的
FP_growth_rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

#計算fg_growth結束時間
fp_growth_end_time = time.time()

for i,k in FP_growth_rules.items():
    print("選擇的話 : ",i)
    print("會被推測為會購買 : ",k)
    print("-"*20)

print("FP-Growth cost time = %.2f" %(fp_growth_end_time-fp_growth_start_time))