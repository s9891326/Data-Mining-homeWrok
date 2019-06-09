import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
# from efficient_apriori import apriori
from apyori import apriori
import pyfpgrowth
import time

df_list = []
df = pd.read_excel("交易資料集.xlsx")

##刪除數量 = 0 及負值
df = df[df['QUANTITY'] > 0]

#抓取不重複的INVOICE_NO值
df = df.drop_duplicates(['INVOICE_NO','PRODUCT_TYPE'],'first')

#把INVOICE_NO重複的值PRODUCT_TYPE相加
df = df.groupby(by='INVOICE_NO').apply(lambda x:[','.join(x['PRODUCT_TYPE'])])

for i in df:
    df_list.append(i[0].split(','))

print(df_list)

#計算apriori開始時間
apriori_start_time = time.time()

te = TransactionEncoder()
#建立表格
te_ary = te.fit(df_list).transform(df_list)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)
print(df.columns)

#依照支持度 信心度 進行apriori的關聯規則探索
rules = list(apriori(df, min_support=0.7,min_confidence=0.7))

#計算apriori結束時間
apriori_end_time = time.time()

for i in range(0,len(rules)):
    for j in range(0,len(rules[i][2])):
        for k in range(0,len(rules[i][2][j])):
            print(rules[i][2][j][k])
        print("-"*20)

print("apriori cost time = %.2f" %(apriori_end_time-apriori_start_time))

#計算fg_growth開始時間
fp_growth_start_time = time.time()

#選擇支持度在100以上的
patterns = pyfpgrowth.find_frequent_patterns(df_list,50)

#選擇信心度在0.7以上的
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

#計算fg_growth結束時間
fp_growth_end_time = time.time()

for i,k in rules.items():
    print("選擇的話 : ",i)
    print("會被推測為會購買 : ",k)
    print("-"*20)

print("FP-Growth cost time = %.2f" %(fp_growth_end_time-fp_growth_start_time))