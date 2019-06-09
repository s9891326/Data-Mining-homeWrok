import pandas as pd
import pyfpgrowth
from time import time
import xlsxwriter

# 加载数据集
def loadDataSet():
    df_init = pd.read_excel("交易資料集.xlsx", dtype={'ITEM_NO':str, 'INVOICE_NO':str})
    ##刪除數量 = 0 及負值
    df_init = df_init[df_init['QUANTITY'] > 0]
    #抓取不重複的INVOICE_NO值
    df_init = df_init.drop_duplicates(['INVOICE_NO','ITEM_NO'],'first')
    #把INVOICE_NO重複的值PRODUCT_TYPE相加
    df_init = df_init.groupby(by='INVOICE_NO').apply(lambda x:[(' , '.join(x['ITEM_NO']).split(' , '))][0])
    
    df_list = []
    for i in df_init:
        if(len(i) > 1):
            df_list.append(i)
    
    return df_list

# 创建集合 C1。即对 dataSet 进行去重，排序，放入 list 中，然后转换所有的元素为 frozenset
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item]) 
    C1.sort()
    
    return list(map(frozenset, C1))

# 计算候选数据集 CK 在数据集 D 中的支持度，并返回支持度大于最小支持度（minSupport）的数据
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(D))
    
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support
    return retList, supportData

# 输入频繁项集列表 Lk 与返回的元素个数 k，然后输出所有可能的候选项集 Ck
def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            L1 = list(Lk[i])[: k-2]
            L2 = list(Lk[j])[: k-2]
            L1.sort()
            L2.sort()
            # 第一次 L1,L2 为空，元素直接进行合并，返回元素两两合并的数据集
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])
    return retList

# 找出数据集 dataSet 中支持度 >= 最小支持度的候选项集以及它们的支持度。即我们的频繁项集。
def apriori(dataSet, minSupport=0.5):
    C1 = createC1(dataSet)
    D = list(dataSet)
    # 计算候选数据集 C1 在数据集 D 中的支持度，并返回支持度大于 minSupport 的数据
    L1, supportData = scanD(D, C1, minSupport)

    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)

        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        if len(Lk) == 0:
            break
        
        L.append(Lk)
        k += 1
    return L, supportData

# 计算可信度（confidence）
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    # 记录可信度大于最小可信度（minConf）的集合
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet]/supportData[freqSet-conseq] 
        if conf >= minConf:
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

# 递归计算频繁项集的规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)):
        Hmp1 = aprioriGen(H, m+1)
        # 返回可信度大于最小可信度的集合
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 计算可信度后，还有数据大于最小可信度的话，那么继续递归调用，否则跳出递归
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

# 生成关联规则
def generateRules(L, supportData, minConf=0.7):
    bigRuleList = []
    for i in range(1, len(L)):
        # 获取频繁项集中每个组合的所有元素
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


dataSet = loadDataSet()
print('load success')

apriori_start_time = time()
# Apriori 算法生成频繁项集以及它们的支持度
L1, supportData1 = apriori(dataSet, minSupport=0.0025)

apriori_time = time() - apriori_start_time
print('Apriori time: ', apriori_time)
# 生成关联规则
apriori_rules = generateRules(L1, supportData1, minConf=0.5)

for item in apriori_rules:
    rule = [x for x in item[0]]
    result = [x for x in item[1]]
    print("選擇的話 : ",rule)
    print("會被推測為會購買 : ",result)
    print("-"*20)
print("\n\n")

fp_growth_start_time = time()
patterns = pyfpgrowth.find_frequent_patterns(dataSet, 29)
fp_rules = pyfpgrowth.generate_association_rules(patterns, 0.5)

#計算fg_growth結束時間
fp_growth_time = time() - fp_growth_start_time
print("FP-Growth cost time = %.2f" %(fp_growth_time))

for i,k in fp_rules.items():
    print("選擇的話 : ",i)
    print("會被推測為會購買 : ",k)
    print("-"*20)

workbook = xlsxwriter.Workbook('rule.xlsx')

worksheet1 = workbook.add_worksheet('Apriori')
row = 1
worksheet1.write(0, 0, '選擇')
worksheet1.write(0, 2, '推薦')

for item in apriori_rules:
    rule = [x for x in item[0]]
    result = [x for x in item[1]]
    worksheet1.write(row, 0, ', '.join(rule))
    worksheet1.write(row, 1, '->')
    worksheet1.write(row, 2, ', '.join(result))
    row += 1

worksheet2 = workbook.add_worksheet('FP-Growth')
row = 1
worksheet2.write(0, 0, '選擇')
worksheet2.write(0, 2, '推薦')

for i, k in fp_rules.items():
    if (str(i)[-2] == "'"):
        worksheet2.write(row, 0, str(i)[1:-1])
    else:
        worksheet2.write(row, 0, str(i)[1:-2])
        
    worksheet2.write(row, 1, '->')
    
    if (str(k[0])[-2] == "'"):
        worksheet2.write(row, 2, str(k[0])[1:-1])
    else:
        worksheet2.write(row, 2, str(k[0])[1:-2])
    row = row + 1

workbook.close()
