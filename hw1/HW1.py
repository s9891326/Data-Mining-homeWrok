from sklearn import *
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pydotplus
import xlsxwriter
import matplotlib.pyplot as plt

#讀取原始資料
original_dataset = np.loadtxt('letter-recognition.data', dtype = 'str', delimiter  = ',')

#指定屬性
feature_attribute = original_dataset[:, 1::].astype('float')

#指定類別
class_attribute = original_dataset[:, 0]

#標準化屬性
#standardized_feature_attribute = preprocessing.scale(feature_attribute)

#切割訓練及測試資料
feature_attributes_for_training, feature_attributes_for_test, class_attribute_for_training, class_attribute_for_test = train_test_split(feature_attribute, class_attribute, test_size = 0.5)

#採用標準化的方式進行預處理
scale = preprocessing.StandardScaler()

#計算平均值和標準差
scale.fit(feature_attributes_for_training)

#對訓練資料標準化
standardized_feature_attributes_for_training = scale.transform(feature_attributes_for_training)

#對測試資料標準化
standardized_feature_attributes_for_test = scale.transform(feature_attributes_for_test)



#方法一 - Decision Tree

#紀錄決策樹的深度
depth_list = range(2, 31)

#紀錄決策樹的訓練正確率
accuracy_of_decision_tree_training_list = []

#記錄決策樹的測試正確率
accuracy_of_decision_tree_test_list = []

#計算決策樹的最佳深度
for depth in depth_list:

    #採取決策樹演算法
    decision_tree_classifier = DecisionTreeClassifier(max_depth = depth)

    #訓練決策樹
    decision_tree = decision_tree_classifier.fit(standardized_feature_attributes_for_training, class_attribute_for_training)

    #用決策樹的模型來預測訓練資料
    predict_decision_tree_training_data = decision_tree.predict(standardized_feature_attributes_for_training)
    
    #計算訓練資料的分類正確率
    accuracy_of_decision_tree_training_data = metrics.accuracy_score(class_attribute_for_training, predict_decision_tree_training_data)
    
    #紀錄正確率
    accuracy_of_decision_tree_training_list.append(accuracy_of_decision_tree_training_data)

    #用決策樹的模型來預測測試資料
    predict_decision_tree_test_data = decision_tree.predict(standardized_feature_attributes_for_test)
    
    #計算測試資料的分類正確率
    accuracy_of_decision_tree_test_data = metrics.accuracy_score(class_attribute_for_test, predict_decision_tree_test_data)
    
    #記錄正確率
    accuracy_of_decision_tree_test_list.append(accuracy_of_decision_tree_test_data)

#產生圖片
plt.figure()

#設定x軸刻度
plt.xticks(np.arange(2, 32, step = 2))

#設定y軸刻度
plt.yticks(np.arange(0.1, 1.2, step = 0.1))

#畫出決策樹訓練曲線
plt.plot(depth_list, accuracy_of_decision_tree_training_list, marker = 'o', label = 'training')

#畫出決策樹預測曲線
plt.plot(depth_list, accuracy_of_decision_tree_test_list, marker = 'o', label = 'test')

#設定圖片標題
plt.title('Decision Tree')

#設定x軸標題
plt.xlabel('the depth of decision tree')

#設定y軸標題
plt.ylabel('accuracy')

#產生圖例
plt.legend(loc = 'best')

#儲存圖片
plt.savefig('DecisionTree.png')

#設定Excel檔的原始類別欄位
colunm_of_original_class_attribute_for_excel = class_attribute_for_test.reshape(class_attribute_for_test.shape[0], 1)

#設定Excel檔的箭頭欄位
colunm_of_arrow_for_excel = np.full(((class_attribute_for_test.shape[0]), 1), '→')

#設定Excel檔的預測類別欄位
colunm_of_predicted_class_attribute_for_excel = predict_decision_tree_test_data.reshape(class_attribute_for_test.shape[0], 1)

#組合多個欄位
result = np.concatenate((colunm_of_original_class_attribute_for_excel, standardized_feature_attributes_for_test, colunm_of_arrow_for_excel, colunm_of_predicted_class_attribute_for_excel), axis = 1)

#建立Excel檔案
workbook = xlsxwriter.Workbook('result.xlsx')

#新增工作表
worksheet1 = workbook.add_worksheet('Decision Tree')

#寫檔
for row, data in enumerate(result):
    worksheet1.write_row(row, 0, data)

#列出屬性名稱
feature_names = np.array([['x-box'], ['y-box'], ['width'], ['high'], ['onpix'], ['x-bar'], ['y-bar'], ['x2bar'], ['y2bar'], ['xybar'], ['x2ybr'], ['xy2br'], ['x-egg'], ['xegvy'], ['y-egg'], ['yegvx']])

#找出測試紀錄中的最高正確率
highest_accuracy_in_list = max(accuracy_of_decision_tree_test_list)

#找出最高正確率的索引值
index_of_highest_accuracy_in_list = accuracy_of_decision_tree_test_list.index(highest_accuracy_in_list)

#採取決策樹演算法
decision_tree_classifier = DecisionTreeClassifier(max_depth = index_of_highest_accuracy_in_list)

#訓練決策樹
decision_tree = decision_tree_classifier.fit(standardized_feature_attributes_for_training, class_attribute_for_training)

#繪製決策樹(DOT格式)
dot_data = tree.export_graphviz(decision_tree, feature_names = feature_names, class_names = class_attribute, filled = True)

#取得圖片(DOT格式)
graph = pydotplus.graph_from_dot_data(dot_data)

#轉存為pdf格式
graph.write_pdf('decision_tree.pdf')



#方法二 - KNN

#紀錄最近鄰居法的鄰居個數
number_of_neighbors_list = range(1, 21)

#紀錄最近鄰居法的訓練正確率
accuracy_of_k_nearest_neighbors_training_list = []

#紀錄最近鄰居法的測試正確率
accuracy_of_k_nearest_neighbors_test_list = []

#計算最近鄰居法的最佳鄰居個數
for number_of_neighbors in number_of_neighbors_list:

    #採取最近鄰居法演算法
    k_nearest_neighbors_classifier = KNeighborsClassifier(n_neighbors = number_of_neighbors, weights = 'distance', algorithm = 'auto')

    #訓練最近鄰居法
    k_nearest_neighbors = k_nearest_neighbors_classifier.fit(standardized_feature_attributes_for_training, class_attribute_for_training)

    #用最近鄰居法的模型來預測訓練資料
    predict_k_nearest_neighbors_train_data = k_nearest_neighbors.predict(standardized_feature_attributes_for_training)
    
    #計算訓練資料的分類正確率
    accuracy_of_k_nearest_neighbors_training_data = metrics.accuracy_score(class_attribute_for_training, predict_k_nearest_neighbors_train_data)
    
    #記錄正確率
    accuracy_of_k_nearest_neighbors_training_list.append(accuracy_of_k_nearest_neighbors_training_data)

    #用最近鄰居法的模型來預測測試資料
    predict_k_nearest_neighbors_test_data = k_nearest_neighbors.predict(standardized_feature_attributes_for_test)
    
    #計算預測資料的分類正確率
    accuracy_of_k_nearest_neighbors_test_data = metrics.accuracy_score(class_attribute_for_test, predict_k_nearest_neighbors_test_data)
    
    #記錄正確率
    accuracy_of_k_nearest_neighbors_test_list.append(accuracy_of_k_nearest_neighbors_test_data)

#產生圖片
plt.figure()

#設定x軸刻度
plt.xticks(np.arange(1, 21, step = 1))

#設定y軸刻度
plt.yticks(np.arange(0.91, 1.03, step = 0.02))

#畫出最近鄰居法訓練曲線
plt.plot(number_of_neighbors_list, accuracy_of_k_nearest_neighbors_training_list, marker = 'o', label = 'training')

#畫出最近鄰居法測試曲線
plt.plot(number_of_neighbors_list, accuracy_of_k_nearest_neighbors_test_list, marker = 'o', label = 'test')

#設定圖片標題
plt.title('K Nearest Neighbors')

#設定x軸標題
plt.xlabel('the number of k')

#設定y軸標題
plt.ylabel('accuracy')

#設定圖例
plt.legend(loc = 'best')

#儲存圖片
plt.savefig('KNN.png')

#設定Excel檔的預測類別欄位
colunm_of_predicted_class_attribute_for_excel = predict_k_nearest_neighbors_test_data.reshape(class_attribute_for_test.shape[0], 1)

#組合多個欄位
result = np.concatenate((colunm_of_original_class_attribute_for_excel, standardized_feature_attributes_for_test, colunm_of_arrow_for_excel, colunm_of_predicted_class_attribute_for_excel), axis = 1)

#新增工作表
worksheet2 = workbook.add_worksheet('K Nearest Neighbors')

#寫檔
for row, data in enumerate(result):
    worksheet2.write_row(row, 0, data)
    
#關閉Excel檔
workbook.close()