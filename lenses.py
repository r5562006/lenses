# -*- coding: UTF-8 -*-
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd
import pydotplus
from io import StringIO

if __name__ == '__main__':
    with open('lenses.txt', 'r', encoding='utf-8') as fr:  # 載入檔案
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]  # 處理檔案
    
    lenses_target = []  # 提取每組資料的類別，保存在列表中
    for each in lenses:
        lenses_target.append(each[-1])
    
    lenses_labels = ['年齡', '處方', '散光', '淚液量']  # 特徵標籤
    lenses_dict = {label: [] for label in lenses_labels}  # 保存 lenses 資料的字典

    for each in lenses:
        for label in lenses_labels:
            lenses_dict[label].append(each[lenses_labels.index(label)])

    lenses_pd = pd.DataFrame(lenses_dict)  # 生成 pandas.DataFrame
    le = LabelEncoder()  # 建立 LabelEncoder() 物件，用於序列化

    for col in lenses_pd.columns:  # 進行序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])

    clf = tree.DecisionTreeClassifier(max_depth=4)  # 建立 DecisionTreeClassifier() 類
    clf = clf.fit(lenses_pd.values.tolist(), lenses_target)  # 使用資料，構建決策樹

    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data,  # 繪製決策樹
                          feature_names=lenses_pd.columns,
                          class_names=clf.classes_,
                          filled=True, rounded=True,
                          special_characters=True)
    
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")  # 保存繪製好的決策樹，以 PDF 的形式存儲

    print(clf.predict([[1, 1, 1, 0]]))  # 預測
