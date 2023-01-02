# coding = utf-8
# python3
# Author: yrt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_total = pd.read_csv("./HomeworkData.csv")
df_total.drop(columns=['Unnamed: 0','funny','useful','review_id','business_id','date','user_id','cool'],inplace=True)
# print(df_total['stars'].value_counts())

df_total['text'] = df_total['text'].str.replace('[^\w\s]','') # 去除标点等
df_total['text']= df_total['text'].apply(lambda sen:" ".join(x.lower() for x in sen.split())) # 转为小写

x_train, x_test, y_train, y_test = train_test_split(df_total['text'],df_total['stars'],test_size=1/9) # 区分训练集和测试集

vectorizer = CountVectorizer(ngram_range=(1,1), min_df=3, max_df=0.9, max_features=100000)

x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)
y_train = y_train - 1 # 类别默认是从0开始的，所以这里需要-1

lg = LogisticRegression()
lg.fit(x_train, y_train)

y_test_res = lg.predict(x_test)
y_test_res2 = y_test_res.tolist()
y_test2 = y_test.tolist()

df_result = pd.DataFrame({"真实值":y_test2 ,"预测值":y_test_res2})
df_result["预测值"] = df_result["预测值"] + 1 # 把之前减去的1加回来
print(f"均方误差：{mean_squared_error(df_result['真实值'],df_result['预测值'])}")
df_result.to_csv("result.csv")