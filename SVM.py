from sklearn import model_selection, preprocessing, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer # 使用TD-IDF
import os,csv # 读取文件

'''
程序使用方法：
1.需要更改root作为打开文件的路径，更改cvname作为csv等表格文件的名称，文件格式和之前提供的表格格式相同，在30th左右
2.需要重新训练SVM，训练：测试=8：1，下面代码中也有注释，如果指定测试集需要修改test_x和test_y，在36th行左右
3.经过ti-idf和svm处理，准确率和MSE的结果会打印出来，作为最终结果
4.包含2种svm：线性SVM和rbf核函数SVM，其中的参数经过调参不用修改
程序是可以运行的！有问题请及时联系！感谢！
'''

def load_csv(root, csvname):
    '''
    root:文件夹路径
    csvname：csv文件名
    '''
    comments, labels = [], []
    f=open(os.path.join(root, csvname),encoding='utf-8')
    reader = csv.reader(f)
    next(reader) # 跳过第一行
    for row in reader:
        # 读取评论和类别
        cmt = row[4].lower() # 转换小写
        label = row[6]
        comments.append(cmt)
        labels.append(label)
    return comments, labels

#。。。。。。。。。。。。。。。。此处要改。。。。。。。。。。。。。。。。。。。。。。。。。。
root = 'D:/文档/东南大学本科/2021-2022学年/1学期/人工智能研讨/作业/大作业/TextClassification'
cvname = 'HomeworkData.csv'
# 读取评论和评分
comments, labels = load_csv(root, cvname)

# 划分训练集测试集 训练集8：1测试集
train_x, test_x, train_y, test_y = model_selection.train_test_split(comments, labels, test_size=1/9)
print('训练集有 %s 条句子 , 测试集有 %s 条句子' % (len(train_x), len(test_x)))
# 标签编码
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
test_y = encoder.fit_transform(test_y)

# 数据预处理
# 词语级tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', norm='l2', max_features=5000)
tfidf_vect.fit(comments)
xtrain_tfidf = tfidf_vect.transform(train_x)
xtest_tfidf = tfidf_vect.transform(test_x)

# ngram 级tf-idf： {1,}匹配 n 个前面表达式 \w匹配字母数字及下划线、英文停用词、2-3个组成词组、归一化、最多使用多少个词语
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2, 3), norm='l2', max_features=5000)
tfidf_vect_ngram.fit(comments)
xtrain_tfidf_ngram = tfidf_vect_ngram.transform(train_x)
xtest_tfidf_ngram = tfidf_vect_ngram.transform(test_x)


# 线性SVM
model=svm.SVC(kernel='linear', C=1)
model=model.fit(xtrain_tfidf, train_y)
pred_y = model.predict(xtest_tfidf)
mse = 0
test = list(zip(test_y, pred_y))
for y in test:
    mse += pow(y[0] - y[1], 2)
accuracy = metrics.accuracy_score(pred_y, test_y)
print("SVM, linear, accuracy of tfidf: ", accuracy)
print("SVM, linear, mse of tfidf: ", mse/len(test_y))

# rbfSVM
model=svm.SVC(kernel='rbf', gamma=0.1, C=8)
model=model.fit(xtrain_tfidf, train_y)
pred_y = model.predict(xtest_tfidf)
mse = 0
test = list(zip(test_y, pred_y))
for y in test:
     mse += pow(y[0] - y[1], 2)
accuracy = metrics.accuracy_score(pred_y, test_y)
print("SVM, rbf, accuracy of tfidf: ", accuracy)
print("SVM, rbf, mse of tfidf: ", mse/len(test_y))