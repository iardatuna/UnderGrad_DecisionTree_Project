#Ismail Arda Tuna
#240201031
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus 




def MaxAccDepthLevel(alist):
    value = max(alist)
    index = alist.index(value)
    return index+1,value

data_frame = pd.read_csv("bank_customer.csv")
#Part a
data_frame.job = data_frame.job.replace(to_replace =["management","admin."], value = "white collar") 
data_frame.job = data_frame.job.replace(to_replace =["services","housemaid"], value = "pink collar") 
data_frame.job= data_frame.job.replace(to_replace =["retired","student","unemployed","unknown"], value = "other")
data_frame.poutcome= data_frame.poutcome.replace(to_replace =["other","unknown"], value = "unknown")

print("Job Counts for each value")
print("-------------------------")
print(data_frame["job"].value_counts())
print("")
print("Poutcome Counts for each value")
print("------------------------------")
print(data_frame["poutcome"].value_counts())
print("")
#Part b

label_encoder = preprocessing.LabelEncoder()
categorical_columns = data_frame.select_dtypes(['object']).columns
for i in range(len(categorical_columns)):
    data_frame[categorical_columns[i]] = label_encoder.fit_transform(data_frame[categorical_columns[i]])
print(data_frame[categorical_columns])

#Part c
dataset = data_frame[data_frame.columns[0:16]]
target = data_frame["deposit"]
x_train,x_test, y_train, y_test  = train_test_split(dataset,target, test_size=0.3, random_state = 123) 

print("")
print ("Train size" + " ------------>  " + str(x_train.shape))
print("")
print ("Test size" + "  ------------>  " + str(x_test.shape))
print("")
#Part d

data_1 = x_train[["age","job","marital","education","balance","housing","duration","poutcome"]]
data_1_test = x_test[["age","job","marital","education","balance","housing","duration","poutcome"]]
data_2 = x_train[["job","marital","education","housing"]]
data_2_test = x_test[["job","marital","education","housing"]]

#Part e
print("Decision Tree with Entropy")
print("")
entropy_data_1 = DecisionTreeClassifier(criterion = "entropy",random_state = 123, max_depth=5) 
entropy_data_1 = entropy_data_1.fit(data_1, y_train)
entropy_y_pred_data_1 = entropy_data_1.predict(data_1_test)
print("Accuracy for Data 1: ",accuracy_score(y_test, entropy_y_pred_data_1))
print("")

entropy_data_2 = DecisionTreeClassifier(criterion = "entropy",random_state = 123, max_depth=5) 
entropy_data_2 = entropy_data_2.fit(data_2, y_train)
entropy_y_pred_data_2 = entropy_data_2.predict(data_2_test)
print("Accuracy for Data 2: ",accuracy_score(y_test, entropy_y_pred_data_2))
print("")
    
#Part f
print("Decision Tree with Gini")
print("")
gini_data_1 = DecisionTreeClassifier(criterion = "gini",random_state = 123, max_depth=5) 
gini_data_1 = gini_data_1.fit(data_1, y_train)
gini_y_pred_data_1 = gini_data_1.predict(data_1_test)
print("Accuracy for Data 1: ",accuracy_score(y_test, gini_y_pred_data_1))
print("")

gini_data_2 = DecisionTreeClassifier(criterion = "gini",random_state = 123, max_depth=5) 
gini_data_2 = gini_data_2.fit(data_2, y_train)
gini_y_pred_data_2 = gini_data_2.predict(data_2_test)
print("Accuracy for Data 2: ",accuracy_score(y_test, gini_y_pred_data_2))
print("")


#Part g
entropy_acc_data1_list=[]
entropy_acc_data2_list=[]
gini_acc_data1_list=[]
gini_acc_data2_list=[]

for i in range(10):
    entropy_data_1 = DecisionTreeClassifier(criterion = "entropy",random_state = 123, max_depth=i+1) 
    entropy_data_1 = entropy_data_1.fit(data_1, y_train)
    entropy_y_pred_data_1 = entropy_data_1.predict(data_1_test)
    entropy_data_2 = DecisionTreeClassifier(criterion = "entropy",random_state = 123, max_depth=i+1) 
    entropy_data_2 = entropy_data_2.fit(data_2, y_train)
    entropy_y_pred_data_2 = entropy_data_2.predict(data_2_test)
    entropy_acc_data1_list.append(accuracy_score(y_test, entropy_y_pred_data_1))
    entropy_acc_data2_list.append(accuracy_score(y_test, entropy_y_pred_data_2))
    gini_data_1 = DecisionTreeClassifier(criterion = "gini",random_state = 123, max_depth=i+1) 
    gini_data_1 = gini_data_1.fit(data_1, y_train)
    gini_y_pred_data_1 = gini_data_1.predict(data_1_test)
    gini_data_2 = DecisionTreeClassifier(criterion = "gini",random_state = 123, max_depth=i+1) 
    gini_data_2 = gini_data_2.fit(data_2, y_train)
    gini_y_pred_data_2 = gini_data_2.predict(data_2_test)        
    gini_acc_data1_list.append(accuracy_score(y_test, gini_y_pred_data_1))
    gini_acc_data2_list.append(accuracy_score(y_test, gini_y_pred_data_2))

entropy_acc_data1 = MaxAccDepthLevel(entropy_acc_data1_list)[1]
entropy_acc_data2 = MaxAccDepthLevel(entropy_acc_data2_list)[1]
gini_acc_data1 = MaxAccDepthLevel(gini_acc_data1_list)[1]
gini_acc_data2 = MaxAccDepthLevel(gini_acc_data2_list)[1]


print("Maximum Accuracy with Depth Level")
print("")
print("Data 1 with Entropy -->",MaxAccDepthLevel(entropy_acc_data1_list))
print("")
print("Data 2 with Entropy -->",MaxAccDepthLevel(entropy_acc_data2_list))
print("")
print("Data 1 with Gini -->",MaxAccDepthLevel(gini_acc_data1_list))
print("")
print("Data 2 with Gini -->", MaxAccDepthLevel(gini_acc_data2_list))
print("")
#Part h

N = len(y_test)
entropy_lower_p_data1 = (2*N*entropy_acc_data1 + 1.96*1.96 - np.sqrt(1.96*1.96 + 4*N*entropy_acc_data1 - 4*N*entropy_acc_data1*entropy_acc_data1)) / (2*(N+1.96*1.96))
entropy_upper_p_data1 = (2*N*entropy_acc_data1 + 1.96*1.96 + np.sqrt(1.96*1.96 + 4*N*entropy_acc_data1 - 4*N*entropy_acc_data1*entropy_acc_data1)) / (2*(N+1.96*1.96))
entropy_lower_p_data2 = (2*N*entropy_acc_data2 + 1.96*1.96 - np.sqrt(1.96*1.96 + 4*N*entropy_acc_data2 - 4*N*entropy_acc_data2*entropy_acc_data2)) / (2*(N+1.96*1.96))
entropy_upper_p_data2 = (2*N*entropy_acc_data2 + 1.96*1.96 + np.sqrt(1.96*1.96 + 4*N*entropy_acc_data2 - 4*N*entropy_acc_data2*entropy_acc_data2)) / (2*(N+1.96*1.96))
gini_lower_p_data1 = (2*N*gini_acc_data1 + 1.96*1.96 - np.sqrt(1.96*1.96 + 4*N*gini_acc_data1 - 4*N*gini_acc_data1*gini_acc_data1)) / (2*(N+1.96*1.96))
gini_upper_p_data1 = (2*N*gini_acc_data1 + 1.96*1.96 + np.sqrt(1.96*1.96 + 4*N*gini_acc_data1 - 4*N*gini_acc_data1*gini_acc_data1)) / (2*(N+1.96*1.96))
gini_lower_p_data2 = (2*N*gini_acc_data2 + 1.96*1.96 - np.sqrt(1.96*1.96 + 4*N*gini_acc_data2 - 4*N*gini_acc_data2*gini_acc_data2)) / (2*(N+1.96*1.96))
gini_upper_p_data2 = (2*N*gini_acc_data2 + 1.96*1.96 + np.sqrt(1.96*1.96 + 4*N*gini_acc_data2 - 4*N*gini_acc_data2*gini_acc_data2)) / (2*(N+1.96*1.96))

print("Data 1's Lower p value with Entropy -> ", entropy_lower_p_data1)
print("")
print("Data 1's Upper p value with Entropy -> ", entropy_upper_p_data1)
print("")
print("Data 2's Lower p value with Entropy -> ", entropy_lower_p_data2)
print("")
print("Data 2's Upper p value with Entropy -> ", entropy_upper_p_data2)
print("")
print("Data 1's Lower p value with Gini -> ", gini_lower_p_data1)
print("")
print("Data 1's Upper p value with Gini -> ", gini_upper_p_data1)
print("")
print("Data 2's Lower p value with Gini -> ", gini_lower_p_data2)
print("")
print("Data 2's Upper p value with Gini -> ", gini_upper_p_data2)
print("")

#Part i

data_1_col_names= ["age","job","marital","education","balance","housing","duration","poutcome"]
data_2_col_names = ["job","marital","education","housing"]
dot_data1_entropy = StringIO()
dot_data2_entropy = StringIO()
dot_data1_gini = StringIO()
dot_data2_gini = StringIO()
export_graphviz(entropy_data_1, out_file=dot_data1_entropy,  
                filled=True, rounded=True,
                special_characters=True, feature_names = data_1_col_names, class_names=["0","1"])
data_1_entropy_graph = pydotplus.graph_from_dot_data(dot_data1_entropy.getvalue())  
data_1_entropy_graph.write_png('data_1_entropy.png')
Image(data_1_entropy_graph.create_png())

export_graphviz(entropy_data_2, out_file=dot_data2_entropy,  
                filled=True, rounded=True,
                special_characters=True, feature_names = data_2_col_names, class_names=["0","1"])
data_2_entropy_graph = pydotplus.graph_from_dot_data(dot_data2_entropy.getvalue())  
data_2_entropy_graph.write_png('data_2_entropy.png')
Image(data_2_entropy_graph.create_png())


export_graphviz(gini_data_1, out_file=dot_data1_gini,  
                filled=True, rounded=True,
                special_characters=True, feature_names = data_1_col_names, class_names=["0","1"])
data_1_gini_graph = pydotplus.graph_from_dot_data(dot_data1_gini.getvalue())  
data_1_gini_graph.write_png('data_1_gini.png')
Image(data_1_gini_graph.create_png())


export_graphviz(gini_data_2, out_file=dot_data2_gini,  
                filled=True, rounded=True,
                special_characters=True, feature_names = data_2_col_names, class_names=["0","1"])
data_2_gini_graph = pydotplus.graph_from_dot_data(dot_data2_gini.getvalue())  
data_2_gini_graph.write_png('data_2_gini.png')
Image(data_2_gini_graph.create_png())
   



