from sre_constants import MIN_UNTIL
from tkinter import *
from tkinter import filedialog
import pandas as pd
import math
from tkinter import ttk
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import operator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import itertools
import numpy as np
from sklearn.tree import _tree
from sklearn import metrics
from tkinter import messagebox as mb
from sklearn.neighbors import KNeighborsClassifier
from joblib.numpy_pickle_utils import xrange
from numpy import log,dot,exp,shape
from sklearn.datasets import make_classification
import random
import sys
from urllib.request import urljoin
from bs4 import BeautifulSoup
import requests
from urllib.request import urlparse
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
def browseDataset():
    filename = filedialog.askopenfilename(initialdir="/",title="Select dataset", filetypes=(("CSV files", "*.csv*"), ("all files", "*.*")))
    label_file_explorer.configure(text="File Opened: "+filename)
    newfilename = ''
    print(filename)
    for i in filename:
        if i == "/":
            newfilename = newfilename + "/"
        newfilename = newfilename + i
    print(newfilename)
    data = pd.read_csv(filename)
    d = pd.read_csv(filename)
    w = Tk()
    w.title("Census Income Predictor")
    w.geometry("600x500")
    
    tv1 = ttk.Treeview(w)
    tv1.place(relheight=1, relwidth=1)

    treescrolly = Scrollbar(w, orient="vertical", command=tv1.yview) 
    treescrollx = Scrollbar(w, orient="horizontal", command=tv1.xview)
    tv1.configure(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    treescrollx.pack(side="bottom", fill="x")
    treescrolly.pack(side="right", fill="y") 
    tv1["column"] = list(data.columns)
    tv1["show"] = "headings"
    for column in tv1["columns"]:
        tv1.heading(column, text=column) 

    df_rows = data.to_numpy().tolist() 
    for row in df_rows:
        tv1.insert("", "end", values=row)
        
    dataset = data.copy()
    dataset.drop(dataset.tail(30000).index,
        inplace = True)
    x=dataset.iloc[:,:-1].values
    y=dataset.iloc[:,-1].values
    imputer=SimpleImputer(missing_values=np.nan,strategy='most_frequent')
    imputer.fit(x[:,1:])
    x[:,1:]=imputer.transform(x[:,1:])
    le1=LabelEncoder()
    le3=LabelEncoder()
    le5=LabelEncoder()
    le6=LabelEncoder()
    le7=LabelEncoder()
    le8=LabelEncoder()
    le9=LabelEncoder()
    le13=LabelEncoder()
    le=LabelEncoder()
    x[:,1]=le1.fit_transform(x[:,1])
    x[:,3]=le3.fit_transform(x[:,3])
    x[:,5]=le5.fit_transform(x[:,5])
    x[:,6]=le6.fit_transform(x[:,6])
    x[:,7]=le7.fit_transform(x[:,7])
    x[:,8]=le8.fit_transform(x[:,8])
    x[:,9]=le9.fit_transform(x[:,9])
    x[:,13]=le13.fit_transform(x[:,13])
    y=le.fit_transform(y)
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    
    dict_workclass = {}
    for i in range(len(x[:,1])):
        dict_workclass[dataset['workclass'][i]] = x[:,1][i]
    print(dict_workclass)
    
    dict_education = {}
    for i in range(len(x[:,3])):
        dict_education[dataset['education'][i]] = x[:,3][i]
    print(dict_education)
    
    dict_maritalStatus = {}
    for i in range(len(x[:,5])):
        dict_maritalStatus[dataset['marital.status'][i]] = x[:,5][i]
    print(dict_maritalStatus)
    
    dict_occupation = {}
    for i in range(len(x[:,6])):
        dict_occupation[dataset['occupation'][i]] = x[:,6][i]
    print(dict_occupation)
    
    dict_relationship = {}
    for i in range(len(x[:,7])):
        dict_relationship[dataset['relationship'][i]] = x[:,7][i]
    print(dict_relationship)
    
    dict_race = {}
    for i in range(len(x[:,8])):
        dict_race[dataset['race'][i]] = x[:,8][i]
    print(dict_race)
    
    dict_sex = {}
    for i in range(len(x[:,9])):
        dict_sex[dataset['sex'][i]] = x[:,9][i]
    print(dict_sex)
    
    dict_nativeCountry = {}
    for i in range(len(x[:,13])):
        dict_nativeCountry[dataset['native.country'][i]] = x[:,13][i]
    print(dict_nativeCountry)
    
    dict_income= {}
    for i in range(len(y)):
        dict_income[dataset['income'][i]] = y[i]
    print(dict_income)
    
    
    # Creating Menubar
    menubar = Menu(w)
    
    # Adding File Menu and commands
    topics = Menu(menubar, tearoff = 0)
    menubar.add_cascade(label ='Topics', menu = topics)
    topics.add_command(label ='Describe Data', command = lambda: ImplementTopic("Describe Data"))
    
    univariategraphs = Menu(menubar, tearoff=0)
    menubar.add_cascade(label ='Univariate Analysis', menu=univariategraphs)
    univariategraphs.add_command(label='Distribution of Income', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Income"))
    univariategraphs.add_command(label='Distribution of Age', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Age"))
    univariategraphs.add_command(label='Distribution of Education', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Education"))
    univariategraphs.add_command(label='Distribution of Years of Education', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Years of Education"))
    univariategraphs.add_command(label='Marital Distribution', command = lambda: ImplementTopic("Univariate Analysis and Marital Distribution"))
    univariategraphs.add_command(label='Relationship Distribution', command = lambda: ImplementTopic("Univariate Analysis and Relationship Distribution"))
    univariategraphs.add_command(label='Distribution of Sex', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Sex"))
    univariategraphs.add_command(label='Race Distribution', command = lambda: ImplementTopic("Univariate Analysis and Race Distribution"))
    univariategraphs.add_command(label='Distribution of Hours of work per week', command = lambda: ImplementTopic("Univariate Analysis and Distribution of Hours of work per week"))
    
    bivariategraphs = Menu(menubar, tearoff=0)
    menubar.add_cascade(label ='Bivariate Analysis', menu=bivariategraphs)
    bivariategraphs.add_command(label='Distribution of Income vs Age', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Age"))
    bivariategraphs.add_command(label='Distribution of Income vs Education', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Education"))
    bivariategraphs.add_command(label='Distribution of Income vs Years of Education', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Years of Education"))
    bivariategraphs.add_command(label='Distribution of Income vs Marital Status', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Marital Status"))
    bivariategraphs.add_command(label='Distribution of Income vs Race', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Race"))
    bivariategraphs.add_command(label='Distribution of Income vs Sex', command = lambda: ImplementTopic("Bivariate Analysis and Distribution of Income vs Sex"))
   
    multivariategraphs = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Multivariate Analysis", menu=multivariategraphs)
    multivariategraphs.add_command(label="Heatmap", command = lambda: ImplementTopic("Multivariate Analysis and Heatmap"))
    
    trainingModel = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Model", menu=trainingModel)
    trainingModel.add_command(label="Logistic Regression", command= lambda: ImplementTopic("Logistic Regression"))
    trainingModel.add_command(label="kNN Classifier", command= lambda: ImplementTopic("kNN Classifier"))
    
    
    # display Menu
    w.config(menu = menubar,bg='#18253f')
    def ImplementTopic(topic):
        if topic == "Describe Data":
            print(data.describe().T,type(data.describe().T))
            dt = data.describe().T
            print(dt.columns)
            w1 = Tk()
            w1.title("Census Income Predictor")
            w1.geometry("600x500")
            frame = Frame(w1)
            frame.pack()

            cols = list(data.describe().T.columns)
            cols.insert(0,"Attributes")
            for i in range(len(cols)):
                Label(frame,text=cols[i],font=("Verdana",10),background="black",foreground="white",width=18).grid(row = 0, column = i, sticky = W, padx=2, pady = 2)
            i = 1
            df_rows = (data.describe().T).to_numpy().tolist() 
            attributes = ["Age","Fnlwgt","Education","Capital Gain","Capital Loss","Hours per week"]
            for r in df_rows:
                r.insert(0,attributes[i-1])
                for j in range(len(r)):
                    Label(frame,text=r[j],font=("Verdana",10),background="grey",foreground="white",width=18).grid(row = i, column = j, sticky = W, padx=2, pady = 2)
                i += 1
            w1.mainloop()
        elif topic == "Univariate Analysis and Distribution of Income":
            # Creating a barplot for 'Income'
            income = data['income'].value_counts()

            plt.style.use('seaborn-whitegrid')
            plt.figure(figsize=(7, 5))
            sns.barplot(income.index, income.values, palette='bright')
            plt.title('Distribution of Income', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Income', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=10)
            plt.show()
        elif topic == "Univariate Analysis and Distribution of Age":
            # Creating a distribution plot for 'Age'
            age = data['age'].value_counts()

            plt.figure(figsize=(10, 5))
            plt.style.use('fivethirtyeight')
            sns.distplot(data['age'], bins=20)
            plt.title('Distribution of Age', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=10)
            plt.show()
        elif topic == "Univariate Analysis and Distribution of Education":
            # Creating a barplot for 'Education'
            edu = data['education'].value_counts()

            plt.style.use('seaborn')
            plt.figure(figsize=(10, 5))
            sns.barplot(edu.values, edu.index, palette='Paired')
            plt.title('Distribution of Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.show()
        elif topic == "Univariate Analysis and Distribution of Years of Education":
            # Creating a barplot for 'Years of Education'
            edu_num = data['education.num'].value_counts()

            plt.style.use('ggplot')
            plt.figure(figsize=(10, 5))
            sns.barplot(edu_num.index, edu_num.values, palette='colorblind')
            plt.title('Distribution of Years of Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Years of Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.show()
        elif topic == "Univariate Analysis and Marital Distribution":
            # Creating a pie chart for 'Marital status'
            marital = data['marital.status'].value_counts()

            plt.style.use('default')
            plt.figure(figsize=(10, 7))
            plt.pie(marital.values, labels=marital.index, startangle=10, explode=(
                0, 0.20, 0, 0, 0, 0, 0), shadow=True, autopct='%1.1f%%')
            plt.title('Marital distribution', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.legend()
            plt.legend(prop={'size': 7})
            plt.axis('equal')
            plt.show()
        elif topic == "Univariate Analysis and Relationship Distribution":
            # Creating a donut chart for 'Age'
            relation = data['relationship'].value_counts()

            plt.style.use('bmh')
            plt.figure(figsize=(20, 10))
            plt.pie(relation.values, labels=relation.index,
                    startangle=50, autopct='%1.1f%%')
            centre_circle = plt.Circle((0, 0), 0.7, fc='white')
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.title('Relationship distribution', fontdict={
                    'fontname': 'Monospace', 'fontsize': 30, 'fontweight': 'bold'})
            plt.axis('equal')
            plt.legend(prop={'size': 15})
            plt.show()
        elif topic == "Univariate Analysis and Distribution of Sex":
            # Creating a barplot for 'Sex'
            sex = data['sex'].value_counts()

            plt.style.use('default')
            plt.figure(figsize=(7, 5))
            sns.barplot(sex.index, sex.values)
            plt.title('Distribution of Sex', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Sex', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=10)
            plt.grid()
            plt.show()
        elif topic == "Univariate Analysis and Race Distribution":
            # Creating a Treemap for 'Race'
            import squarify
            race = data['race'].value_counts()

            plt.style.use('default')
            plt.figure(figsize=(7, 5))
            squarify.plot(sizes=race.values, label=race.index, value=race.values)
            plt.title('Race distribution', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.show()
        elif topic == "Univariate Analysis and Distribution of Hours of work per week":
            # Creating a barplot for 'Hours per week'
            hours = data['hours.per.week'].value_counts().head(10)

            plt.style.use('bmh')
            plt.figure(figsize=(15, 7))
            sns.barplot(hours.index, hours.values, palette='colorblind')
            plt.title('Distribution of Hours of work per week', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Hours of work', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Age":
            # Creating a countplot of income across age
            plt.style.use('default')
            plt.figure(figsize=(20, 7))
            sns.countplot(data['age'], hue=data['income'])
            plt.title('Distribution of Income across Age', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Age', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 15})
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Education":
            # Creating a countplot of income across education
            plt.style.use('seaborn')
            plt.figure(figsize=(20, 7))
            sns.countplot(data['education'],
                        hue=data['income'], palette='colorblind')
            plt.title('Distribution of Income across Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Education', fontdict={'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 15})
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Years of Education":
            # Creating a countplot of income across years of education
            plt.style.use('bmh')
            plt.figure(figsize=(20, 7))
            sns.countplot(data['education.num'],
                        hue=data['income'])
            plt.title('Income across Years of Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Years of Education', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 15})
            plt.savefig('bi2.png')
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Marital Status":
            # Creating a countplot of income across Marital Status
            plt.style.use('seaborn')
            plt.figure(figsize=(20, 7))
            sns.countplot(data['marital.status'], hue=data['income'])
            plt.title('Income across Marital Status', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Marital Status', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 15})
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Race":
            # Creating a countplot of income across race
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(20, 7))
            sns.countplot(data['race'], hue=data['income'])
            plt.title('Distribution of income across race', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Race', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 15})
            plt.show()
        elif topic == "Bivariate Analysis and Distribution of Income vs Sex":
            # Creating a countplot of income across sex
            plt.style.use('fivethirtyeight')
            plt.figure(figsize=(7, 3))
            sns.countplot(data['sex'], hue=data['income'])
            plt.title('Distribution of income across sex', fontdict={
                    'fontname': 'Monospace', 'fontsize': 20, 'fontweight': 'bold'})
            plt.xlabel('Sex', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.ylabel('Number of people', fontdict={
                    'fontname': 'Monospace', 'fontsize': 15})
            plt.tick_params(labelsize=12)
            plt.legend(loc=1, prop={'size': 10})
            plt.savefig('bi3.png')
            plt.show()
        elif topic == "Multivariate Analysis and Heatmap":
            le = LabelEncoder()
            data['income'] = le.fit_transform(data['income'])
            corr = data.corr()
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
            with sns.axes_style("white"):
                f, ax = plt.subplots(figsize=(7, 5))
                ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True,
                                annot=True, cmap='RdYlGn')
            plt.savefig('multi2.png')
            plt.show()
        elif topic == "Logistic Regression":
            class LogisticRegression:
                def sigmoid(self,z):
                    sig = 1/(1+exp(-z))
                    return sig
                def initialize(self,X):
                    weights = np.zeros((shape(X)[1]+1,1))
                    X = np.c_[np.ones((shape(X)[0],1)),X]
                    return weights,X
                def fit(self,X,y,alpha=0.001,iter=400):
                    weights,X = self.initialize(X)
                    def cost(theta):
                        z = dot(X,theta)
                        cost0 = y.T.dot(log(self.sigmoid(z)))
                        cost1 = (1-y).T.dot(log(1-self.sigmoid(z)))
                        cost = -((cost1 + cost0))/len(y)
                        return cost
                    cost_list = np.zeros(iter,)
                    for i in range(iter):
                        weights = weights - alpha*dot(X.T,self.sigmoid(dot(X,weights))-np.reshape(y,(len(y),1)))
                        cost_list[i] = cost(weights)
                    self.weights = weights
                    return cost_list
                def predict(self,X):
                    z = dot(self.initialize(X)[1],self.weights)
                    lis = []
                    for i in self.sigmoid(z):
                        if i>0.5:
                            lis.append(1)
                        else:
                            lis.append(0)
                    return lis
            obj1 = LogisticRegression()
            print(X_train)
            model= obj1.fit(X_train,Y_train)
            y_pred = obj1.predict(X_test)
            print(X_test,"xtest")
            cnt = 0
            for i in range(len(Y_test)):
                if Y_test[i]==y_pred[i]:
                    cnt += 1

            print(cnt/len(Y_test))
            accuracy_scratch_code = str(cnt/len(Y_test)*100)+" %"
            # in built fn
            cnt = 0
            from sklearn.linear_model import LogisticRegression
            log_reg = LogisticRegression(random_state=42)
            log_reg.fit(X_train, Y_train)
            Y_pred_log_reg = log_reg.predict(X_test)
            for i in range(len(Y_test)):
                if Y_test[i]==Y_pred_log_reg[i]:
                    cnt += 1
            print(cnt/len(Y_test))   
            accuracy_inbuilt_code = str(cnt/len(Y_test)*100)+" %"      
            w2 = Tk()
            w2.title("Census Income Predictor-Logistic Regression")
            w2.geometry("600x500")
            def predictClass(answerAge,answerWorkclass,answerFnlwgt,answerEducation,answerEducationNumber,answerMaritalStatus,answerOccupation,answerRelationship,answerRace,answerSex,answerCapitalgain,answerCapitalloss,answerHoursperweek,answerNativeCountry):
                print(answerAge,answerWorkclass,answerFnlwgt,answerEducation,answerEducationNumber,answerMaritalStatus,answerOccupation,answerRelationship,answerRace,answerSex,answerCapitalgain,answerCapitalloss,answerHoursperweek,answerNativeCountry)
                obj2 = LogisticRegression()
                print(X_train)
                model= obj2.fit(X_train,Y_train)
                y_pred = obj2.predict(sc.transform([[answerAge,dict_workclass[answerWorkclass],answerFnlwgt,dict_education[answerEducation],answerEducationNumber,dict_maritalStatus[answerMaritalStatus],dict_occupation[answerOccupation],dict_relationship[answerRelationship],dict_race[answerRace],dict_sex[answerSex],answerCapitalgain,answerCapitalloss,answerHoursperweek,dict_nativeCountry[answerNativeCountry]]]))
                print(y_pred)
                Label(w2,text="Predicted Class: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=18,column=1, sticky = W,padx=4,pady=2)
                Label(w2,text='>50k' if y_pred[0] else '<=50k',justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=18,column=2, sticky = W,padx=2,pady=2)
            
            Label(w2,text="Accuracy from scratch code: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=1,column=1, sticky = W,padx=4,pady=2)
            Label(w2,text=accuracy_scratch_code,justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=1,column=2, sticky = W,padx=2,pady=2)
            Label(w2,text="Accuracy from in-built fn: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=2,column=1, sticky = W,padx=4,pady=2)
            Label(w2,text=accuracy_inbuilt_code,justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=2,column=2, sticky = W,padx=2,pady=2)
            Label(w2,text="Age",font=('Verdana', 10),height=2,width=45).grid(row=3,column=1,padx=4,pady=2)
            answerAge = Entry(w2)
            answerAge.grid(row=3,column=2,padx=4,pady=2)
            Label(w2,text="Workclass",font=('Verdana', 10),height=2,width=45).grid(row=4,column=1,padx=4,pady=2)
            answerWorkclass = Entry(w2)
            answerWorkclass.grid(row=4,column=2,padx=4,pady=2)
            Label(w2,text="Fnlwgt",font=('Verdana', 10),height=2,width=45).grid(row=5,column=1,padx=4,pady=2)
            answerFnlwgt = Entry(w2)
            answerFnlwgt.grid(row=5,column=2,padx=4,pady=2)
            Label(w2,text="Education",font=('Verdana', 10),height=2,width=45).grid(row=6,column=1,padx=4,pady=2)
            answerEducation = Entry(w2)
            answerEducation.grid(row=6,column=2,padx=4,pady=2)
            Label(w2,text="Education Number",font=('Verdana', 10),height=2,width=45).grid(row=7,column=1,padx=4,pady=2)
            answerEducationNumber = Entry(w2)
            answerEducationNumber.grid(row=7,column=2,padx=4,pady=2)
            Label(w2,text="Marital Status",font=('Verdana', 10),height=2,width=45).grid(row=8,column=1,padx=4,pady=2)
            answerMaritalStatus = Entry(w2)
            answerMaritalStatus.grid(row=8,column=2,padx=4,pady=2)
            Label(w2,text="Occupation",font=('Verdana', 10),height=2,width=45).grid(row=9,column=1,padx=4,pady=2)
            answerOccupation = Entry(w2)
            answerOccupation.grid(row=9,column=2,padx=4,pady=2)
            Label(w2,text="Relationship",font=('Verdana', 10),height=2,width=45).grid(row=10,column=1,padx=4,pady=2)
            answerRelationship = Entry(w2)
            answerRelationship.grid(row=10,column=2,padx=4,pady=2)
            Label(w2,text="Race",font=('Verdana', 10),height=2,width=45).grid(row=11,column=1,padx=4,pady=2)
            answerRace = Entry(w2)
            answerRace.grid(row=11,column=2,padx=4,pady=2)
            Label(w2,text="Sex",font=('Verdana', 10),height=2,width=45).grid(row=12,column=1,padx=4,pady=2)
            answerSex = Entry(w2)
            answerSex.grid(row=12,column=2,padx=4,pady=2)
            Label(w2,text="Capital gain",font=('Verdana', 10),height=2,width=45).grid(row=13,column=1,padx=4,pady=2)
            answerCapitalgain = Entry(w2)
            answerCapitalgain.grid(row=13,column=2,padx=4,pady=2)
            Label(w2,text="Capital loss",font=('Verdana', 10),height=2,width=45).grid(row=14,column=1,padx=4,pady=2)
            answerCapitalloss = Entry(w2)
            answerCapitalloss.grid(row=14,column=2,padx=4,pady=2)
            Label(w2,text="Hours per week",font=('Verdana', 10),height=2,width=45).grid(row=15,column=1,padx=4,pady=2)
            answerHoursperweek = Entry(w2)
            answerHoursperweek.grid(row=15,column=2,padx=4,pady=2)
            Label(w2,text="Native Country",font=('Verdana', 10),height=2,width=45).grid(row=16,column=1,padx=4,pady=2)
            answerNativeCountry = Entry(w2)
            answerNativeCountry.grid(row=16,column=2,padx=4,pady=2)
            computeButton = Button(w2,text="Compute", justify='center', width=20, height=2, font=("Verdana", 8),command=lambda:predictClass(int(answerAge.get()),answerWorkclass.get(),int(answerFnlwgt.get()),answerEducation.get(),int(answerEducationNumber.get()),answerMaritalStatus.get(),answerOccupation.get(),answerRelationship.get(),answerRace.get(),answerSex.get(),int(answerCapitalgain.get()),int(answerCapitalloss.get()),int(answerHoursperweek.get()),answerNativeCountry.get()))
            computeButton.grid(row=17,column=2,padx=4,pady=2)
            w2.mainloop()
        elif topic == "kNN Classifier":
            k = 3
            Y_pred = []
            for i in range(len(X_test)):
                distance = []
                for j in range(len(X_train)):
                    val = (X_train[j]-X_test[i])**2
                    euclidean_distance = math.sqrt(np.sum(val))
                    distance.append((euclidean_distance,Y_train[j]))
                distance = sorted(distance)[:k]
                freq1 = 0 
                freq2 = 0 
                for d in distance:
                    if d[1] == 0:
                        freq1 += 1
                    elif d[1] == 1:
                        freq2 += 1
                Y_pred.append(0 if freq1>freq2 else 1)
            cnt = 0
            for i in range(len(Y_test)):
                if Y_test[i]==Y_pred[i]:
                    cnt += 1

            print(cnt/len(Y_test))
            accuracy_scratch_code = str(cnt/len(Y_test)*100)+" %"
            
            #in built fn
            cnt = 0
            knn = KNeighborsClassifier()
            knn.fit(X_train, Y_train)
            Y_pred_knn = knn.predict(X_test)
            for i in range(len(Y_test)):
                if Y_test[i]==Y_pred_knn[i]:
                    cnt += 1
            print(cnt/len(Y_test))
            
            accuracy_inbuilt_code = str(cnt/len(Y_test)*100)+" %"      
            w2 = Tk()
            w2.title("Census Income Predictor-k-NN Classifier")
            w2.geometry("600x500")
            def predictClass(answerAge,answerWorkclass,answerFnlwgt,answerEducation,answerEducationNumber,answerMaritalStatus,answerOccupation,answerRelationship,answerRace,answerSex,answerCapitalgain,answerCapitalloss,answerHoursperweek,answerNativeCountry):
                print(answerAge,answerWorkclass,answerFnlwgt,answerEducation,answerEducationNumber,answerMaritalStatus,answerOccupation,answerRelationship,answerRace,answerSex,answerCapitalgain,answerCapitalloss,answerHoursperweek,answerNativeCountry)
                unknownTuple = sc.transform([[answerAge,dict_workclass[answerWorkclass],answerFnlwgt,dict_education[answerEducation],answerEducationNumber,dict_maritalStatus[answerMaritalStatus],dict_occupation[answerOccupation],dict_relationship[answerRelationship],dict_race[answerRace],dict_sex[answerSex],answerCapitalgain,answerCapitalloss,answerHoursperweek,dict_nativeCountry[answerNativeCountry]]])
                Y_pred = []
                distance = []
                print(X_train[0])
                print(unknownTuple)
                
                for j in range(len(X_train)):
                    val = (X_train[j]-unknownTuple)**2
                    euclidean_distance = math.sqrt(np.sum(val))
                    distance.append((euclidean_distance,Y_train[j]))
                distance = sorted(distance)[:k]
                freq1 = 0 
                freq2 = 0 
                for d in distance:
                    if d[1] == 0:
                        freq1 += 1
                    elif d[1] == 1:
                        freq2 += 1
                Y_pred.append(0 if freq1>freq2 else 1)
                Label(w2,text="Predicted Class: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=18,column=1, sticky = W,padx=4,pady=2)
                Label(w2,text='>50k' if Y_pred[0] else '<=50k',justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=18,column=2, sticky = W,padx=2,pady=2)
            
            Label(w2,text="Accuracy from scratch code: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=1,column=1, sticky = W,padx=4,pady=2)
            Label(w2,text=accuracy_scratch_code,justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=1,column=2, sticky = W,padx=2,pady=2)
            Label(w2,text="Accuracy from in-built fn: ",justify='center',font=("Verdana", 10),height=2,width=45,fg="white",bg="black").grid(row=2,column=1, sticky = W,padx=4,pady=2)
            Label(w2,text=accuracy_inbuilt_code,justify='center',font=("Verdana", 10),height=2,fg="green").grid(row=2,column=2, sticky = W,padx=2,pady=2)
            Label(w2,text="Age",font=('Verdana', 10),height=2,width=45).grid(row=3,column=1,padx=4,pady=2)
            answerAge = Entry(w2)
            answerAge.grid(row=3,column=2,padx=4,pady=2)
            Label(w2,text="Workclass",font=('Verdana', 10),height=2,width=45).grid(row=4,column=1,padx=4,pady=2)
            answerWorkclass = Entry(w2)
            answerWorkclass.grid(row=4,column=2,padx=4,pady=2)
            Label(w2,text="Fnlwgt",font=('Verdana', 10),height=2,width=45).grid(row=5,column=1,padx=4,pady=2)
            answerFnlwgt = Entry(w2)
            answerFnlwgt.grid(row=5,column=2,padx=4,pady=2)
            Label(w2,text="Education",font=('Verdana', 10),height=2,width=45).grid(row=6,column=1,padx=4,pady=2)
            answerEducation = Entry(w2)
            answerEducation.grid(row=6,column=2,padx=4,pady=2)
            Label(w2,text="Education Number",font=('Verdana', 10),height=2,width=45).grid(row=7,column=1,padx=4,pady=2)
            answerEducationNumber = Entry(w2)
            answerEducationNumber.grid(row=7,column=2,padx=4,pady=2)
            Label(w2,text="Marital Status",font=('Verdana', 10),height=2,width=45).grid(row=8,column=1,padx=4,pady=2)
            answerMaritalStatus = Entry(w2)
            answerMaritalStatus.grid(row=8,column=2,padx=4,pady=2)
            Label(w2,text="Occupation",font=('Verdana', 10),height=2,width=45).grid(row=9,column=1,padx=4,pady=2)
            answerOccupation = Entry(w2)
            answerOccupation.grid(row=9,column=2,padx=4,pady=2)
            Label(w2,text="Relationship",font=('Verdana', 10),height=2,width=45).grid(row=10,column=1,padx=4,pady=2)
            answerRelationship = Entry(w2)
            answerRelationship.grid(row=10,column=2,padx=4,pady=2)
            Label(w2,text="Race",font=('Verdana', 10),height=2,width=45).grid(row=11,column=1,padx=4,pady=2)
            answerRace = Entry(w2)
            answerRace.grid(row=11,column=2,padx=4,pady=2)
            Label(w2,text="Sex",font=('Verdana', 10),height=2,width=45).grid(row=12,column=1,padx=4,pady=2)
            answerSex = Entry(w2)
            answerSex.grid(row=12,column=2,padx=4,pady=2)
            Label(w2,text="Capital gain",font=('Verdana', 10),height=2,width=45).grid(row=13,column=1,padx=4,pady=2)
            answerCapitalgain = Entry(w2)
            answerCapitalgain.grid(row=13,column=2,padx=4,pady=2)
            Label(w2,text="Capital loss",font=('Verdana', 10),height=2,width=45).grid(row=14,column=1,padx=4,pady=2)
            answerCapitalloss = Entry(w2)
            answerCapitalloss.grid(row=14,column=2,padx=4,pady=2)
            Label(w2,text="Hours per week",font=('Verdana', 10),height=2,width=45).grid(row=15,column=1,padx=4,pady=2)
            answerHoursperweek = Entry(w2)
            answerHoursperweek.grid(row=15,column=2,padx=4,pady=2)
            Label(w2,text="Native Country",font=('Verdana', 10),height=2,width=45).grid(row=16,column=1,padx=4,pady=2)
            answerNativeCountry = Entry(w2)
            answerNativeCountry.grid(row=16,column=2,padx=4,pady=2)
            computeButton = Button(w2,text="Compute", justify='center', width=20, height=2, font=("Verdana", 8),command=lambda:predictClass(int(answerAge.get()),answerWorkclass.get(),int(answerFnlwgt.get()),answerEducation.get(),int(answerEducationNumber.get()),answerMaritalStatus.get(),answerOccupation.get(),answerRelationship.get(),answerRace.get(),answerSex.get(),int(answerCapitalgain.get()),int(answerCapitalloss.get()),int(answerHoursperweek.get()),answerNativeCountry.get()))
            computeButton.grid(row=17,column=2,padx=4,pady=2)
            w2.mainloop()
    w.mainloop()


def Usage():
    mb.showinfo("Product Information", "1.Browse dataset .csv file from file explorer \n2.First select assignment number from menu dropdown \n3.Perform data analysis of your choice from menu\n")

window = Tk()
window.title("Data Mining Lab-Mini Project")
window.geometry("600x500")
labelHead = Label(window,text="Census Income Predictor",justify='center',font=("Verdana", 34),background='#c345fc',foreground='#fff')
label_file_explorer = Label(window,text="Choose Dataset from File Explorer",justify='center',font=("Verdana", 14),height=4,fg="blue")
button_explore = Button(window,text="Browse Dataset", justify='center', width=20, height=4, font=("Verdana", 8),command=browseDataset)
labelHead.place(relx=0.5, rely=0.1, anchor=CENTER)
label_file_explorer.place(relx=0.5, rely=0.3, anchor=CENTER)
button_explore.place(relx=0.5, rely=0.5, anchor=CENTER)

menubar = Menu(window)
helps = Menu(menubar, tearoff = 0)
menubar.add_cascade(label ='Usage', menu = helps)
helps.add_command(label ='HowToUse?', command = Usage)
window.config(menu = menubar,bg='#18253f')
window.config(bg='#18253f')
window.mainloop()