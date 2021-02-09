
import sys
import os
import collections
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import re
import seaborn as sns
import math
from sklearn import neighbors, datasets
from numpy.random import permutation
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support


BASE_DIR = Path(__file__).resolve().parent.parent

class Recommender:
    # %%

    # %%

    def make_recommendation(self, GREv, GREq, GREa, CGPA):
        # cs_file = BASE_DIR/ "csv_data/student_admission_data.csv"
        # data = pd.read_csv(cs_file)
        # data.shape
        #
        # # %%
        #
        # data.drop(data.columns[data.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        # data.head()
        #
        # # %%
        #
        # data = data[data['admit'] > 0]
        # data = data.drop('admit', 1)
        #
        # data.shape
        #
        # # %%
        #
        # data = data[pd.notnull(data['greQ'])]
        # data.shape
        #
        # # %%
        #
        # data['greQ'] = data['greQ'].fillna(130)
        # data['greV'] = data['greV'].fillna(130)
        # data['greA'] = data['greA'].fillna(0)
        # data.greA.head()
        #
        # # %%
        #
        # import matplotlib.pyplot as plt
        # data.isna().sum()
        #
        # # %%
        #
        # uni_names = data['univName'].unique()
        #
        # similar_univs = pd.DataFrame({'univName': uni_names})
        # similar_univs
        #
        # # %%
        #
        # data.describe()
        #
        # # %%
        #
        #
        # data['greQ'] = self.convert_quant_score(data['greQ'])
        # data['greV'] = self.convert_verbal_score(data['greV'])
        #
        # # %%
        #
        # sns.pairplot(data, palette="husl", x_vars=["greV", "cgpa", "greQ"], y_vars=["greV", "cgpa", "greQ"], height=8)
        # plt.show()
        #
        # # %%
        #
        # # %%
        #
        # data = data.drop('gmatA', 1)
        # data = data.drop('gmatQ', 1)
        # data = data.drop('gmatV', 1)
        # data = data.drop('specialization', 1)
        # data = data.drop('department', 1)
        # data = data.drop('program', 1)
        # data = data.drop('toeflEssay', 1)
        # data = data.drop('userProfileLink', 1)
        # data = data.drop('topperCgpa', 1)
        # data = data.drop('termAndYear', 1)
        # data = data.drop('userName', 1)
        # data = data.drop('toeflScore', 1)
        # data = data.drop('major', 1)
        #
        # data = data.dropna()
        # data2 = data.drop('ugCollege', 1)
        # data2 = self.normalize_gpa(data2, 'cgpa', 'cgpaScale')
        #
        # data2 = data2.drop('industryExp', 1)
        # data2 = data2.drop('internExp', 1)
        # data2 = data2.drop('researchExp', 1)
        # data2 = data2.drop('confPubs', 1)
        # data2 = data2.drop('cgpaScale', 1)
        # # data2 = data2.drop('toeflScore', 1)
        # data2 = data2.drop('journalPubs', 1)
        # data = data2
        # university_list = list(set(data['univName'].tolist()))
        # for i in range(len(university_list)):
        #     if (len(data[data['univName'] == university_list[i]]) < 100):
        #         data = data[data['univName'] != university_list[i]]
        # data = data.dropna()
        #
        # data.head()
        #
        # # %%
        #
        # processed_data = data[['greV', 'greQ', 'greA', 'cgpa', 'univName']]
        #
        # processed_data.to_csv('Processed_data_v2_a.csv')
        # processed_data.head()

        # %%

        cs_file = BASE_DIR /"home/Processed_data_v2_a.csv"
        processed_data = pd.read_csv(cs_file)
        processed_data.index = pd.RangeIndex(start=0, step=1, stop=len(processed_data))

        # %%

        # processed_data.shape

        # %%



        # %%

        # similar_univs = pandas.read_csv('similar_universities.csv')
        # random_indices = permutation(data.index)
        # test_cutoff = math.floor(len(data) / 5)
        # print(test_cutoff)
        # test = processed_data.loc[random_indices[1:test_cutoff]]
        # train = processed_data.loc[random_indices[test_cutoff:]]
        # train_output_data = train['univName']
        # print("train Output data", train_output_data)
        # train_input_data = train
        # train_input_data = train_input_data.drop('univName', 1)
        # print("train input data", train_input_data)
        # test_output_data = test['univName']
        # print("test Output data", test_output_data)
        # test_input_data = test
        # test_input_data = test_input_data.drop('univName', 1)
        # print("test input data", test_input_data)

        # %%



        # %%

        testSet = [[GREv, GREq, GREa, CGPA]]
        test = pd.DataFrame(testSet)
        test.shape

        # %%

        # %%

        k = 5

        result, neigh = self.knn( processed_data, test, k )

        list1 = []
        list2 = []
        for i in result:
            list1.append(i[0])
            list2.append(i[1])
        for i in list1:
            print(i)


        from sklearn.neighbors import KNeighborsClassifier
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(processed_data.iloc[:, 0:4], processed_data['univName'])

        print(neigh.predict(test))
        return list1
        # %%

        # from sklearn.model_selection import train_test_split
        # from sklearn.metrics import accuracy_score
        # from sklearn.metrics import classification_report
        # from sklearn.metrics import confusion_matrix
        # from sklearn.neighbors import KNeighborsClassifier
        # from sklearn import neighbors, datasets, preprocessing
        # cs_file = "Processed_data_v2_a.csv"
        #
        # data_with_index = pd.read_csv(cs_file)
        # data_with_index.drop(data_with_index.columns[data_with_index.columns.str.contains('unnamed', case=False)], axis=1, inplace=True)
        # data_with_index.head()
        #
        # X = data_with_index.iloc[:, :-1].values
        # y = data_with_index.iloc[:, 4].values
        # Xtrain, Xtest, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1, train_size=0.8)
        # scaler = preprocessing.StandardScaler().fit(Xtrain)
        # Xtrain = scaler.transform(Xtrain)
        # Xtest = scaler.transform(Xtest)
        #
        # knn = neighbors.KNeighborsClassifier(n_neighbors=7)
        # knn.fit(Xtrain, y_train)
        # y_pred = knn.predict(Xtest)
        #
        # print(accuracy_score(y_test, y_pred))

        # %%


    # def convert_quant_score(self,quant_score):
    #     quant_list = []
    #     quant_score = quant_score.tolist()
    #     for old_quant in quant_score:
    #         if old_quant <= 170:
    #             quant_list.append(old_quant)
    #             continue
    #         else:
    #             old_quant = old_quant / 4.7
    #             if old_quant <= 130:
    #                 quant_list.append(130)
    #             else:
    #                 quant_list.append(old_quant)
    #     return quant_list
    #
    # def convert_verbal_score(self,verbal_score):
    #     verbal_list = []
    #     verbal_score = verbal_score.tolist()
    #     for old_verbal in verbal_score:
    #         if old_verbal <= 170:
    #             verbal_list.append(old_verbal)
    #             continue
    #         else:
    #             old_verbal = old_verbal / 4.7
    #             if old_verbal <= 130:
    #                 verbal_list.append(130)
    #             else:
    #                 verbal_list.append(old_verbal)
    #     return verbal_list
    #
    # def normalize_gpa(self, data2, cgpa, totalcgpa):
    #     cgpa = data2[cgpa].tolist()
    #     totalcgpa = data2[totalcgpa].tolist()
    #     for i in range(len(cgpa)):
    #         if totalcgpa[i] != 0:
    #             cgpa[i] = cgpa[i] / totalcgpa[i]
    #         else:
    #             cgpa[i] = 0
    #     data2['cgpa'] = cgpa
    #     return data2

    def euclideanDistance(self ,data1, data2, length):
        distance = 0
        for x in range(length):
            distance += np.square(data1[x] - data2[x])
        return np.sqrt(distance)

    def knn(self ,trainingSet, testInstance, k):
        print(k)
        distances = {}
        sort = {}
        length = testInstance.shape[1]

        for x in range(len(trainingSet)):
            dist = self.euclideanDistance(testInstance, trainingSet.iloc[x], length)

            distances[x] = dist[0]

        sorted_d = sorted(distances.items(), key=lambda x: x[1])

        neighbors = []

        for x in range(k):
            neighbors.append(sorted_d[x][0])

        classVotes = {}

        for x in range(len(neighbors)):
            response = trainingSet.iloc[neighbors[x]][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1

        sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)

        return (sortedVotes, neighbors)
