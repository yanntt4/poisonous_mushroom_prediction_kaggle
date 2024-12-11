# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 13:12:32 2023

@author: ythiriet
"""


# Global importation
import math
import matplotlib.pyplot as plot
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
import optuna
import random
import tensorflow as tf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance
import shap
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTEN
import joblib
import numbers

from sklearn.impute import KNNImputer

from sklearn.metrics import explained_variance_score, max_error, root_mean_squared_error
from sklearn.metrics import mean_squared_log_error, root_mean_squared_log_error
from sklearn.metrics import median_absolute_error, mean_poisson_deviance
from sklearn.metrics import mean_gamma_deviance, mean_absolute_percentage_error
from sklearn.metrics import d2_absolute_error_score
from sklearn.metrics import matthews_corrcoef


# Class containing all parameters
class Parameters():
    def __init__(self):
        self.CLEAR_MODE = True
        
        self.NAME_DATA_PREDICT = "class"
        self.GENERIC_NAME_DATA_PREDICT = "Poisonous Mushroom ?" # for plot

        self.SWITCH_REMOVING_DATA = True
        self.List_data_drop = ["DepTime"]
        self.SWITCH_DATA_REDUCTION = False
        self.SWITCH_DATA_NOT_ENOUGHT = False
        self.NB_DATA_NOT_ENOUGHT = 1500
        self.SWITCH_ABERRANT_IDENTICAL_DATA = True
        self.SWITCH_RELATION_DATA = False
        self.ARRAY_RELATION_DATA = np.array([["Height", 2],["Age", 2]], dtype = object)
        self.SWITCH_ENCODE_DATA_PREDICT = True
        self.ARRAY_DATA_ENCODE_PREDICT = np.array([[self.NAME_DATA_PREDICT,"p",1],[self.NAME_DATA_PREDICT,"e",0]], dtype = object)
        self.SWITCH_ENCODE_DATA = True
        self.SWITCH_ENCODE_DATA_ONEHOT = False
        self.LIST_DATA_ENCODE = ["Geography", "Gender"]
        self.ARRAY_DATA_ENCODE_REPLACEMENT = np.zeros(9, dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[0] = np.array(
            [["season","u",1,"printemps"],["season","s",2,"ete"],["season","a",3,"automne"],["season","w",4,"hiver"]], dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[1] = np.array(
            [["does-bruise-or-bleed","t",1,"visqueux"],["does-bruise-or-bleed","f",0,"fluide"]], dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[2] = np.array(
            [["has-ring","t",1,"oui"],["has-ring","f",0,"non"]], dtype = object)
        self.ARRAY_DATA_ENCODE_REPLACEMENT[3] = np.array(
            [["cap-shape","f",1,"membraneux"],["cap-shape","x",2,"obconique"],["cap-shape","p",3,"turbune"],
             ["cap-shape","b",4,"cornucopie"],["cap-shape","o",5,"infundibuliforme"],["cap-shape","c",6,"campanule"],
             ["cap-shape","s",7,"conique"],["cap-shape","d",8, "convexe"],["cap-shape","e",9,"deprime"],
             ["cap-shape","n",10,"plat"],["cap-shape","w",11,"gibbeux"],["cap-shape","k",12,"ogival"],
             ["cap-shape","l",13,"excentre"],["cap-shape","t",14,"ovale"],["cap-shape","g",15,"ombilique"],
             ["cap-shape","z",16,"umbone"],["cap-shape","a",17,"papille"],["cap-shape","r",18,"flabelliforme"],
             ["cap-shape","u",19,"petaloide"],["cap-shape","y",20,"pulvine"],["cap-shape","i",21,"cylindrique"],
             ["cap-shape","m",22,"coalescents"],["cap-shape","h",23,"autre"]], dtype = object)
        
        self.ARRAY_DATA_ENCODE_REPLACEMENT[4] = np.array(
            [["cap-color","u",1,"brun"],["cap-color","o",2,"orange"],["cap-color","b",3,"noir"],
             ["cap-color","g",4,"vert"],["cap-color","w",5,"blanc"],["cap-color","n",6,"bleu fonce"],
             ["cap-color","e",7,"emeraude"],["cap-color","y",8,"jaune"],["cap-color","r",9,"rouge"],
             ["cap-color","p",10,"rose"],["cap-color","k",11,"kaki"],["cap-color","l",12,"lavande"],
             ["cap-color","f",13,"fuchsia"],["cap-color","d",14,"dallas"],["cap-color","i",15,"indigo"],
             ["cap-color","h",16,"or"],["cap-color","s",17,"saumon"],["cap-color","a",18,"ambre"],
             ["cap-color","z",19,"turquoise"],["cap-color","t",20,"bleu"],["cap-color","x",21,"violet"],
             ["cap-color","season",22,"gris"],["cap-color","c",23,"chocolat"],["cap-color","m",24,"autre"]], dtype = object)

        self.ARRAY_DATA_ENCODE_REPLACEMENT[5] = np.array(
            [["gill-color","u",1,"brun"],["gill-color","o",2,"orange"],["gill-color","b",3,"noir"],
             ["gill-color","g",4,"vert"],["gill-color","w",5,"blanc"],["gill-color","n",6,"bleu fonce"],
             ["gill-color","e",7,"emeraude"],["gill-color","y",8,"jaune"],["gill-color","r",9,"rouge"],
             ["gill-color","p",10,"rose"],["gill-color","k",11,"kaki"],["gill-color","l",12,"lavande"],
             ["gill-color","f",13,"fuchsia"],["gill-color","d",14,"dallas"],["gill-color","i",15,"indigo"],
             ["gill-color","h",16,"or"],["gill-color","s",17,"saumon"],["gill-color","a",18,"ambre"],
             ["gill-color","z",19,"turquoise"],["gill-color","t",20,"bleu"],["gill-color","x",21,"violet"],
             ["gill-color","season",22,"gris"],["gill-color","c",23,"chocolat"],["gill-color","m",24,"autre"]], dtype = object)
        
        self.ARRAY_DATA_ENCODE_REPLACEMENT[6] = np.array(
            [["stem-color","u",1,"brun"],["stem-color","o",2,"orange"],["stem-color","b",3,"noir"],
             ["stem-color","g",4,"vert"],["stem-color","w",5,"blanc"],["stem-color","n",6,"bleu fonce"],
             ["stem-color","e",7,"emeraude"],["stem-color","y",8,"jaune"],["stem-color","r",9,"rouge"],
             ["stem-color","p",10,"rose"],["stem-color","k",11,"kaki"],["stem-color","l",12,"lavande"],
             ["stem-color","f",13,"fuchsia"],["stem-color","d",14,"dallas"],["stem-color","i",15,"indigo"],
             ["stem-color","h",16,"or"],["stem-color","s",17,"saumon"],["stem-color","a",18,"ambre"],
             ["stem-color","z",19,"turquoise"],["stem-color","t",20,"bleu"],["stem-color","x",21,"violet"],
             ["stem-color","season",22,"gris"],["stem-color","c",23,"chocolat"],["stem-color","m",24,"autre"]], dtype = object)

        self.ARRAY_DATA_ENCODE_REPLACEMENT[7] = np.array(
            [["habitat","d",1,"marais"],["habitat","l",2,"marecage"],["habitat","g",3,"tourbiere"],
             ["habitat","h",4,"prairie"],["habitat","p",5,"plaine"],["habitat","m",6,"vallee"],
             ["habitat","u",7,"ruisseau"],["habitat","w",8,"mangrove"],["habitat","y",9,"montagne"],
             ["habitat","a",10,"foret"],["habitat","s",11,"jungle"],["habitat","k",12,"desert"],
             ["habitat","z",13,"savane"],["habitat","b",14,"steppe"],["habitat","t",15,"toundra"],
             ["habitat","c",16,"taiga"],["habitat","e",17,"ville"],["habitat","n",18,"friche"],
             ["habitat","r",19,"sol pollue"],["habitat","f",20,"neige"],["habitat","o",21,"grotte"],
             ["habitat","x",22,"caverne"],["habitat","i",23,"autre"]], dtype = object)
        
        self.ARRAY_DATA_ENCODE_REPLACEMENT[8] = np.array(
            [["ring-type","f",1,"carre"],["ring-type","z",2,"rectanglulaire"],["ring-type","e",3,"circulaire"],
              ["ring-type","p",4,"simple anneau"],["ring-type","l",5,"double anneau"],["ring-type","g",6,"triple anneau"],
              ["ring-type","r",7,"elliptique"],["ring-type","m",8,"oval"],["ring-type","y",9,"demi cercle"],
              ["ring-type","h",10,"trapeze"],["ring-type","o",11,"parallelogramme"],["ring-type","t",12,"pentagone"],
              ["ring-type","a",13,"hexagone"],["ring-type","d",14,"croix"],["ring-type","s",15,"etoile"],
              ["ring-type","x",16,"triangle"],["ring-type","n",17,"diamant"],["ring-type","u",18,"heptagone"],
              ["ring-type","w",19,"octogone"],["ring-type","b",20,"nuage"],["ring-type","k",21,"coeur"],
              ["ring-type","c",22,"feuille"],["ring-type","i",23,"autre"]], dtype = object)

        
        self.SWITCH_PLOT_DATA = True
        self.SWITCH_EQUILIBRATE_DATA = False
        self.SWITCH_SMOTEN_DATA = False
        self.SWITCH_REPLACING_NAN = False
        self.SWITCH_SAMPLE_DATA = False
        self.Fraction_Sample_Data = 0.5

        self.RF_MODEL = False
        self.RF_MODEL_OPTI = False
        self.RF_MODEL_TRIAL = 60
        
        self.GB_MODEL = False
        self.GB_MODEL_OPTI = False
        self.GB_MODEL_TRIAL = 50

        self.NN_MODEL = False
        self.NN_MODEL_OPTI = False
        self.NN_MODEL_TRIAL = 50

        self.XG_MODEL = True
        self.XG_MODEL_OPTI = False
        self.XG_MODEL_TRIAL = 500

        self.MULTI_CLASSIFICATION = False

        self.N_SPLIT = 5
        self.k_folds = KFold(n_splits=self.N_SPLIT)


    # Determining if multi-classification
    def Multi_Classification_Analysis(self, UNIQUE_PREDICT_VALUE):

        if UNIQUE_PREDICT_VALUE.shape[0] > 2:
            self.MULTI_CLASSIFICATION = True
    
    
    def regression_analysis(self, TRAIN_DATAFRAME):
        if isinstance(TRAIN_DATAFRAME[self.NAME_DATA_PREDICT][0], numbers.Number):
            self.REGRESSION = True
        else:
            self.REGRESSION = False
    
    def saving_array_replacement(self):
        joblib.dump(self.ARRAY_DATA_ENCODE_REPLACEMENT, "./data_replacement/array_data_encode_replacement.joblib")
        

class Data_Preparation():
    def __init__(self):
        self.TRAIN_DATAFRAME = []
        self.TEST_DATAFRAME = []
        self.TRAIN_STATS = []
        self.UNIQUE_PREDICT_VALUE = []
        self.TRAIN_CORRELATION = []
        self.DUPLICATE_LINE = []

        self.ARRAY_REPLACEMENT_ALL = np.zeros([0], dtype = object)
        self.INDEX_REPLACEMENT_ALL = np.zeros([0], dtype = object)


    def data_import(self, NAME_DATA_PREDICT):

        self.TRAIN_DATAFRAME = pd.read_csv("./data/train.csv")
        self.TEST_DATAFRAME = pd.read_csv("./data/test.csv")
        self.TRAIN_STATS = self.TRAIN_DATAFRAME.describe()


    def real_value_replacement(self):
        TABLE_REPLACEMENT = np.array([["9E","Easyjet"], ["AA","AirFrance"], ["AQ","Ryanair"], ["AS","Vueling"],
                             ["B6","Transavia"], ["CO","Lufthansa"], ["DH","Iberia"], ["DL","Wizzair"],
                             ["EV","Etihad"], ["F9","Emirates"], ["FL","Luxair"], ["HA","Peach"],
                             ["HP","StarFlyer"], ["MQ","MyAir"], ["NW","CityJet"], ["OH","AirIndia"],
                             ["OO","Afrijet"], ["TZ","HOP"], ["UA","UnitedAirlines"], ["US","ContinentalAirlines"],
                             ["WN","AmericanAirlines"], ["XE","BritishAirways"], ["YV","Aeroflot"]])

        for i in range(TABLE_REPLACEMENT.shape[0]):
            self.TRAIN_DATAFRAME["UniqueCarrier"] = self.TRAIN_DATAFRAME["UniqueCarrier"].replace(
                TABLE_REPLACEMENT[i,0], TABLE_REPLACEMENT[i,1])
            self.TEST_DATAFRAME["UniqueCarrier"] = self.TEST_DATAFRAME["UniqueCarrier"].replace(
                TABLE_REPLACEMENT[i,0], TABLE_REPLACEMENT[i,1])


    def data_predict_description(self, NAME_DATA_PREDICT):
        self.UNIQUE_PREDICT_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        
        # Printing first values
        print(self.TRAIN_DATAFRAME.head())


    def data_encoding_replacement(self, ARRAY_REPLACEMENT, NAN_VALUES = False):
        
        for i_encoding, DataFrame in enumerate([self.TRAIN_DATAFRAME, self.TEST_DATAFRAME]):
    
            # Replacement
            for j in range(ARRAY_REPLACEMENT.shape[0]):
                for k in range(ARRAY_REPLACEMENT[j].shape[0]):
                    DataFrame[ARRAY_REPLACEMENT[j][k][0]] = DataFrame[ARRAY_REPLACEMENT[j][k][0]].replace(
                        ARRAY_REPLACEMENT[j][k][1], int(ARRAY_REPLACEMENT[j][k][2]))
                
            # Replacing nan values
            if NAN_VALUES:
                DataFrame[ARRAY_REPLACEMENT[j][0][0]] = DataFrame[ARRAY_REPLACEMENT[j][0][0]].fillna(0)

            # Recording the replacement
            if i_encoding == 0:
                self.TRAIN_DATAFRAME = DataFrame
            else:
                self.TEST_DATAFRAME = DataFrame


    def data_encoding_replacement_important(self, COLUMN_NAME):

        # Init
        self.ARRAY_REPLACEMENT_ALL = np.append(
            self.ARRAY_REPLACEMENT_ALL, np.zeros([1], dtype = object), axis = 0)
        self.INDEX_REPLACEMENT_ALL = np.append(
            self.INDEX_REPLACEMENT_ALL, np.zeros([1], dtype = object), axis = 0)

        DF_TRAIN_TEST = pd.concat([Global_Data.TRAIN_DATAFRAME, Global_Data.TEST_DATAFRAME],
                                  ignore_index = True)
        UNIQUE_DF_TRAIN_TEST = DF_TRAIN_TEST.groupby(COLUMN_NAME)[COLUMN_NAME].count()
        ARRAY_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).to_numpy()
        INDEX_REPLACEMENT = pd.DataFrame(UNIQUE_DF_TRAIN_TEST.index).index.tolist()

        for i_encoding, DataFrame in enumerate([self.TRAIN_DATAFRAME, self.TEST_DATAFRAME]):

            # Replacement
            for j in range(ARRAY_REPLACEMENT.shape[0]):
                DataFrame[COLUMN_NAME] = DataFrame[COLUMN_NAME].replace(
                    ARRAY_REPLACEMENT[j], INDEX_REPLACEMENT[j])

            # Recording the replacement
            if i_encoding == 0:
                self.TRAIN_DATAFRAME[COLUMN_NAME] = DataFrame[COLUMN_NAME]
            else:
                self.TEST_DATAFRAME[COLUMN_NAME] = DataFrame[COLUMN_NAME]

        # Recording the replacement
        self.ARRAY_REPLACEMENT_ALL[-1] = ARRAY_REPLACEMENT
        self.INDEX_REPLACEMENT_ALL[-1] = INDEX_REPLACEMENT


    def data_encoding_replacement_predict(self, ARRAY_REPLACEMENT):
        for j in range(ARRAY_REPLACEMENT.shape[0]):
            self.TRAIN_DATAFRAME[ARRAY_REPLACEMENT[j,0]] = self.TRAIN_DATAFRAME[ARRAY_REPLACEMENT[j,0]].replace(
                ARRAY_REPLACEMENT[j,1],ARRAY_REPLACEMENT[j,2])


    def data_encoding_onehot(self, Name_Data_Encode):
        Enc = OneHotEncoder(handle_unknown='ignore')
        Data_Encode_Train = self.TRAIN_DATAFRAME.loc[:,[Name_Data_Encode]]
        Data_Encode_Test = self.TEST_DATAFRAME.loc[:,[Name_Data_Encode]]
        Data_Encode_Name = Name_Data_Encode + Data_Encode_Train.groupby(Name_Data_Encode)[Name_Data_Encode].count().index
        Enc.fit(Data_Encode_Train)

        Data_Encode_Train = Enc.transform(Data_Encode_Train).toarray()
        Data_Encode_Train = pd.DataFrame(Data_Encode_Train,
                                         columns = Data_Encode_Name)

        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(columns = Name_Data_Encode)
        self.TRAIN_DATAFRAME = pd.concat([self.TRAIN_DATAFRAME, Data_Encode_Train],
                                         axis = 1)

        Data_Encode_Test = Enc.transform(Data_Encode_Test).toarray()
        Data_Encode_Test = pd.DataFrame(Data_Encode_Test,
                                         columns = Data_Encode_Name)
        Data_Encode_Test = Data_Encode_Test.set_index(self.TEST_DATAFRAME.index)

        self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(columns = Name_Data_Encode)
        self.TEST_DATAFRAME = pd.concat([self.TEST_DATAFRAME, Data_Encode_Test],
                                         axis = 1)
    
    
    def encode_data_error_removal(self, ARRAY_REPLACEMENT):
        for ARRAY in ARRAY_REPLACEMENT:
            Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]] = pd.to_numeric(
                Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]],errors="coerce", downcast = 'integer')
    
    
    def data_format_removal(self, ARRAY_REPLACEMENT, Type = str, Len = 2):
        for ARRAY in ARRAY_REPLACEMENT:
            Global_Data.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME.loc[(
                Global_Data.TRAIN_DATAFRAME[ARRAY[0][0]].astype(Type).str.len() < Len)]
        

    def data_drop(self, Name_data_drop):
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop([Name_data_drop],axis=1)
        self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop([Name_data_drop],axis=1)


    def data_pow(self, Name_Data_Duplicate, Number_Duplicate):
        self.TRAIN_DATAFRAME[Name_Data_Duplicate] = (
            self.TRAIN_DATAFRAME[Name_Data_Duplicate].pow(Number_Duplicate))
        self.TEST_DATAFRAME[Name_Data_Duplicate] = (
            self.TEST_DATAFRAME[Name_Data_Duplicate].pow(Number_Duplicate))


    def data_duplicate_removal(self, NAME_DATA_PREDICT, Column_Drop = ""):

        if len(Column_Drop) == 0:
            Duplicated_Data_All = self.TRAIN_DATAFRAME.drop(NAME_DATA_PREDICT, axis = 1).duplicated()
        else:
            Duplicated_Data_All = self.TRAIN_DATAFRAME.drop([Column_Drop, NAME_DATA_PREDICT],axis = 1).duplicated()
        self.DUPLICATE_LINE = Duplicated_Data_All.loc[Duplicated_Data_All == True]
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.DUPLICATE_LINE.index)
        
        # Information to the user
        print(f"{self.DUPLICATE_LINE.shape[0]} has been removed because of duplicates")
        plot.pause(3)


    def remove_low_data(self, NB_DATA_NOT_ENOUGHT, NAME_DATA_NOT_ENOUGHT, LIST_NAME_DATA_REMOVE_MULTIPLE = []):

        # Searching for data with low values
        TRAIN_GROUP_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_NOT_ENOUGHT)[NAME_DATA_NOT_ENOUGHT].count().sort_values(ascending = False)

        # Adding values only inside NAME DATA REMOVE MULTIPLE
        for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
            TRAIN_GROUP_VALUE_OTHER = self.TRAIN_DATAFRAME.groupby(NAME_DATA_REMOVE_MULTIPLE)[NAME_DATA_REMOVE_MULTIPLE].count().index

        for VALUE_OTHER in TRAIN_GROUP_VALUE_OTHER:
            if np.sum(VALUE_OTHER == np.array(TRAIN_GROUP_VALUE.index)) == 0:
                TRAIN_GROUP_VALUE = pd.concat([TRAIN_GROUP_VALUE, pd.Series(0, index = [VALUE_OTHER])])

        # Searching for values to drop following number of elements
        REMOVE_TRAIN_GROUP_VALUE = TRAIN_GROUP_VALUE.drop(TRAIN_GROUP_VALUE[TRAIN_GROUP_VALUE > NB_DATA_NOT_ENOUGHT].index)

        # Removing data inside train and test dataframe
        for DATA_REMOVE in REMOVE_TRAIN_GROUP_VALUE.index:
            self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.TRAIN_DATAFRAME[self.TRAIN_DATAFRAME[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)
            self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(self.TEST_DATAFRAME[self.TEST_DATAFRAME[NAME_DATA_NOT_ENOUGHT] == DATA_REMOVE].index)

            for NAME_DATA_REMOVE_MULTIPLE in LIST_NAME_DATA_REMOVE_MULTIPLE:
                self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.drop(self.TRAIN_DATAFRAME[self.TRAIN_DATAFRAME[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)
                self.TEST_DATAFRAME = self.TEST_DATAFRAME.drop(self.TEST_DATAFRAME[self.TEST_DATAFRAME[NAME_DATA_REMOVE_MULTIPLE] == DATA_REMOVE].index)

        # Reseting index
        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.reset_index(drop = True)
        self.TEST_DATAFRAME = self.TEST_DATAFRAME.reset_index(drop = True)


    def oversampling(self, NAME_DATA_PREDICT, NB_DATA_NOT_ENOUGHT, Name_Data_Oversample = ""):

        self.UNIQUE_PREDICT_VALUE = self.TRAIN_DATAFRAME.groupby(NAME_DATA_PREDICT)[NAME_DATA_PREDICT].count()
        Max_Nb_Data = np.amax(self.UNIQUE_PREDICT_VALUE.to_numpy())

        if len(Name_Data_Oversample) > 1:
            Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[(
                self.UNIQUE_PREDICT_VALUE.index == "Overweight_Level_II")]
        else:
            # Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[
            #     self.UNIQUE_PREDICT_VALUE > NB_DATA_NOT_ENOUGHT]
            Global_Table_Train_Equilibrate = self.UNIQUE_PREDICT_VALUE.loc[self.UNIQUE_PREDICT_VALUE < Max_Nb_Data]

        for i in range(Global_Table_Train_Equilibrate.shape[0]):
            Matrix_To_Add = np.zeros(
                [0, self.TRAIN_DATAFRAME.shape[1]],
                dtype=object)
            DF_Reference = self.TRAIN_DATAFRAME.loc[self.TRAIN_DATAFRAME[NAME_DATA_PREDICT] == pd.DataFrame(
                Global_Table_Train_Equilibrate.index).iloc[i][0]]
            for j in range(Max_Nb_Data - Global_Table_Train_Equilibrate.iloc[i]):
                Matrix_To_Add = np.append(
                    Matrix_To_Add,
                    np.zeros([1, self.TRAIN_DATAFRAME.shape[1]],
                              dtype=object),
                    axis=0)

                Matrix_To_Add[-1, :] = DF_Reference.iloc[
                    random.randint(0, DF_Reference.shape[0] - 1), :].to_numpy()

            DataFrame_To_Add = pd.DataFrame(
                Matrix_To_Add,
                columns=self.TRAIN_DATAFRAME.columns)

            self.TRAIN_DATAFRAME = pd.concat(
                [self.TRAIN_DATAFRAME, DataFrame_To_Add],
                ignore_index=True)


    def data_sample(self, Sample_Fraction):

        self.TRAIN_DATAFRAME = self.TRAIN_DATAFRAME.sample(
            frac = Sample_Fraction, replace = False, random_state = 42)


    def nan_replacing(self, COLUMN_NAMES):
        

        # Creating a column indicating missing value
        for COLUMN in COLUMN_NAMES:
            self.TRAIN_DATAFRAME[f"{COLUMN} Missing"] = self.TRAIN_DATAFRAME[f"{COLUMN}"].isnull()
        
        # Replacing missing values with nearest neightboor
        COLUMNS = self.TRAIN_DATAFRAME.columns
        imputer = KNNImputer(n_neighbors=20, weights="uniform")
        TRAIN_ARRAY = imputer.fit_transform(self.TRAIN_DATAFRAME)
        
        # Turning numpy array into dataframe
        self.TRAIN_DATAFRAME = pd.DataFrame(TRAIN_ARRAY, columns = COLUMNS)
    
    
    def saving_data_names(self):
        joblib.dump(self.TEST_DATAFRAME.columns, "./data_replacement/data_names.joblib")
        

class Data_Plot():
    def __init__(self):
        self.Box_Plot_Data_Predict = ""
        self.Box_Plot_Data_Available = ""
        self.Correlation_Plot = ""
        self.TRAIN_DATAFRAME = []
        self.TRAIN_CORRELATION = []
        self.UNIQUE_PREDICT_VALUE = []


    def Box_Plot_Data_Predict_Plot(
            self, GENERIC_NAME_DATA_PREDICT):

        # Init
        fig, self.Box_Plot_Data_Predict = plot.subplots(2)
        plot.suptitle(f"Data count following {GENERIC_NAME_DATA_PREDICT}",
                      fontsize = 25,
                      color = "gold",
                      fontweight = "bold")

        # Horizontal bars for each possibilities
        self.Box_Plot_Data_Predict[0].barh(
            y = self.UNIQUE_PREDICT_VALUE.index,
            width=self.UNIQUE_PREDICT_VALUE,
            height=0.03,
            label=self.UNIQUE_PREDICT_VALUE.index)

        # Cumulative horizontal bars
        Cumulative_Value = 0
        for i in range(self.UNIQUE_PREDICT_VALUE.shape[0]):
            self.Box_Plot_Data_Predict[1].barh(
                y=1,
                width=self.UNIQUE_PREDICT_VALUE.iloc[i],
                left = Cumulative_Value)
            self.Box_Plot_Data_Predict[1].text(
                x = Cumulative_Value + 100,
                y = 0.25,
                s = self.UNIQUE_PREDICT_VALUE.index[i])
            Cumulative_Value += self.UNIQUE_PREDICT_VALUE.iloc[i]
        self.Box_Plot_Data_Predict[1].set_ylim(0, 2)
        self.Box_Plot_Data_Predict[1].legend(
            self.UNIQUE_PREDICT_VALUE.index.to_numpy(),
            ncol=int(self.UNIQUE_PREDICT_VALUE.shape[0]/2),
            fontsize=6)


    def plot_data_repartition(self):

        # Init
        NB_LINE = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)
        NB_COLUMN = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)

        # Box Plot for all data
        fig, self.Box_Plot_Data_Available = plot.subplots(NB_LINE, NB_COLUMN)
        plot.suptitle("Boxplot for all data into the TRAIN dataset",
                      fontsize = 25,
                      color = "chartreuse",
                      fontweight = "bold")

        for i in range(NB_LINE):
            for j in range(NB_COLUMN):
                if i*NB_COLUMN + j < self.TRAIN_DATAFRAME.shape[1]:
                    try:
                        self.Box_Plot_Data_Available[i, j].boxplot(
                            self.TRAIN_DATAFRAME.iloc[:, [i*NB_COLUMN + j]])
                        self.Box_Plot_Data_Available[i, j].set_title(
                            self.TRAIN_DATAFRAME.iloc[:, i*NB_COLUMN + j].name,
                            fontweight = "bold",
                            fontsize = 15)
                    except:
                        continue
    
    
    def plot_data_hist(self):

        # Init
        NB_LINE = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)
        NB_COLUMN = math.ceil((self.TRAIN_DATAFRAME.shape[1] - 1)/3)

        # Box Plot for all data
        fig, self.Box_Plot_Data_Available = plot.subplots(NB_LINE, NB_COLUMN)
        plot.suptitle("Boxplot for all data into the TRAIN dataset",
                      fontsize = 25,
                      color = "chartreuse",
                      fontweight = "bold")

        for i in range(NB_LINE):
            for j in range(NB_COLUMN):
                if (i*NB_COLUMN + j < self.TRAIN_DATAFRAME.shape[1]):
                    try:
                        self.Box_Plot_Data_Available[i, j].hist(
                            self.TRAIN_DATAFRAME.iloc[:, [i*NB_COLUMN + j]],
                            bins = 100)
                        self.Box_Plot_Data_Available[i, j].set_title(
                            self.TRAIN_DATAFRAME.iloc[:, i*NB_COLUMN + j].name,
                            fontweight = "bold",
                            fontsize = 15)
                    except:
                        continue


    def Plot_Data_Relation(self, Name_Data_x, Name_Data_y):

        plot.figure()
        plot.scatter(self.TRAIN_DATAFRAME[Name_Data_x],
                     self.TRAIN_DATAFRAME[Name_Data_y])
        plot.suptitle(
            f"Relation between {Name_Data_x} and {Name_Data_y} variables",
            fontsize = 25,
            color = "darkorchid",
            fontweight = "bold")


    def Correlation_Plot_Plot(self):

        fig2, self.Correlation_Plot = plot.subplots()
        im = self.Correlation_Plot.imshow(
            self.TRAIN_CORRELATION,
            vmin=-1,
            vmax=1,
            cmap="bwr")
        self.Correlation_Plot.figure.colorbar(im, ax=self.Correlation_Plot)
        self.Correlation_Plot.set_xticks(np.linspace(
            0, self.TRAIN_DATAFRAME.shape[1] - 1, self.TRAIN_DATAFRAME.shape[1]))
        self.Correlation_Plot.set_xticklabels(np.array(self.TRAIN_DATAFRAME.columns, dtype = str),
                                              rotation = 45)
        self.Correlation_Plot.set_yticks(np.linspace(
            0, self.TRAIN_DATAFRAME.shape[1] - 1, self.TRAIN_DATAFRAME.shape[1]))
        self.Correlation_Plot.set_yticklabels(np.array(self.TRAIN_DATAFRAME.columns, dtype = str))


class Data_Modelling():
    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.X_test = []
        self.Y_test = []
        self.Y_predict = []
        self.Y_predict_proba = []
        self.K_predict = []
        self.K_predict_proba = []

        self.MODEL = ""
        self.Model_Name = ""
        self.Y_predict = []
        self.Y_test = []
        self.Nb_Correct_Prediction = 0
        self.Percentage_Correct_Prediction = 0
        self.score = 0
        self.Best_Params = np.zeros([1], dtype = object)


    def Splitting_Data(self, TRAIN_DATAFRAME, GENERIC_NAME_DATA_PREDICT, MULTI_CLASSIFICATION, REGRESSION):

        # Split data creation
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
            TRAIN_DATAFRAME.drop([GENERIC_NAME_DATA_PREDICT], axis=1),
            TRAIN_DATAFRAME.loc[:, [GENERIC_NAME_DATA_PREDICT]].iloc[:,0],
            test_size=0.2,
            random_state=0)

        # Turning Y_train and Y_test into boolean if needed
        if MULTI_CLASSIFICATION == False and REGRESSION == False:
            self.Y_train = self.Y_train.astype(bool)
            self.Y_test = self.Y_test.astype(bool)


    def Smoten_Sampling(self):

        sampler = SMOTEN(random_state = 0)
        self.X_train, self.Y_train = sampler.fit_resample(self.X_train, self.Y_train)
    
    
    def extract_max_diff_regression(self):
        
        # Finding data with highest difference between real and prediction
        X_TEST_EXTREMA_ANALYSIS = self.X_test.copy()
        X_TEST_EXTREMA_ANALYSIS["Real"] = self.Y_test
        X_TEST_EXTREMA_ANALYSIS["Predicted"] = self.Y_predict
        X_TEST_EXTREMA_ANALYSIS["Difference"] = abs(self.Y_test - self.Y_predict)
        self.X_TEST_EXTREMA_INDEX = X_TEST_EXTREMA_ANALYSIS.sort_values(by = ["Difference"]).nlargest(10, "Difference").index
        
        # Data standardisation using max value
        self.X_TEST_STANDARD = self.X_test.copy()
        
        for COLUMN in self.X_test.columns:
            self.X_TEST_STANDARD[COLUMN] = self.X_TEST_STANDARD[COLUMN]/self.X_TEST_STANDARD[COLUMN].max()
        
        # Finiding closest points using euclidian distance
        X_TEST_STANDARD_EXTREMA = self.X_TEST_STANDARD[self.X_TEST_STANDARD.index == self.X_TEST_EXTREMA_INDEX[0]]
        
        EUCLIDIAN_DISTANCE_CALCULATION_ARRAY = np.zeros([self.X_TEST_STANDARD.shape[0]], dtype = float)
        X_TEST_STANDARD_ARRAY = np.array(self.X_TEST_STANDARD)
        X_TEST_STANDARD_EXTREMA_ARRAY = np.array(X_TEST_STANDARD_EXTREMA)
        
        for i in range(X_TEST_STANDARD_ARRAY.shape[0]):
            for j in range(X_TEST_STANDARD_ARRAY.shape[1]):
                EUCLIDIAN_DISTANCE_CALCULATION_ARRAY[i] += (
                    (X_TEST_STANDARD_ARRAY[i,j] - X_TEST_STANDARD_EXTREMA_ARRAY[0,j])*(X_TEST_STANDARD_ARRAY[i,j] - X_TEST_STANDARD_EXTREMA_ARRAY[0,j]))
        
        self.X_TEST_STANDARD["Euclidian Distance"] = EUCLIDIAN_DISTANCE_CALCULATION_ARRAY
        self.X_TEST_CLOSEST_EXTREMA = self.X_test.loc[np.array(self.X_TEST_STANDARD.sort_values(by = ["Euclidian Distance"]).nlargest(6, "Euclidian Distance").index)]
        self.X_TEST_CLOSEST_EXTREMA["SalePrice"] = self.Y_test.loc[np.array(self.X_TEST_STANDARD.sort_values(by = ["Euclidian Distance"]).nlargest(6, "Euclidian Distance").index)]
        

    def result_plot_classification(self):

        # Plotting results
        X_plot = np.linspace(1, self.Y_test.shape[0], self.Y_test.shape[0])

        fig, ax = plot.subplots(2)
        ax[0].scatter(X_plot, self.Y_predict, color="green")
        ax[0].scatter(X_plot, self.Y_test, color="orange")
        ax[0].legend([f"Prediction from {self.Model_Name} Model", "Real Results"])
        ax[1].scatter(X_plot, np.sort(abs(self.Y_test.astype(int) - self.Y_predict.astype(int))))
        ax[1].set_title("Difference between predict and real results")
        ax[1].legend([f"Score for {self.Model_Name} Model : {round(self.Percentage_Correct_Prediction,2)}"])
        plot.grid()
    
    
    def result_plot_regression(self):
        
        # Preparation
        self.Y_predict = np.array(self.Y_predict)
        if self.Y_predict.ndim == 2:
            self.Y_predict = self.Y_predict[:,0]

        # Plotting results
        X_plot = np.linspace(1, self.Y_test.shape[0], self.Y_test.shape[0])

        fig, ax = plot.subplots(3)
        ax[0].scatter(X_plot, self.Y_predict, color="green")
        ax[0].scatter(X_plot, self.Y_test, color="orange")
        ax[0].legend([f"Prediction from {self.Model_Name} Model", "Real Results"])
        ax[1].scatter(X_plot, np.sort(abs(self.Y_test - self.Y_predict)))
        ax[1].set_title("Difference between predict and real results")
        ax[1].legend([f"Score for {self.Model_Name} Model : {round(self.AVERAGE_DIFFERENCE,2)}"])
        plot.grid()
        ax[2].boxplot(abs(self.Y_test - self.Y_predict), vert = False)


    def result_report_classification_calculation(self):
        self.CONFUSION_MATRIX = confusion_matrix(self.Y_test, self.Y_predict)
        self.ACCURACY = accuracy_score(self.Y_test, self.Y_predict)
        self.BALANCED_ACCURACY = balanced_accuracy_score(self.Y_test, self.Y_predict)
        
        # Precision
        self.MICRO_PRECISION = precision_score(self.Y_test, self.Y_predict, average='micro')
        self.MACRO_PRECISION = precision_score(self.Y_test, self.Y_predict, average='macro')
        self.WEIGHTED_PRECISION = precision_score(self.Y_test, self.Y_predict, average='weighted')
        
        # Recall
        self.MICRO_RECALL = recall_score(self.Y_test, self.Y_predict, average='micro')
        self.MACRO_RECALL = recall_score(self.Y_test, self.Y_predict, average='macro')
        self.WEIGHTED_RECALL = recall_score(self.Y_test, self.Y_predict, average='weighted')
        
        # F1-score
        self.MICRO_F1_SCORE = f1_score(self.Y_test, self.Y_predict, average='micro')
        self.MACRO_F1_SCORE = f1_score(self.Y_test, self.Y_predict, average='macro')
        self.WEIGHTED_F1_SCORE = f1_score(self.Y_test, self.Y_predict, average='weighted')
        
        # Matthews correlation coefficient (MCC)
        self.MCC = matthews_corrcoef(self.Y_test, self.Y_predict)


    def result_report_classification_print(self):

        print('\n------------------ Confusion Matrix -----------------\n')
        print(self.CONFUSION_MATRIX)

        print('\n-------------------- Key Metrics --------------------')
        print('\nAccuracy: {:.3f}'.format(self.ACCURACY))
        print('Balanced Accuracy: {:.3f}\n'.format(self.BALANCED_ACCURACY))

        print('Micro Precision: {:.3f}'.format(self.MICRO_PRECISION))
        print('Micro Recall: {:.3f}'.format(self.MICRO_RECALL))
        print('Micro F1-score: {:.3f}\n'.format(self.MICRO_F1_SCORE))

        print('Macro Precision: {:.3f}'.format(self.MACRO_PRECISION))
        print('Macro Recall: {:.3f}'.format(self.MACRO_RECALL))
        print('Macro F1-score: {:.3f}\n'.format(self.MACRO_F1_SCORE))

        print('Weighted Precision: {:.3f}'.format(self.WEIGHTED_PRECISION))
        print('Weighted Recall: {:.3f}'.format(self.WEIGHTED_RECALL))
        print('Weighted F1-score: {:.3f}\n'.format(self.WEIGHTED_F1_SCORE))
        
        print('Matthews correlation coefficient: {:.3f}\n'.format(self.MCC))

        print('\n--------------- Classification Report ---------------\n')
        print(classification_report(self.Y_test, self.Y_predict))

        print('\n--------------- Imbalanced Report ---------------\n')
        print(classification_report_imbalanced(self.Y_test, self.Y_predict))


    def result_report_classification_plot(self):

        plot.figure(figsize = (10,8))
        plot.ylim(1,40)
        plot.text(0.02,39,'------------------ Confusion Matrix -----------------')
        plot.text(0.02,29, self.CONFUSION_MATRIX)
        plot.text(0.4,28,'-------------------- Key Metrics --------------------')
        plot.text(0.4,26,'Accuracy: {:.3f}'.format(self.ACCURACY))
        plot.text(0.4,24,'Balanced Accuracy: {:.3f}\n'.format(self.BALANCED_ACCURACY))
        plot.text(0.4,22,'Micro Precision: {:.3f}'.format(self.MICRO_PRECISION))
        plot.text(0.4,20,'Micro Recall: {:.3f}'.format(self.MICRO_RECALL))
        plot.text(0.4,18,'Micro F1-score: {:.3f}\n'.format(self.MICRO_F1_SCORE))
        plot.text(0.4,16,'Macro Precision: {:.3f}'.format(self.MACRO_PRECISION))
        plot.text(0.4,14,'Macro Recall: {:.3f}'.format(self.MACRO_RECALL))
        plot.text(0.4,12,'Macro F1-score: {:.3f}\n'.format(self.MACRO_F1_SCORE))
        plot.text(0.4,10,'Weighted Precision: {:.3f}'.format(self.WEIGHTED_PRECISION))
        plot.text(0.4,8,'Weighted Recall: {:.3f}'.format(self.WEIGHTED_RECALL))
        plot.text(0.4,6,'Weighted F1-score: {:.3f}'.format(self.WEIGHTED_F1_SCORE))
        plot.text(0.4,4,'Matthews correlation coefficient: {:.3f}'.format(self.MCC))
        plot.text(0.02,15,'--------------- Classification Report ---------------')
        plot.text(0.02,1,classification_report(self.Y_test, self.Y_predict))
        plot.suptitle(f"Various Result Score for {self.Model_Name}")
        plot.text(0.4,39,'--------------- Imbalanced Report ---------------')
        plot.text(0.4,29,classification_report_imbalanced(self.Y_test, self.Y_predict))
    
    
    def result_report_regression_calculation(self):
        
        # Calculation
        self.DIFF_PREDICT_REAL = self.Y_test - self.Y_predict
        self.MEAN_DIFF_PREDICT_REAL = np.mean(self.DIFF_PREDICT_REAL)
        self.STD_DIFF_PREDICT_REAL = np.std(self.DIFF_PREDICT_REAL)
        self.STANDARD_DIFF_PREDICT_REAL = (self.DIFF_PREDICT_REAL -self. MEAN_DIFF_PREDICT_REAL)/self.STD_DIFF_PREDICT_REAL
        
        # R2 score (coefficient of determination)
        self.R2_SCORE = r2_score(self.Y_test, self.Y_predict)
        
        # Relative Squared Error (RSE)
        self.RSE = np.sum(self.DIFF_PREDICT_REAL*self.DIFF_PREDICT_REAL)/np.sum((self.Y_predict-np.mean(self.Y_predict))*(self.Y_predict-np.mean(self.Y_predict)))
        
        # Mean Squared Error (MSE)
        self.MSE = mean_squared_error(self.Y_test, self.Y_predict)
        
        # Mean Absolute Error (MAE)
        self.MAE = mean_absolute_error(self.Y_test, self.Y_predict)
        
        # Explained variance score
        self.EVC = explained_variance_score(self.Y_test, self.Y_predict)
        
        # Max Error
        self.MAX_ERROR = max_error(self.Y_test, self.Y_predict)
        
        # Root mean squared error (RMSE)
        self.RMSE = root_mean_squared_error(self.Y_test, self.Y_predict)
        
        # Mean Squared Log Error (MSLE)
        self.MSLE = mean_squared_log_error(self.Y_test, self.Y_predict)
        
        # Root Mean Squared Log Error (RMSLE)
        self.RMSLE = root_mean_squared_log_error(self.Y_test, self.Y_predict)
        
        # Median absolute error 
        self.MEDIAN_ABSOLUTE_ERROR = median_absolute_error(self.Y_test, self.Y_predict)
        
        # Mean Poisson deviance
        self.MPD = mean_poisson_deviance(self.Y_test, self.Y_predict)
        
        # Mean Gamma deviance
        self.MGD = mean_gamma_deviance(self.Y_test, self.Y_predict)
        
        # Mean Absolute Percentage Error
        self.MAPE = mean_absolute_percentage_error(self.Y_test, self.Y_predict)
        
        # D2 Absolute Error Score
        self.D2 = d2_absolute_error_score(self.Y_test, self.Y_predict)
    
    
    def result_report_regression_print(self):
        
        print('\n-------------------- Key Metrics --------------------')
        print('\nR2 SCORE: {:.3f} %'.format(100*self.R2_SCORE))
        print('Represent the percentage of observed variation that can be explained by model inputs')
        
        print('\nMaximum Prediction Error: {:.0f}'.format(self.MAX_ERROR))

        print('\nRelative Squared Error (RSE): {:.3f}'.format(self.RSE))
        print('0 means perfectly fit/overfitting')

        print('\nMean Squared Error (MSE): {:.0f} €'.format(self.MSE))
        print('Mean Absolute Error (MAE): {:.0f} €'.format(self.MAE))
        print('Root Mean Squared Error (RMSE): {:.0f} €'.format(self.RMSE))
        print('Mean Absolute Percentage Error (MAPE): {:.3f} %'.format(100*self.MAPE))
        
        print('\nMean Squared Log Error (MSLE): {:.3f}'.format(self.MSLE))
        print('Root Mean Squared Log Error (RMSLE): {:.3f}'.format(self.RMSLE))
        
        print('\nExplained variance score: {:.3f}'.format(self.EVC))
        print('D2 Absolute Error Score: {:3f} / 1.0'.format(self.D2))
        print('Median absolute error : {:.0f} €'.format(self.MEDIAN_ABSOLUTE_ERROR))
        print('Mean Poisson deviance: {:.0f}'.format(self.MPD))
        print('Mean Gamma deviance: {:.3f}'.format(self.MGD))

    
    def result_report_regression_plot(self):
    
        # Plot
        fig,ax = plot.subplots(2)
        ax[0].hist(self.DIFF_PREDICT_REAL, bins = int(self.DIFF_PREDICT_REAL.shape[0]/4))
        ax[0].plot([self.STD_DIFF_PREDICT_REAL,self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "black")
        ax[0].plot([-self.STD_DIFF_PREDICT_REAL,-self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "black")
        ax[0].text(self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "σ", color = "black")
        ax[0].text(-self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "σ", color = "black")
        ax[0].plot([2*self.STD_DIFF_PREDICT_REAL,2*self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "orange")
        ax[0].plot([-2*self.STD_DIFF_PREDICT_REAL,-2*self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "orange")
        ax[0].text(2*self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "2σ", color = "orange")
        ax[0].text(-2*self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "2σ", color = "orange")
        ax[0].plot([3*self.STD_DIFF_PREDICT_REAL,3*self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "green")
        ax[0].plot([-3*self.STD_DIFF_PREDICT_REAL,-3*self.STD_DIFF_PREDICT_REAL],[0,int(self.DIFF_PREDICT_REAL.shape[0]/10)], color = "green")
        ax[0].text(3*self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "3σ", color = "green")
        ax[0].text(-3*self.STD_DIFF_PREDICT_REAL, int(self.DIFF_PREDICT_REAL.shape[0]/10) + 2, "3σ", color = "green")
        ax[0].set_title("Histogram on the difference between prediction and reality",)
        ax[1].scatter(self.Y_predict, self.STANDARD_DIFF_PREDICT_REAL)
        ax[1].set_ylim([-10,10])
        ax[1].set_title("Standard deviation regarding predicted values")
        plot.grid()
        
        # Print plot
        plot.figure(figsize = (10,8))
        plot.ylim(1,40)
        plot.text(0.1,38,'-------------------- Key Metrics --------------------')
        plot.text(0.1,36,'R2 SCORE: {:.3f} %'.format(100*self.R2_SCORE))
        plot.text(0.1,35,'Represent the percentage of observed variation that can be explained by model inputs')
        plot.text(0.1,33,'Relative Squared Error (RSE): {:.3f}'.format(self.RSE))
        plot.text(0.1,32,'0 means perfectly fit/overfitting')
        plot.text(0.1,30,'Mean Squared Error (MSE): {:.0f} €'.format(self.MSE))
        plot.text(0.1,29,'Mean Absolute Error (MAE): {:.0f} €'.format(self.MAE))
        plot.text(0.1,28,'Root Mean Squared Error (RMSE): {:.0f} €'.format(self.RMSE))
        plot.text(0.1,27,'Mean Absolute Percentage Error (MAPE): {:.3f} %'.format(100*self.MAPE))
        plot.text(0.1,25,'Maximum prediction error : {:.0f} €'.format(self.MAX_ERROR))
        plot.text(0.1,23,'Mean Squared Log Error (MSLE): {:.3f}'.format(self.MSLE))
        plot.text(0.1,22,'Root Mean Squared Log Error (RMSLE): {:.3f}'.format(self.RMSLE))
        plot.text(0.1,20,'Explained variance score: {:.3f} %'.format(100*self.EVC))
        plot.text(0.1,19,'D2 Absolute Error Score: {:3f}'.format(self.D2))
        plot.text(0.1,18,'Median absolute error : {:.0f} €'.format(self.MEDIAN_ABSOLUTE_ERROR))
        plot.text(0.1,17,'Mean Poisson deviance: {:.0f}'.format(self.MPD))
        plot.text(0.1,16,'Mean Gamma deviance: {:.3f}'.format(self.MGD))
        

class Data_Modelling_Random_Forest(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Random_Forest, self).__init__()

        self.Nb_Tree = 146
        self.min_samples_leaf = 16
        self.min_samples_split = 7
        self.min_weight_fraction_leaf = 0.00007276912136637689
        self.max_depth = 33
        
        self.Nb_Tree = 140
        self.min_samples_leaf = 5
        self.min_samples_split = 22
        self.min_weight_fraction_leaf = 0.00021061239694571002
        self.max_depth = 32

        self.Start_Point = 0
        self.End_Point_1 = 20
        self.End_Point_2 = 100
        self.shap_explainer = 0


    def Random_Forest_Modellisation(self, k_folds, REGRESSION):

        # Setting the model with parameters
        if REGRESSION:
            self.MODEL = RandomForestRegressor(
                n_estimators=self.Nb_Tree,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                verbose=2,
                random_state=0)
        
        else:
            self.MODEL = RandomForestClassifier(
                n_estimators=self.Nb_Tree,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                max_depth=self.max_depth,
                verbose=2,
                random_state=0)

        # Cross validation
        self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train)

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        if REGRESSION:
            self.AVERAGE_DIFFERENCE = np.mean(abs(self.Y_predict - Y_test))
            print(f"\n Moyenne des différences : {round(self.AVERAGE_DIFFERENCE,2)} €")
            self.PERCENTAGE_AVERAGE_DIFFERENCE = 100*self.AVERAGE_DIFFERENCE / np.mean(self.Y_test)
            print(f"\n Pourcentage de différence : {round(self.PERCENTAGE_AVERAGE_DIFFERENCE,2)} %")
            
        else:
            self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
            self.Nb_Correct_Prediction = np.count_nonzero(
                Y_test.astype(int) - self.Y_predict.astype(int))
            self.Percentage_Correct_Prediction = (1 -
                self.Nb_Correct_Prediction / Y_test.shape[0])
            print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")


    def Feature_Importance_Plot(self):

        # Feature Importance
        RF_Feature_Importance = pd.DataFrame(
            {'Variable': self.X_train.columns,
             'Importance': self.MODEL.feature_importances_}).sort_values(
                 'Importance', ascending=False)

        fig, ax = plot.subplots()
        ax.barh(RF_Feature_Importance.Variable,
                RF_Feature_Importance.Importance)
        plot.grid()
        plot.suptitle("Feature Importance for Random Forest Model")


    def Permutation_Importance(self, TEST_DATAFRAME):

         # Permutation Importance
        Permutation_Importance_Train = permutation_importance(self.MODEL, self.X_train, self.Y_train,
                                                              n_repeats=10, random_state=0, n_jobs=2)
        Permutation_Importance_Test = permutation_importance(self.MODEL, self.X_test, self.Y_test,
                                                             n_repeats=10, random_state=0, n_jobs=2)

        # Init
        fig, ax = plot.subplots(2)
        max_importances = 0

        # Loop for Train/Test Data
        for i, Permutation_Importance in enumerate(
                [Permutation_Importance_Train, Permutation_Importance_Test]):

            # Calculating permutaion importance
            sorted_importances_idx = Permutation_Importance.importances_mean.argsort()
            importances = pd.DataFrame(
                Permutation_Importance.importances[sorted_importances_idx].T,
                columns=self.X_test.columns[sorted_importances_idx],)
            max_importances = max([max_importances,importances.max().max()])

            # Plotting results
            ax[i].boxplot(importances, vert=False)
            ax[i].set_title("Permutation Importances")
            ax[i].axvline(x=0, color="k", linestyle="--")
            ax[i].set_xlabel("Decrease in accuracy score")
            ax[i].set_xlim([-0.01,max_importances + 0.1])
            ax[i].set_yticks(np.linspace(1,importances.shape[1],importances.shape[1]))
            ax[i].set_yticklabels(importances.columns)
            ax[i].figure.tight_layout()


    def Shap_Value_Analysis_Single_Point(self):

        # Init
        Nb_Analysed = random.randint(0,self.X_test.shape[0])

        # Create object that can calculate shap values
        self.shap_explainer = shap.TreeExplainer(self.MODEL)

        # Calculate Shap values for one point
        self.MODEL.predict(np.array(self.X_test.iloc[Nb_Analysed,:]).reshape(-1,1).T)
        shap_values_1 = self.shap_explainer.shap_values(self.X_test.iloc[Nb_Analysed,:])

        # Plotting results
        shap.initjs()
        shap.force_plot(
            self.shap_explainer.expected_value[0],
            shap_values_1[1],
            self.X_test.iloc[Nb_Analysed,:].index,
            matplotlib=True)
        plot.suptitle(f"Prediction attendue : {self.Y_test[Nb_Analysed]}")


    def Shap_Value_Analysis_Multiple_Point(self):

        # Calculate Shap values for 10 points
        Wrong_pred = self.MODEL.predict(self.X_test) != np.array(self.Y_test)
        shap_values_10 = self.shap_explainer.shap_values(
            self.X_test.iloc[self.Start_Point:self.End_Point_1,:])

        # Plotting results
        plot.figure()
        shap.decision_plot(
            self.shap_explainer.expected_value[0],
            shap_values_10[0],
            self.X_test.iloc[self.Start_Point:self.End_Point_1,:],
            feature_names = np.array(self.X_test.columns),
            highlight=Wrong_pred[self.Start_Point:self.End_Point_1])


    def Shap_Value_Analysis_Multiple_Massive_Point(self):

        # Calculate Shap values for 100 points
        shap_value_100 = self.shap_explainer.shap_values(
            self.X_test.iloc[self.Start_Point:self.End_Point_2,:])

        # Plotting results
        shap.dependence_plot(
            2,
            shap_value_100[0],
            self.X_test.iloc[self.Start_Point:self.End_Point_2,:])


class Data_Modelling_Gradient_Boosting(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Gradient_Boosting, self).__init__()

        self.learning_rate = 0.4968286095170373
        self.Nb_Tree = 192
        self.min_samples_leaf = 37
        self.min_samples_split = 35
        self.min_weight_fraction_leaf = 0.00018800850129667424
        self.max_depth = 24
        self.validation_fraction = 0.1   # Early Stopping
        self.n_iter_no_change = 10   # Early Stopping
        self.train_errors = []   # Early Stopping
        self.test_errors = []   # Early Stopping


    def Gradient_Boosting_Modellisation(self, N_SPLIT, REGRESSION):

        # Setting the model with parameters
        if REGRESSION:
            self.MODEL = GradientBoostingRegressor(
                learning_rate=self.learning_rate,
                n_estimators=self.Nb_Tree,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                verbose=2,
                random_state=0,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change)

        else:
            self.MODEL = GradientBoostingClassifier(
                learning_rate=self.learning_rate,
                n_estimators=self.Nb_Tree,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                verbose=2,
                random_state=0,
                validation_fraction=self.validation_fraction,
                n_iter_no_change=self.n_iter_no_change)

        # Init
        k_folds = KFold(n_splits=N_SPLIT)

        # Cross validation
        self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train)

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        if REGRESSION:
            self.AVERAGE_DIFFERENCE = np.mean(abs(self.Y_predict - Y_test))
            print(f"\n Moyenne des différences : {round(self.AVERAGE_DIFFERENCE,2)} €")
            self.PERCENTAGE_AVERAGE_DIFFERENCE = 100*self.AVERAGE_DIFFERENCE / np.mean(Y_test)
            print(f"\n Pourcentage de différence : {round(self.PERCENTAGE_AVERAGE_DIFFERENCE,2)} %")
        else:
            self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
            self.Nb_Correct_Prediction = np.count_nonzero(
                Y_test.astype(int) - self.Y_predict.astype(int))
            self.Percentage_Correct_Prediction = (1 -
                self.Nb_Correct_Prediction / Y_test.shape[0])
            print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")


    def Plot_Training_Validation_Error(self):

        for i, (train_pred, test_pred) in enumerate(
            zip(
                self.MODEL.staged_predict(self.X_train),
                self.MODEL.staged_predict(self.X_test),
            )
        ):

            if isinstance(self.Y_train.iloc[0], bool):
                self.train_errors.append(mean_squared_error(
                    self.Y_train, train_pred))
                self.test_errors.append(mean_squared_error(
                    self.Y_test, test_pred))
            else:
                self.train_errors.append(mean_squared_error(
                    self.Y_train.astype(int), train_pred.astype(int)))
                self.test_errors.append(mean_squared_error(
                    self.Y_test.astype(int), test_pred.astype(int)))


        fig, ax = plot.subplots(ncols=2, figsize=(12, 4))

        ax[0].plot(self.train_errors, label="Gradient Boosting with Early Stopping")
        ax[0].set_xlabel("Boosting Iterations")
        ax[0].set_ylabel("MSE (Training)")
        ax[0].set_yscale("log")
        ax[0].legend()
        ax[0].set_title("Training Error")

        ax[1].plot(self.test_errors, label="Gradient Boosting with Early Stopping")
        ax[1].set_xlabel("Boosting Iterations")
        ax[1].set_ylabel("MSE (Validation)")
        ax[1].set_yscale("log")
        ax[1].legend()
        ax[1].set_title("Validation Error")


class Data_Modelling_Neural_Network(Data_Modelling):
    def __init__(self):
        super(Data_Modelling_Neural_Network, self).__init__()

        self.n_hidden = 3
        self.n_neurons = 68
        self.n_trials = 20
        self.History = 0

        self.monitor ="val_accuracy"
        self.min_delta = 0.002
        self.patience = 25
    
    
    def Neural_Network_Modellisation(self, N_SPLIT, REGRESSION):
        
        tf.random.set_seed(0)

        if REGRESSION:
            
            # Neural Network model
            self.MODEL = tf.keras.models.Sequential()
            self.MODEL.add(tf.keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)))
            for layers in range(self.n_hidden):
                self.MODEL.add(tf.keras.layers.Dense(self.n_neurons, activation = "relu"))
            self.MODEL.add(tf.keras.layers.Dense(1))
    
            # Optimizer used
            OPTIMIZER = tf.keras.optimizers.Nadam()
    
            self.MODEL.compile(loss = "mse",
                               optimizer = OPTIMIZER,
                               metrics = ["accuracy"])
        
        else:

            # Neural Network model
            self.MODEL = tf.keras.models.Sequential()
            self.MODEL.add(tf.keras.layers.InputLayer(input_shape=(self.X_train.shape[1],)))
            for layers in range(self.n_hidden):
                self.MODEL.add(tf.keras.layers.Dense(self.n_neurons, activation = "relu"))
            self.MODEL.add(tf.keras.layers.Dense(Global_Data.UNIQUE_PREDICT_VALUE.shape[0], activation = "softmax"))
    
            # Optimizer used
            OPTIMIZER = tf.keras.optimizers.Nadam()
    
            self.MODEL.compile(
                loss = "sparse_categorical_crossentropy", optimizer = OPTIMIZER, metrics = ["accuracy"])

        # Init
        k_folds = KFold(n_splits=N_SPLIT)

        # Cross validation
        # self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds, scoring = "accuracy")
        # Early stopping init
        callback = tf.keras.callbacks.EarlyStopping(monitor = self.monitor,
                                                    min_delta = self.min_delta,
                                                    patience = self.patience,
                                                    verbose = 0,
                                                    restore_best_weights=True)
        self.History = self.MODEL.fit(np.asarray(self.X_train).astype("float32"),
                                      np.asarray(self.Y_train).astype("float32"),
                                      epochs = 500,
                                      validation_split = 0.01,
                                      initial_epoch = 0,
                                      callbacks=[callback])
        
        # Plot learning evolution for Neural Network
        pd.DataFrame(self.History.history).plot(figsize = (8,5))
        plot.grid(True)
        plot.title("Learning Evolution for Neural Network")

        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        if REGRESSION:
            self.AVERAGE_DIFFERENCE = np.mean(abs(self.Y_predict - Y_test))
            print(f"\n Moyenne des différences : {round(self.AVERAGE_DIFFERENCE,2)} €")
            self.PERCENTAGE_AVERAGE_DIFFERENCE = 100*self.AVERAGE_DIFFERENCE / np.mean(Y_test)
            print(f"\n Pourcentage de différence : {round(self.PERCENTAGE_AVERAGE_DIFFERENCE,2)} %")
        else:
            self.Y_predict_proba = self.Y_predict
            self.Y_predict = np.argmax(self.Y_predict_proba,axis=1)
            self.Nb_Correct_Prediction = np.count_nonzero(
                Y_test.astype(int) - self.Y_predict.astype(int))
            self.Percentage_Correct_Prediction = (1 -
                self.Nb_Correct_Prediction / Y_test.shape[0])
            print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")


class Data_Modelling_XGBoosting(Data_Modelling):
    def __init__(self,num_class):
        super(Data_Modelling_XGBoosting, self).__init__()

        self.objective_classification='multi:softmax'
        self.objective_regression='reg:linear'
        self.num_class=num_class
        # self.learning_rate=0.1
        # self.max_depth=5
        # self.gamma=0
        # self.reg_lambda=1
        self.learning_rate=0.2578
        self.max_depth=5
        self.gamma=0.940859
        self.reg_lambda=5
        self.min_child_weight = 1
        self.early_stopping_rounds=25
        self.eval_metric_classification=['merror','mlogloss']
        self.min_child_weight = 1

        self.x_axis = []
        self.results_metric_plot = []


    def XGBoosting_Modellisation(self, k_folds, REGRESSION):

        # Setting the model with parameters
        if REGRESSION:
            self.MODEL = xgb.XGBRegressor(
                objective=self.objective_regression,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                early_stopping_rounds=self.early_stopping_rounds,
                min_child_weight=self.min_child_weight,
                seed=42)
        
        else:
            self.MODEL = xgb.XGBClassifier(
                objective=self.objective_classification,
                num_class=self.num_class,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
                early_stopping_rounds=self.early_stopping_rounds,
                eval_metric=self.eval_metric_classification,
                min_child_weight=self.min_child_weight,
                seed=42)

        # Cross validation
        # self.score = cross_val_score(self.MODEL, self.X_train, self.Y_train, cv=k_folds)
        self.MODEL.fit(self.X_train, self.Y_train,
                       verbose = 1,
                       eval_set = [(self.X_train, self.Y_train),
                                   (self.X_test, self.Y_test)])
 
        # Predicting results
        self.Y_predict = self.MODEL.predict(self.X_test)
        Y_test = np.squeeze(self.Y_test.to_numpy())

        # Percentage calculation for correct prediction
        if REGRESSION:
            self.AVERAGE_DIFFERENCE = np.mean(abs(self.Y_predict - Y_test))
            print(f"\n Moyenne des différences : {round(self.AVERAGE_DIFFERENCE,2)} €")
            self.PERCENTAGE_AVERAGE_DIFFERENCE = 100*self.AVERAGE_DIFFERENCE / np.mean(Y_test)
            print(f"\n Pourcentage de différence : {round(self.PERCENTAGE_AVERAGE_DIFFERENCE,2)} %")
        else:
            self.Y_predict_proba = self.MODEL.predict_proba(self.X_test)
            self.Nb_Correct_Prediction = np.count_nonzero(
                Y_test.astype(int) - self.Y_predict.astype(int))
            self.Percentage_Correct_Prediction = (1 -
                self.Nb_Correct_Prediction / Y_test.shape[0])
            print(f"\n Pourcentage de predictions correctes {self.Model_Name} : {100*round(self.Percentage_Correct_Prediction,5)} %")

            # Preparing evaluation metric plots
            self.results_metric_plot = self.MODEL.evals_result()
            epochs = len(self.results_metric_plot['validation_0']['mlogloss'])
            self.x_axis = range(0, epochs)


    def Evaluation_Metric_Plot_Mlogloss(self):

        # xgboost 'mlogloss' plot
        fig, ax = plot.subplots(figsize=(9,5))
        ax.plot(self.x_axis, self.results_metric_plot['validation_0']['mlogloss'], label='Train')
        ax.plot(self.x_axis, self.results_metric_plot['validation_1']['mlogloss'], label='Test')
        ax.legend()
        plot.ylabel('mlogloss')
        plot.title('GridSearchCV XGBoost mlogloss')
        plot.show()


    def Evaluation_Metric_Plot_Merror(self):

        # xgboost 'merror' plot
        fig, ax = plot.subplots(figsize=(9,5))
        ax.plot(self.x_axis, self.results_metric_plot['validation_0']['merror'], label='Train')
        ax.plot(self.x_axis, self.results_metric_plot['validation_1']['merror'], label='Test')
        ax.legend()
        plot.ylabel('merror')
        plot.title('GridSearchCV XGBoost merror')
        plot.show()


    def Feature_Importance_Plot(self):

        fig, ax = plot.subplots(figsize=(9,5))
        plot_importance(self.MODEL, ax=ax)
        plot.show()





# -- ////////// --
# -- ////////// --
# -- ////////// --





# Init for global parameters
Global_Parameters = Parameters()

if Global_Parameters.CLEAR_MODE:

    # Removing data
    from IPython import get_ipython
    get_ipython().magic('reset -sf')

    # Closing all figures
    plot.close("all")


Global_Data = Data_Preparation()
Global_Data.data_import(Global_Parameters.NAME_DATA_PREDICT)
Global_Parameters.regression_analysis(Global_Data.TRAIN_DATAFRAME)

# Droping some columns
if Global_Parameters.SWITCH_REMOVING_DATA:
    for name_drop in ["id", "cap-surface", "gill-attachment", "gill-spacing",
                      "stem-root","stem-surface","veil-type","veil-color",
                      "spore-print-color"]:
        Global_Data.data_drop(name_drop)

# # Cheat
# Global_Data.real_value_replacement()


# Removing variable with too low data
if Global_Parameters.SWITCH_DATA_REDUCTION:
    Global_Data.remove_low_data(Global_Parameters.NB_DATA_NOT_ENOUGHT, "Origin",
                                LIST_NAME_DATA_REMOVE_MULTIPLE = ["Dest"])

# Data description
Global_Data.data_predict_description(Global_Parameters.NAME_DATA_PREDICT)

# Multi classification identification
Global_Parameters.Multi_Classification_Analysis(Global_Data.UNIQUE_PREDICT_VALUE)


# Sample Data
if Global_Parameters.SWITCH_SAMPLE_DATA:
    Global_Data.data_sample(Global_Parameters.Fraction_Sample_Data)


# Encoding data for entry variables
if Global_Parameters.SWITCH_ENCODE_DATA:
    if Global_Parameters.SWITCH_ENCODE_DATA_ONEHOT:
        for Name_Data_Encode in Global_Parameters.LIST_DATA_ENCODE:
            Global_Data.data_encoding_onehot(Name_Data_Encode)

    else:
        # Removing data with incorrect format before encoding
        Global_Data.data_format_removal(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT)
        
        # Encoding
        Global_Data.data_encoding_replacement(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT, True)
    
        # Removing error data after encoding
        Global_Data.encode_data_error_removal(Global_Parameters.ARRAY_DATA_ENCODE_REPLACEMENT)
        Global_Data.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME.dropna()


# Encoding data for predict variable
if Global_Parameters.SWITCH_ENCODE_DATA_PREDICT:
    Global_Data.data_encoding_replacement_predict(Global_Parameters.ARRAY_DATA_ENCODE_PREDICT)


# Searching for and removing aberrant/identical values
if Global_Parameters.SWITCH_ABERRANT_IDENTICAL_DATA:
    Global_Data.data_duplicate_removal(Global_Parameters.NAME_DATA_PREDICT)


# Oversampling to equilibrate data
if (Global_Parameters.SWITCH_EQUILIBRATE_DATA and Global_Parameters.SWITCH_SMOTEN_DATA == False):
    Global_Data.oversampling(Global_Parameters.NAME_DATA_PREDICT, Global_Parameters.NB_DATA_NOT_ENOUGHT)


# Searching for repartition on data to predict
if Global_Parameters.SWITCH_PLOT_DATA:

    Global_Data_Plot = Data_Plot()
    Global_Data_Plot.TRAIN_DATAFRAME = Global_Data.TRAIN_DATAFRAME
    Global_Data_Plot.UNIQUE_PREDICT_VALUE = Global_Data.UNIQUE_PREDICT_VALUE
    Global_Data_Plot.Box_Plot_Data_Predict_Plot(Global_Parameters.GENERIC_NAME_DATA_PREDICT)
    Global_Data_Plot.plot_data_repartition()
    Global_Data_Plot.plot_data_hist()
    plot.pause(1)
    # Global_Data_Plot.Plot_Data_Relation("Height", "Gender")
    plot.pause(1)
    Global_Data.TRAIN_CORRELATION = Global_Data.TRAIN_DATAFRAME.iloc[
        :,:Global_Data.TRAIN_DATAFRAME.shape[1] - 1].corr()
    Global_Data_Plot.TRAIN_CORRELATION = Global_Data.TRAIN_CORRELATION
    Global_Data_Plot.Correlation_Plot_Plot()


# Modifying linear relation between data
if Global_Parameters.SWITCH_RELATION_DATA:
    for i in range(Global_Parameters.List_Relation_Data.shape[0]):
        Global_Data.data_pow(Global_Parameters.List_Relation_Data[i,0],
                             Global_Parameters.List_Relation_Data[i,1])


# Replacing Nan values
if Global_Parameters.SWITCH_REPLACING_NAN:
    Global_Data.nan_replacing(["BsmtQual", "BsmtCond", "BsmtExposure"])

# Generic Data Model
Data_Model = Data_Modelling()
Data_Model.Splitting_Data(Global_Data.TRAIN_DATAFRAME,
                          Global_Parameters.NAME_DATA_PREDICT,
                          Global_Parameters.MULTI_CLASSIFICATION,
                          Global_Parameters.REGRESSION)
if (Global_Parameters.SWITCH_SMOTEN_DATA and Global_Parameters.SWITCH_EQUILIBRATE_DATA):
    Data_Model.Smoten_Sampling()


#
# Random Forest

if Global_Parameters.RF_MODEL:
    DATA_MODEL_RF = Data_Modelling_Random_Forest()
    DATA_MODEL_RF.X_train = Data_Model.X_train
    DATA_MODEL_RF.Y_train = Data_Model.Y_train
    DATA_MODEL_RF.X_test = Data_Model.X_test
    DATA_MODEL_RF.Y_test = Data_Model.Y_test
    DATA_MODEL_RF.Model_Name = "Random Forest"

    # Building a Random Forest Model with adjusted parameters
    if Global_Parameters.REGRESSION:
        def build_model_RF(Nb_Tree=1, min_samples_leaf=2, min_samples_split=10,
                           max_depth=2, min_weight_fraction_leaf=0.5):
    
            MODEL_RF = RandomForestRegressor(
                n_estimators=Nb_Tree,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=0,
                max_depth=max_depth,
                min_weight_fraction_leaf=min_weight_fraction_leaf,)
    
            return MODEL_RF
        
    else:
        def build_model_RF(Nb_Tree=1, min_samples_leaf=2, min_samples_split=10,
                           max_depth=2, min_weight_fraction_leaf=0.5):
    
            MODEL_RF = RandomForestClassifier(
                n_estimators=Nb_Tree,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=0,
                max_depth=max_depth,
                min_weight_fraction_leaf=min_weight_fraction_leaf,)
    
            return MODEL_RF


    # Searching for Optimized Hyperparameters
    if Global_Parameters.RF_MODEL_OPTI:

        # Building function to minimize
        def objective_RF(trial):
            params = {'Nb_Tree': trial.suggest_int('Nb_Tree', 10, 250),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 20),
                      'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                      'max_depth': trial.suggest_int('max_depth', 1, 50),
                      'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5)}

            MODEL_RF = build_model_RF(**params)
            scores = cross_val_score(MODEL_RF, DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train, cv=Global_Parameters.k_folds)
            MODEL_RF.fit(DATA_MODEL_RF.X_train, DATA_MODEL_RF.Y_train)
            prediction_score = MODEL_RF.score(DATA_MODEL_RF.X_test, DATA_MODEL_RF.Y_test)

            return 0.1*np.amax(scores) + prediction_score

        # Search for best parameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_RF, n_trials=Global_Parameters.RF_MODEL_TRIAL,
                       catch=(ValueError,))
        Best_params_RF = np.zeros([1], dtype=object)
        Best_params_RF[0] = study.best_params
        DATA_MODEL_RF.Nb_Tree = int(Best_params_RF[0].get("Nb_Tree"))
        DATA_MODEL_RF.min_samples_leaf = int(Best_params_RF[0].get("min_samples_leaf"))
        DATA_MODEL_RF.min_samples_split = int(Best_params_RF[0].get("min_samples_split"))
        DATA_MODEL_RF.min_weight_fraction_leaf = float(Best_params_RF[0].get("min_weight_fraction_leaf"))
        DATA_MODEL_RF.max_depth = int(Best_params_RF[0].get("max_depth"))

    DATA_MODEL_RF.Random_Forest_Modellisation(
        Global_Parameters.k_folds, Global_Parameters.REGRESSION)
    
    DATA_MODEL_RF.Feature_Importance_Plot()
    DATA_MODEL_RF.Permutation_Importance(Global_Data.TEST_DATAFRAME)
    if Global_Parameters.REGRESSION == False:
        DATA_MODEL_RF.result_plot_classification()
        DATA_MODEL_RF.result_report_classification_calculation()
        DATA_MODEL_RF.result_report_classification_print()
        DATA_MODEL_RF.result_report_classification_plot()
        
        DATA_MODEL_RF.Shap_Value_Analysis_Single_Point()
        DATA_MODEL_RF.Shap_Value_Analysis_Multiple_Point()
        DATA_MODEL_RF.Shap_Value_Analysis_Multiple_Massive_Point()
    else:
        DATA_MODEL_RF.result_plot_regression()
        DATA_MODEL_RF.result_report_regression_calculation()
        DATA_MODEL_RF.result_report_regression_print()
        DATA_MODEL_RF.result_report_regression_plot()
        
        DATA_MODEL_RF.extract_max_diff_regression()


#
# Gradient Boosting

if Global_Parameters.GB_MODEL:
    DATA_MODEL_GB = Data_Modelling_Gradient_Boosting()
    DATA_MODEL_GB.X_train = Data_Model.X_train
    DATA_MODEL_GB.Y_train = Data_Model.Y_train
    DATA_MODEL_GB.X_test = Data_Model.X_test
    DATA_MODEL_GB.Y_test = Data_Model.Y_test
    DATA_MODEL_GB.Model_Name = "Gradient Boosting"

    # Building a Gradient Boosting Model with adjusted parameters
    if Global_Parameters.REGRESSION:
        
        def build_model_GB(
                learning_rate=0.1,
                Nb_Tree=1,
                min_samples_split=10,
                min_samples_leaf=2,
                min_weight_fraction_leaf=0.5,
                max_depth=2):

            MODEL_GB = GradientBoostingRegressor(
                learning_rate=learning_rate,
                n_estimators=Nb_Tree,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=0,
                max_depth=max_depth,
                min_weight_fraction_leaf=min_weight_fraction_leaf)

            return MODEL_GB
    
    else:
        
        def build_model_GB(
                learning_rate=0.1,
                Nb_Tree=1,
                min_samples_split=10,
                min_samples_leaf=2,
                min_weight_fraction_leaf=0.5,
                max_depth=2):
    
            MODEL_GB = GradientBoostingClassifier(
                learning_rate=learning_rate,
                n_estimators=Nb_Tree,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=0,
                max_depth=max_depth,
                min_weight_fraction_leaf=min_weight_fraction_leaf)
    
            return MODEL_GB

    # Searching for Optimized Hyperparameters
    if Global_Parameters.GB_MODEL_OPTI:

        # Building function to minimize
        def objective_GB(trial):
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.5),
                      'Nb_Tree': trial.suggest_int('Nb_Tree', 2, 200),
                      'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
                      'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
                      'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0, 0.5),
                      'max_depth': trial.suggest_int('max_depth', 2, 50)}

            MODEL_GB = build_model_GB(**params)
            scores = cross_val_score(MODEL_GB, Data_Model.X_train, Data_Model.Y_train, cv=Global_Parameters.k_folds)
            MODEL_GB.fit(Data_Model.X_train, Data_Model.Y_train)
            prediction_score = MODEL_GB.score(Data_Model.X_test, Data_Model.Y_test)

            return 0.1*np.amax(scores) + prediction_score

        # Search for best parameters
        study = optuna.create_study(direction='maximize')
        study.optimize(objective_GB, n_trials=Global_Parameters.GB_MODEL_TRIAL, catch=(ValueError,))
        Best_params_GB = np.zeros([1], dtype=object)
        Best_params_GB[0] = study.best_params
        DATA_MODEL_GB.learning_rate = float(Best_params_GB[0].get("learning_rate"))
        DATA_MODEL_GB.Nb_Tree = int(Best_params_GB[0].get("Nb_Tree"))
        DATA_MODEL_GB.min_samples_leaf = int(Best_params_GB[0].get("min_samples_leaf"))
        DATA_MODEL_GB.min_samples_split = int(Best_params_GB[0].get("min_samples_split"))
        DATA_MODEL_GB.min_weight_fraction_leaf = float(Best_params_GB[0].get("min_weight_fraction_leaf"))
        DATA_MODEL_GB.max_depth = int(Best_params_GB[0].get("max_depth"))

    DATA_MODEL_GB.Gradient_Boosting_Modellisation(
        Global_Parameters.N_SPLIT, Global_Parameters.REGRESSION)
    
    if Global_Parameters.REGRESSION == False:
        DATA_MODEL_GB.result_plot_classification()
        DATA_MODEL_GB.Plot_Training_Validation_Error()
        DATA_MODEL_GB.result_report_classification_calculation()
        DATA_MODEL_GB.result_report_classification_print()
        DATA_MODEL_GB.result_report_classification_plot()
        
    else:
        DATA_MODEL_GB.result_plot_regression()
        DATA_MODEL_GB.result_report_regression_calculation()
        DATA_MODEL_GB.result_report_regression_print()
        DATA_MODEL_GB.result_report_regression_plot()
        
        DATA_MODEL_GB.extract_max_diff_regression()


#
# Neural Network

if Global_Parameters.NN_MODEL:
    DATA_MODEL_NN = Data_Modelling_Neural_Network()
    DATA_MODEL_NN.X_train = Data_Model.X_train
    DATA_MODEL_NN.Y_train = Data_Model.Y_train
    DATA_MODEL_NN.X_test = Data_Model.X_test
    DATA_MODEL_NN.Y_test = Data_Model.Y_test
    DATA_MODEL_NN.Model_Name = "Neural Network"
    
    # Building a Gradient Boosting Model with adjusted parameters
    if Global_Parameters.REGRESSION:
        
        def build_model_NN(n_hidden = 1, n_neurons = 100, input_shape = (Data_Model.X_train.shape[1],)):
    
            # Neural Network model
            MODEL = tf.keras.models.Sequential()
            MODEL.add(tf.keras.layers.InputLayer(input_shape=input_shape))
            for layers in range(n_hidden):
                MODEL.add(tf.keras.layers.Dense(n_neurons, activation = "relu"))
            MODEL.add(tf.keras.layers.Dense(1))
    
            # Optimizer used
            OPTIMIZER = tf.keras.optimizers.Nadam()
    
            MODEL.compile(loss = "mse",
                          optimizer = OPTIMIZER,
                          metrics = ["accuracy"])
    
            return MODEL
    
    else:

        def build_model_NN(n_hidden = 1, n_neurons = 100, input_shape = (Data_Model.X_train.shape[1],)):
    
            # Neural Network model
            MODEL = tf.keras.models.Sequential()
            MODEL.add(tf.keras.layers.InputLayer(input_shape=input_shape))
            for layers in range(n_hidden):
                MODEL.add(tf.keras.layers.Dense(n_neurons, activation = "relu"))
            MODEL.add(tf.keras.layers.Dense(Global_Data.UNIQUE_PREDICT_VALUE.shape[0], activation = "softmax"))
    
            # Optimizer used
            OPTIMIZER = tf.keras.optimizers.Nadam()
    
            MODEL.compile(loss = "sparse_categorical_crossentropy",
                          optimizer = OPTIMIZER,
                          metrics = ["accuracy"])
    
            return MODEL


    if Global_Parameters.NN_MODEL_OPTI:
        
        if Global_Parameters.REGRESSION:
            
            # Building function to minimize
            def objective_NN(trial):
                params = {
                    'n_hidden': trial.suggest_int('n_hidden', 2, 4),
                    'n_neurons': trial.suggest_int('n_neurons', 10, 100)}
    
                model = build_model_NN(**params)
    
                model.fit(np.asarray(DATA_MODEL_NN.X_train).astype("float32"),
                          np.asarray(DATA_MODEL_NN.Y_train).astype("float32"),
                          epochs = 50,
                          validation_split = 0.01,
                          initial_epoch = 0)
    
                Preds_NN = model.predict(np.asarray(DATA_MODEL_NN.X_test).astype("float32"))
                Score_NN = mean_squared_error(Data_Model.Y_test, Preds_NN)
    
                return Score_NN
            
        else:

            # Building function to minimize
            def objective_NN(trial):
                params = {
                    'n_hidden': trial.suggest_int('n_hidden', 2, 4),
                    'n_neurons': trial.suggest_int('n_neurons', 10, 100)}
    
                model = build_model_NN(**params)
    
                model.fit(np.asarray(DATA_MODEL_NN.X_train).astype("float32"),
                          np.asarray(DATA_MODEL_NN.Y_train).astype("float32"),
                          epochs = 50,
                          validation_split = 0.01,
                          initial_epoch = 0)
    
                Preds_NN_proba = model.predict(np.asarray(DATA_MODEL_NN.X_test).astype("float32"))
                Preds_NN = np.zeros([Preds_NN_proba.shape[0]], dtype = int)
    
                # Turning probability prediction into prediction
                for i in range(Preds_NN_proba.shape[0]):
                    Preds_NN[i] = np.where(Preds_NN_proba[i,:] == np.amax(Preds_NN_proba[i,:]))[0][0]
    
                Score_NN = accuracy_score(Data_Model.Y_test, Preds_NN)
    
                return Score_NN


        # Search for best parameters
        if Global_Parameters.REGRESSION:
            study = optuna.create_study(direction='minimize')
        else:
            study = optuna.create_study(direction='maximize')
        study.optimize(objective_NN,
                       n_trials=Global_Parameters.NN_MODEL_TRIAL,
                       catch=(ValueError,))
        Best_params_NN = np.zeros([1], dtype = object)
        Best_params_NN[0] = study.best_params
        DATA_MODEL_NN.n_hidden = int(Best_params_NN[0].get("n_hidden"))
        DATA_MODEL_NN.n_neurons = int(Best_params_NN[0].get("n_neurons"))
        
    DATA_MODEL_NN.Neural_Network_Modellisation(Global_Parameters.N_SPLIT, Global_Parameters.REGRESSION)
    
  
    if Global_Parameters.REGRESSION == False:
        DATA_MODEL_NN.result_plot_classification()
        DATA_MODEL_NN.result_report_classification_calculation()
        DATA_MODEL_NN.result_report_classification_print()
        DATA_MODEL_NN.result_report_classification_plot()
        
        # DATA_MODEL_NN.Shap_Value_Analysis_Single_Point()
        # DATA_MODEL_NN.Shap_Value_Analysis_Multiple_Point()
        # DATA_MODEL_NN.Shap_Value_Analysis_Multiple_Massive_Point()
    else:
        DATA_MODEL_NN.result_plot_regression()
        DATA_MODEL_NN.result_report_regression_calculation()
        DATA_MODEL_NN.result_report_regression_print()
        DATA_MODEL_NN.result_report_regression_plot()
        
        DATA_MODEL_NN.extract_max_diff_regression()
        


#
# XGBoosting

if Global_Parameters.XG_MODEL:
    DATA_MODEL_XG = Data_Modelling_XGBoosting(pd.unique(Global_Data.TRAIN_DATAFRAME[Global_Parameters.NAME_DATA_PREDICT]).shape[0])
    DATA_MODEL_XG.X_train = Data_Model.X_train
    DATA_MODEL_XG.Y_train = Data_Model.Y_train
    DATA_MODEL_XG.X_test = Data_Model.X_test
    DATA_MODEL_XG.Y_test = Data_Model.Y_test
    DATA_MODEL_XG.Model_Name = "XG Boosting"
    
    # Building a Random Forest Model with adjusted parameters
    if Global_Parameters.REGRESSION:
        
        def build_model_XG(objective='reg:linear', learning_rate=0.1, max_depth=5,
                           gamma=0, reg_lambda=1, early_stopping_rounds=25,
                           min_child_weight=1):
        
            MODEL_XG = xgb.XGBRegressor(
                objective=objective,
                learning_rate=learning_rate,
                max_depth=max_depth,
                gamma=gamma,
                reg_lambda=reg_lambda,
                early_stopping_rounds=early_stopping_rounds,
                min_child_weight = min_child_weight,
                seed=42)
            
            return MODEL_XG
    
    else:
    
        def build_model_XG(objective='multi:softmax', num_class=16, learning_rate=0.1, max_depth=5,
                           gamma=0, reg_lambda=1, early_stopping_rounds=25,
                           eval_metric = ['merror','mlogloss'], min_child_weight=1):
    
            MODEL_XG = xgb.XGBClassifier(
                objective=objective,
                num_class=num_class,
                learning_rate=learning_rate,
                max_depth=max_depth,
                gamma=gamma,
                reg_lambda=reg_lambda,
                early_stopping_rounds=early_stopping_rounds,
                eval_metric=eval_metric,
                min_child_weight = min_child_weight,
                seed=42)
            
            return MODEL_XG
    
        
    # Searching for Optimized Hyperparameters
    if Global_Parameters.XG_MODEL_OPTI:

        # Building function to minimize
        def objective_XG(trial):
            params = {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.5),
                      'max_depth': trial.suggest_int('max_depth', 1, 15),
                      'gamma': trial.suggest_float('gamma', 0, 1),
                      'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
                      'reg_lambda': trial.suggest_int('reg_lambda', 1, 10)}

            MODEL_XG = build_model_XG(**params)
            # scores = cross_val_score(
            #     MODEL_XG, DATA_MODEL_XG.X_train, DATA_MODEL_XG.Y_train, cv=Global_Parameters.k_folds)
            MODEL_XG.fit(DATA_MODEL_XG.X_train, DATA_MODEL_XG.Y_train,
                           verbose = 1,
                           eval_set = [(DATA_MODEL_XG.X_train, DATA_MODEL_XG.Y_train),
                                       (DATA_MODEL_XG.X_test, DATA_MODEL_XG.Y_test)])
            
            # AVERAGE_DIFFERENCE = np.mean(abs(MODEL_XG.predict(DATA_MODEL_XG.X_test) - DATA_MODEL_XG.Y_test))
            MSLE = tf.keras.losses.MSLE(DATA_MODEL_XG.Y_test, MODEL_XG.predict(DATA_MODEL_XG.X_test))

            # return prediction_score
            # return AVERAGE_DIFFERENCE
            return MSLE

        # Search for best parameters
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_XG, n_trials=Global_Parameters.XG_MODEL_TRIAL, catch=(ValueError,))
        Best_params_XG = np.zeros([1], dtype=object)
        Best_params_XG[0] = study.best_params
        DATA_MODEL_XG.learning_rate = float(Best_params_XG[0].get("learning_rate"))
        DATA_MODEL_XG.max_depth = int(Best_params_XG[0].get("max_depth"))
        DATA_MODEL_XG.gamma = int(Best_params_XG[0].get("gamma"))
        DATA_MODEL_XG.min_child_weights = int(Best_params_XG[0].get("min_child_weight"))
        DATA_MODEL_XG.reg_lambda = int(Best_params_XG[0].get("reg_lambda"))

    DATA_MODEL_XG.XGBoosting_Modellisation(Global_Parameters.k_folds, Global_Parameters.REGRESSION)
    DATA_MODEL_XG.Feature_Importance_Plot()
    if Global_Parameters.REGRESSION == False:
        DATA_MODEL_XG.Evaluation_Metric_Plot_Mlogloss()
        DATA_MODEL_XG.Evaluation_Metric_Plot_Merror()
        
        DATA_MODEL_XG.result_plot_classification()
        DATA_MODEL_XG.result_report_classification_calculation()
        DATA_MODEL_XG.result_report_classification_print()
        DATA_MODEL_XG.result_report_classification_plot()
        
        # DATA_MODEL_XG.Shap_Value_Analysis_Single_Point()
        # DATA_MODEL_XG.Shap_Value_Analysis_Multiple_Point()
        # DATA_MODEL_XG.Shap_Value_Analysis_Multiple_Massive_Point()
    else:  
        DATA_MODEL_XG.result_plot_regression()
        DATA_MODEL_XG.result_report_regression_calculation()
        DATA_MODEL_XG.result_report_regression_print()
        DATA_MODEL_XG.result_report_regression_plot()
        
        DATA_MODEL_XG.extract_max_diff_regression()


# Saving model and information
Global_Parameters.saving_array_replacement()
Global_Data.saving_data_names()

if Global_Parameters.RF_MODEL:
    with open('./models/rf_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_RF.MODEL, f)
elif Global_Parameters.NN_MODEL:
    with open('./models/nn_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_NN.MODEL, f)
elif Global_Parameters.GB_MODEL:
    with open('./models/gb_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_GB.MODEL, f)
elif Global_Parameters.XG_MODEL:
    with open('./models/xg_model.sav', 'wb') as f:
        joblib.dump(DATA_MODEL_XG.MODEL, f)

# # Kaggle competition
# for NAME in ["cap-shape","cap-color","does-bruise-or-bleed","gill-color","stem-color","has-ring","ring-type","habitat"]:
#     Global_Data.TEST_DATAFRAME[NAME] = pd.to_numeric(Global_Data.TEST_DATAFRAME[NAME], errors = "coerce").fillna(0)

# A = pd.DataFrame(DATA_MODEL_XG.MODEL.predict(Global_Data.TEST_DATAFRAME), columns = ["class"])
# # A = pd.DataFrame(np.argmax(DATA_MODEL_NN.MODEL.predict(Global_Data.TEST_DATAFRAME), axis = 1), columns = ["class"])
# A = A.replace(0,"e")
# A = A.replace(1,"p")
# Global_Data.data_import(Global_Parameters.NAME_DATA_PREDICT)
# A.index = Global_Data.TEST_DATAFRAME.id
# A.to_csv("kaggle_compet.csv", index_label = "id")