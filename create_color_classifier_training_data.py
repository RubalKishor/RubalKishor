import json
from webcolors import (CSS3_HEX_TO_NAMES,hex_to_rgb,)
import pandas as pd 
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle

class RGBTOCOLOR(object):
    def __init__(self, client):
        color_conf_filepath = f"configs/{client}/color_conf.json"
        with open(color_conf_filepath, "r") as rj:
            self.color_conf = json.load(rj)
        self.css3_db = { k : v for k, v in CSS3_HEX_TO_NAMES.items()
                if v in self.color_conf["color_classes"]
                }

    def convert_rgb_to_color(self, rgb_tuple):
        names = []
        rgb_values = []
        for color_hex, color_name in self.css3_db.items():
            names.append(color_name)
            rgb_values.append(hex_to_rgb(color_hex))
        kdt_db = KDTree(rgb_values)
        distance, index = kdt_db.query(rgb_tuple)
        return names[index]


    def run(self, df):
        df["color"] = df.apply(lambda row :
                self.convert_rgb_to_color(
                    (row["red"], row["green"], row["blue"])
                    ),
                axis=1
                )
        return df


class BaseTrainColorClassifier(object):
    def __init__(self, args):
        self.creative_obj = locals()
        self.client = args["client"]
        conf_filepath = f"configs/{self.client}/config.json"
        with open(conf_filepath, "r") as rj:
            config = json.load(rj)
        self.creative_obj.update(args)
        self.creative_obj.update(config)
        self.rgb_to_color_obj = RGBTOCOLOR(self.client)
        color_conf_filepath = f"configs/{self.client}/color_conf.json"
        with open(color_conf_filepath, "r") as rj:
            self.color_conf = json.load(rj)



    def prepare_data(self):
        raw_df = self.dataframe
        raw_df["RESPONSE"] = raw_df["RESPONSE"].apply(lambda x: json.loads(x))        
        li = []
        for i, row in raw_df.iterrows():
            for color_di in row["RESPONSE"]["dominantColors"]["colors"]:
                #di = {"ID" : row["ID"]}
                di = color_di["color"]
                li.append(di)
        df = pd.DataFrame(li)
        df = self.rgb_to_color_obj.run(df)
        df = pd.get_dummies(df, columns=['color'])
        train_df = df.sample(frac=0.8, random_state=8)
        test_df = df.drop(train_df.index)
        return train_df, test_df


    def train(self, df):
        clf = KNeighborsClassifier(n_neighbors=1)
        label_cols = [col 
               for col in df.columns 
               if col not in ("red", "green", "blue")
               ]
        labels_df = pd.DataFrame([
            df.pop(x) for x in label_cols
            ]).T

        clf.fit(df, labels_df)
        accuracy = clf.score(df, labels_df)
        
        print(accuracy)
        return clf, labels_df
    
    def sav_model(self,df):
        model=self.train(df)
        with open('model_pkl','wb') as file:
            pickle.dump(model,file)
        return file
    
    def load_model(self):
        load_model=self.sav_model(df)
        pickle.load(open(load,'rb'))
        result=load_model.score(train_dataset, train_labels)
        print(result)

    def evaluate(self, pred_df, act_df, train_df, train_labels_df):
        print(classification_report(act_df, pred_df, 
            target_names=self.color_conf["color_classes"]))
           
        knn2 = KNeighborsClassifier()
        param_grid = {"n_neighbors": np.arange(1, 25)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        knn_gscv.fit(train_df, train_labels_df)
        print(knn_gscv.best_params_)
        print(knn_gscv.best_score_)


    def test(self, df, clf):
        label_cols = [col
                for col in df.columns
                if col not in ("red", "green", "blue")
                ]
        labels_df = pd.DataFrame([
            df.pop(x) for x in label_cols 
            ]).T
        test_predictions = clf.predict(df)
        predicted_encoded_test_labels = np.argmax(test_predictions, axis=1)
        pred_enc_label_df = pd.DataFrame(
                predicted_encoded_test_labels, columns=['Predicted Labels'])
        actual_encoded_test_labels = np.argmax(labels_df.to_numpy(), axis=1)
        act_enc_label_df = pd.DataFrame(
        actual_encoded_test_labels, columns=['Actual Labels'])
        return pred_enc_label_df, act_enc_label_df, labels_df



    def run(self):
        train_df, test_df = self.prepare_data()
        #import pdb;pdb.set_trace()
        clf, train_labels_df = self.train(train_df) 
        pred_enc_label_df, act_enc_label_df, test_labels_df = self.test(test_df, clf)
        self.evaluate(pred_enc_label_df, act_enc_label_df, train_df, train_labels_df)
        


    
