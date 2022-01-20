from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account
import json


class BaseFeaturizer(object):
    
    def __init__(self,creative_obj):
        self.creative_obj=creative_obj
        self.project_id=self.creative_obj["big_query"]["project_name"]
        self.creds=service_account.Credentials.from_service_account_file(
            self.creative_obj["big_query"]["cred_filepath"],)
        self.feature_dict= {
            "score" : self.find_img_score,
            "pixel_fraction" :self.find_pixel_fraction
            }
        self.schema =[
            {"name":"ID", "type": "INTIGER"},
            {"name":"score","type":"FLOAT"},
            {"name":"Pixel_fraction","type":"FLOAT"},
            {"name":"color","type":"STRING"}
            ]
        
        self.bq_dataset=creative_obj["big_query"]["dataset"]
    
    
    def find_img_score(self,_id,img_data):
        score=img_data.get("score",0)
        score=round(score,2)
        return score
    
    def find_pixel_fraction(self,_id,img_data):
        pixel_fraction=img_data("pixel_fraction",0)
        pixel_fraction=round(pixel_fraction,2)
        return pixel_fraction
    
    def store_in_gbq(self,img_data):
        #TODO: new record
        table_name="COLOR_FEATURE"
        table_path=f"{self.bq_dataset}.{table_name}"
        
        img_data.to_gbq(
            table_path,self.project_id,
            if_exist="replace",Credentials=self.creds,
            table_schema=self.schema
            )
        
    def transform(self,img_data):
        cols_to_keep = list(img_data.columns)
        cols_to_keep.sort()
        cols_to_keep.remove("ID")
        cols_to_keep = ["ID"] + cols_to_keep
        img_data = img_data[cols_to_keep]
        return img_data
    
    def run(self,img_prop_df):
        self.img_prop_df = img_prop_df
        raw_df = self.dataframe
        img_df = []
        raw_df["RESPONSE"] = raw_df["RESPONSE"].apply(lambda x: json.loads(x))
        import pdb;
        pdb.set_trace()
        for i, row in raw_df.iterrows():
            for resp in row["RESPONSE"]:
                tmp_di = {k : None for k in self.feature_dict}
                tmp_di["ID"] = row["ID"]
                self.img_obj = self.img_prop_df.loc[self.img_prop_df["ID"] == row["ID"], "image_obj"].values[0]
              
            for feat_name in self.feature_dict:
                tmp_di[feat_name] = self.feature_dict[feat_name](row["ID"], resp)
            img_df.append(tmp_di)
    
    
        img_df = pd.DataFrame(img_df)
        img_df = self.transformations(img_df)
        self.store_in_gbq(img_df)