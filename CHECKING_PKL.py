import pickle
import pandas as pd
import os
from datetime import datetime

if __name__ == '__main__':
    #df = pd.read_pickle("Action-net/data/EMG_data/S00_2.pkl")
    #df = pd.read_pickle("Action-net/data/annotations/ActionNet_test.pkl")
    #df = df[df['file'] == 'S04_1.pkl']
    df = pd.read_pickle("Action-net/data/S04_test.pkl") #S04_train.pkl #S04_test.pkl
    print(df)
    print(df.shape)
    print(df.columns)  
    #print(df['features_EMG']) 
    print(df[['description_class', 'description']].drop_duplicates(subset=['description_class']).sort_values(by='description_class'))
    #unique_samples = df[['description',  "description_class" ]].drop_duplicates(subset='description_class')
    #print(unique_samples.sort_values(by='description_class'))
    
    #print to txt
    # columns_to_print = ['index', 'description', "labels", "description_class", "start_frame", "stop_frame", "features_EMG"]
    # output_name = "S04_train"
    # file_path = 'C:/Users/emanu/Downloads/S04_video/'+ output_name + '.csv'
    # df[columns_to_print].head(1).to_csv(file_path, sep='\t', index=False)




    # initial_timestamp = '14 June 2022 at 16:38:20'
    # initial_datetime = datetime.strptime(initial_timestamp, '%d %B %Y at %H:%M:%S')
    # print(initial_datetime.timestamp())
    
    # timestamp = 1655239114.118433 #1655239123.020082 #1655239114.118433

    # dt_object = datetime.fromtimestamp(timestamp)
    # print(dt_object)