#Importing all the packages needed
import pandas as pd
import numpy as np
import re
import dask.dataframe as dd
from pandarallel import pandarallel
import datetime

#pandarallel.initialize(progress_bar=True)

#We need to specifiy which log file we want to upload and its path
df_rts_dd = dd.read_csv('C:/Users/laura.finarell/OneDrive - HESSO/Polarizzazione/rts_accesslogs/www.rts.ch_accesslogs_2023-03.0/www.rts.ch_accesslogs.log',sep = ' ', header = None,)
print('Step 1: Uploaded log file')


df_rts = df_rts_dd.compute()
print('Step 2: Convert dask df into pandas df')
print('Shape')
print(df_rts.shape)

df_rts = df_rts.drop([0,1,3,5,6,7,8,10,11,12,13,14,16,17,18],axis = 1)
print('Step 3: Useless columns deleted')

df_rts.columns = ['Timestamp','User_IP','Url','User_Agent']

date = []
for ts in df_rts['Timestamp']:
    date.append(datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'))
    
df_rts['Access_date'] = date
print('Computed date and added to the df')

df_rts.drop('Timestamp',axis =1,inplace=True)
'''
index_video = df_rts.query('Url.str.contains("video", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_video)]
print('Video deleted')
print(len(index_video))
df_rts.reset_index(inplace=True, drop=True)
index_audio = df_rts.query('Url.str.contains("audio", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_audio)]
print('Audio deleted')
print(len(index_audio))
df_rts.reset_index(inplace=True, drop=True)
index_image = df_rts.query('Url.str.contains("image", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_image)]
print('Image deleted')
print(len(index_image))
df_rts.reset_index(inplace=True, drop=True)
index_404 = df_rts.query('Url.str.contains("404", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_404)]
print('404 deleted')
print(len(index_404))
df_rts.reset_index(inplace=True, drop=True)
index_index = df_rts.query('Url.str.contains("index", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_index)]
print('Index deleted')
print(len(index_index))
df_rts.reset_index(inplace=True, drop=True)
'''
index_bot = df_rts.query('User_Agent.str.contains("bot", na= False,case=False)').index
df_rts = df_rts[~df_rts.index.isin(index_bot)]
print('bot deleted')
print(len(index_bot))
df_rts.reset_index(inplace=True, drop=True)

df_rts.drop('User_Agent',axis =1,inplace=True)
print('User agent columns deleted')
#We filter according to the .html urls
index_html = df_rts.query('Url.str.contains(".html$", na= False)').index
print('Html number')
print(len(index_html))
df_rts_html = df_rts[df_rts.index.isin(index_html)]
df_rts_html.reset_index(inplace=True, drop=True)
print('.html extracted')

topics = {'culture': r'culture',
          'economie': r'economie',
          'suisse': r'suisse',
          'monde': r'monde',
          'sciences-tech': r'sciences-tech',
          'sport': r'sport',
          'environnement' : r'environnement'}

def extract_topic_and_id(url):
    if pd.isna(url):
        return 'other', None
    for topic, pattern in topics.items():
        if re.search(pattern, url):
            escenic_id_match = re.search(r'/(\d+)-', url)
            escenic_id = escenic_id_match.group(1) if escenic_id_match else None
            return topic, escenic_id
    return 'other', None  # If no topic matches, return 'other' and None for Escenic ID

# Apply the function to extract the topic and Escenic ID
df_rts_html[['Topic', 'Escenic_ID']] = df_rts_html['Url'].apply(lambda x: pd.Series(extract_topic_and_id(x)))

print('df shape')
print(df_rts_html.shape)
df_rts_html.to_csv('Code_Laura/Reduced_logs/rts_cleaned_0.csv') 

'''
start_indexes = []
for i in range(df_rts_html.shape[0]):
    if re.search('/1',df_rts_html.loc[i,'Url']) is None:
        #print(i)
        start_indexes.append(0)
    else:
        start_indexes.append(re.search('/1',df_rts_html.loc[i,'Url']).span()[0]+1)

df_rts_html['start_indexes'] = start_indexes
print('start indexes extracted') 
df_rts_html = df_rts_html[df_rts_html['start_indexes'] != 0]
df_rts_html.reset_index(inplace=True, drop=True)
escenicID = []
for i in range(len(df_rts_html)):
    item = df_rts_html.loc[i,'start_indexes']
    escenicID.append(df_rts_html.iloc[i].Url[item:item+8])
#We add a column with escenicID
df_rts_html['escenicID']=escenicID
print('escenicID added')
df_rts_html_reduced = df_rts_html.drop('start_indexes',axis = 1)
#A first filtering for escenicID
control = []
for i in df_rts_html_reduced['escenicID']:
    control.append(i.find('1'))
print('If there is no 1 in the escenicID, then drop'
df_rts_html_reduced['control'] = control
df_rts_html_reduced = df_rts_html_reduced.drop(df_rts_html_reduced[df_rts_html_reduced.control != 0].index)
df_rts_html_reduced = df_rts_html_reduced.drop('control', axis=1)
df_rts_html_reduced.reset_index(inplace=True, drop=True)
print('Checked')
'''