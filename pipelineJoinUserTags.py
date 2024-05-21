import pandas as pd
import numpy as np
import re
from pandarallel import pandarallel

#pandarallel.initialize(progress_bar=True)

df_complete = pd.read_csv('C:/Users/laura.finarell/OneDrive - HESSO/Polarizzazione/Code_Laura/pipeline/Dataset_User_Articles_all.csv',usecols=['User_IP','Access_date','publishDate','iptcTags','escenicID'],index_col=[0])

#We want to have a correspondance 1-1 user-article
df_complete = df_complete.drop_duplicates(subset=['User_IP','escenicID'],keep='first')
df_complete = df_complete.reset_index(drop=True)
print('Duplicates deleted, number of rows')
print(df_complete.shape[0])


df_group_by = df_complete.groupby(['User_IP','publishDate'],as_index = False).count()

df_group_by = df_group_by.drop(df_group_by[df_group_by['iptcTags'] > 50].index)

User_to_keep = df_group_by['User_IP'].unique()

df_complete_filtered = df_complete[df_complete['User_IP'].isin(User_to_keep)]
print('filtered users with more than 50 articles per day')
print(df_complete_filtered.shape[0])

df_user_tags = pd.concat([df_complete_filtered['User_IP'],df_complete_filtered['iptcTags']],axis=1)

df_user_tags.to_csv('user_alltags.csv')

df_user_tags['iptcTags'] = df_user_tags['iptcTags'].str.replace('[','')
df_user_tags['iptcTags'] = df_user_tags['iptcTags'].str.replace(']','')

df_user_tags['iptcTags'] = df_user_tags.groupby(['User_IP'])['iptcTags'].transform(','.join)

tags_to_keep = ['Arts, culture, divertissement','Conflit, guerre et paix','Criminalité, droit et justice','Désastres et accidents','Environnement','Gens animaux insolite','Météo','Politique','Religion et croyance','Santé','Science et technologie','Social','Société','Sport','Vie quotidienne et loisirs','Économie et finances','Éducation']

df_user_tags = df_user_tags.drop(df_user_tags[df_user_tags['iptcTags'].str.contains('|'.join(tags_to_keep))==False].index)
df_user_tags = df_user_tags.reset_index( drop=True)

iptcTags_filtered = []
for i in range(df_user_tags.shape[0]):
    iptcTags_filtered.append(re.findall('|'.join(tags_to_keep),df_user_tags.iloc[i,1]))

df_user_tags['iptcTags_filtered'] = iptcTags_filtered
print('filtering tags')

df_user_tags_filtered = df_user_tags.drop('iptcTags',axis=1)

#Now we have to count the occurences of each tag for each user and store the frequency of a tag in the corresponding column
counter = []
for i in range(df_user_tags_filtered.shape[0]):
    unique,counts = np.unique(df_user_tags_filtered.iloc[i,1],return_counts=True)
    counter.append(dict(zip(unique, counts)))
print('frequency computed')
df_user_tags_filtered['counter'] = counter

df_to_cluster = pd.DataFrame(df_user_tags_filtered['counter'].values.tolist(),index=df_user_tags_filtered['User_IP']).fillna(0).astype(int)

df_to_cluster.to_csv('user_17tagsfrequency.csv')