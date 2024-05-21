pipelines:

1. pipelineUploadCleanReduce.py to upload, filter and reduce the logs files with only the logs about articles.
This passage must be done for each log file.

2. join_log_article.ipynb will join all the cleaned logs into one file 'logs_cleaned_all.csv' + will merge them with the article dataset
into a new file 'Dataset_Users_Articles_all.csv'

3. pipelineJoinUserTags.py load the previous dataset and delete the duplicates row so that we have the same article per user once.
We also can filter the number of articles per day. The we create a dataset with users and all tags in 'user_alltags.csv'.
We then chose 17 of them and we compute for them the relative frequencies. The final dataset is 'user_17tagsfrequency.csv'.

4. clustering.ipynb is a jupyter notebook with a preliminary clustering

Articles folder:

script to open the article file and .csv and .xlsx file. 
There is also an improved version of the file with the field UserNeeds cleaned and ready to use.

Debug folder:

Old codes with preliminary data analysis

Graph folder:

Codes for graph visualization

Reduced_logs:

files containing the server logs cleaned

UserNeed folder:

Analysis on user need per topic 
