Data_Preparation folder:

1. pipelineUploadCleanReduce.py to upload, filter and reduce the logs files with only the logs about articles.
This passage must be done for each log file.

2. join_log_article.ipynb will join all the cleaned logs into one file 'logs_cleaned_all.csv' + will merge them with the article dataset
into a new file 'Dataset_Users_Articles_all.csv'

Articles folder:

script to open the article file and .csv and .xlsx file. 
There is also an improved version of the file with the field UserNeeds cleaned and ready to use.

