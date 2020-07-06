# -*- coding: utf-8 -*-
import os
from os import scandir
import time
import sys
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

if len(sys.argv) == 1 :
  print("Missing path of the documents to summarize")
  exit()

if len(sys.argv) > 2 :
  print("Too many arguments. Indicate path of documents to summarize")
  exit()

init_t = time.perf_counter()

# List reports function
def ls(path):
    return [report.name for report  in scandir(path)]

# Source annual reports path
path = sys.argv[1]
#path = 'testing/annual_reports/'


#get report names list
report_list = ls(path)

# Generated summaries destination path
new_dir_path = 'HULAT_Summaries'

# Create directory to upload the summaries
os.mkdir(new_dir_path)

# Summarize each report from the source path
for r in report_list:
   
   # Report complete path
   r_path = os.path.join(path, r)

   # Call Summarizer component to summarize each report specifying source and destination path
   os.system('python3 FNS_Summarizer1.py '+ r_path + ' ' + new_dir_path)  

# Print processing time
end_t = time.perf_counter()
print(f"Total time:     {end_t - init_t:0.4f} seconds")
