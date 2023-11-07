import os
os.chdir("/usr/local/micapollo01/MIC/DATA/STAFF/bbuyuk0/VisualAttentionProjects/DentalStagingVIT/")
import shutil

roots= ['./02 Results/training_attentions','./02 Results/test_results','./02 Results/models']

for root in roots:
    folders = [i for i in os.listdir(root) if i.__contains__('1107')]
    for folder in folders:
        shutil.rmtree(root+"/"+folder)
