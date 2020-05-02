import shutil
import os


N_CATEGORIES=15
CLASS_NAMES = ['None', 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
LABEL_PATH='./labels/test_list.txt'
IMAGE_PATH='./images/'

files_per_category=[]



files_per_category = [ [] for i in range(N_CATEGORIES) ]


with open(LABEL_PATH) as f:
    for line in f:
        has_diagnosis= False
        line_data=line.split(' ')
        for i in range(line_data[1:]):
            if line_data[i]:
                files_per_category[i].append(line_data)
                has_diagnosis=True
        if not has_diagnosis:
            files_per_category[0].append(line_data)


for i in range(N_CATEGORIES):
    if not os.path.exists(IMAGE_PATH+CLASS_NAMES[i]):
        os.mkdir(IMAGE_PATH+CLASS_NAMES[i])
    for file in files_per_category[i]:
        shutil.copy(IMAGE_PATH+file,IMAGE_PATH+CLASS_NAMES[i]+'/'+file)
        
