from vqaTools.vqa import VQA
import json
import pandas as pd
import numpy as np
from images_encoder import get_image_features
from questions_encoder import get_question_features


data_dir = 'Dataset'
version_type = '' # this should be '' when using VQA v2.0 dataset
task_type = 'MultipleChoice' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
data_type = 'abstract_v002'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.
# dataSubType ='train2015'
vqa_train_file = 'vqa_train.json'
vqa_train_images = 'vqa_train_images.pkl'
vqa_train_questions = 'vqa_train_question.pkl'
vqa_val_file = 'vqa_val.json'
vqa_val_images = 'vqa_val_images.pkl'
vqa_val_questions = 'vqa_val_questions.pkl'


def create_combined_dataset(sub_type, output_file):

    ann_file = '%s/Annotations/%s%s_%s_annotations.json'%(data_dir, version_type, data_type, sub_type)
    ques_file = '%s/Questions/%s%s_%s_%s_questions.json'%(data_dir, version_type, task_type, data_type, sub_type)
    img_dir = '%s/Images/scene_img_%s_%s/' %(data_dir, data_type, sub_type)

    train_data = []

    vqa = VQA(ann_file, ques_file)
    im_ids = vqa.getImgIds() 

    for im_id in list(set(im_ids)):
        ann_ids = vqa.getQuesIds(imgIds=im_id, ansTypes='yes/no')
        anns = vqa.loadQA(ann_ids)
        for ann in anns:
            # print(ann)
            row = {}
            row['question_id'] = ann['question_id']
            row['image_id'] = ann['image_id']
            ques, multiple_choices, ans = vqa.getQA(ann) 
            row['question'] = ques
            # row['mc_answers'] = multiple_choices
            row['answer'] = ans
            row['image_file'] = img_dir + data_type + '_' + sub_type + '_'+ str(im_id).zfill(12) + '.png'
            train_data.append(row)

    json.dump(train_data, open(output_file, 'w'))
    return pd.DataFrame(train_data)


def main():

    #Prepare data for training
    vqa_train_df = create_combined_dataset('train2015', vqa_train_file)
    get_image_features(vqa_train_file, vqa_train_images)
    get_question_features(vqa_train_file, vqa_train_questions)

    #Prepare data for validation
    vqa_val_df = create_combined_dataset('val2015', vqa_val_file)
    get_image_features(vqa_val_file, vqa_val_images)
    get_question_features(vqa_val_file, vqa_val_questions)