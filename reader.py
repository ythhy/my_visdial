import os

import sys

sys.path.append(os.getcwd())

import json
import h5py
import numpy as np



# result_f = open('/home/yth/visDial/challenge_result/challenge/result8-10-20-25valcoatt-rnd3.json')
# result = json.load(result_f)

data_f = open('/home/yth/visDial/data/visdial_1.0_val.json')
data = json.load(data_f)['data']
dialogs = data['dialogs']
questions = data['questions']
answers = data['answers']

i=0
samples=[]
for i in range(0, 2000):

    sample={}

    sample['i'] = i
    if i ==85:
        opt = dialogs[i]['dialog'][3]['answer_options']
        for j in range(100):
            print(answers[opt[j]])
        print('done')
    sample['caption'] = dialogs[i]['caption']
    sample['image'] = str(dialogs[i]['image_id'])
    sample['dialog'] = []
    for rnd in range(10):
        ques_id = dialogs[i]['dialog'][rnd]['question']
        question_i = questions[ques_id]+'?    '
        ans_id = dialogs[i]['dialog'][rnd]['answer']
        ans_i = answers[ans_id]
        sample['dialog'].append(question_i+ans_i)

    samples.append(sample)
file_dir = 'val_iccv.json'
with open(file_dir, 'w') as f:
    result_w = json.dumps(samples,indent=2)
    f.write(result_w)
f.close()
print('done')