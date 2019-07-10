import json
from pathlib import Path

from configuration.config import data_dir

output_path = (Path(data_dir)/'submission_subject.json').open('w')

group = json.load((Path(data_dir)/ 'el_group_word.json').open())

d1 = [json.loads(l) for l in (Path(data_dir)/'ensemble'/'result_9.json').open()]
d2 = [json.loads(l) for l in (Path(data_dir)/'ensemble'/'submission_subject_0706.json').open()]
d3 = [json.loads(l) for l in (Path(data_dir)/'ensemble'/'submission_subject_0707.json').open()]
d4 = [json.loads(l) for l in (Path(data_dir)/'ensemble'/'submission_subject_0710.json').open()]

for x1, x2, x3,x4 in zip(d1, d2, d3,d4):
    x1_mention = [(x['mention'], int(x['offset']), int(x['offset']) + len(x['mention'])) for x in x1['mention_data']]
    x2_mention = [(x[0], x[1], x[1]+len(x[0])) for x in x2['mention_data']]
    x3_mention = [(x[0], x[1], x[1]+len(x[0])) for x in x3['mention_data']]
    x4_mention = [(x[0], x[1], x[1]+len(x[0])) for x in x4['mention_data']]

    mention = list(set(x1_mention + x2_mention + x3_mention+x4_mention))

    new_mention_data = mention.copy()
    for m1 in mention:
        for m2 in mention:
            if m1[1] == m2[1] and m1[2] != m2[2] and len(m1[0]) > len(m2[0]) and m2 in new_mention_data:
                # if len(m1[0]) == 4 and len(m2[0]) == 2 and m1 in new_mention_data and m1[0] in group and (group[m1[0]]['group_labeled_per']==0 or
                #                                             group[m1[0]]['group_labeled_per'] < group[m1[0]]['s_same_per'] or
                #                                             group[m1[0]]['group_labeled_per'] < group[m1[0]]['e_same_per']):
                #     print(f'**{m1} - {new_mention_data}')
                #     new_mention_data.remove(m1)
                # else:
                new_mention_data.remove(m2)

            if m1[1] != m2[1] and m1[2] == m2[2] and len(m1[0]) > len(m2[0]) and m2 in new_mention_data:
                # if len(m1[0]) == 4 and len(m2[0]) == 2 and m1 in new_mention_data and m1[0] in group and (group[m1[0]]['group_labeled_per']==0 or
                #                                             group[m1[0]]['group_labeled_per'] < group[m1[0]]['s_same_per'] or
                #                                             group[m1[0]]['group_labeled_per'] < group[m1[0]]['e_same_per']):
                #     print(f'**{m1} - {new_mention_data}')
                #
                #     new_mention_data.remove(m1)
                # else:
                new_mention_data.remove(m2)

    new_mention_data = [[m[0], m[1]] for m in new_mention_data]

    doc = {
        'text_id': x1['text_id'],
        'text': x1['text'],
        'mention_data': new_mention_data
    }

    output_path.write(json.dumps(doc, ensure_ascii=False) + '\n')




