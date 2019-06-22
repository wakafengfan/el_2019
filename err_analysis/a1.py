import json
from pathlib import Path

from configuration.config import data_dir

err = json.load((Path(data_dir)/'err_log/err_log_0_2.json').open())['err']

p_less, t_less, eq = 0,0,0
p_less_list, t_less_list, eq_list = [],[],[]
for e in err:
    if len(e['predict'])<len(e['mention_data']):
        p_less+=1
        p_less_list.append(e['predict'])
    elif len(e['mention_data'])< len(e['predict']):
        t_less+=1
        t_less_list.append(e['mention_data'])
    else:
        eq+=1
        eq_list.append((e['text'],e['mention_data'],e['predict']))

print(f'p_less:{p_less}, t_less:{t_less}, eq:{eq}')
print('done')
