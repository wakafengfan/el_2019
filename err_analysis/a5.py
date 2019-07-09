import json
from pathlib import Path

from configuration.config import data_dir


def remove_duplication():
    # res_path = (Path(data_dir)/'result.json').open('w')
    for l in (Path(data_dir)/'submission_object.json').open():
        l = json.loads(l)
        mention_data = l['mention_data']
        for m in mention_data:
            m.update({'end_idx':int(m['offset'])+len(m['mention'])})

        new_mention_data = mention_data.copy()
        for m1 in mention_data:
            for m2 in mention_data:
                if m1['offset'] == m2['offset'] and m1['end_idx'] != m2['end_idx'] and len(m1['mention']) > len(m2['mention']) and m2 in new_mention_data:
                    print(f'**{m2} ---- {m1}')
                    new_mention_data.remove(m2)

                if m1['offset'] != m2['offset'] and m1['end_idx'] == m2['end_idx'] and len(m1['mention']) > len(m2['mention']) and m2 in new_mention_data:
                    print(f'**{m2} ---- {m1}')
                    new_mention_data.remove(m2)

        for m in new_mention_data:
            del m['end_idx']

        l['mention_data'] = [m for m in new_mention_data if m['kb_id']!='NIL']

        # res_path.write(json.dumps(l, ensure_ascii=False) + '\n')


def remove_nil():
    res_path = (Path(data_dir) / 'result_.json').open('w')
    for l in (Path(data_dir) / 'result.json').open():
        l = json.loads(l)
        l['mention_data'] = [m for m in l['mention_data'] if m['kb_id'] != 'NIL']

        res_path.write(json.dumps(l, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    remove_duplication()

