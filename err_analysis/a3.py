import collections

import pymongo

train_db = pymongo.MongoClient()['local_db'].get_collection('train')

all_m_list = []
all_m_list_s = []
for doc in train_db.find({}):
    old = doc.copy()
    m_list = [m['mention'] for m in doc['mention_data']]
    all_m_list.extend(m_list)
    all_m_list_s.extend(list(set(m_list)))
    doc.update({'mention_list': m_list})

    print('tst')
    train_db.update_one(filter=old,update={'$set':doc},upsert=True)

for a in collections.Counter(all_m_list).most_common(100):
    print(a)

print('*'*20)
for b in collections.Counter(all_m_list_s).most_common(100):
    print(b)
