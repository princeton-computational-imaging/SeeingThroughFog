import os
import pickle

from pathlib import Path

dir_path = Path(os.path.dirname(os.path.realpath(__file__)))

pkl_dir = dir_path.parent.parent.parent / 'data' / 'dense'

save_file = f'{pkl_dir}/dense_dbinfos_train_clear.pkl'

day_file = f'{pkl_dir}/dense_dbinfos_train_clear_day.pkl'
night_file = f'{pkl_dir}/dense_dbinfos_train_clear_night.pkl'

with open(str(day_file), 'rb') as df:
    day_dict = pickle.load(df)

with open(str(night_file), 'rb') as nf:
    night_dict = pickle.load(nf)

save_dict = {}

for key in day_dict:

    save_dict[key] = day_dict[key] + night_dict[key]

with open(save_file, 'wb') as f:
    pickle.dump(save_dict, f)

for stage in ['train', 'val', 'trainval']:

    save_file = f'{pkl_dir}/dense_infos_{stage}_clear.pkl'

    day_file = f'{pkl_dir}/dense_infos_{stage}_clear_day.pkl'
    night_file = f'{pkl_dir}/dense_infos_{stage}_clear_night.pkl'

    with open(str(day_file), 'rb') as df:
        day_infos = pickle.load(df)

    with open(str(night_file), 'rb') as nf:
        night_infos = pickle.load(nf)

    with open(save_file, 'wb') as f:
        pickle.dump(day_infos + night_infos, f)


for condition in ['clear', 'light_fog', 'dense_fog', 'snow']:

    save_file = f'{pkl_dir}/dense_infos_test_{condition}.pkl'

    day_file = f'{pkl_dir}/dense_infos_test_{condition}_day.pkl'
    night_file = f'{pkl_dir}/dense_infos_test_{condition}_night.pkl'

    with open(str(day_file), 'rb') as df:
        day_infos = pickle.load(df)

    with open(str(night_file), 'rb') as nf:
        night_infos = pickle.load(nf)

    with open(save_file, 'wb') as f:
        pickle.dump(day_infos + night_infos, f)