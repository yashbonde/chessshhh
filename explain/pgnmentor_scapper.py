import requests
import os
import json
import re

html = open('pgnmentor_files.html', 'r', encoding = 'utf-8').read()
rows = [line for line in html.split('\n') if 'events/' in line or 'openings/' in line or 'players/' in line]

all_files = []
re_pat_zip = r'\".*\.zip\"\>'
re_pat_pgn = r'\".*\.pgn\"\>'
for row in rows:
    if re.search(re_pat_zip, row) or re.search(re_pat_pgn, row):
        for pat in re.findall(re_pat_zip, row):
            file_ = re.sub(r'\"|\>', '', pat)
            all_files.append(file_)
            
        for pat in re.findall(re_pat_pgn, row):
            file_ = re.sub(r'\"|\>', '', pat)
            all_files.append(file_)
            
sub_folders = ['players', 'openings', 'events']
master_folder = 'games_data_pgns'
if not os.path.exists(master_folder):
    os.makedirs(master_folder)
    for sf in sub_folders:
        sf_full = master_folder + '/' + sf
        if not os.path.exists(sf):
            print("Making : {}".format(sf_full))
            os.makedirs(sf_full)

base_url = 'https://www.pgnmentor.com/{file_ext}'

for ext_idx, ext in enumerate(all_files):
    url = base_url.format(file_ext = ext)
    mode = 'wb'
    encoding = 'utf-8' if mode == 'w' else None
    fpath = '{master_folder}/{ext}'.format(master_folder = master_folder, ext = ext)
    if os.path.exists(fpath) or os.path.exists(fpath.replace('zip', 'pgn')):
        print('{}/{} Continuing {} --> Already Present'.format(ext_idx, len(all_files), ext))
        continue
    print('{}/{} -> {} Downloading...'.format(ext_idx, len(all_files), mode), url, end = ' ...')
    try:
        r = requests.get(url)
        open('{master_folder}/{ext}'.format(master_folder = master_folder, ext = ext), mode, encoding = encoding).write(r.content)
        print('Done')
    except Exception as e:
        print('Skipping: {} --> Issues: {}'.format(ext, e))