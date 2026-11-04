"""Download the RamanSTS dataset from Harvard Dataverse into ./data/dataset.

Usage:
    python download_data.py              # download, verify MD5, extract, delete the zips
    python download_data.py --keep-zips  # keep the downloaded zips in ./data/downloads

Dataset: RamanSTS, https://doi.org/10.7910/DVN/NGEZPA (CC BY-NC 4.0).
Only the Python standard library is used, so this can run before installing dependencies.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
import zipfile

# Same locations as config.py (not imported here because it needs pandas)
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
SERVER = 'https://dataverse.harvard.edu'
DOI = 'doi:10.7910/DVN/NGEZPA'
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'downloads')
EXTRA_FILES = ('map_index.csv',)  # copied to data/dataset/ when present in the record
# Dataverse rejects urllib's default User-Agent with HTTP 403
USER_AGENT = 'RamanCNN-download/1.0 (+https://github.com/VahidDanesh/RamanCNN)'


def urlopen(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': USER_AGENT}),
                                  timeout=60)


def list_files():
    """Return the files of the latest published version: [{id, name, size, md5}]."""
    url = f'{SERVER}/api/datasets/:persistentId/versions/:latest-published/files?persistentId={DOI}'
    with urlopen(url) as r:
        files = json.load(r)['data']
    return [dict(id=f['dataFile']['id'], name=f['label'], size=f['dataFile']['filesize'],
                 md5=f['dataFile'].get('md5')) for f in files]


def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def download(file, dest):
    """Download one file (original format, not Dataverse's ingested copy) and check its MD5."""
    if os.path.exists(dest) and md5sum(dest) == file['md5']:
        print(f'{file["name"]}: already downloaded')
        return
    url = f'{SERVER}/api/access/datafile/{file["id"]}?format=original'
    part = dest + '.part'
    with urlopen(url) as r, open(part, 'wb') as out:
        done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            print(f'\r{file["name"]}: {done / 1e6:,.0f} / {file["size"] / 1e6:,.0f} MB',
                  end='', flush=True)
    print()
    got = md5sum(part)
    if file['md5'] and got != file['md5']:
        os.remove(part)
        sys.exit(f'{file["name"]}: MD5 mismatch (expected {file["md5"]}, got {got}); '
                 'please run the script again.')
    os.replace(part, dest)


def extract(zip_path, out_dir):
    with zipfile.ZipFile(zip_path) as z:
        for member in z.infolist():
            target = os.path.realpath(os.path.join(out_dir, member.filename))
            if not target.startswith(os.path.realpath(out_dir) + os.sep):
                sys.exit(f'Unsafe path in {zip_path}: {member.filename}')
        z.extractall(out_dir)
    print(f'{os.path.basename(zip_path)}: extracted to {out_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--keep-zips', action='store_true',
                    help='keep the downloaded zip files in data/downloads')
    args = ap.parse_args()

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)
    files = list_files()
    zips = [f for f in files if f['name'].endswith('.zip')]
    extras = [f for f in files if f['name'] in EXTRA_FILES]
    if not zips:
        sys.exit(f'No zip files found in {DOI}')
    print(f'Downloading {len(zips)} archives '
          f'({sum(f["size"] for f in zips) / 1e9:.1f} GB) from {SERVER}')

    for f in zips + extras:
        dest = os.path.join(DOWNLOAD_DIR, f['name'])
        download(f, dest)
        if f in zips:
            extract(dest, DATASET_DIR)
            if not args.keep_zips:
                os.remove(dest)
        else:
            shutil.copy2(dest, os.path.join(DATASET_DIR, f['name']))

    npy = [os.path.join(d, n) for d in sorted(os.listdir(DATASET_DIR))
           if os.path.isdir(os.path.join(DATASET_DIR, d))
           for n in sorted(os.listdir(os.path.join(DATASET_DIR, d))) if n.endswith('.npy')]
    print(f'Done: {len(npy)} .npy files in {DATASET_DIR}')


if __name__ == '__main__':
    main()
