""" This script is used to download the DL3DV-10 dataset for all resolution levels from the huggingface repo.
    As the whole dataset is too large for most users, we provide this script so that you can download the dataset efficiently based on your needs.
    We provide several options to download the dataset (image frames with poses):
        - [X] Resolution level: 4K, 2K, 960P, 480P  
        - [X] Subset of the 10K, e.g. 1K(0~1K), 2K(1K~2K), 3K(2K~3K), etc
        - [X] specific hash 
        - [X] file_type: raw video | images+poses | colmap cache 

    Notes:
        - file_type + resolution will decide which dataset repo to download the files 
        - subset will decide which subdir will be used 
        - if hash is set, only the specific hash will be downloaded

"""

import os
import threading
from os.path import join
import csv
from tqdm import tqdm
from huggingface_hub import HfApi 
import argparse
import re
import traceback
import shutil
import urllib.request
import zipfile
from huggingface_hub import HfFileSystem

api = HfApi()

# Serialize hub download + optional cache wipe: concurrent threads (streaming prefetch)
# cannot share hf_hub_download or shutil.rmtree(.cache) on the same odir.
_HF_HUB_DOWNLOAD_LOCK = threading.Lock()

resolution2repo = {
    '480P': 'DL3DV/DL3DV-ALL-480P',
    '960P': 'DL3DV/DL3DV-ALL-960P',
    '2K': 'DL3DV/DL3DV-ALL-2K',
    '4K': 'DL3DV/DL3DV-ALL-4K'
}
SAFE_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_path_component(value: str, label: str) -> str:
    """Allow one plain filename/path component from remote metadata."""
    if not value or not SAFE_PATH_COMPONENT.fullmatch(value):
        raise ValueError(f"Unsafe {label}: {value!r}")
    if value in {".", ".."}:
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value

def verify_access(repo: str):
    """ This function can be used to verify if the user has access to the repo. 

    :param repo: the repo name  
    :return: True if the user has access, False otherwise
    """    
    fs = HfFileSystem()
    try:
        fs.ls(f'datasets/{repo}')
        return True
    except BaseException as e:
        return False


def hf_download_path(repo: str, rel_path: str, odir: str, max_try: int = 5):
    """ hf api is not reliable, retry when failed with max tries

    :param repo: The huggingface dataset repo 
    :param rel_path: The relative path in the repo
    :param odir: output path 
    :param max_try: As the downloading is not a reliable process, we will retry for max_try times
    """	
    counter = 0
    while True:
        if counter >= max_try:
            print(f"ERROR: Download {repo}/{rel_path} failed.")
            return False
        try:
            api.hf_hub_download(
                repo_id=repo,
                filename=rel_path,
                repo_type="dataset",
                local_dir=odir,
                cache_dir=join(odir, ".cache"),
            )
            return True

        except KeyboardInterrupt:
            print('Keyboard Interrupt. Exit.')
            exit()
        except BaseException as e:
            traceback.print_exc()
            counter += 1
            # print(f'Downloading summary {counter}')
    

def download_from_url(url: str, ofile: str):
    """ Download a file from the url to ofile 

    :param url: The url link 
    :param ofile: The output path 
    :return: True if download success, False otherwise
    """    
    try:
        # Use urllib.request.urlretrieve to download the file from `url` and save it locally at `local_file_path`
        urllib.request.urlretrieve(url, ofile)
        return True
    except Exception as e:
        print(f"An error occurred while downloading the file: {e}") 
        return False


def clean_huggingface_cache(output_dir: str, repo: str):
    """ Huggingface cache may take too much space, we clean the cache to save space if necessary

        Current huggingface hub does not provide good practice to clean the space.  
        We mannually clean the cache directory if necessary. 

    :param output_dir: the current output directory 
    :param output_dir: the huggingface repo 
    """    
    repo_cache_dir = repo.replace('/', '--')
    # cur_cache_dir = join(output_dir, '.cache', f'datasets--{repo_cache_dir}')
    cur_cache_dir = join(output_dir, '.cache')

    if os.path.exists(cur_cache_dir):
        shutil.rmtree(cur_cache_dir)


def safe_extract_zip(zip_ref: zipfile.ZipFile, output_dir: str):
    """Extract a zip archive after blocking path traversal entries."""
    output_root = os.path.abspath(output_dir)
    for info in zip_ref.infolist():
        member = info.filename
        target_path = os.path.abspath(os.path.join(output_root, member))
        if target_path != output_root and not target_path.startswith(output_root + os.sep):
            raise RuntimeError(f"Unsafe zip entry blocked: {member}")
        file_type = (info.external_attr >> 16) & 0o170000
        if file_type == 0o120000:
            raise RuntimeError(f"Unsafe zip symlink blocked: {member}")
    zip_ref.extractall(output_root)


def get_download_list(subset_opt: str, hash_name: str, reso_opt: str, file_type: str, output_dir: str):
    """ Get the download list based on the subset and hash name

        1. Get the meta file   
        2. Select the subset. Based on reso_opt, get the downloading list prepared. 
        3. Return the download list.

    :param subset_opt: Subset of the 10K, e.g. 1K(0~1K), 2K(1K~2K), 3K(2K~3K), etc
    :param hash_name: If provided a non-empty string, ignore the subset_opt and only download the specific hash 
    :param reso_opt: The resolution to download. 
    :param file_type: The file type to download: video | images+poses | colmap_cache  
    :param output_dir: The output directory. 
    """    
    def to_download_item(hash_name, reso, batch, file_type):
        hash_name = validate_path_component(hash_name, "scene hash")
        batch = validate_path_component(batch, "batch")
        if file_type == 'images+poses':
            repo = resolution2repo[reso]
            rel_path = f'{batch}/{hash_name}.zip'
        elif file_type == 'video':
            repo = 'DL3DV/DL3DV-ALL-video'
            rel_path = f'{batch}/{hash_name}/video.mp4'
        elif file_type == 'colmap_cache':
            repo = 'DL3DV/DL3DV-ALL-ColmapCache'
            rel_path = f'{batch}/{hash_name}.zip'

        # return f'{repo}/{batch}/{hash_name}'
        return { 'repo': repo, 'rel_path': rel_path }

    ret = []

    meta_link = 'https://raw.githubusercontent.com/DL3DV-10K/Dataset/main/cache/DL3DV-valid.csv'
    cache_folder = join(output_dir, '.cache') 
    meta_file = join(cache_folder, 'DL3DV-valid.csv')
    os.makedirs(cache_folder, exist_ok=True)
    if not os.path.exists(meta_file):
        assert download_from_url(meta_link, meta_file), 'Download meta file failed.'

    with open(meta_file, newline="") as f:
        rows = list(csv.DictReader(f))

    # if hash is set, ignore the subset_opt
    if hash_name != '':
        match = next((r for r in rows if r.get('hash') == hash_name), None)
        assert match is not None, f'Hash {hash_name} not found in the meta file.'

        batch = match['batch']
        link = to_download_item(hash_name, reso_opt, batch, file_type)
        ret = [link]
        return ret

    # if hash not set, we download the whole subset
    for r in rows:
        if r.get('batch') != subset_opt:
            continue
        hash_name = r['hash']
        ret.append(to_download_item(hash_name, reso_opt, subset_opt, file_type))

    return ret


def download(download_list: list, output_dir: str, is_clean_cache: bool):
    """ Download the dataset based on the download_list and user options.

    :param download_list: the list of files to download, [{'repo', 'rel_path'}]
    :param output_dir: the output directory 
    :param reso_opt: the resolution option 
    :param is_clean_cache: if set, will clean the huggingface cache to save space 
    """	
    succ_count = 0
    
    for item in tqdm(download_list, desc='Downloading'):
        repo = item['repo']
        rel_path = item['rel_path']
        
        output_path = os.path.join(output_dir, rel_path)
        output_path = output_path.replace('.zip', '')
        # skip if already exists locally
        if os.path.exists(output_path):
            succ_count += 1
            continue
        with _HF_HUB_DOWNLOAD_LOCK:
            succ = hf_download_path(repo, rel_path, output_dir)
            if succ and is_clean_cache:
                clean_huggingface_cache(output_dir, repo)

        if succ:
            succ_count += 1
            # unzip the file 
            if rel_path.endswith('.zip'):
                zip_file = join(output_dir, rel_path)
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    ofile = join(output_dir, os.path.dirname(rel_path))
                    safe_extract_zip(zip_ref, ofile)
                os.remove(zip_file)
        else:
            print(f'Download {rel_path} failed')

    print(f'Summary: {succ_count}/{len(download_list)} files downloaded successfully')
    return succ_count == len(download_list)


def download_dataset(args):
    """ Download the dataset based on the user inputs.

    :param args: argparse args. Used to decide the subset.
    :return: download success or not
    """	
    output_dir = args.odir
    subset_opt = args.subset
    reso_opt   = args.resolution
    hash_name  = args.hash
    file_type  = args.file_type
    is_clean_cache = args.clean_cache

    os.makedirs(output_dir, exist_ok=True)

    download_list = get_download_list(subset_opt, hash_name, reso_opt, file_type, output_dir)
    return download(download_list, output_dir, is_clean_cache)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--odir', type=str, help='output directory', required=True)
    parser.add_argument('--subset', choices=['1K', '2K', '3K', '4K', '5K', '6K', '7K', '8K', '9K', '10K', '11K'], help='The subset of the benchmark to download', required=True)
    parser.add_argument('--resolution', choices=['4K', '2K', '960P', '480P'], help='The resolution to donwnload', required=True)
    parser.add_argument('--file_type', choices=['images+poses', 'video', 'colmap_cache'], help='The file type to download', required=True, default='images+poses')
    parser.add_argument('--hash', type=str, help='If set subset=hash, this is the hash code of the scene to download', default='')
    parser.add_argument('--clean_cache', action='store_true', help='If set, will clean the huggingface cache to save space')
    params = parser.parse_args()

    assert params.file_type in ['images+poses', 'video', 'colmap_cache'], 'Check the file_type input.'

    if params.file_type == 'images+poses':
        repo = resolution2repo[params.resolution]
    elif params.file_type == 'video':
        repo = 'DL3DV/DL3DV-ALL-video'
    elif params.file_type == 'colmap_cache':
        repo = 'DL3DV/DL3DV-ALL-ColmapCache'

    if not verify_access(repo):
        print(f'You have not grant the access yet. Go to relevant huggingface repo (https://huggingface.co/datasets/{repo}) and apply for the access.')
        exit(1)

    if download_dataset(params):
        print('Download Done. Refer to', params.odir)
    else:
        print(f'Download to {params.odir} failed. See error messsage.')
