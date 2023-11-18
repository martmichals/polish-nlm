import os
import wandb
import shutil
import zipfile
import requests
from tqdm import tqdm

# Temporary directory to download data into
TMP_DIR = '/tmp/wandb-klej-tmp/'

# KLEJ task names
KLEJ_TASK_NAMES = [
    'nkjp-ner', 
    'cdsc-e',
    'cdsc-r',
    'cbd',
    'polemo2.0-in',
    'polemo2.0-out',
    'dyk',
    'psc',
    'ar'
]

# Download a file
def download_file(url: str, fpath: str) -> None:
    response = requests.get(url, stream=True)
    with open(fpath, 'wb') as fout:
        for chunk in response.iter_content(chunk_size=128):
            fout.write(chunk)

# Extract a file 
def extract_file(zip_path: str, extraction_path: str, delete_zip: bool=True) -> None:
    # Create output path if it does not exist
    if not os.path.exists(extraction_path):
        os.makedirs(extraction_path)

    # Extract
    with zipfile.ZipFile(zip_path, 'r') as inzip:
        inzip.extractall(extraction_path)

    # Conditional deletion
    if delete_zip:
        os.remove(zip_path)

def main():
    # Create the directory if it does not exist
    if not os.path.exists(TMP_DIR):
        os.makedirs(TMP_DIR)

    # Download filepaths
    filepaths = [
        os.path.join(TMP_DIR, f'klej_{task_name}.zip') 
    for task_name in KLEJ_TASK_NAMES]

    # Download files
    for task_name, fpath in tqdm(
        zip(KLEJ_TASK_NAMES, filepaths), 
        total=len(KLEJ_TASK_NAMES), 
        desc='Downloading data'
    ):
        download_file(
            url=f'https://klejbenchmark.com/static/data/klej_{task_name}.zip',
            fpath=fpath
        )  

    # Unzip files
    extraction_paths = [
        os.path.join(TMP_DIR, f'klej_{task_name}') 
    for task_name in KLEJ_TASK_NAMES]
    for filepath, extraction_path in tqdm(
        zip(filepaths, extraction_paths),
        total=len(KLEJ_TASK_NAMES),
        desc='Unzipping downloaded archives'
    ):
        extract_file(
            zip_path=filepath,
            extraction_path=extraction_path
        )

    # Initialize W&B upload run
    run = wandb.init(name='upload-klej-files', job_type='upload')

    # Create and upload artifact for each of the KLEJ tasks
    for task_name, extraction_path in tqdm(
        zip(KLEJ_TASK_NAMES, extraction_paths),
        total=len(KLEJ_TASK_NAMES),
        desc='Uploading data to W&B'
    ):
        # Create artifact
        task_artifact = wandb.Artifact(f'klej_{task_name}_raw', type='raw_data')

        # Add files
        for dirpath, _, fnames in os.walk(extraction_path):
            for fname in fnames:
                fpath = os.path.join(dirpath, fname)
                task_artifact.add_file(
                    fpath,
                    name=os.path.relpath(fpath, extraction_path)
                )

        # Log the artifact
        wandb.log_artifact(task_artifact)

    # Finish the run
    run.finish()

    # Delete the download directory
    shutil.rmtree(TMP_DIR)
    

if __name__ == '__main__':
    main()