import tarfile
from urllib.request import urlretrieve
from depccg.instance_models import MODEL_DIRECTORY

URL = 'https://qnlp.quantinuum.com/models/tri_headfirst.tar.gz'

print('Please consider using Bobcat, the parser included with lambeq,\n'
      'instead of depccg.')


def print_progress(chunk: int, chunk_size: int, size: int) -> None:
    percentage = chunk * chunk_size / size
    mb_size = size / 10**6
    print(f'\rDownloading model... {percentage:.1%} of {mb_size:.1f} MB',
          end='')


print('Downloading model...', end='')
download, _ = urlretrieve(URL, reporthook=print_progress)

print('\nExtracting model...')
tarfile.open(download).extractall(MODEL_DIRECTORY)

print('Download successful')
