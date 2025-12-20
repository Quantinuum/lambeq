# Copyright 2021-2024 Cambridge Quantum Computing Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from shutil import copytree
import sys
import tarfile
import tempfile
from typing import IO
import warnings

import requests
from tqdm import TqdmWarning
from tqdm.auto import tqdm

from lambeq.core.globals import VerbosityLevel
from lambeq.typing import StrPathT

MODELS_URL = 'https://qnlp.quantinuum.com/models'
MODELS = {'bobcat', 'oncilla'}
VERSION_FNAME = 'version.txt'
CHECKSUM_FNAME = 'model_checksum.sha256'
# The server expects a user-agent header in each request, even if the
# field is empty.
HEADERS = {'user-agent': ''}


class ModelDownloaderError(Exception):
    def __init__(self, error_msg: str) -> None:
        super().__init__(error_msg)
        self.error_msg = error_msg

    def __str__(self) -> str:
        return f'ModelDownloader raised error: {self.error_msg}'


class ModelDownloader:
    """Class to handle updates to parser models."""

    def __init__(self,
                 model_name: str,
                 cache_dir: StrPathT | None = None) -> None:
        """Instantiate a model downloader.

        Parameters
        ----------
        model_name : str
            The name of a pre-trained model.
        cache_dir : str or os.PathLike, optional
            The directory to which a downloaded pre-trained model should
            be cached instead of the standard cache
            (`$XDG_CACHE_HOME` or `~/.cache`).

        """

        if model_name not in MODELS:
            raise ValueError(f'Invalid model name: {model_name!r}')

        self.model = model_name
        self.model_dir = self.get_dir(cache_dir)
        self.model_url = self.get_url()

        try:
            self.remote_version: str | None = self.get_latest_remote_version()
        except Exception as e:
            # It is possible in some cases to continue even if the
            #  version could not be retrieved from remote.
            # The error is saved to potentially be raised later.

            self.version_retrieval_error = e
            self.remote_version = None

    def get_url(self) -> str:
        """Get URL for the latest version of specified model."""

        model_path = self.model
        if self.model == 'bobcat':
            model_path = 'bert'
        return f'{MODELS_URL}/{model_path}/latest'

    def get_dir(self,
                cache_dir: StrPathT | None = None) -> Path:
        """Return a directory in which to cache the model locally.
         Create parent directory if it does not exist."""

        if cache_dir is None:
            cache_dir = Path(os.getenv('XDG_CACHE_HOME',
                                       Path.home() / '.cache'))
        else:
            cache_dir = Path(cache_dir)
        models_dir = cache_dir / 'lambeq' / self.model
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except FileExistsError as e:
            raise FileExistsError('Cache directory location '
                                  f'(`{models_dir}`) already exists and'
                                  ' is not a directory.') from e
        return models_dir / self.model

    def model_is_stale(self) -> bool:
        """Check if the locally cached model is older than the
         latest model available remotely."""

        if self.remote_version is None:
            return False

        local_version = self.get_local_model_version()

        return bool(self.remote_version != local_version)

    def get_local_model_version(self) -> str | None:
        """Get version of model cached locally."""

        try:
            with open(self.model_dir / VERSION_FNAME) as f:
                local_version = f.read().strip()
        except FileNotFoundError:
            # Fallback: Check for nested directory (e.g. bobcat/bobcat)
            try:
                with open(self.model_dir / self.model / VERSION_FNAME) as f:
                    local_version = f.read().strip()
            except Exception:
                local_version = None
        except Exception:
            local_version = None

        return local_version

    def download_model(self,
                       verbose: str = VerbosityLevel.PROGRESS.value
                       ) -> None:
        """Download model from remote, compare checksum of tarball,
         and then extract the model to `model_dir`"""

        if self.remote_version is None:
            if self.get_local_model_version() is not None:
                print('Failed to retrieve remote model version. '
                      'Using local model.', file=sys.stderr)
                # Check if model is in nested directory and update path
                nested_dir = self.model_dir / self.model
                if nested_dir.exists() and (nested_dir / 'config.json').exists():
                    self.model_dir = nested_dir
                return
            raise self.version_retrieval_error

        expected_checksum = self._get_remote_checksum()

        download_url = self.model_url + '/model.tar.gz'

        if verbose == VerbosityLevel.TEXT.value:
            print('Downloading model...', file=sys.stderr)

        model_file = tempfile.NamedTemporaryFile()

        if verbose == VerbosityLevel.PROGRESS.value:
            response = requests.get(download_url, headers=HEADERS, stream=True)

            if response.status_code != 200:
                raise ModelDownloaderError(
                    f'Failed to download model {self.model}. Received '
                    f'response status code: {response.status_code}')

            size = int(response.headers.get('content-length', 0))
            block_size = 1024

            warnings.filterwarnings('ignore', category=TqdmWarning)
            progress_bar = tqdm(
                bar_format='Downloading model: {percentage:3.1f}%|'
                           '{bar}|{n:.3f}/{total:.3f}GB '
                           '[{elapsed}<{remaining}]',
                total=size/1e9)

            for data in response.iter_content(block_size):
                progress_bar.update(len(data)/1e9)
                model_file.write(data)

        else:
            response = requests.get(download_url, headers=HEADERS)

            if response.status_code != 200:
                raise ModelDownloaderError(
                    f'Failed to download model {self.model}. Received '
                    f'response status code: {response.status_code}')

            content = response.content
            model_file.write(content)

        checksum = self._calculate_sha256_checksum(model_file,
                                                   verbose)

        if checksum != expected_checksum:
            raise ModelDownloaderError(
                f'Checksum of downloaded model {self.model} does not '
                'match checksum from remote')

        # Extract model
        model_file.seek(0)
        if verbose != VerbosityLevel.SUPPRESS.value:
            print('Extracting model...')

        with tempfile.TemporaryDirectory() as extraction_target:
            # Extract model to temporary directory first

            try:
                tar = tarfile.open(fileobj=model_file)
                tar.extractall(extraction_target)
            except tarfile.TarError as e:
                model_file.close()
                raise ModelDownloaderError(
                    'Failed to extract compressed model {self.model}'
                ) from e

            model_file.close()

            # Copy extracted model to model_dir
            copytree(extraction_target,
                     str(self.model_dir),
                     dirs_exist_ok=True)

        with open(self.model_dir / VERSION_FNAME, 'w') as w:
            w.write(self.remote_version)

    def get_latest_remote_version(self) -> str:
        """Retrieve the latest model version number from server."""

        ver_url = self.model_url + '/' + VERSION_FNAME

        try:
            remote_version: str = requests.get(ver_url,
                                               headers=HEADERS).text.strip()
        except Exception as e:
            raise ModelDownloaderError(
                'Failed to retrieve remote version number from '
                f'{ver_url}'
            ) from e

        return remote_version

    def _get_remote_checksum(self) -> str:
        """Retrieve the sha256 checksum from server."""

        checksum_url = self.model_url + '/' + CHECKSUM_FNAME

        try:
            checksum_response: str = requests.get(checksum_url,
                                                  headers=HEADERS).text
        except Exception as e:
            raise ModelDownloaderError(
                'Failed to retrieve remote checksum from '
                f'{checksum_url}'
            ) from e

        return checksum_response.split(' ')[0]

    def _calculate_sha256_checksum(self,
                                   file_obj: IO[bytes],
                                   verbose: str) -> str:
        """Generate the SHA256 hash of specified file."""

        file_obj.seek(0, os.SEEK_END)
        file_size = file_obj.tell()
        file_obj.seek(0)

        digest = hashlib.sha256()

        progress_bar = tqdm(
            bar_format='Evaluating checksum: {percentage:3.1f}%|'
                       '{bar}|{n:.3f}/{total:.3f}GB '
                       '[{elapsed}<{remaining}]',
            total=file_size/1e9,
            disable=verbose == VerbosityLevel.SUPPRESS.value)

        while (chunk := file_obj.read(1024)):
            digest.update(chunk)
            progress_bar.update(len(chunk) / 1e9)

        return digest.hexdigest()
