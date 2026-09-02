import pytest
from unittest import mock

from lambeq import VerbosityLevel
from lambeq.text2diagram.model_based_reader.model_downloader import (
    DEFAULT_MODELS_URL,
    ModelDownloader,
    ModelDownloaderError,
    MODELS,
    MODELS_URL_ENV_VAR,
)


class MockRequestGetResponse:
    def __init__(self, status_code, content=None, text=None):
        self.status_code = status_code
        self.content = content
        self.text = text


def generate_mock_requests_get_fn(status_code,
                                  tar_response=None,
                                  checksum_response=None,
                                  version_response='1.2.3'):

    def mock_requests_get(url, *args, **kwargs):
        if 'model.tar.gz' in url:
            return MockRequestGetResponse(status_code, content=tar_response)
        if 'model_checksum' in url:
            return MockRequestGetResponse(status_code, text=checksum_response)
        if 'version.txt' in url:
            return MockRequestGetResponse(status_code, text=version_response)

    return mock_requests_get


invalid_checksum_params = {'status_code': 200,
                               'tar_response': b'12345',
                               'checksum_response': ''}


invalid_extraction_params = {'status_code': 200,
                             'tar_response': b'12345',
                             'checksum_response': ('5994471abb01112afcc18159f6cc7'
                                                   '4b4f511b99806da59b3caf5a9c173cacfc5')}


def test_invalid_model_name():
    with pytest.raises(ValueError):
        downloader = ModelDownloader('incorrect model name')


def test_default_models_url():
    downloader = ModelDownloader('bobcat')
    assert downloader.model_url == f'{DEFAULT_MODELS_URL}/bobcat/latest'


def test_models_url_env_override(monkeypatch):
    monkeypatch.setenv(MODELS_URL_ENV_VAR, 'https://mirror.example.com/models/')
    downloader = ModelDownloader('bobcat')
    # The override is honoured and a trailing slash is not doubled.
    assert downloader.model_url == 'https://mirror.example.com/models/bobcat/latest'


def test_invalid_url():
    downloader = ModelDownloader('bobcat')

    downloader.model_url += '/incorrect_url_suffix'

    with pytest.raises(ModelDownloaderError, match='Failed to download'):
        downloader.download_model(verbose=VerbosityLevel.SUPPRESS.value)

    with pytest.raises(ModelDownloaderError, match='Failed to download'):
        downloader.download_model(verbose=VerbosityLevel.PROGRESS.value)


@mock.patch('requests.get',
            side_effect=generate_mock_requests_get_fn(**invalid_checksum_params))
def test_invalid_checksum(mock_get_fn):

    downloader = ModelDownloader('bobcat')
    with pytest.raises(ModelDownloaderError, match='does not match checksum'):
        downloader.download_model(verbose=VerbosityLevel.SUPPRESS.value)


@mock.patch('requests.get',
            side_effect=generate_mock_requests_get_fn(**invalid_extraction_params))
def test_invalid_extraction(mock_get_fn):

    downloader = ModelDownloader('bobcat')
    with pytest.raises(ModelDownloaderError, match='Failed to extract'):
        downloader.download_model(verbose=VerbosityLevel.SUPPRESS.value)


def raise_error(*args, **kwargs):
    raise Exception('ERR')

@mock.patch('requests.get', side_effect=raise_error)
def test_remote_version_error(mock_get_fn):

    downloader = ModelDownloader('bobcat')

    assert downloader.remote_version == None
    assert downloader.model_is_stale() == False

    with pytest.raises(ModelDownloaderError,
                       match='Failed to retrieve remote version'):
        downloader.download_model()


@mock.patch('requests.get', side_effect=raise_error)
def test_remote_checksum_error(mock_get_fn):

    downloader = ModelDownloader('bobcat')

    with pytest.raises(ModelDownloaderError,
                       match='Failed to retrieve remote checksum'):
        downloader._get_remote_checksum()
