from typing import Tuple, List, Dict, Callable, Any, Union, Generator

import sys
import os

import subprocess as process
from pathlib import Path

import glob

import re

import io

from zipfile import ZipFile, ZipInfo

import json
import csv

import pandas as pd

from datetime import datetime
import time

import traceback

import requests

from .exceptions import InvalidConfigurations

__all__ = ['Logger', 'OS', 'Sys', 'Terminal', 'Reader', 'Writer', 'IO',
           'Time', 'ResManger', 'StatusWatch']

OS_ROOT_DIR = str(Path.home()) + '/../../'


def validate_arguments(func: Callable) -> Callable:

    def inner(*args, **kwargs) -> None:

        try:

            return func(*args, **kwargs)

        except TypeError as err:

            Logger.fail('Invalid arguments::', func.__name__, ', arguments : ')
            Logger.error(err)

    return inner


class Logger:

    @staticmethod
    def write_messages_json(content: Any, log_file: str = 'logs.json', **kwargs) -> None:

        json_record: Dict[Any, Any] = dict()

        json_record['update_time'] = datetime.now().strftime("%I.%M:%S %p")
        json_record[json_record['update_time']] = content

        if OS.file_exists(log_file):

            Writer.dict_to_json(json_filename=log_file, content=json_record, overwrite=True, **kwargs)

        else:

            Writer.dict_to_json(json_filename=log_file, content=json_record, overwrite=False, **kwargs)

    @staticmethod
    def info(message: Any, *args, end: str = '\n') -> None:

        print(Formatter.GREEN + str(message) + Formatter.END + Logger.__format_args__(*args), end=end)

    @staticmethod
    def info_r(message: Any, *args) -> None:

        sys.stdout.write('\r ' + Formatter.GREEN + str(message) + Formatter.END + Logger.__format_args__(*args))
        sys.stdout.flush()

    @staticmethod
    def fail(message: Any, *args, end: str = '\n') -> None:

        print(Formatter.FAIL + str(message) + Formatter.END + Logger.__format_args__(*args), end=end)

    @staticmethod
    def error(err: Any, end: str = '\n') -> None:

        err_traceback = traceback.format_exc()

        Logger.fail(str(err))
        Logger.fail(str(err_traceback), end=end)

    @staticmethod
    def warning(message: Any, *args, end: str = '\n') -> None:

        print(Formatter.WARNING + str(message) + Formatter.END + Logger.__format_args__(*args), end=end)

    @staticmethod
    def set_line(length: int = 100) -> None:

        Logger.info(Formatter.BLUE + '=' * length + Formatter.END)

    @staticmethod
    def wait_key(message: str, key_value: str, valid_keys: list) -> [None, bool]:

        Logger.warning(message, end=':')

        ok = input()

        if ok.lower() == key_value:

            return True

        elif ok in valid_keys:

            Logger.fail('Aborted!')

            return False

        else:

            Logger.fail(f'key={key_value} is invalid, Aborted!')

            return None

    @staticmethod
    def __format_args__(*args) -> str:

        _format = '{}, {}' * (len(args) // 2)

        if len(args) % 2:
            _format += '{}'

        return _format.format(*args)


class OS:

    KB = 2 ** 10
    MB = 2 ** 20
    GB = 2 ** 30

    @staticmethod
    def file_exists(path: str) -> bool:

        return Path(path).is_file()

    @staticmethod
    def dir_exists(path: str) -> bool:

        return Path(path).is_dir()

    @staticmethod
    def make_dir(path: str) -> None:

        os.mkdir(path)

    @staticmethod
    def make_dirs(path: str) -> None:

        os.makedirs(path)

    @staticmethod
    def cwd() -> str:

        return os.getcwd()

    @staticmethod
    def realpath(path: str) -> str:

        return os.path.realpath(path)

    @staticmethod
    def join(*args) -> str:

        return os.path.join(*args)

    @staticmethod
    def split(path: str) -> Tuple[str, str]:

        return os.path.split(path)

    @staticmethod
    def splitdir(path: str, index: int = 0, sep: str = '/') -> str:

        return path.split(sep)[index]

    @staticmethod
    def splitext(path: str) -> Tuple[str, str]:

        return os.path.splitext(path)

    @staticmethod
    def dirname(path: str) -> str:

        return os.path.dirname(path)

    @staticmethod
    def filename(path: str, ext_include: bool = False, sep: str = '/') -> str:

        fname = path.split(sep)[-1]

        if not ext_include:
            return OS.splitext(fname)[0]

        return fname

    @staticmethod
    def listdir(path: str) -> List[str]:

        return os.listdir(path)

    @staticmethod
    def match_ext(path: str, ext_list: Union[None, list] = None) -> bool:

        if ext_list is None:
            return True

        fext = OS.splitext(path)[-1]

        if fext in ext_list:
            return True

        return False

    @staticmethod
    def rfind_ext(path: str, ext_list: list, include_ext=True,
                  join=False) -> Generator[Tuple[str, str, str], None, None]:

        for (dirpath, _, filenames) in os.walk(path):

            for i in range(len(filenames)):

                fname, fext = OS.splitext(filenames[i])

                if fext in ext_list:

                    ret = (dirpath, fname)

                    if include_ext:
                        ret += (fext,)

                    if join:

                        yield OS.join(*ret)

                    else:

                        yield ret

    @staticmethod
    def file_at(path: str, *args) -> str:

        for i in range(len(args)):

            if path in os.listdir(args[i]):
                return args[i]

        raise ValueError(f'file : {path}, is not found')

    @staticmethod
    def file_size(path) -> float:

        if not OS.file_exists(path):
            raise ValueError(f'file: {OS.filename(path)}, is not found, or not a regular file')

        return os.path.getsize(path)

    @staticmethod
    def windows() -> bool:

        return 'win' in Sys.platform()

    @staticmethod
    def linux() -> bool:

        return 'linux' in Sys.platform()

    @staticmethod
    def environ(name) -> Any:

        return os.environ.get(name)


class Sys:

    @staticmethod
    def insert_path(index: int, path: str) -> None:

        sys.path.insert(index, path)

    @staticmethod
    def maxint() -> int:

        return sys.maxsize

    @staticmethod
    def max(nbits) -> int:

        return 2 ** nbits - 1

    @staticmethod
    def platform() -> str:

        return sys.platform

    @staticmethod
    def virtualenv() -> List[str]:

        return [sys.base_prefix, sys.prefix, sys.exec_prefix]

    @staticmethod
    def __on_platform__(signature, **kwargs) -> None:

        for key, value in kwargs.items():

            if value and (key not in Sys.platform()):

                raise ValueError(f'Not yet supported, {signature}, Platform={Sys.platform()}')


class Terminal:

    @staticmethod
    def locate_file(pattern: str, params: str = '', updatedb: bool = False, **kwargs) -> List[str]:

        Sys.__on_platform__('locate_file(...)', linux=True)

        if updatedb:

            Terminal.run_command('nice -n 19 ionice -c 3 updatedb', **kwargs)

        file_dirs = Terminal.run_command(f'locate {params} {pattern}',
                                         multi_outputs=True, multi_output_sep='\n', **kwargs)

        if (file_dirs is not None) \
                and (len(file_dirs) > 1):

            file_dirs.pop()

        return file_dirs

    @staticmethod
    def run_command(command: str, multi_outputs: bool = True, multi_output_sep: str = '\n',
                    signature='', **kwargs) -> Union[None, bytes, str, list]:

        output: Union[bytes, str, list]

        try:

            if OS.linux():

                if kwargs.get('as_root', False):

                    if kwargs.get('password', None) is None:
                        raise ValueError(f'password=?, is required {signature}')

                    command = f'echo {kwargs["password"]} | sudo -S {command}'

                output = process.check_output(command, shell=True)

            elif OS.windows():

                bash_dir = os.environ.get('bash')

                if bash_dir is None:
                    raise InvalidConfigurations('bash environmental variable doesn\'t exist')

                output = process.check_output([bash_dir, '-c', command], shell=True)

            else:

                raise InvalidConfigurations(f'Invalid Platform : {Sys.platform()}')

        except process.CalledProcessError as error:

            content = {f'CalledProcessError': 'OS::run_command(...), ' + str(error)}
            Logger.write_messages_json(content)

            return None

        else:

            output = output.decode()

        if multi_outputs:
            output = output.split(multi_output_sep)

        return output

    @staticmethod
    def pip_install(packname: str, params: str = '', signature: str = '', **kwargs) -> None:

        _ = Terminal.run_command(f'pip install {params} {packname}', signature=signature, **kwargs)

    @staticmethod
    def pip_uninstall(packname: str, params: str = '', signature: str = '', **kwargs) -> None:

        _ = Terminal.run_command(f'pip uninstall {params} {packname}', signature=signature, **kwargs)

    @staticmethod
    def conda_activate(envname, signature: str = '', **kwargs) -> None:

        _ = Terminal.run_command(f'conda activate {envname}', signature=signature, **kwargs)

    @staticmethod
    def pip_info(signature='', **kwargs) -> str:

        envname = Terminal.run_command(f'pip -V', signature=signature, **kwargs)

        return envname

    @staticmethod
    def kill_gpu_processes(process_keyword: str = 'python') -> None:

        command = f"nvidia-smi | grep '{process_keyword}'"

        output = Terminal.run_command(command=command, multi_outputs=True)

        output = list(map(lambda l: l.split(), output))
        output = list(map(lambda l: l[4] if len(l) > 4 and l[4].isnumeric() else None, output))

        for pid in output:

            if pid is not None:
                Logger.info(f'kill : {pid}')

                Terminal.run_command(command=f'kill -9 {pid}')

        Logger.info('kill_gpu_processes::', output)


class Reader:

    @staticmethod
    def json_to_dict(json_filename: str, **kwargs) -> Union[None, Dict[Any, Any]]:

        if not OS.file_exists(json_filename):

            Logger.warning(f'File: {json_filename} Doesn\'t Exist')

            return None

        content: dict

        with open(json_filename, 'r') as buffer:

            content = json.load(buffer, **kwargs)

        return content


class Writer:

    @staticmethod
    def dict_to_json(json_filename: str, content: Any, overwrite: bool = False,
                     indent_level: int = 3, sort_keys: bool = False, separators: Tuple = (',', ':'), **kwargs) -> None:

        is_file_exist = OS.file_exists(json_filename)

        if not is_file_exist and overwrite:

            Logger.warning(f'overwrite=True, File: {json_filename} is Not Exists')

        elif is_file_exist and not overwrite:

            Logger.warning(f'File: {json_filename} Already Exists')

            ok = input('Do you want to continue - [y/n]: ')

            if ok.lower() == 'n':

                return None

            elif ok.lower() != 'y':

                Logger.error(f'Abort')

                return None

        if not is_file_exist:

            with open(json_filename, 'w+') as buffer_writer:

                json.dump(content, buffer_writer, indent=indent_level,
                          separators=separators, sort_keys=sort_keys, **kwargs)
        else:

            new_content: dict

            with open(json_filename, 'r+') as buffer:

                new_content = json.load(buffer)
                new_content.update(content)

                buffer.seek(0)

                json.dump(new_content, buffer, indent=indent_level,
                          separators=separators, sort_keys=sort_keys, **kwargs)

                buffer.truncate()

    @staticmethod
    def dict_to_csv(csv_filename: str, content: Any, overwrite: bool = False, use_pandas: bool = True) -> None:

        is_file_exist = OS.file_exists(csv_filename)

        if not is_file_exist and overwrite:

            Logger.warning(f'overwrite=True, File: {csv_filename} is Not Exists')

        elif is_file_exist and not overwrite:

            Logger.warning(f'File: {csv_filename} Already Exists')

            ok = input('Do you want to continue - [y/n]: ')

            if ok.lower() == 'n':

                return None

            elif ok.lower() != 'y':

                Logger.error(f'Abort')

                return None

        if not use_pandas:

            if not is_file_exist:

                with open(csv_filename, 'w+') as buffer_writer:

                    csv_writer = csv.writer(buffer_writer)
                    csv_writer.writerow(content.keys())
                    csv_writer.writerow(content.values())

            else:

                new_content: dict

                with open(csv_filename, 'a') as buffer_writer:

                    csv_writer = csv.writer(buffer_writer)
                    csv_writer.writerow(content.values())
        else:

            dataframe = pd.DataFrame(content)
            dataframe.to_csv(csv_filename, index=False, encoding='utf-8')


class IO:

    @staticmethod
    def download(url: str, dest: str, fname: str, chunk_size: int = 1024) -> str:

        """
        Parameter
        ---------
        url (str): file url

        dest (str): director

        fname (str): filename + [file extension]

        chunk_size: request chunk_size, default = 1024
        """

        req = requests.get(url, stream=True)

        if req.status_code != 200:

            raise req.raise_for_status()

        else:

            path = os.path.join(dest, fname)

            with open(path, 'wb') as buffer:

                for block in req.iter_content(chunk_size=chunk_size):

                    if block:
                        buffer.write(block)

            return path

    @staticmethod
    def get_zipfile_content(url: str) -> ZipFile:

        """
        Parameter
        ---------
        url (str): zip-file url

        Returns
        -------
        ZipFile: ZipFile Object
        """

        res = requests.get(url, stream=True)

        d_bytes = io.BytesIO(res.content)

        zfile = ZipFile(d_bytes)

        return zfile

    @staticmethod
    def get_zipfile_bytes(url, fname) -> List[bytes]:

        """
        Parameter
        ---------
        url (str): zip-file url

        fname (str): zip inner filename

        Returns
        -------
        list: List[bytes]
        """

        files = IO.get_zipfile_content(url)
        d_bytes = None

        try:

            d_bytes = files.open(fname)

        except KeyError as err:

            print(err, '\nAvailable files : \n', '=' * 50, files.filelist)

        bytes_data = d_bytes.readlines()

        return bytes_data

    @staticmethod
    def zipfile_read(zipfile_path: str, include_dirs: bool = False, ext_list: list = None,
                     **kwargs) -> Tuple[ZipFile, List[ZipInfo]]:

        # -----------------------------------------------------

        def filelist_filter(_zipfile: ZipFile):

            _filelist = []

            for fileinfo in _zipfile.filelist:

                if (not fileinfo.is_dir() or include_dirs) \
                        and OS.match_ext(fileinfo.filename, ext_list):
                    _filelist.append(fileinfo)

            return _filelist

        # -----------------------------------------------------

        _zipfile = ZipFile(zipfile_path, 'r', **kwargs)

        filelist = filelist_filter(_zipfile)

        return _zipfile, filelist


class Formatter:

    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


class Time:

    @staticmethod
    def now(sep: str = ':') -> str:
        return datetime.now().strftime(f'%I{sep}%M{sep}%S')

    @staticmethod
    def sleep(seconds: float) -> None:

        time.sleep(seconds)


class ResManger:

    @staticmethod
    def get_intersect_path(ddir: str, current: str, index: int = -1) -> str:

        idx = list(re.finditer(ddir, current))[index].span()[0]

        return current[:idx + len(ddir)]

    @staticmethod
    def get_vars_name(cls: Any, instance: Any) -> Dict[Any, Any]:

        """
        Parameters
        ----------
        cls [class]
        instance [class instance]
        ------
        """
        vars_name = {}

        for key, value in vars(cls).items():

            if isinstance(value, instance):
                vars_name[value] = key

        return vars_name


class StatusWatch:

    @staticmethod
    def fsize(path: str) -> float:

        size = OS.file_size(path) / OS.MB

        return size

    @staticmethod
    def download(ddir: str, ongoing_indicator: str = '*.crdownload') -> Callable:

        ddir = OS.realpath(ddir)

        pattern = OS.join(ddir, ongoing_indicator)

        files = glob.glob(pattern)

        def on_watch(dtype: Any = float) -> Tuple[bool, Dict[str, Any]]:

            mp = {}

            for path in files:

                if OS.file_exists(path):

                    mp[OS.filename(path, ext_include=False)] = dtype(StatusWatch.fsize(path))

            return bool(len(mp)), mp

        return on_watch
