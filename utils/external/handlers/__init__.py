import importlib

from tqdm import tqdm

from ..common import Sys, Terminal, OS, Logger


def __selenium_installed__() -> bool:

    return importlib.util.find_spec('selenium') is not None


def __selenium_install__(**kwargs) -> None:

    sig = kwargs.get('signature', '')

    Sys.__on_platform__(sig, linux=True)

    path = '../scripts/chromedriver.sh'

    if not OS.file_exists(path):

        raise ValueError(f'path={path}, does not exist')

    cmd_seq = [f'sh {path}']

    for cmd in tqdm(cmd_seq):

        _ = Terminal.run_command(cmd, **kwargs)

    Terminal.pip_install('selenium')

    Logger.info('__selenium_install__(...) <--> Done!')


def __selenium_uninstall__(**kwargs) -> None:

    sig = kwargs.get('signature', '')

    Sys.__on_platform__(sig, linux=True)

    path = '../scripts/rm_chromedriver.sh'

    if not OS.file_exists(path):
        raise ValueError(f'path={path}, does not exist')

    cmd_seq = [f'sh {path}']

    for cmd in tqdm(cmd_seq):

        _ = Terminal.run_command(cmd, **kwargs)

    Terminal.pip_uninstall('selenium', '--yes')

    Logger.info('__selenium_uninstall__(...) <--> Done!')


def install_dependencies(**kwargs):

    sig = 'install_dependencies(...)'

    if not __selenium_installed__():

        if kwargs.get('notebook', False):

            __selenium_install__(signature=sig, **kwargs)

        else:

            __selenium_install__(signature=sig, as_root=True, **kwargs)


def reinstall_dependencies(**kwargs):

    sig = 'reinstall_dependencies(...)'

    if kwargs.get('notebook', False):

        __selenium_uninstall__(signature=sig, **kwargs)
        __selenium_install__(signature=sig, **kwargs)

    else:
        __selenium_uninstall__(signature=sig, as_root=True, **kwargs)
        __selenium_install__(signature=sig, as_root=True, **kwargs)
