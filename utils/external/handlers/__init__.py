import importlib

from tqdm import tqdm

from ..common import Sys, Terminal, OS, Logger


def __selenium_installed__() -> bool:

    return importlib.util.find_spec('selenium') is not None


def __selenium_install__(**kwargs) -> None:

    sig = kwargs.get('signature', '')

    Sys.__on_platform__(sig, linux=True)

    path = '../../../scripts/chromedriver.sh'
    path = OS.join(OS.dirname(__file__), path)

    if not OS.file_exists(path):

        raise ValueError(f'path={path}, does not exist')

    cmd_seq = [f'sh {path}']

    if kwargs.get('notebook', False):

        cmd_seq[0] += ' -n'

    for cmd in tqdm(cmd_seq):

        _ = Terminal.run_command(cmd, **kwargs)

    Terminal.pip_install('selenium')

    Logger.info('\n__selenium_install__(...) <--> Done!\n')


def __selenium_uninstall__(**kwargs) -> None:

    sig = kwargs.get('signature', '')

    Sys.__on_platform__(sig, linux=True)

    path = '../../../scripts/rm_chromedriver.sh'
    path = OS.join(OS.dirname(__file__), path)

    if not OS.file_exists(path):

        raise ValueError(f'path={path}, does not exist')

    cmd_seq = [f'sh {path}']

    if kwargs.get('notebook', False):

        cmd_seq[0] += ' -n'

    for cmd in tqdm(cmd_seq):

        _ = Terminal.run_command(cmd, **kwargs)

    Terminal.pip_uninstall('selenium', '--yes')

    Logger.info('\n__selenium_uninstall__(...) <--> Done!\n')


def install_dependencies(**kwargs):

    sig = 'install_dependencies(...)'

    if not __selenium_installed__():

        if kwargs.get('notebook', False):

            __selenium_install__(signature=sig, **kwargs)

        else:

            __selenium_install__(signature=sig, as_root=True, **kwargs)


def uninstall_dependencies(**kwargs):

    sig = 'reinstall_dependencies(...)'

    if kwargs.get('notebook', False):

        __selenium_uninstall__(signature=sig, **kwargs)

    else:
        __selenium_uninstall__(signature=sig, as_root=True, **kwargs)
