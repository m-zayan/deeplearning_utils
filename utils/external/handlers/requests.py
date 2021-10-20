from typing import Tuple, List, Union, Any, Callable, Optional

from copy import deepcopy

import pickle

from threading import Thread

from selenium import webdriver

from selenium.webdriver import ChromeOptions

from selenium.common.exceptions import TimeoutException, WebDriverException

from selenium.webdriver.remote.webelement import WebElement

from selenium.webdriver.common.by import By

from ..common import Terminal, Logger, Time, OS, StatusWatch

__all__ = ['Chrome', 'By']


def from_kwargs_callback(key, **kwargs):

    if key in kwargs:

        if not isinstance(kwargs[key], Callable):

            raise ValueError(f'{key}, is not callable')

        elif 'meta' in kwargs:

            return kwargs[key](**kwargs.get('meta', {}))

        else:

            return kwargs[key]()

    raise ValueError(f'from_kwargs_callback(...), key={key}, does not exist')


class Chrome(webdriver.Chrome):

    cache_dir = OS.realpath('./tmp/chrome/')
    profile_dir = 'Default'

    default_options = {'headless': True, 'no-sandbox': True, 'disable-dev-shm-usage': True}

    def __init__(self, path: str = '/usr/bin/chromedriver', options: dict = None, extensions: dict = None,
                 detach=False, **kwargs):

        # ---------------------------------------------------------------------------

        self.mutable_kwargs = deepcopy(kwargs)

        if options is None:

            options = Chrome.default_options

        if extensions is None:

            extensions = {}

        # ---------------------------------------------------------------------------

        self.options: ChromeOptions = ChromeOptions()

        self.set_options(**options)
        self.set_extensions(**extensions)

        # ---------------------------------------------------------------------------

        if self.mutable_kwargs.pop('use_cache', False):

            if not OS.dir_exists(Chrome.cache_dir):

                OS.make_dirs(Chrome.cache_dir)

            self.set_user_data_directory(self.mutable_kwargs.pop('data_dir', None), extend_session=False)
            self.set_profile_directory(self.mutable_kwargs.pop('profile_dir', None), extend_session=False)

        if detach:

            self.detach(extend_session=False)

        # ---------------------------------------------------------------------------

        if path is None:

            self.path: str = Chrome.find_path(**self.mutable_kwargs.pop('locator_options', {}))

        else:

            self.path: str = path

        # ---------------------------------------------------------------------------

        super(Chrome, self).__init__(options=self.options, executable_path=self.path, **self.mutable_kwargs)

        # ---------------------------------------------------------------------------

    # noinspection PyMethodOverriding
    def get(self, url: str, load_timeout: float = 30.0, max_reloads: int = 1, **kwargs):

        self.set_page_load_timeout(load_timeout)

        for i in range(max_reloads):

            try:

                super(Chrome, self).get(url=url)

                break

            except TimeoutException as exc:

                if i < max_reloads - 1:

                    Logger.fail(str(i + 1) + ': timeout::page has been reloaded')
                    Logger.set_line(length=60)

                else:

                    Logger.fail(str(i + 1) + ': timeout::page reloads limit has been exceed\n'
                                             '\tdo you want to try again - [y/n]: ', end='')
                    ok = input()

                    Logger.set_line(length=60)

                    if ok.lower() == 'y':

                        self.get(url, load_timeout, max_reloads, **kwargs)

                    elif ok.lower() == 'n':

                        from_kwargs_callback('timeout_callback', **kwargs)

                        Logger.error(exc)

                    else:

                        from_kwargs_callback('timeout_callback', **kwargs)

                        Logger.fail('Abort')
                        Logger.error(exc)

    # noinspection PyMethodOverriding
    def find_elements(self, value: str, by: By = By.XPATH, raise_exc: bool = True) -> Union[None, List[WebElement]]:

        try:

            return super(Chrome, self).find_elements(by=by, value=value)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:
                return []

    # noinspection PyMethodOverriding
    def find_element(self, value: str, by: By = By.XPATH, raise_exc: bool = True) -> Union[None, WebElement]:

        try:

            return super(Chrome, self).find_element(by=by, value=value)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:

                return None

    def find_element_by_text(self, text: str, raise_exc=True) -> Union[None, WebElement]:

        return self.find_element(f"//*[contains(text(), '{text}')]", raise_exc=raise_exc)

    def scroll_down(self, delay: float = 0.5, limit: int = 1, **kwargs) -> Any:

        for i in range(limit):

            # scroll to - document.body.scrollHeight
            self.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            Logger.info_r(f'steps : {i + 1}/{limit}')

            Time.sleep(delay)

        Logger.info('', end='\n')
        Logger.set_line(length=60)

        return from_kwargs_callback('callback', **kwargs)

    def quit(self) -> None:

        super(Chrome, self).quit()

    def close(self) -> None:

        super(Chrome, self).close()

    def get_scroll_height(self, time_to_wait: float = 1) -> int:

        self.set_script_timeout(time_to_wait=time_to_wait)

        return self.execute_script("return document.body.scrollHeight")

    def add_option(self, option: str, extend_session: bool = True):

        self.options.add_argument(option)

        if extend_session:

            self.extend_session()

    def add_experimental_option(self, name: str, value: Any, extend_session: bool = True):

        self.options.add_experimental_option(name, value)

        if extend_session:

            self.extend_session()

    def add_extension(self, extension_path, extend_session: bool = True):

        self.options.add_extension(extension_path)

        if extend_session:

            self.extend_session()

    def set_options(self, *args, **kwargs) -> None:

        options = set(self.options.arguments).union(set(args))

        options = set(options)

        for option, setop in kwargs.items():

            if (not setop) and (option in options):

                options.remove(option)

            else:

                options.add(option)

        self.options._arguments = list(options)

    def set_extensions(self, *args, **kwargs) -> None:

        for path in args:

            self.options.add_extension(path)

        for path in kwargs.values():

            self.options.add_extension(path)

    def set_user_data_directory(self, path: Optional[str] = None,  extend_session: bool = True) -> None:

        if path is None:

            path = OS.join(Chrome.cache_dir, 'user-data')

        self.add_option(f'--user-data-dir={path}', extend_session)

    def set_profile_directory(self, profile_directory: Optional[str] = None,  extend_session: bool = True):

        if profile_directory is None:

            profile_directory = Chrome.profile_dir

        self.add_option(f'--profile-directory={profile_directory}', extend_session)

    def set_download_directory(self, dest: Optional[str] = None, extend_session: bool = True) -> None:

        if dest is None:

            dest = Chrome.cache_dir

        prefs = {'download.default_directory': dest}

        self.add_experimental_option('prefs', prefs, extend_session)

    def detach(self, extend_session: bool = True) -> None:

        self.add_experimental_option('detach', True, extend_session)

    def pickle_cookies(self, dest: Optional[str] = None) -> None:

        if dest is None:

            dest = Chrome.cache_dir

        dest = OS.realpath(dest)

        path = OS.join(dest, 'cookies.pkl')

        buffer = open(path, 'wb')

        pickle.dump(self.get_cookies(), buffer)

        buffer.close()

    def unpickle_cookies(self, src: Optional[str] = None, append: bool = True) -> Optional[List]:

        if src is None:

            src = Chrome.cache_dir

        src = OS.realpath(src)

        path = OS.join(src, 'cookies.pkl')

        buffer = open(path, 'rb')

        cookies = pickle.load(buffer)

        buffer.close()

        if append:

            for cookie in cookies:

                self.add_cookie(cookie)

        else:

            return cookies

    def extend_session(self, capabilities=None, browser_profile=None):

        self.close()

        if capabilities is None:

            self.start_session(self.options.to_capabilities(), browser_profile)

        else:

            self.start_session(capabilities, browser_profile)

    def login(self, url: str, username: str, password: str, uname_key: str = 'username',
              passwd_key: str = 'password', submit_key: str = 'submit', load_timeout=60.0):

        self.get(url, load_timeout=load_timeout)

        uname = self.find_element(by=By.NAME, value=uname_key)
        passwd = self.find_element(by=By.NAME, value=passwd_key)
        submit = self.find_element(by=By.NAME, value=submit_key)

        uname.send_keys(username)
        passwd.send_keys(password)

        submit.click()

    def download_request(self, url: str, cache_dir: str, load_timeout: float) -> Tuple[str, Callable]:

        # ---------------------------------------------------------

        cache_dir = OS.realpath(cache_dir)

        if not OS.dir_exists(cache_dir):

            OS.make_dirs(cache_dir)

        # ---------------------------------------------------------

        def request() -> None:

            self.get(url, load_timeout=load_timeout)

        # ---------------------------------------------------------

        return cache_dir, request

    @staticmethod
    def watch_download(cache_dir: str) -> None:

        Logger.set_line(length=60)

        on_watch = StatusWatch.download(cache_dir)

        ongoing, current_size = True, 0.0

        while ongoing:

            ongoing, current_size = on_watch(int)

            if ongoing:

                Logger.info_r(current_size)

        Logger.info('\n', end='')
        Logger.set_line(length=60)

    def download(self, url, cache_dir: str, load_timeout=60, logs_delay=5) -> str:

        # -----------------------------------------------------

        cache_dir, request = self.download_request(url, cache_dir, load_timeout)

        # -----------------------------------------------------

        if len(OS.listdir(cache_dir)):

            Logger.set_line(length=60)
            Logger.fail('A non-empty caching directory is not supported')
            Logger.set_line(length=60)
            Logger.warning('Download aborted!')
            Logger.set_line(length=60)

            return cache_dir

        # -----------------------------------------------------

        download_th = Thread(target=request, args=())
        watch_th = Thread(target=Chrome.watch_download, args=(cache_dir, ))

        # -----------------------------------------------------

        download_th.start()

        Time.sleep(max(5, logs_delay))

        watch_th.start()

        # -----------------------------------------------------

        download_th.join()
        watch_th.join()

        # -----------------------------------------------------

        return cache_dir

    @staticmethod
    def node_find_element_by_xpath(node: WebElement, xpath: str, raise_exc: bool = True) -> Union[None, WebElement]:

        if node is None:
            return None

        try:

            return node.find_element_by_xpath(xpath)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:

                return None

    @staticmethod
    def node_find_elements_by_xpath(node: WebElement, xpath: str,
                                    raise_exc: bool = True) -> Union[None, List[WebElement]]:

        if node is None:
            return []

        try:

            return node.find_elements_by_xpath(xpath)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:

                return None

    @staticmethod
    def safe_get_attribute(element, attr_name, default) -> Any:

        return getattr(element, attr_name, default)

    @staticmethod
    def map_by_property(elems, name):

        mp = {}

        for elem in elems:

            mp[elem.get_property(name)] = elem

        return mp

    @staticmethod
    def find_path(**kwargs) -> str:

        path = Terminal.locate_file(pattern='/chromedriver$', params='-i --regexp',
                                    signature='Chrome.find_path(...)', **kwargs)

        if len(path):

            return path[0]

        raise ValueError('find_path(...)')

    # noinspection PyProtectedMember
    @property
    def command_executor_url(self):

        return self.command_executor._url

    scroll_height = property(get_scroll_height)
