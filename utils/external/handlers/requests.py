from typing import List, Union, Any, Callable

from selenium import webdriver

from selenium.webdriver import Remote, ChromeOptions

from selenium.common.exceptions import TimeoutException, WebDriverException

from selenium.webdriver.remote.webelement import WebElement

from ..common import Terminal, Logger, Time

__all__ = ['Chrome']


def from_kwargs_callback(key, kwargs_key='meta', **kwargs):

    if key in kwargs:

        if not isinstance(kwargs[key], Callable):

            raise ValueError(f'{key}, is not callable')

        elif kwargs_key in kwargs:

            return kwargs[key](**kwargs.get(kwargs_key, {}))

        else:

            return kwargs[key]()

    raise ValueError(f'from_kwargs_callback(...), key={key}, does not exist')


class Chrome:

    default_options = ['headless', 'no-sandbox', 'disable-dev-shm-usage']

    def __init__(self, path: str = '/usr/bin/chromedriver', **kwargs):

        # --------------------------------------

        self.options: ChromeOptions = ChromeOptions()

        self.__set_options__(*kwargs.get('options', Chrome.default_options))

        # --------------------------------------

        if path is None:

            self.path: str = Chrome.find_path(**kwargs)

        else:

            self.path: str = path

        self.driver: Remote = webdriver.Chrome(options=self.options, executable_path=self.path)

        # --------------------------------------

    def get(self, url: str, load_timeout: float = 0.0, max_reloads: int = 1, **kwargs):

        self.driver.set_page_load_timeout(time_to_wait=load_timeout)

        for i in range(max_reloads):

            try:

                self.driver.get(url=url)

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

    def find_elements_by_xpath(self, xpath, raise_exc=True) -> Union[None, List[WebElement]]:

        try:

            return self.driver.find_elements_by_xpath(xpath)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:
                return []

    def find_element_by_xpath(self, xpath, raise_exc=True) -> Union[None, WebElement]:

        try:

            return self.driver.find_element_by_xpath(xpath)

        except WebDriverException as exc:

            if raise_exc:

                raise exc

            else:

                return None

    def find_element_by_text(self, text: str, raise_exc=True) -> Union[None, WebElement]:

        return self.find_element_by_xpath(f"//*[contains(text(), '{text}')]", raise_exc=raise_exc)

    def scroll_down(self, delay=0.5, limit: int = 1, **kwargs) -> Any:

        for i in range(limit):

            # scroll to - document.body.scrollHeight
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            Logger.info_r(f'steps : {i + 1}/{limit}')

            Time.sleep(delay)

        Logger.info('', end='\n')
        Logger.set_line(length=60)

        return from_kwargs_callback('callback', **kwargs)

    def quit(self) -> None:

        self.driver.quit()

    def close(self) -> None:

        self.driver.close()

    def get_scroll_height(self, time_to_wait=1) -> int:

        self.driver.set_script_timeout(time_to_wait=time_to_wait)

        return self.driver.execute_script("return document.body.scrollHeight")

    def __set_options__(self, *args, **kwargs) -> None:

        options = set(self.options.arguments).union(set(args))

        options = set(options)

        for option, setop in kwargs.items():

            if not setop:

                options.remove(option)

            else:

                options.add(option)

        self.options._arguments = list(options)

        if hasattr(self, 'driver'):

            self.driver.start_session(self.options.to_capabilities())

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

    scroll_height = property(get_scroll_height)

    @staticmethod
    def find_path(**kwargs) -> str:

        path = Terminal.locate_file(pattern='/chromedriver$', params='-i --regexp',
                                    signature='Chrome.find_path(...)', **kwargs)

        if len(path):

            return path[0]

        raise ValueError('find_path(...)')
