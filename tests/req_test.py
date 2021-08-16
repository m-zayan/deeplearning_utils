from utils.external import handlers

handlers.reinstall_dependencies(notebook=False)

from utils.external.handlers.requests import Chrome

chrome = Chrome(install=False, notebook=False, password=None)

chrome.driver.get('https://www.cityscapes-dataset.com/downloads/')

# chrome.__set_options__(headless=False)

print(chrome.driver.current_url)
