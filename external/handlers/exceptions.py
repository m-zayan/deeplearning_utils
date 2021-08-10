__all__ = ['InvalidConfigurations']


class InvalidConfigurations(Exception):

    def __init__(self, message):

        super().__init__(message)
