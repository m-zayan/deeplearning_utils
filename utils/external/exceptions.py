from socket import error

__all__ = ['SocketError', 'InvalidConfigurations']


class InvalidConfigurations(Exception):

    def __init__(self, message):

        super().__init__(message)


class SocketError(error):

    def __init__(self, *args, **kwargs):
        pass
