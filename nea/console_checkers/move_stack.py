from nea.console_checkers.consts import USER_DISPLAY_ACTION


class MoveStack:
    def __init__(self) -> None:
        self._data = []
        self._removed_items = []

    @property
    def isempty(self):
        return True if len(self._data) == 0 else False

    def push(self, move: USER_DISPLAY_ACTION) -> None:
        self._data.append(move)

    def pop(self) -> None:
        self._removed_items.append(self._data[-1])
        self._data.pop()

    def unpop(self) -> None:
        self._data.append(self._removed_items[-1])

    def peek(self) -> USER_DISPLAY_ACTION:
        return self._data[-1]

    def __getitem__(self, start: int, stop: int) -> USER_DISPLAY_ACTION:
        return self._data[start, stop]
