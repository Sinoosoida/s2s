from queue import Queue, Empty


class ProcessIterator:
    """
    Итератор, который хранит и выдает чанки данных.
    """

    def __init__(self):
        self.chunk_queue = Queue()
        self._sentinel = object()

    def put(self, chunk):
        """
        Добавляет чанк в итератор.
        """
        self.chunk_queue.put(chunk)

    def close(self):
        """
        Помечает итератор как закрытый (больше данных не будет).
        """
        self.chunk_queue.put(self._sentinel)

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self.chunk_queue.get()
        if chunk is self._sentinel:
            raise StopIteration
        return chunk