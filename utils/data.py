import threading
import queue
import threading

class ImmutableDataChain:
    def __init__(self, value=None, key=None, save_data=True, previous=None, index=None):
        """
        key - ключ значения. Может быть None, но это означает, что данные будут потеряны после add_data.
        value - сохраняемое значение.
        previous - ссылка на предыдущий экземпляр данных.
        index - индекс экземпляра класса в цепочке экземпляров, созданных из того же экземпляра с помощью add_data.
        save_data - если False, предыдущие данные не будут доступны в новом экземпляре.
        """
        self._key = key
        self._value = value
        self._previous = previous
        self.index = index
        self._counter = -1
        self._save_data = save_data
        self._lock = threading.Lock()  # Блокировка для синхронизации

    def add_data(self, value, key=None, save_data=True):
        with self._lock:
            self._counter += 1
            if self._key is None or not self._save_data:
                return ImmutableDataChain(value, key, save_data, self._previous, self._counter)
            else:
                return ImmutableDataChain(value, key, save_data, self, self._counter)

    def get_data(self, key=None):
        with self._lock:
            # Рекурсивно ищем значение по ключу
            if key is None or self._key == key:
                return self._value
            elif self._previous is not None:
                return self._previous.get_data(key)
            else:
                return None

    def to_dict(self):
        with self._lock:
            data_dict = {}
            current = self
            while current is not None:
                if current._key is not None and current._key not in data_dict:
                    data_dict[current._key] = current._value
                current = current._previous
            return data_dict

    def get(self, key, default=None):
        value = self.get_data(key)
        return value if value is not None else default

    def __getitem__(self, key):
        value = self.get_data(key)
        if value is None:
            raise KeyError(f"Ключ '{key}' не найден")
        return value

    def get_index(self, key):
        """Возвращает индекс, соответствующий ключу, или None, если ключ не найден."""
        with self._lock:
            if self._key == key:
                return self.index
            elif self._previous is not None:
                return self._previous.get_index(key)
            else:
                return None

    def get_counter(self, key=None):
        """Возвращает значение counter, соответствующее ключу, или None, если ключ не найден."""
        with self._lock:
            if self._key == key:
                return self._counter+1
            elif self._previous is not None:
                return self._previous.get_counter(key)+1
            else:
                return None


example_data = {
    "user_audio":None,
    "text":None,
    "language_code":None,
    "start_phrase":None,
    "llm_sentence":None,
    "output_audio_iterator": None,
    "output_audio_chunk":None
}


class FilteredQueue:
    def __init__(self):
        self._queue = queue.Queue()
        self._user_phrase_id = 0
        self._put_lock = threading.Lock()
        self._get_lock = threading.Lock()

    def set_user_phrase_id(self, phrase_id: int):
        """Sets the user_phrase_id that will be used for comparison."""
        self._user_phrase_id = phrase_id

    def put(self, item: ImmutableDataChain):
        """Puts the item in the queue if it passes the validation check."""
        with self._put_lock:
            if self._validate_item(item):
                self._queue.put(item)
            else:
                print(f"Item rejected: 'user_audio' not matching {self._user_phrase_id} or missing.")

    def get(self):
        """Gets the next item from the queue, blocking until one is available."""
        return self._queue.get()

    def remove_non_matching(self):
        """Removes all items in the queue that don't match the user_phrase_id."""
        temp_queue = queue.Queue()
        while not self._queue.empty():
            item = self._queue.get()
            if self._validate_item(item):
                temp_queue.put(item)

        while not temp_queue.empty():
            item = temp_queue.get()
            self._queue.put(item)

    def _validate_item(self, item: ImmutableDataChain) -> bool:
        """Checks if the item has 'user_audio' matching the current user_phrase_id."""
        user_phrase_id = item.get_index('user_audio')
        if user_phrase_id is None:
            return False
        return user_phrase_id >= self._user_phrase_id

    def filter(self, phrase_id: int):
        """Sets the user_phrase_id and removes non-matching items in one step."""
        with self._put_lock, self._get_lock:  # Lock both get and put actions while modifying the queue
            self.set_user_phrase_id(phrase_id)
            self.remove_non_matching()
