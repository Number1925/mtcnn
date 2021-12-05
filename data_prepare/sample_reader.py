# coding=utf-8

class SampleReader:

    def __init__(self, files):
        """
        Init sample reader which supports some ability of reading data. 

        Args:
            image_dir ([list]): [Image file collection.]
        """
        self.files = files
        self.batch = 1
        self.map_functions = []
        self._reset_batch_status()

    def _reset_batch_status(self):
        """
        Reset variables that is used in iterator

        Raises:
            Exception : Raise Exception when collection is empty
        """
        self._mod = False
        self._cursor = 0
        self._num = len(self.files)
        if self._cursor > self._num:
            raise Exception

    def set_batch(self, num):
        """
        The way to set iterator's batch size

        Args:
            num ([int]): [the number of batch size]

        Raises:
            Exception: [Raise Exception when the iterator is working]
        """
        if self._mod:
            raise Exception
        self.batch = num

    def register_map(self, func):
        """
        Register map function into function queue, all registered map function will execute when iterate data

        Args:
            func ([type]): [map function]

        Raises:
            Exception : Raise Exception when the iterator is working
        """
        if self._mod:
            raise Exception
        self.map_functions.append(func)
        
    def __iter__(self):
        if self._mod:
            raise Exception
        self._mod = True
        return self

    def __next__(self):
        if self._cursor >= self._num:
            raise StopIteration
        next_step = self._cursor + self.batch
        if next_step > self._num:
            next_step = self._num
        tmp_row = self.files[self._cursor:next_step]
        rows = []
        rows = tmp_row
        if len(self.map_functions) > 0:
            for map in self.map_functions:
                rows = map(rows)
        self._cursor = next_step
        return rows
            
