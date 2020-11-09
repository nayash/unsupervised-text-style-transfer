#
# Copyright (c) 2020. Asutosh Nayak (nayak.asutosh@ymail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#


from collections import deque
import threading
import os
import time


class Logger:
    LOG_QUEUE_SIZE = 10
    LOGGING_LEVELS = {0: "INFO", 1: "DEBUG",
                      2: "WARN", 3: "ERROR", 4: "CRITICAL"}

    def __init__(self, log_file_path, log_file_name_prefix,
                 log_queue_size=LOG_QUEUE_SIZE, std_out=False):
        if not os.path.exists(log_file_path):
            os.makedirs(log_file_path)

        LOG_QUEUE_SIZE = log_queue_size
        self.std_out = std_out
        self.init_file_writing(log_file_path, log_file_name_prefix)

    def init_file_writing(self, LOG_PATH, LOG_FILE_NAME_PREFIX):
        self.file_path = self.cleanup_file_path(LOG_PATH + os.sep +
                                                LOG_FILE_NAME_PREFIX + "_" +
                                                self.get_readable_ctime() +
                                                ".log")
        self.log_queue = deque([])
        self.append_log("Initialized logging at path {}".format(
            self.file_path), level=Logger.LOGGING_LEVELS[0])

    def cleanup_file_path(self, path):
        # return path.replace('\\', '/').replace(" ", "_").replace(':', '_')
        return path

    def get_log_prefix_format(self, level=None):
        level = Logger.LOGGING_LEVELS[1] if level is None else level
        return " ".join([self.get_readable_ctime(),
                         threading.current_thread().name, level])

    def append_log(self, *text, level=None):
        level = Logger.LOGGING_LEVELS[1] if level is None else level
        text = ' '.join(str(t) for t in text)
        log_str = self.get_log_prefix_format(level) + r"\ " + text
        self.log_queue.append(log_str)
        if self.std_out:
            print(log_str)
        if len(self.log_queue) >= self.LOG_QUEUE_SIZE:
            log_file = open(self.cleanup_file_path(self.file_path), "a+")
            while len(self.log_queue) > 0:
                log_file.write(self.log_queue.popleft() + "\n")

            log_file.close()
            if self.std_out:
                print("logs written...")

    def flush(self):
        log_file = open(self.cleanup_file_path(self.file_path), "a+")
        while len(self.log_queue) > 0:
            log_file.write(self.log_queue.popleft() + "\n")

        log_file.close()
        if self.std_out:
            print("logs flushed...")

    def get_readable_ctime(self):
        return time.strftime("%d-%m-%Y %H_%M_%S")



