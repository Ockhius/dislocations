import os


class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = os.path.join(path, 'log.txt')
        self.init_logs()

    def init_logs(self):

        text_file = open(self.path, "w")
        text_file.close()

    def log_string(self, string):
        """Stores log string in log file."""
        text_file = open(self.path, "a")
        text_file.write(str(string)+'\n')
        text_file.close()