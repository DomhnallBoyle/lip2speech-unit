import sqlite3


class DB:
    
    def __init__(self, path='server.db'):
        self.path = path

    def __enter__(self):
        self.con = sqlite3.connect(self.path)

        return self.con.cursor()

    def __exit__(self, *args, **kwargs):
        self.con.commit()
        self.con.close()
