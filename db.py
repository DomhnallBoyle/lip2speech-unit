import os
import sqlite3


class DB:
    
    def __init__(self, path='server.db'):
        self.path = path

    def delete(self):
        os.remove(self.path)

    def __enter__(self):
        self.con = sqlite3.connect(self.path)
        self.cur = self.con.cursor()

        return self.cur

    def __exit__(self, *args, **kwargs):
        self.cur.close()
        self.con.commit()
        self.con.close()
