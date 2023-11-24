import argparse

from db import DB


class Migration:

    def __init__(self):
        self.db = DB()

    def update(self):
        raise NotImplementedError

    def revert(self):
        raise NotImplementedError


class InitMigration(Migration):
    
    def __init__(self):
        super().__init__()

    def update(self):
        with self.db as cur:
            cur.execute('''
                CREATE TABLE audio( 
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE
                )'''
            )
            cur.execute('''
                CREATE TABLE usage(
                    id TEXT PRIMARY KEY,
                    video_id TEXT, 
                    audio_id TEXT,
                    date TIMESTAMP,
                    FOREIGN KEY(audio_id) REFERENCES audio(id)
                )'''
            )
            cur.execute('''
                CREATE TABLE asr_transcription(
                    id TEXT PRIMARY KEY, 
                    usage_id TEXT, 
                    transcription TEXT,
                    FOREIGN KEY(usage_id) REFERENCES usage(id)
                )'''
            )

    def revert(self):
        with self.db as cur:
            cur.execute('DROP TABLE asr_transcription')
            cur.execute('DROP TABLE usage')
            cur.execute('DROP TABLE audio')
        
        self.db.delete()


class ModelMigration(Migration):

    def __init__(self):
        super().__init__()

    def update(self):
        with self.db as cur:
            cur.execute('CREATE TABLE model(id TEXT PRIMARY KEY, name TEXT UNIQUE)')
            cur.execute('ALTER TABLE usage ADD COLUMN model_id TEXT REFERENCES model(id)')
    
    def revert(self):
        with self.db as cur:
            cur.execute('ALTER TABLE usage DROP COLUMN model_id')
            cur.execute('DROP TABLE model')


class VSGServiceMigration(Migration):

    def __init__(self):
        super().__init__()

    def update(self):
        with self.db as cur:
            cur.execute('''
                CREATE TABLE vsg_service_usage(
                    id TEXT PRIMARY KEY, 
                    usage_id TEXT, 
                    email TEXT,
                    FOREIGN KEY(usage_id) REFERENCES usage(id)
                )'''
            )

    def revert(self):
        with self.db as cur:
            cur.execute('DROP TABLE vsg_service_usage')


def main(args):
    if args.run_type == 'init':
        # run migrations in specific order
        for migration in [InitMigration, ModelMigration, VSGServiceMigration]:
            migration = migration()
            migration.update()
    else:
        migration = globals().get(args.name)()
        if not migration:
            return
        
        getattr(migration, args.method)()


if __name__ == '__main__':
    # NOTE: a newer version of sqlite3 may be required for some functions to work e.g. DROP COLUMN
    # install the newest version of sqlite3 and export LD_LIBRARY_PATH=/usr/local/lib to point to this new version
    # before running this script

    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='run_type')
    
    parser_1 = sub_parsers.add_parser('init')

    parser_2 = sub_parsers.add_parser('single')
    parser_2.add_argument('name')
    parser_2.add_argument('method', choices=['update', 'revert'])

    main(parser.parse_args())
