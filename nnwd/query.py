
import functools
import os
import pdb
import pickle
import psycopg2 as postgres
import sqlite3 as sqlite

sqlite.register_adapter(tuple, lambda t: pickle.dumps(t))
sqlite.register_converter("tuple", pickle.loads)
postgres.extensions.register_adapter(tuple, lambda t: postgres.extensions.Binary(pickle.dumps(t)))

POSTGRES = "postgres"
SQLITE = "sqlite"
QUERY_DB = "query-db."


def postgres_db():
    db = postgres.connect("dbname=postgres user=sawatzky")
    # I could not for the life of me get the postgres -> python casting working.
    #cursor = db.cursor()
    #cursor.execute("select NULL::bytea")
    #bytea_oid = cursor.description[0][1]
    #TUPLE = postgres.extensions.new_type(psycopg2.BINARY.values, "bytea", lambda v, c: None if v is None else pickle.loads(v))
    #TUPLE = postgres.extensions.new_type((bytea_oid,), "bytea", lambda v, c: None if v is None else pickle.loads(v))
    #TUPLE = postgres.extensions.new_type(psycopg2.BINARY.values, "TUPLE", lambda v, c: None if v is None else pickle.loads(v))
    #TUPLE = postgres.extensions.new_type((bytea_oid,), "TUPLE", lambda v, c: None if v is None else pickle.loads(v))
    #postgres.extensions.register_type(TUPLE)
    #postgres.extensions.register_type(TUPLE, db)
    return db


def sqlite_db(query_dir, key):
    os.makedirs(query_dir, exist_ok=True)
    db_path = os.path.join(query_dir, QUERY_DB + key)
    db = sqlite.connect(db_path, detect_types=sqlite.PARSE_DECLTYPES)
    return db


def database_for(query_dir, db_kind, lstm, key):
    if db_kind == POSTGRES:
        db = postgres_db()
        table_suffix = "_%s" % key.replace("-", "_")
    else:
        db = sqlite_db(query_dir, key)
        table_suffix = ""

    dimensions = lstm.part_width(key)
    cursor = db.cursor()

    try:
        id_type = "integer" if db_kind == SQLITE else "serial"
        sequence_type = "tuple" if db_kind == SQLITE else "bytea"
        cursor.execute("""create table sequences
            (id %s primary key,
            sequence %s not null unique)""" % (id_type, sequence_type))
        cursor.execute("""create index sequence on sequences(sequence)""")
    except (sqlite.OperationalError, postgres.errors.UniqueViolation, postgres.errors.DuplicateTable) as e:
        if "already exists" in str(e):
            db.rollback()
        else:
            raise e

    try:
        # The (sequence_id, sequence_index) serve as a composite unique key.
        # This key can uniquely define the point (hidden state).
        suffix = " on conflict ignore" if db_kind == SQLITE else ""
        cursor.execute("""create table activations%s
            (sequence_id integer not null,
            sequence_index integer not null,
            %s,
            foreign key(sequence_id) references sequences(id),
            unique (sequence_id, sequence_index)%s)""" % (table_suffix, ", ".join(["axis_%d real" % d for d in range(dimensions)]), suffix))

        for d in range(dimensions):
            cursor.execute("""create index axis_%d_index%s on activations%s(axis_%d)""" % (d, table_suffix, table_suffix, d))
    except (sqlite.OperationalError, postgres.errors.UniqueViolation, postgres.errors.DuplicateTable) as e:
        if "already exists" in str(e):
            pass
        else:
            raise e

    db.commit()
    return QueryDatabase(db, db_kind, table_suffix, dimensions)


def get_databases(query_dir, db_kind, lstm):
    databases = {}

    if db_kind == POSTGRES:
        for key in lstm.keys():
            db = postgres.connect("dbname=postgres user=sawatzky")
            databases[key] = QueryDatabase(db, db_kind, "_%s" % key.replace("-", "_"), lstm.part_width(key))
    else:
        for name in os.listdir(query_dir):
            if name.startswith(QUERY_DB):
                db_path = os.path.join(query_dir, name)
                db = sqlite.connect(db_path, detect_types=sqlite.PARSE_DECLTYPES)
                key = name[len(QUERY_DB):]
                databases[key] = QueryDatabase(db, db_kind, "", lstm.part_width(key))

    return databases


class QueryDatabase:
    BATCH_SIZE = 500

    def __init__(self, db, db_kind, table_suffix, dimensions):
        self.db = db
        self.db_kind = db_kind
        self.table_suffix = table_suffix
        self.dimensions = dimensions
        suffix = "" if self.db_kind == SQLITE else " on conflict do nothing"
        self._insert_activations = "insert into activations%s values (?, ?, %s)%s" % (self.table_suffix, ",".join(["?"] * self.dimensions), suffix)
        self._select_point = ", ".join(["activations.axis_%d" % d for d in range(dimensions)])

    def wrap(self, statement):
        if self.db_kind == SQLITE:
            return statement
        else:
            return statement.replace("?", "%s")

    def _converter(self):
        if self.db_kind == SQLITE:
            return lambda item: item
        else:
            return lambda item: tuple((pickle.loads(item[0]), *item[1:]))

    def insert_sequence(self, sequence):
        self.commit()
        cursor = self.db.cursor()

        try:
            suffix = "" if self.db_kind == SQLITE else " returning id"
            cursor.execute(self.wrap("insert into sequences(sequence) values (?)%s" % suffix), (sequence,))

            if self.db_kind == SQLITE:
                return cursor.lastrowid
            else:
                return cursor.fetchone()[0]
        except (sqlite.IntegrityError, postgres.errors.UniqueViolation) as e:
            self.db.rollback()
            cursor.execute(self.wrap("select id from sequences where sequence = ?"), (sequence,))
            return cursor.fetchone()[0]

    def insert_activations(self, data):
        assert isinstance(data[0][0], int), "%s (%s) is not an int" % (data[0][0], type(data[0][0]))
        assert isinstance(data[0][1], int), "%s (%s) is not an int" % (data[0][1], type(data[0][1]))
        assert isinstance(data[0][2], float), "%s (%s) is not a float" % (data[0][2], type(data[0][2]))
        self.db.cursor().executemany(self.wrap(self._insert_activations), data)

    def commit(self):
        self.db.commit()

    @functools.lru_cache()
    def select_activations_range(self, axis, lower_bound, upper_bound):
        activations_suffix = "" if self.db_kind == SQLITE else self.table_suffix + " as activations"
        cursor = self.db.cursor()
        cursor.execute(self.wrap("""select sequences.sequence, activations.sequence_index, %s
            from sequences inner join activations%s on sequences.id = activations.sequence_id
            where activations.axis_%d >= ? and activations.axis_%d <= ?""" % (self._select_point, activations_suffix, axis, axis)), (lower_bound, upper_bound))
        # Roll out the stream here so the result may be cached.
        return [r for r in streaming_convert(cursor.fetchall(), self._converter())]

    def select_activations(self, sequences):
        activations_suffix = "" if self.db_kind == SQLITE else self.table_suffix + " as activations"
        cursor = self.db.cursor()
        matched_sequences = [sequence for sequence in sequences]
        points = []
        offset = 0

        while offset < len(matched_sequences):
            batch = matched_sequences[offset:offset + QueryDatabase.BATCH_SIZE]
            cursor.execute(self.wrap("""select sequences.sequence, activations.sequence_index, %s
                from sequences inner join activations%s on sequences.id = activations.sequence_id
                where sequences.sequence in (%s)""" % (self._select_point, activations_suffix, ", ".join(["?"] * len(batch)))), tuple(batch))
            points += cursor.fetchall()
            offset += QueryDatabase.BATCH_SIZE

        return streaming_convert(points, self._converter())


def streaming_convert(items, converter):
    for item in items:
        yield converter(item)

