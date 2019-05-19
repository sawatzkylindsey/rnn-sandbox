
import os
import pdb
import pickle
import sqlite3

sqlite3.register_adapter(tuple, lambda t: pickle.dumps(t))
sqlite3.register_converter("tuple", pickle.loads)


QUERY_DB = "query-db."


def database_for(query_dir, lstm, key):
    os.makedirs(query_dir, exist_ok=True)
    db_path = os.path.join(query_dir, QUERY_DB + key)
    db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    dimensions = lstm.part_width(key)
    cursor = db.cursor()

    cursor.execute("""create table sequences
        (id integer primary key,
        sequence tuple not null unique)""")

    # The (sequence_id, sequence_index) serve as a composite unique key.
    # This key can uniquely define the point (hidden state).
    cursor.execute("""create table activations
        (sequence_id integer not null,
        sequence_index integer not null,
        %s,
        foreign key(sequence_id) references sequences(id),
        unique (sequence_id, sequence_index) on conflict ignore)""" % ", ".join(["axis_%d real" % d for d in range(dimensions)]))

    for d in range(dimensions):
        cursor.execute("""create index axis_%d_index on activations(axis_%d)""" % (d, d))

    db.commit()
    return QueryDatabase(db, dimensions)


def get_databases(query_dir, lstm):
    databases = {}

    for name in os.listdir(query_dir):
        if name.startswith(QUERY_DB):
            db_path = os.path.join(query_dir, name)
            db = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
            key = name[len(QUERY_DB):]
            databases[key] = QueryDatabase(db, lstm.part_width(key))

    return databases


class QueryDatabase:
    BATCH_SIZE = 500

    def __init__(self, db, dimensions):
        self.db = db
        self.dimensions = dimensions
        self._insert_activations = "insert into activations values (?, ?, %s)" % ",".join(["?"] * self.dimensions)
        self._select_point = ", ".join(["activations.axis_%d" % d for d in range(dimensions)])

    def insert_sequence(self, sequence):
        cursor = self.db.cursor()
        cursor.execute("insert into sequences values (null, ?)", (sequence,))
        return cursor.lastrowid

    def insert_activations(self, data):
        assert isinstance(data[0][0], int), "%s (%s) is not an int" % (data[0][0], type(data[0][0]))
        assert isinstance(data[0][1], int), "%s (%s) is not an int" % (data[0][1], type(data[0][1]))
        assert isinstance(data[0][2], float), "%s (%s) is not a float" % (data[0][2], type(data[0][2]))
        self.db.cursor().executemany(self._insert_activations, data)

    def commit(self):
        self.db.commit()

    def select_activations_range(self, axis, lower_bound, upper_bound):
        cursor = self.db.cursor()
        cursor.execute("""select sequences.sequence, activations.sequence_index, %s
            from sequences inner join activations on sequences.id = activations.sequence_id
            where activations.axis_%d >= ? and activations.axis_%d <= ?""" % (self._select_point, axis, axis), (lower_bound, upper_bound))
        return cursor.fetchall()

    def select_activations(self, sequences):
        cursor = self.db.cursor()
        matched_sequences = [sequence for sequence in sequences]
        points = []
        offset = 0

        while offset < len(matched_sequences):
            batch = matched_sequences[offset:offset + QueryDatabase.BATCH_SIZE]
            cursor.execute("""select sequences.sequence, activations.sequence_index, %s
                from sequences inner join activations on sequences.id = activations.sequence_id
                where sequences.sequence in (%s)""" % (self._select_point, ", ".join(["?"] * len(batch))), tuple(batch))
            points += cursor.fetchall()
            offset += QueryDatabase.BATCH_SIZE

        return points

