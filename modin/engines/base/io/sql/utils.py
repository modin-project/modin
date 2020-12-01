class SqlQuery:
    engine = None

    def __init__(self, sa_driver=None):
        self.engine = BaseSqlQuery()
        if sa_driver == "pymssql":
            self.engine = MssqlSqlQuery()

    def empty(self, sql):
        return self.engine.get_empty(sql)

    def row_cnt(self, sql):
        return self.engine.get_row_cnt(sql)

    def partitioned(self, sql, limit, offset):
        return self.engine.get_partitioned(sql, limit, offset)


class BaseSqlQuery:
    @staticmethod
    def get_empty(sql):
        return "SELECT * FROM ({}) as foo LIMIT 0".format(sql)

    @staticmethod
    def get_row_cnt(sql):
        return "SELECT COUNT(*) FROM ({}) as foo".format(sql)

    @staticmethod
    def get_partitioned(sql, limit, offset):
        return "SELECT * FROM ({0}) as foo LIMIT {1} OFFSET {2}".format(
            sql, limit, offset
        )


class MssqlSqlQuery(BaseSqlQuery):
    @staticmethod
    def get_empty(sql):
        return "SELECT Top 0 * FROM ({}) as foo".format(sql)

    @staticmethod
    def get_partitioned(sql, limit, offset):
        return "SELECT * FROM ({0}) as foo ORDER BY(SELECT NULL) OFFSET {1} ROWS FETCH NEXT {2} ROWS ONLY".format(
            sql, offset, limit
        )
