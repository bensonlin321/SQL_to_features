from sql_metadata import Parser

# Example SQL query with a subquery and joins
sql = """SELECT * 
         FROM (
            SELECT id, col1, col2
            FROM TableA
         ) a
         LEFT JOIN TableB b ON a.id = b.id 
         LEFT JOIN thingdata td ON b.id = td.id"""

# Parse the query using sql_metadata
parser = Parser(sql)

# Extract and print all the tables
tables = parser.tables

columns = parser.columns

print(f"parser: {dir(parser)}")

join_clauses = [clause.strip() for clause in sql.split() if "JOIN" in clause.upper()]
join_count = len(join_clauses)

print(f"Tables involved: {tables}")
print(f"JOIN clauses: {join_clauses}")
print(f"columns: {columns}")