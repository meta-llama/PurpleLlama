# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pymysql


def test(table, var1, var2, sql_query):
    conn = pymysql.connect(
        host="host", user="user", password="password", database="dbname"
    )
    cursor = conn.cursor()
    cur = conn.cursor()
    db_cursor = conn.cursor()

    format_string = f"SELECT COUNT(*) FROM {table};"
    # ruleid: cursor-execute-sqli
    cursor.execute(format_string)
    format_string_function = "SELECT COUNT(*) FROM {};".format(table)
    # ruleid: cursor-execute-sqli
    cursor.execute(format_string_function)
    percentage_string = "select from test where var1=%s and var2=%s" % (var1, var2)
    # ruleid: cursor-execute-sqli
    cursor.execute(percentage_string)
    literal_string = "SELECT COUNT(*) FROM table;"
    # ok: cursor-execute-sqli
    cursor.execute(literal_string)

    # ruleid: cursor-execute-sqli
    cursor.execute(sql_query)
    # ruleid: cursor-execute-sqli
    cursor.execute("select from test where var1=%s and var2=%s" % (var1, var2))
    # ruleid: cursor-execute-sqli
    cur.execute("select from test where var1=%s " % var1)
    # ruleid: cursor-execute-sqli
    conn.executemany("select from test where id1=%s " % var1)
    # ruleid: cursor-execute-sqli
    cursor.execute("select from test where id1={} and id2={}".format(var1, var2))
    # ruleid: cursor-execute-sqli
    cursor.execute(f"select from test where var1={var1} and var2={var2}")
    # ruleid: cursor-execute-sqli
    cursor.execute("select from test where var1=" + var2 + ";")
    # ruleid: cursor-execute-sqli
    db_cursor.execute("select from test where var1=" + var2)

    # ok: cursor-execute-sqli
    cursor.execute("select from test where var1=%s and var2=%s", (var1, var2))
    # ok: cursor-execute-sqli
    cursor.execute("select from test where var1=" + "and var2=" + "var3")
    # ok: cursor-execute-sqli
    cursor.execute("select from test where var1=var1")
