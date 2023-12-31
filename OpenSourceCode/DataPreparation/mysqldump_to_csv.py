#!/usr/bin/env python
import fileinput
import csv
import sys

# This prevents prematurely closed pipes from raising
# an exception in Python
# from signal import signal, SIGPIPE, SIG_DFL
# signal(SIGPIPE, SIG_DFL)

# allow large content in the dump
csv.field_size_limit(2**31-1)

def is_insert(line):
    """
    Returns true if the line begins a SQL insert statement.
    """
    return line.startswith('INSERT INTO')


def get_values(line):
    """
    Returns the portion of an INSERT statement containing values
    """
    return line.partition(' VALUES ')[2]


def values_sanity_check(values):
    """
    Ensures that values from the INSERT statement meet basic checks.
    """
    assert values
    assert values[0] == '('
    # Assertions have not been raised
    return True


def parse_values(values, outfile,field_num,del_index):
    """
    Given a file handle and the raw values from a MySQL INSERT
    statement, write the equivalent CSV to the file
    """
    latest_row = []

    reader = csv.reader([values], delimiter=',',
                        doublequote=False,
                        escapechar='\\',
                        quotechar="'",
                        strict=True
    )

    writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
    for reader_row in reader:
        for column in reader_row:
            # If our current string is empty...
            if len(column) == 0 or column == 'null':
                latest_row.append(chr(0))
                continue
            # If our string starts with an open paren
            if column[0] == "(":
                # If we've been filling out a row
                if len(latest_row) > 0:
                    # Check if the previous entry ended in
                    # a close paren. If so, the row we've
                    # been filling out has been COMPLETED
                    # as:
                    #    1) the previous entry ended in a )
                    #    2) the current entry starts with a (
                    if latest_row[-1][-1] == ")":
                        # Remove the close paren.
                        latest_row[-1] = latest_row[-1][:-1]
                        writer.writerow(latest_row)
                        latest_row = []
                # If we're beginning a new row, eliminate the
                # opening parentheses.
                if len(latest_row) == 0:
                    column = column[1:]
            # Add our column to the row we're working on.
            latest_row.append(column)
        # At the end of an INSERT statement, we'll
        # have the semicolon.
        # Make sure to remove the semicolon and
        # the close paren.
        if latest_row[-1][-2:] == ");":
            latest_row[-1] = latest_row[-1][:-2]
        while len(latest_row) > field_num:
            del latest_row[del_index]
        writer.writerow(latest_row)


def main(input_file,output_file,field_num,del_index):
    """
    Parse arguments and start the program
    """
    # Iterate over all lines in all files
    # listed in sys.argv[1:]
    # or stdin if no args given.
    try:
        i = 0
        with open(output_file, "w",encoding='utf-8',newline='') as datacsv:
            for line in fileinput.input(input_file, openhook=fileinput.hook_encoded('utf-8', ) ):
                if i % 50000 == 0:
                    print("已经处理至第{0}行".format(i))
                # Look for an INSERT statement and parse it.
                if not is_insert(line):
                    raise Exception("SQL INSERT statement could not be found!")
                values = get_values(line)
                if not values_sanity_check(values):
                    raise Exception("Getting substring of SQL INSERT statement after ' VALUES ' failed!")
                parse_values(values, datacsv,field_num,del_index)
                i+=1
    except KeyboardInterrupt:
        sys.exit(0)

if __name__ == "__main__":
    # 第一个事故数据表
    main(input_file="_1109.sql",
         output_file="_1109.csv",
         field_num=383, del_index=7)
