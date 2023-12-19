#!/bin/bash

# MySQL credentials
DB_USER="root"
DB_PASS="root"
DB_NAME="python_ea"

# SQL file path

# Determine the project root directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# SQL file path relative to the project root
SQL_FILE="$PROJECT_ROOT/database/init.sql"

# Create a new database
mysql -u"$DB_USER" -p"$DB_PASS" -e "CREATE DATABASE IF NOT EXISTS $DB_NAME;"

# Check if the SQL file exists
if [ -f "$SQL_FILE" ]; then
    # Run SQL file
    mysql -u"$DB_USER" -p"$DB_PASS" "$DB_NAME" < "$SQL_FILE"
    echo "Database '$DB_NAME' set up and initialized from '$SQL_FILE'."
else
    echo "SQL file '$SQL_FILE' not found."
fi
