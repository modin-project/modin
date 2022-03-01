# This script sets up a SQL server listening at 0.0.0.0:1234.

# If any step fails, we can't set up a valid SQL server for unit tests.
set -e

# Pull the 2019 SQL server docker container image by following:
# https://docs.microsoft.com/en-us/sql/linux/quickstart-install-connect-docker?view=sql-server-ver15&pivots=cs1-powershell#pullandrun2019
sudo docker pull mcr.microsoft.com/mssql/server:2019-latest
sudo docker run -d --name example_sql_server -e 'ACCEPT_EULA=Y' -e 'SA_PASSWORD=Strong.Pwd-123' -p 1433:1433 mcr.microsoft.com/mssql/server:2019-latest


# Wait 10 seconds because if we don't the server typically will not be ready
# to accept connections by the time we want to make them.
sleep 10
