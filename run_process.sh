#!/bin/bash

# Start the MySQL service
service mysql start

# Run MySQL setup commands
mysql -u root -e "CREATE DATABASE MetLifeChallenge;"
mysql -u root -e "CREATE USER 'lucas'@'%' IDENTIFIED BY 'foo123';"
mysql -u root -e "GRANT ALL PRIVILEGES ON MetLifeChallenge.* TO 'lucas'@'%';"
mysql -u root -e "FLUSH PRIVILEGES;"

python3 main.py