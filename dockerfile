FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install MySQL client and server non-interactively
RUN apt update && apt install -y python3 pip mysql-server vim mc wget curl && apt-get clean

# Set the working directory
WORKDIR /app

# Copy the Conda environment file to the container
COPY requirements.txt .

# Install the Python packages from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 3306

# Start the MySQL service and configure it
#RUN service mysql start && \
#    mysql -u root -e "CREATE DATABASE MetLifeChallenge;" && \
#    mysql -u root -e "CREATE USER 'lucas'@'%' IDENTIFIED BY 'foo123';" && \
#    mysql -u root -e "GRANT ALL PRIVILEGES ON MetLifeChallenge.* TO 'lucas'@'%';" && \
#    mysql -u root -e "FLUSH PRIVILEGES;"

# Copy your application code to the container
COPY . /app/

RUN chmod +x /app/run_process.sh

# Add Python to the PATH
ENV PATH="/usr/local/bin:${PATH}"

# Define the command to run your application
CMD ["/app/run_process.sh"]