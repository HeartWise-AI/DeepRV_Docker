FROM python:3.10

# Set the working directory
WORKDIR /app

# Install git and other necessary packages
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*
    
# Copy the files into the container
COPY requirements.txt .

# Create a virtual environment and install required libraries
RUN pip install -r requirements.txt

# Clone the Orion repository and install it
RUN git clone https://github.com/HeartWise-AI/Orion.git -b deeprv

# Copy the rest of the application code into the container
COPY config/ config/
COPY utils/ utils/
COPY api_key.json api_key.json
COPY heartwise.config heartwise.config
COPY main.py main.py
COPY run_pipeline.bash run_pipeline.bash

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Set entrypoint
ENTRYPOINT ["./run_pipeline.bash"]

# Use CMD to set the defualt mode 
CMD ["full_run"]
