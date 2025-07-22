FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy source code and requirements
COPY ../src/ /app/
COPY ../requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the ports for Flask web server
EXPOSE 5000

# Run both port_server and blockchain_node concurrently
CMD ["sh", "-c", "python port_server.py 5000 & python blockchain_node.py"]
