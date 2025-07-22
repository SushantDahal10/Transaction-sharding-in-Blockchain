# Python script to generate docker-compose.yml for 30 nodes

def generate_docker_compose():
    docker_compose = """version: '3.8'

services:
"""
    
    # Loop to create 30 nodes
    for i in range(1, 31):
        docker_compose += f"""  node{i}:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: node{i}
    networks:
      - blockchain_network
    ports:
      - "{5000 + i}:5000"
    environment:
      - NODE_ID={i}
      - PEERS={','.join([f'http://node{j}:5000' for j in range(1, 31) if j != i])}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000"]
      interval: 10s
      retries: 5

"""
    
    docker_compose += """networks:
  blockchain_network:
    driver: bridge
"""
    
    # Save the generated content to a file
    with open('docker-compose.yml', 'w') as file:
        file.write(docker_compose)
    print("docker-compose.yml file generated successfully.")

# Call the function to generate the file
generate_docker_compose()
