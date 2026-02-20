
## Usage  

1. **Build the Docker image**:

The Dockerfile referenced in the instructions is ``Dockerfile_CGN``. You can build the Docker image by running the following command:
```bash
docker build -t cuda118:contact_graspnet -f Dockerfile_CGN .
```
Alternatively, you may use the following command to pull the docker image for contact-graspnet from docker hub:
```
docker pull zhaohuajing/cuda118:contact_graspnet
```

2. **Start the Docker container** (if not already running):  
   ```bash
   ./run_docker.sh
   ```
This script launches the Contact-GraspNet container with the proper environment and names it:
	```contact_graspnet_container
	```

