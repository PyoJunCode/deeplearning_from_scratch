# Model

Using YOLOV3 in cvlib and tensorflow

Deploy Predict server by using Server.ipynb, Consume it by Client.ipynb



Server using FastAPI to deploy, after execute server codes then

http://localhost:8000/docs -> Click `/predict` -> `Try it out` to test server is working.



# Docker

### 1. Pulling the image from Docker hub

```bash
docker pull deeplearningai/mlepc1w1-ugl:jupyternb
```

above container contain jupyter notebook environment on ubuntu (provided by deeplearning.ai)

### 2. Running a container out of the image:

```bash
docker run -it --rm -p 8888:8888 -p 8000:8000 --mount type=bind,source="$(pwd)",target=/home/jovyan/work deeplearningai/mlepc1w1-ugl:jupyternb
```

go to https://localhost:8888/  and run the server and test with client
