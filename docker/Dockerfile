FROM ubuntu:18.04
COPY mnist_mlops /exp/mnist_mlops
RUN mkdir /exp/models
COPY requirements.txt /exp/requirements.txt
RUN apt-get update && apt-get install -y python3 python3-pip 
RUN pip3 install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python3","./mnist_mlops/plot_graphs.py"]
COPY models /exp/models