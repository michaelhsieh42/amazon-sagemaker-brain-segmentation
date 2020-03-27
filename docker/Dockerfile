FROM python:3.6-slim-buster

# Install libraries
RUN pip3 install imageio==2.3.0 numpy==1.16.4 nibabel==3.0.2 scikit-learn==0.20.3 pillow==5.1.0

# Add a python script and configure Docker to run it
ADD processing_script.py /
ENTRYPOINT ["python3", "/processing_script.py"]

