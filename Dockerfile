#docker build -t image_name . -> budowanie image'u
#docker run -p 8098:8098 image_name -> uruchom nowy kontener z danego widoku
    # -i - interactive mode
    # -t - pseudo terminal 
    # -p port_from_outside:port_in_container -> żeby działało wystawienie serwera

#docker ps - działające kontenery

#docker container ls -a - wylistuje wszystkie kontenery
#docker container start -ai <container_id>  - uruchomi wcześniej utworzony kontener

#docker exec -it <RTS> /bin/sh - cli w kontenerze

#docker system prune -a - usuwanie z dysku 
#  - all stopped containers
#  - all networks not used by at least one container
#  - all images without at least one container associated to them
#  - all build cache



FROM python:3.8

WORKDIR /color_classification

COPY app.py .

COPY templates templates

COPY static static

COPY ml/data ml/data
COPY ml/model ml/model
COPY ml/classification.py ml/classification.py
COPY ml/configuration.properties ml/configuration.properties
COPY ml/relearning.py ml/relearning.py

RUN pip install flask
RUN pip install flask_wtf
RUN pip install wtforms
RUN pip install sklearn
RUN pip install pandas
RUN pip install numpy
RUN pip install Pillow
RUN pip install tensorflow==2.2.0
RUN pip install keras==2.3.1


CMD ["python", "./app.py"]