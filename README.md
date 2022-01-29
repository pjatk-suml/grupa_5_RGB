# Color Classification Project - ML Implementation

*SUML gr 14c/5: Michał Nowicki, Mikołaj Ziółek, Jakub Zowczak*

## About

Color Classification is a service which presents the end user with 2 possibilities related to the ML model trained to classify a given colour (RGB values) to the colour name (enum) which best describes it.
 
The first one is the possibility to train the model - the user, based on the randomly presented colour, classifies it to a given colour. This user input is stored in the training data set and the model is trained.
 
The second possibility is to perform a classification task by an already trained model. The user inputs a colour and in response receives the result of the classification performed by the ML model as one of the available colors (`Beige`, 
`Blue`, 
`Brown`, 
`Green`, 
`Grey`, 
`Orange`, 
`Red`, 
`Violet`, 
`Yellow`)
 
The fulfilment of these functionalities will take place through a web page, which can be accessed by the user through a web browser.

## Usage

Requirements: `Docker` installed.

The easiest way to run the application is with Docker:

```bash
docker run michalnowi/color_classify:0.0.1
```

This will create a container locally from our project docker image available on docker hub.

After you created the container it will automatically start. All changes will be saved to this container.

If you want to create another container, you can run previous command again.

To run the previous container, you need to first find it:

```bash
docker container ls -a
```

Then copy its id, and start it with:

```bash
docker container start -ai <container_id>
```

While the container is running, you can enter its bash terminal with:

```bash
docker exec -it <container_id> /bin/sh
```

It is a good idea to download vim to the container with:

```bash
apt-get update
apt-get install vim
```

With vim you can access `ml/configuration.properties` and change properties during runtime of the service.

## Manual installation

Requirements: `Python3` installed, `venv` installed

Run the script `setup.sh` which creates the environment, installs dependencies and starts the app. 

```bash
./setup.sh
```

Afterwards, you can run the application with the following command:
```
./start.sh
```

## Functionalities 
 
### ML model 
- Data and classes
  - The data for the model are the vectors [R,G,B] - corresponding to the - colour values in R,G,B
- Model learning
  - After accumulating a certain number (configurable at the service - runtime) of new data points, the model is trained. 
  - The new model is saved and evaluated. If its accuracy is better than - the previous one - in the configuration it is automatically set as the - model to be used for the classification task.
- Execution of the colour classification task into the class (name)
  - The model is given a vector [R,G,B] - the values set by the user of the - service
  - The model performs the classification task, specifying the class to which it classifies the colour.
 
### ColorPicker 
- The user can select a colour using a tool that presents a colour palette from which the user can choose the colour of interest to be classified by the ML model

### Colour Randomizer
- The program can draw and display a colour to the user (R, G and B - values are drawn
- The user's task is to classify a given colour into one of the classes
- By using randomization - we enrich the data set with possibly valuable information
