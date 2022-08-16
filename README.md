<img src="./src/GetAround_logo.png">

# GetAround Project

Name : Iandro Rakotondrandria

Data : Data files are in "src" folder

Link to app deplyed on Heroku : https://iandrogetaround.herokuapp.com/

Link to MLFlow server : https://getaroundmlfowserver.herokuapp.com/

Link to API documentation : https://iandrogetaroundapi.herokuapp.com/docs

## Context 

[GetAround](https://www.getaround.com/?wpsrc=Google+Organic+Search) is the Airbnb for cars. You can rent cars from any person for a few hours to a few days! Founded in 2009, this company has known rapid growth. In 2019, they count over 5 million users and about 20K available cars worldwide. 

When renting a car, GetAround users have to complete a checkin flow at the beginning of the rental and a checkout flow at the end of the rental in order to:

* Assess the state of the car and notify other parties of pre-existing damages or damages that occurred during the rental.
* Compare fuel levels.
* Measure how many kilometers were driven.

The checkin and checkout of our rentals can be done with three distinct flows:
* **📱 Mobile** rental agreement on native apps: driver and owner meet and both sign the rental agreement on the owner’s smartphone
* **Connect:** the driver doesn’t meet the owner and opens the car with his smartphone
* **📝 Paper** contract (negligible)

When using Getaround, drivers book cars for a specific time period, from an hour to a few days long. They are supposed to bring back the car on time, but it happens from time to time that drivers are late for the checkout.

Late returns at checkout can generate high friction for the next driver if the car was supposed to be rented again on the same day : Customer service often reports users unsatisfied because they had to wait for the car to come back from the previous rental or users that even had to cancel their rental because the car wasn’t returned on time.


## Goals 🎯

In order to mitigate those issues GetAround decided to implement a minimum delay between two rentals. A car won’t be displayed in the search results if the requested checkin or checkout times are too close from an already booked rental.

It solves the late checkout issue but also potentially hurts Getaround/owners revenues: we need to find the right trade off.

**Their Product Manager still needs to decide:**
* **threshold:** how long should the minimum delay be?
* **scope:** should we enable the feature for all cars?, only Connect cars?

In order to help them make the right decision, they are asking for some data insights :

* Which share of our owner’s revenue would potentially be affected by the feature How many rentals would be affected by the feature depending on the threshold and scope we choose?
* How often are drivers late for the next check-in? How does it impact the next driver?
* How many problematic cases will it solve depending on the chosen threshold and scope?

## Web dashboard :bar_chart:

In oreder to help the product Management team with the above questions, we use `streamlit` to display a web dashboard.

Here is the link to the app : https://iandrogetaround.herokuapp.com/ 

**To set up locally :**

1. Go into the `"./streamlit_app"` folder
2. Run :

```shell
docker build . -t iandrogetaround
```

```shell
docker run -it -v "$(pwd):/home/app" -e PORT=80 -p 4000:80 iandrogetaround
```

3. You can run the application in your browser by going to  : `http://0.0.0.0:4000`

## Machine Learning - `/predict` endpoint :computer:

In addition, we worked on *pricing optimization*. We have gathered some data to suggest optimum prices for car owners using Machine Learning. 

### 1. MLFlow

We used MLFlow to track, log and save the model and its training

The URL to the MLFlow server is : https://getaroundmlfowserver.herokuapp.com/

**To set up locally :**

1. Go into the `"./mlflow"` folder
2. Run :

```shell
docker build . -t getaroundmlflowserver
```

```shell
docker run -it\
-p 4000:4000\
-v "$(pwd):/home/app"\
-e PORT=4000\
-e AWS_ACCESS_KEY_ID="YOUR_AWS_ACCESS_KEY_ID"\
-e AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"\
-e BACKEND_STORE_URI="YOUR_BACKEND_STORE_URI"\
-e ARTIFACT_ROOT="YOUR_ARTIFACT_ROOT"\
getaroundmlflowserver
```

You will have to replace *YOUR_AWS_ACCESS_KEY_ID"*, *AWS_SECRET_ACCESS_KEY="YOUR_AWS_SECRET_ACCESS_KEY"*, *BACKEND_STORE_URI="YOUR_BACKEND_STORE_URI"*, and *ARTIFACT_ROOT="YOUR_ARTIFACT_ROOT"* by their actual values

3. You can run the application in your browser by going to  : `http://0.0.0.0:4000`

### 2. API

We built an API you can use to get prediction by unsing our machine learning model at an **endpoint** `/predict`. 

The full URL is : `https://iandrogetaroundapi.herokuapp.com/predict`

This endpoint accepts **POST method** with JSON input data and it should return the predictions. We assume **inputs will be always well formatted**.

The input JSON has to be in the following format :

```JSON
{
  "model_key": String,
  "mileage": Integer,
  "engine_power": Integer,
  "fuel": String,
  "car_type": String,
  "private_parking_available": Boolean,
  "has_gps": Boolean,
  "has_air_conditioning": Boolean,
  "automatic_car": Boolean,
  "has_getaround_connect": Boolean,
  "has_speed_regulator": Boolean,
  "winter_tires": Boolean
}
```

Example in a python request :

```python
model_key = 'Peugeot'
mileage = 126213
engine_power = 225
fuel = 'petrol'
car_type = 'convertible'
private_parking_available = False
has_gps = False
has_air_conditioning = False
automatic_car = False
has_getaround_connect = False
has_speed_regulator = True
winter_tires = True


response = requests.post('https://iandrogetaroundapi.herokuapp.com/predict', json=
    {
        "model_key": model_key,
        "mileage": mileage,
        "engine_power": engine_power,
        "fuel": fuel,
        "car_type": car_type,
        "private_parking_available": private_parking_available,
        "has_gps": has_gps,
        "has_air_conditioning": has_air_conditioning,
        "automatic_car": automatic_car,
        "has_getaround_connect": has_getaround_connect,
        "has_speed_regulator": has_speed_regulator,
        "winter_tires": winter_tires
    }
)
```

The response will be a JSON with one key `prediction` corresponding to the prediction.

Response :

```python
print(response.json())
```
```
{"prediction":116.95750114535788}
```

Example in shell:

```shell
curl -X 'POST' \
  'https://iandrogetaroundapi.herokuapp.com/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model_key": "Peugeot",
  "mileage": 222615,
  "engine_power": 225,
  "fuel": "petrol",
  "car_type": "convertible",
  "private_parking_available": false,
  "has_gps": false,
  "has_air_conditioning": false,
  "automatic_car": false,
  "has_getaround_connect": false,
  "has_speed_regulator": true,
  "winter_tires": true
}'
```

Response : 
```
{"prediction":116.95750114535788}
```

**To set up locally :**

1. Go into the `"./API"` folder
2. Run :

```shell
docker build . -t iandrogetaroundapi
```

```shell
docker run -it \
-v "$(pwd):/home/app" \
-p 4000:4000 \
-e PORT=4000 \
-e MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
-e AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
-e AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
-e BACKEND_STORE_URI=$BACKEND_STORE_URI \
-e ARTIFACT_ROOT=$ARTIFACT_ROOT \
iandrogetaroundapi
```

Your environment variables need to be defined

3. You can run the application in your browser by going to  : `http://0.0.0.0:4000`

### Documentation page

We provided the users with a **documentation** about your API.

The documentation page is located at `https://iandrogetaroundapi.herokuapp.com/docs`

## Deliverable 📬

To sum up, to complete this project, we produced:

- A **dashboard** in production (accessible via a web page for example)
- The **whole code** stored in a **Github repository**.
- A **MLFlow tracking server** on Heroku to track our machine learing model
- An **documented online API** on Heroku server `.