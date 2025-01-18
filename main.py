import torch 
import torch.nn as nn
import numpy as np
import requests

while True:
    print("*"*20)
    print("City future weather prediction")
    cityname = input("Enter city for future weather prediction: ")
    fetch_data = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={cityname}&appid=dd19b95d4b73b36c6a53a3a47cca554e")
    data = fetch_data.json()
    print(f"Current weather of {cityname} : {int(data['main']['temp'] - 273.15)}°")

    X = np.array([
        [281.62, 86, 2.06, 75],  # City1: [temp, humidity, wind_speed, cloudiness]
        [290.0, 60, 3.5, 20],    # City2: [temp, humidity, wind_speed, cloudiness]
        [275.0, 90, 5.0, 100],   # City3: [temp, humidity, wind_speed, cloudiness]
        [280.0, 70, 3.2, 60],    # City4: [temp, humidity, wind_speed, cloudiness]
        [290.5, 55, 2.8, 50],    # City5: [temp, humidity, wind_speed, cloudiness]
        [276.5, 80, 6.0, 80],    # City6: [temp, humidity, wind_speed, cloudiness]
        [284.0, 65, 3.0, 40],    # City7: [temp, humidity, wind_speed, cloudiness]
        [275.8, 85, 4.1, 90],    # City8: [temp, humidity, wind_speed, cloudiness]
        [288.9, 72, 3.6, 30],    # City9: [temp, humidity, wind_speed, cloudiness]
        [282.3, 78, 2.5, 60],    # City10: [temp, humidity, wind_speed, cloudiness]
        [277.4, 69, 5.3, 50],    # City11: [temp, humidity, wind_speed, cloudiness]
        [289.7, 62, 3.8, 20],    # City12: [temp, humidity, wind_speed, cloudiness]
        [285.2, 75, 4.5, 95],    # City13: [temp, humidity, wind_speed, cloudiness]
        [283.1, 68, 3.4, 85],    # City14: [temp, humidity, wind_speed, cloudiness]
        [278.8, 79, 2.9, 70],    # City15: [temp, humidity, wind_speed, cloudiness]
        [277.0, 65, 3.1, 55],    # City16: [temp, humidity, wind_speed, cloudiness]
        [286.3, 60, 4.2, 40],    # City17: [temp, humidity, wind_speed, cloudiness]
        [290.2, 67, 3.9, 30],    # City18: [temp, humidity, wind_speed, cloudiness]
        [282.1, 74, 2.3, 50],    # City19: [temp, humidity, wind_speed, cloudiness]
        [279.3, 71, 5.1, 95],    # City20: [temp, humidity, wind_speed, cloudiness]
        [285.6, 77, 4.0, 60],    # City21: [temp, humidity, wind_speed, cloudiness]
        [276.1, 68, 3.7, 65],    # City22: [temp, humidity, wind_speed, cloudiness]
        [288.1, 65, 3.3, 80],    # City23: [temp, humidity, wind_speed, cloudiness]
        [270.5, 85, 6.3, 70],    # City24: [temp, humidity, wind_speed, cloudiness]
        [290.7, 58, 2.2, 20],    # City25: [temp, humidity, wind_speed, cloudiness]
        [281.0, 64, 3.0, 45],    # City26: [temp, humidity, wind_speed, cloudiness]
        [278.2, 72, 4.4, 65],    # City27: [temp, humidity, wind_speed, cloudiness]
        [287.9, 62, 3.1, 35],    # City28: [temp, humidity, wind_speed, cloudiness]
        [284.3, 71, 2.7, 55],    # City29: [temp, humidity, wind_speed, cloudiness]
        [290.9, 60, 3.5, 40],    # City30: [temp, humidity, wind_speed, cloudiness]
        [276.6, 79, 5.2, 90],    # City31: [temp, humidity, wind_speed, cloudiness]
        [279.8, 69, 4.1, 80],    # City32: [temp, humidity, wind_speed, cloudiness]
        [282.7, 75, 3.2, 50],    # City33: [temp, humidity, wind_speed, cloudiness]
        [283.8, 73, 4.0, 60],    # City34: [temp, humidity, wind_speed, cloudiness]
        [285.5, 65, 3.8, 30],    # City35: [temp, humidity, wind_speed, cloudiness]
        [277.9, 82, 5.3, 75],    # City36: [temp, humidity, wind_speed, cloudiness]
        [278.5, 65, 2.9, 40],    # City37: [temp, humidity, wind_speed, cloudiness]
        [289.1, 58, 3.0, 25],    # City38: [temp, humidity, wind_speed, cloudiness]
        [284.2, 74, 4.4, 85],    # City39: [temp, humidity, wind_speed, cloudiness]
        [280.5, 63, 3.1, 50],    # City40: [temp, humidity, wind_speed, cloudiness]
        [275.3, 76, 2.8, 60],    # City41: [temp, humidity, wind_speed, cloudiness]
        [277.6, 80, 5.2, 90],    # City42: [temp, humidity, wind_speed, cloudiness]
        [282.8, 70, 3.6, 65],    # City43: [temp, humidity, wind_speed, cloudiness]
        [276.2, 66, 2.7, 55],    # City44: [temp, humidity, wind_speed, cloudiness]
        [285.3, 78, 3.9, 80],    # City45: [temp, humidity, wind_speed, cloudiness]
        [278.1, 73, 4.5, 75],    # City46: [temp, humidity, wind_speed, cloudiness]
        [290.1, 63, 3.3, 30],    # City47: [temp, humidity, wind_speed, cloudiness]
        [279.0, 70, 3.7, 50],    # City48: [temp, humidity, wind_speed, cloudiness]
        [282.5, 68, 2.9, 65],    # City49: [temp, humidity, wind_speed, cloudiness]
        [286.0, 77, 4.2, 45],    # City50: [temp, humidity, wind_speed, cloudiness]
    ])

    Y = np.array([0, 2, 1, 2, 1, 1, 0, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 0, 1, 2, 
                0, 1, 1, 2, 1, 0, 2, 1, 2, 0, 2, 1, 1, 0, 2, 1, 0, 1, 2, 
                0, 2, 1, 1, 0, 2, 0, 1, 2, 1, 2])  

    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).long()

    test_data = torch.tensor([[data['main']['temp'] , data["main"]["humidity"] , 
                                data["wind"]["speed"] , data["clouds"]["all"]  ]])

    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4,3)
            self.relu = nn.ReLU(True)

        def forward(self,x):
            x = self.layer1(x)
            x = self.relu(x)
            return x
        
    model = NeuralNetwork()
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    criteration = nn.CrossEntropyLoss()
    num_of_epochs = 100

    for epoch in range(num_of_epochs):
        model.train()
        out = model(X)
        loss = criteration(out,Y)
        optim.zero_grad()
        loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        out = model(test_data)
        max_pred = torch.argmax(out, dim=1)
        weather_labels = {0: 'Rain', 1: 'Sunny', 2: 'Cloudy'}
        print(f"The future predicted weather for {cityname} is : {weather_labels[max_pred.item()]}")
