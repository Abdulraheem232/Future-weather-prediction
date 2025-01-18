
# City Future Weather Prediction

This project is a **City Future Weather Prediction** model using **PyTorch** for basic classification based on current weather data. It predicts the future weather (rain, sunny, or cloudy) for a given city based on the following parameters:

- **Temperature**
- **Humidity**
- **Wind Speed**
- **Cloudiness**

## **How it works:**

1. The user enters the name of a city.
2. The program fetches the current weather data of the city using the [OpenWeatherMap API](https://openweathermap.org/).
3. The model is a simple **neural network** trained to predict weather conditions.
4. The program outputs the predicted weather for the next day as one of the following categories:
   - **Rain** (0)
   - **Sunny** (2)
   - **Cloudy** (1)

## **Requirements:**

- Python 3.x
- PyTorch
- Numpy
- Requests

You can install the required dependencies by running:

```bash
pip install torch numpy requests
```

## **How to Run:**

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Abdulraheem232/Future-weather-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd weather-prediction
   ```
3. Run the Python script:
   ```bash
   python weather_prediction.py
   ```

4. Enter the city name when prompted, and the model will predict the future weather.

---

## **Important Note:**
- **Accuracy**: This is a **basic neural network** model and is not 100% accurate. The model is trained on a small dataset with limited features. The predictions may not always be correct, and it should be used for learning purposes rather than as a reliable source of weather forecasting.
- **Future Improvements**: This project can be improved by training on a larger dataset, adding more features, and implementing a more advanced machine learning model.

## **License:**
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

### **Example Output:**
```text
Enter city for future weather prediction: Los Angeles
Current weather of Los Angeles : 23°
The future predicted weather for Los Angeles is : Cloudy
```

---

