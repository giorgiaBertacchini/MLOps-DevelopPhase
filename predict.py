import requests

prediction = requests.post(
   "http://127.0.0.1:3000/predict",
   headers={"content-type": "application/json"},
   data='[{"Date": "2022-01-14", "Activity ID": "1b2f3d34-af23-4956-bd6d-8a08bd56a0d9", "Type": "Walking", "Distance (km)": 4.01, "Duration": "3:34", "Average Speed (km/h)": 10.58, "Calories Burned": 1400, "Climb (m)": 69, "Average Heart rate (tpm)": 118, "Quality": 10}, {"Date": "2022-01-14", "Activity ID": "1b2f3d34-af23-4956-bd6d-8a08bd56a0d9", "Type": "Walking", "Distance (km)": 4.01, "Duration": "3:34", "Average Speed (km/h)": 10.58, "Calories Burned": 1400, "Climb (m)": 69, "Average Heart rate (tpm)": 178, "Quality": 9}]',
).text

print(prediction)