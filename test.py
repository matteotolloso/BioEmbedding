import requests
import numpy as np


url = "https://api.esmatlas.com/fetchEmbedding/ESM2/MGYP000677884904.bin"
header = {"Accept": "application/octet-stream"}
response = requests.get(url, headers=header)
array = np.frombuffer(response.content, dtype=np.float16)

print(array)