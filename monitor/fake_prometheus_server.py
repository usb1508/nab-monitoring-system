from prometheus_client import start_http_server, Gauge
import time
import random

score_gauge = Gauge('nab_anomaly_score', 'Anomaly score per NAB file', ['model', 'dataset', 'file'])

def generate_spikes():
    models = ['model1', 'model2', 'model3', 'model4']
    datasets = ['data1', 'data2', 'data3']
    files = ['file1.csv', 'file2.csv', 'file3.csv']

    while True:
        model = random.choice(models)
        dataset = random.choice(datasets)
        file = random.choice(files)
        value = random.uniform(0.7, 1.2)  # high values
        score_gauge.labels(model=model, dataset=dataset, file=file).set(value)
        time.sleep(1)

if __name__ == "__main__":
    start_http_server(8001)
    generate_spikes()
