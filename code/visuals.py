"""Creates plot for """
from matplotlib import pyplot as plt
import json


def get_data():
    data_path = "../data/complete_time_series_points.json"
    with open(data_path, 'r', encoding="ascii") as file:
        data = json.load(file)
        return data
    

def visualize_data(data):
    fig, ax = plt.subplots(20, 5, figsize=(20, 40))
    data_index = 0 # Keys to the data dictionary
    for i in range(20):
        for j in range(5):
            time = data[str(data_index)][0]
            amplitude = data[str(data_index)][1]
            ax[i, j].plot(time, amplitude)
            ax[i, j].set_title(f"Observation {data_index} A(t)")
            ax[i, j].set_xlabel("time (seconds)")
            ax[i, j].set_ylabel("Amplitude")
            data_index += 1
    fig.tight_layout()
    
    # Save Img
    img_save_path = "../imgs/raw signals"
    fig.savefig(img_save_path)



def main():
    data = get_data()
    visualize_data(data)




if __name__ == "__main__":
    main()