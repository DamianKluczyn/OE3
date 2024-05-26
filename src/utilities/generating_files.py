import matplotlib.pyplot as plt
import os
class DataSaver:
    def __init__(self):
        pass

    def plot_and_save(self, data, filename):
        epochs, fitness_values = zip(*data)
        plt.plot(epochs, fitness_values, marker='o', linestyle='-', color='blue')
        plt.title('Wykres wartości funkcji celu w kolejnych epokach')
        plt.xlabel('Epoka')
        plt.ylabel('Wartość funkcji celu')
        parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        target_directory = os.path.join(parent_directory, 'data', 'output')
        file_path = os.path.join(target_directory, filename)
        plt.savefig(file_path)
        plt.grid(True)
        plt.show()

    def save_to_file(self, data, filename):
        parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
        target_directory = os.path.join(parent_directory, 'data', 'output')
        file_path = os.path.join(target_directory, filename)
        with open(file_path, 'w') as file:
            for i, item in enumerate(data, 1):
                file.write(f"{i}: {item}\n")

