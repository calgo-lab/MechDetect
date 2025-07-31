import numpy as np
import os


def main():
    error_rates_order = ["0.1", "0.25", "0.5", "0.75", "0.9"]
    base_dir = "./results/error-mechanisms-testing-experiments"
    out_dir = "./data"
    
    dataset_names = sorted([
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ])  # Obtain list of all dataset ids in the base directory
    
    dataset_cubes = {}
    for dataset in dataset_names:
        dataset_path = os.path.join(base_dir, dataset)
        error_rate_arrays = []
        
        for err in error_rates_order:
            error_rate_path = os.path.join(dataset_path, f"{err}")
            if not os.path.isdir(error_rate_path):
                raise ValueError(f"Error rate directory {error_rate_path} does not exist.")
            
            file_path = os.path.join(error_rate_path, "results.npy")
            if not os.path.isfile(file_path):
                raise ValueError(f"File {file_path} does not exist.")
            error_rate_arrays.append(np.load(file_path))
        dataset_array = np.stack(error_rate_arrays, axis=0)
        dataset_cubes[dataset] = dataset_array  # Dimensions for each cube: (error_rate, error_mechanism, column, clean/dirty, scenario, metric)
    
    # Save the data cubes
    print("Dimensions for each cube: (error_rate, error_mechanism, column, clean/dirty, scenario, metric)")
    output_path = os.path.join(out_dir, "data_cubes_dict.npz")
    np.savez_compressed(output_path, **dataset_cubes)
    
    print(f"Data cubes for {len(dataset_cubes)} dataset(s) saved successfully at {output_path}.")
    
if __name__ == "__main__":
    main()