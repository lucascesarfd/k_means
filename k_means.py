import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def centroid_calc(dataset, target_list, centroids_number):
    '''
    This routine uses the dataset information together with targets infortion to calculate the new centroids position.
    The number of centroids is given by "centroids_number".

    Example:
        dataset = pd.read_csv('datset.csv')
        target_list = [0,0,1,1]
        centroids_number = 2
        new_centtroids = centroid_calc(dataset, target_list, centroids_number)
    '''
    new_centroids = np.empty((centroids_number,len(dataset.columns)))
    for cent in range(centroids_number):
        group = separate_group(dataset, target_list, cent)
        if group.size == 0:
            new_centroids[cent] = dataset.loc[[cent]]
        else:
            new_centroids[cent] = np.nanmean(group, axis=1)
    return new_centroids

def calculate_distances(centroids, dataset):
    '''
    This routine calculates the distances between the dataset values and the centroids. It returns a new array containing all the distances.

    Example:
        dataset = pd.read_csv('datset.csv')
        centroids = [[0,1,2],[2,3,4]]
        distances = calculate_distances(centroids, dataset)
    '''
    distances = np.zeros((len(dataset),len(centroids)))

    for cent in range(len(centroids)):
        for point in range(len(dataset)):
            distances[point][cent] = euclidean_dst(dataset.values.tolist()[point], centroids[cent])

    return distances

def group_targets(distances):
    '''
    This routine evaluate the minimun distance between a centroid and a value at dataset using the distance array. It returns the target list.

    Example:
        distances = calculate_distances(centroids, dataset)
        target = group_targets(distances)
    '''
    target = np.zeros(len(distances))

    for index, line in zip(range(len(distances)),distances):
        min_value = np.amin(line, axis=0)
        min_index = np.where(line == min_value)
        target[index] = min_index[0][0]
    return target

def separate_group(dataset, target_list, reference):
    '''
    This function is used to separate the "dataset" into smaller arrays according to the target group choosed in "reference".

    Example:
        dataset = pd.read_csv('datset.csv')
        target_list = [0,0,1,1]
        reference = 0
        group = separate_group(dataset, target_list, reference)
    '''
    num_of_entry = len(dataset.columns)
    data_group = np.empty((num_of_entry, 0))
    new_data = np.empty(0)
    for element in range(len(target_list)):
        new_data = np.empty(0)
        if target_list[element] == reference:
            for entry in range(num_of_entry):
                new_data = np.append(new_data, dataset.at[element, entry])
            data_group = np.append(
                data_group, np.atleast_2d(new_data).T, axis=1)
    return data_group

def euclidean_dst(p1,p2):
    '''
    This routine use euclidean distance algorithmn to evaluate the distance between two given points.

    Example:
        p1 = [1,2,3]
        p2 = [4,5,6]
        distance = euclidean_dst(p1,p2)
    '''
    size1 = len(p1)
    size2 = len(p2)

    if size1 != size2:
        return 0
    
    dist_vector = np.zeros(size1)

    for x in range(size1):
        dist_vector[x] = p2[x] - p1[x]

    dist_vector = np.power(dist_vector, 2)
    dist = np.sum(dist_vector)
    dist = np.power(dist, 0.5)
    return dist

def find_target_group(point, centroids):
    '''
    This routine evaluates the group that a given point is belonging.

    Example:
        point = [1,2,3]
        centroids = [[0,1,2],[2,3,4]]
        group = find_target_group(point, centroids)
    '''
    distances = np.zeros(len(centroids))

    for cent in range(len(centroids)):
        distances[cent] = euclidean_dst(point,centroids.values.tolist()[cent])

    min_value = np.amin(distances, axis=0)
    min_index = np.where(distances == min_value)
    group = min_index[0][0]
    return group

def main():
    file_name = "example.txt"
    colors = ['red', 'blue', 'green', 'orange', 'violet', "olive", "cyan", "gray", "darkviolet"]

    print("Using file: " + file_name)
    num_of_clusters_input = input("Enter the number of clusters: ")
    print()
    try:
        num_of_clusters = int(num_of_clusters_input)
    except ValueError:
        print("Error! Input is not a number!")
        print("Using default value as 2.")
        num_of_clusters = 2

    dataframe = pd.read_csv(file_name, sep=";", header=None)
    normalized_dataframe = (dataframe-dataframe.min()) / (dataframe.max()-dataframe.min())
    max_values = np.amax(normalized_dataframe, axis=0)
    min_values = np.amin(normalized_dataframe, axis=0)
    num_of_columns = normalized_dataframe.shape[1]

    if normalized_dataframe.shape[0] < num_of_clusters:
        num_of_clusters = normalized_dataframe.shape[0]
        print("Number of clusters is bigger than dataset points. Adopting the maximum value of: " + str(num_of_clusters))

    centroids = np.zeros((num_of_clusters,num_of_columns))

    for ax in range(num_of_columns):
        centroid_step = (max_values[ax]-min_values[ax]) / (num_of_clusters + 1)
        for i in range(num_of_clusters):
            centroids[i][ax] = min_values[ax] + ((i+1)*centroid_step)

    print()
    print("Initial Centroids: ")
    print((pd.DataFrame(centroids) * (dataframe.max()-dataframe.min())) + dataframe.min())
 
    centroids_are_equal = False
    iteraction = 0
    while not centroids_are_equal:
        iteraction += 1
        print("iteraction " + str(iteraction))
        distances = calculate_distances(centroids, normalized_dataframe)
        target = group_targets(distances)
        new_centroids = centroid_calc(normalized_dataframe, target, num_of_clusters)

        comparison = new_centroids == centroids
        centroids_are_equal = comparison.all()
        centroids = np.copy(new_centroids)

    centroids = (pd.DataFrame(centroids) * (dataframe.max()-dataframe.min())) + dataframe.min()
    print()
    print("Final Centroids: ")
    print(centroids)
    print()
    #print("Normalized Distances: ")
    #print(distances)
    #print()
    #print("Targets: ")
    #print(target)

    again = input("Do you wanna try a point? (y/n) ")
    while again == "y":
        point_str = input("Enter a point. Ex. 2;3;8  \nATTENTION: WITHOUT PARENTHEIS!!!\n")
        point_list = point_str.split(";")
        point_int = [int(i) for i in point_list] 
        target_point = find_target_group(point_int, centroids)
        print("Target of the inserted point is: " + str(target_point))
        if num_of_columns == 2:
            plt.scatter(point_int[0], point_int[1], c=colors[target_point], marker="+")
        again = input("Do you wanna try another point? (y/n) ")

    if num_of_columns == 2:
        # Separate the groups/classes
        for group_num in range(num_of_clusters):
            group = separate_group(dataframe, target, group_num)

            # Add group to final plot
            plt.scatter(group[0], group[1], c=colors[group_num])

        plt.scatter(centroids[0], centroids[1], c='black', marker="x")

        plt.show()

    return

if __name__ == "__main__":
    main()