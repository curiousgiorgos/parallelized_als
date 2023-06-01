import numpy as np
import pandas as pd 
import random 


data = []


with open('movielens_1M/ratings.dat', 'r') as file:
    for line in file:
        user, movie, rating = line.split("::")[0:3]
        data.append((int(user),int(movie), rating))
        # process the line here

data = sorted(data, key=lambda x: x[0])

random_users = set(random.sample(list(range(1,6041)), 3000))
random_movies = set(random.sample(list(range(1,3953)), 2000))

movie_id_map = {} 
user_id_map = {}

curr_movie = 0 
curr_id = 0 
subset = []

for user, movie, rating in data: 
    if user not in random_users or movie not in random_movies:
        continue 

    if user not in user_id_map: 
        user_id_map[user] = str(curr_id+1) 
        curr_id += 1 

    if movie not in movie_id_map: 
        movie_id_map[movie] = str(curr_movie+1)
        curr_movie += 1

    subset.append(user_id_map[user] + "::" + movie_id_map[movie] + "::" + rating)

avg_data_rating = np.mean([int(i[2]) for i in data]) 
avg_data_subset = np.mean([int(i.split("::")[-1]) for i in subset]) 

std_data_rating = np.std([int(i[2]) for i in data]) 
std_data_subset = np.std([int(i.split("::")[-1]) for i in subset]) 

print("STD data: ", round(std_data_rating,2))
print("STD subset: ", round(std_data_subset,2))

print("Mean data: ", round(avg_data_rating,2))
print("Mean subset: ", round(avg_data_subset,2))

print("Observed ratio data", round(len(data)/ (3952 * 6040),3))
print("Observed ratio subest", round(len(subset) / (1250 * 800),3))

print("Users in dataset " , curr_id)
print("Movies in dataset " , curr_movie)

with open("toydata.dat", "w") as f:
    for item in subset:
        f.write(str(item) + "\n")