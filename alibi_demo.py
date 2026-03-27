from alibi.datasets import fetch_adult

adult = fetch_adult()
data = adult.data
target = adult.target

print("Data:")
print(data)

print("Target: ")
print(target)

print(adult.data.shape, adult.target.shape)