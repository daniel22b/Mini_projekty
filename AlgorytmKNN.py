import numpy as np 
from collections import Counter
import pandas as pd 
from sklearn.model_selection import train_test_split

# Wczytanie danych
df = pd.read_csv("comma_delimited_stock_prices.csv", sep="/")
df.columns = ['Symbol', 'Date', 'Price']

# Usunięcie błędnych danych w 'Price' (np. NaN lub błędy w liczbach)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df = df.dropna(subset=['Price'])

# Dodanie kolumny 'Category' (0 - cena < 1000, 1 - cena >= 1000)
df['Category'] = df['Price'].apply(lambda x: 0 if x < 1000 else 1)

# Dane X (cechy) i y (etykiety)
X_data = df['Price'].values.reshape(-1, 1)  # Przekształcamy na tablicę 2D
y_label = df['Category']

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_data, y_label, test_size=0.25, random_state=42)

# Konwersja do numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Funkcja obliczająca odległość euklidesową
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        # Oblicz odległości
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Znajdź indeksy k najbliższych sąsiadów
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        
        # Znajdź najczęściej występującą etykietę
        most_common = Counter(k_nearest_labels).most_common(1)[0][0]
        return most_common

# Przykład użycia
if __name__ == "__main__":
    clf = KNN(k=3)
    clf.fit(X_train, y_train)

    print("X_test:", X_test)
    print("y_test:", y_test)

    predictions = clf.predict(X_test)
    print("Predictions:", predictions)






example_data = np.array([[1,2],[3,4],[5,6],[2,3],[1,1],[4,5]])
y_data = [1,0,1,0,1,0]

example_data2 = np.array([1,1])


def eukalides(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


distance = [eukalides(example_data2,x2)for x2 in example_data]


sorted_distance = np.argsort(distance)[:3]
choes_neighbour = [y_data[i] for i in sorted_distance]

most_common = Counter(choes_neighbour).most_common(1)[0][0]
print(most_common)