import numpy as np
import tqdm
from scipy.optimize import minimize


class exapmle_class:
    def __init__(self, vector:np.ndarray)->np.ndarray:
        self.vector = vector
        
#DEWIACJA

    def de_mean(self,data: np.ndarray)->np.ndarray:
        mean = np.mean(data, axis=0)
        return data - mean

#NORMALIZACJA WEKTORA
    
    def direction(self) -> np.ndarray:
        mag = np.linalg.norm(self.vector)  
        return self.vector / mag           


#WARIANCJA KIERUNKOWA-MIARA ROZPROSZENIA DANYCH

    def directional_variance(self,data: np.ndarray)->float:
        w_dir = self.direction()
        projection = np.dot(data, w_dir)
        print(projection)
        return np.var(projection)
    
#GRADIENT WARIANCJI KIERUNKOWEJ

    def directional_variance_gradient(self, data: np.ndarray, w:np.ndarray) -> np.ndarray:
        w_dir = self.direction(w)
        gradient = np.sum(2 * np.dot(data, w_dir)[:, np.newaxis] * data, axis=0)
        return gradient

#PIERWSZY SKLADNIK GLOWNY
       
    def first_principal_component(self, data: np.ndarray, n: int = 100, step_size: float = 0.1) -> np.ndarray:
        """Wyszukuje pierwszy składnik główny przy użyciu gradientu."""
        
        guess = np.ones(data.shape[1])

        for _ in range(n):
            gradient = self.directional_variance_gradient(data, guess)
            guess += step_size * gradient  

        return self.direction(guess)

#TEST

data = np.array([[1,2,3],[4,5,6],[7,8,9]])
v = np.array([1,1,1])

exapmle_class_ = exapmle_class(v)
test = exapmle_class_.directional_variance(data)
print(test)

import numpy as np
from sklearn.decomposition import PCA

# # Przykładowe dane
# data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# w = np.array([1, 1, 1])  # Wektor odniesienia

# # PCA oblicza główne kierunki wariancji
# pca = PCA(n_components=2)
# pca.fit(data)

# # Projekcja wektora na główne komponenty
# w_proj = np.dot(pca.components_, w)

# # Wariancja kierunkowa
# directional_variance = np.sum((w_proj * pca.explained_variance_)**2)
# print("Wariancja kierunkowa:", directional_variance)


def solo(v: np.ndarray) -> np.ndarray:
    w_dir = v/ np.linalg.norm(v)
    return w_dir

def solo2(v:np.ndarray, data: np.ndarray) -> np.ndarray:
    w_dir = solo(v)
    projection = np.dot(data, w_dir)
    var = np.var(projection)
    return var

data1 = np.array([[1,4],[3,4]])
v=np.array([1,1])
print(solo2(v, data1))