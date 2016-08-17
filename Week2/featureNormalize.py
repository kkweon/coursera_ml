import numpy as np
def featureNormalize(X):
  """
  Input
  ===========
  X  :  np.array of (N, D)


  Returns
  =========
  X_norm : np.array of (N, D)
  mu  :  np.array of (D, )
  sd  :  np.array of (D, )
  """ 

  X_norm = X

  mu = np.mean(X_norm, axis=0)
  sd = np.std(X_norm, axis=0)

  X_norm = (X_norm - mu) / sd

  return X_norm, mu, sd
