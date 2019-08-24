import gym
from numpy import zeros
from random import randint
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

LIMIT_1 = 0.20943904549550227
LIMIT_2 = 3.1903339148305658
MAX_DATA = 10000  # I removed a zero to prevent overheating.
NUM_CLFS = 4

X = zeros((MAX_DATA,5))
Y = zeros((MAX_DATA,4))

def generate_simualted_data(MAX_DATA):
  env = gym.make("CartPole-v0")
  env.reset()
  for i in range(MAX_DATA):
      input_state = env.state
      action = env.action_space.sample()
      observation, reward, done, info = env.step(action)
      X[i,:4] = input_state
      X[i,4] = action
      Y[i] = observation
      if done:
          env.reset()
  return env, X, Y

def fit_Polynomial(X):
  poly = PolynomialFeatures(degree=5)
  return poly, poly.fit_transform(X)

def fit_clfs(X,Y):
  clfs = [LinearRegression() for _ in range(4)]
  return [clf.fit(X, Y[:,idx]) for idx, clf in enumerate(clfs)]

def move_test(move, clfs, X_init, poly):
  move_count = 0
  X_next = X_init
  while True:
      X_next[4] = move
      X_full_test = poly.fit_transform(X_next)
      X_pred = zeros(5)
      for idx, clf in enumerate(clfs):
        X_pred[idx] = clf.predict(X_full_test.reshape(1, -1))
      if abs(X_pred[2]) > LIMIT_1 or abs(X_pred[3]) > LIMIT_2:
          break
      move_count += 1
      X_next = X_pred
  return move_counts

env, X, Y = generate_simualted_data(MAX_DATA)
poly, X_full = fit_Polynomial(X)
clfs = fit_clfs(X_full, Y)

X_init = zeros(5)
X_init[:4] = env.reset()

for _ in range(1000):
    if move_test(1, clfs, X_init, poly) > move_test(0, clfs, X_init, poly):
        rightchoice = 1
    else:
        rightchoice = 0
    print rightchoice
    X_raw = env.step(rightchoice)
    X_init[:4,] = X_raw[0]
    env.render()
