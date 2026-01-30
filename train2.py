import os
import random
from collections import deque

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import gymnasium as gym
from gymnasium import spaces

import warnings
warnings.filterwarnings('ignore')

# --- Ayarlar / yollar ---
BASE_DIR = '/Users/almilaltintas/Desktop/claude'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Veri yükleme ve hazırlık ---
print("VERİ YÜKLEME VE HAZIRLIK")
df = pd.read_csv(os.path.join(BASE_DIR, 'ai4i2020.csv'))

le = LabelEncoder()
df['Type_encoded'] = le.fit_transform(df['Type'])

feature_columns = ['Air temperature [K]', 'Process temperature [K]',
                   'Rotational speed [rpm]', 'Torque [Nm]',
                   'Tool wear [min]', 'Type_encoded']

X = df[feature_columns].values
y = df['Machine failure'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Eğitim: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")

# --- 1) Makine Öğrenmesi Modelleri ---
print("MAKİNE ÖĞRENMESİ MODELLERİ - Eğitim")

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"RandomForest - Acc: {rf_acc:.4f}, F1: {rf_f1:.4f}")

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
print(f"GradientBoosting - Acc: {gb_acc:.4f}, F1: {gb_f1:.4f}")

svm_model = SVC(kernel='rbf', C=1.0, probability=False, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
print(f"SVM - Acc: {svm_acc:.4f}, F1: {svm_f1:.4f}")

# --- 2) Derin Öğrenme Modeli ---
print("DERİN ÖĞRENME MODELİ - Eğitim")
class_weights = {0: 1.0, 1: len(y_train) / (2 * np.sum(y_train))}

dl_model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

dl_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                 loss='binary_crossentropy',
                 metrics=['accuracy'])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = dl_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

dl_pred_proba = dl_model.predict(X_test_scaled)
dl_pred = (dl_pred_proba > 0.5).astype(int).flatten()
dl_acc = accuracy_score(y_test, dl_pred)
dl_f1 = f1_score(y_test, dl_pred)
print(f"DeepLearning - Acc: {dl_acc:.4f}, F1: {dl_f1:.4f}")

# --- 3) DQN (Eğitim) ---
print("DQN - Eğitim")

class MaintenanceEnvironment(gym.Env):
    def __init__(self, X_data, y_data, grid_size=6):
        super().__init__()
        self.X_data = X_data
        self.y_data = y_data
        self.n_samples = len(X_data)
        self.current_step = 0
        self.grid_size = grid_size
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.agent_pos = [0,0]
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        self.agent_pos = [0,0]; self.trajectory = [self.agent_pos.copy()]
        return self.X_data[self.current_step].astype(np.float32), {}

    def step(self, action):
        true_failure = self.y_data[self.current_step]
        if action == 1 and true_failure == 1:
            reward = 10.0
        elif action == 1 and true_failure == 0:
            reward = -5.0
        elif action == 0 and true_failure == 1:
            reward = -20.0
        else:
            reward = 1.0
        if action == 1:
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        else:
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        self.trajectory.append(self.agent_pos.copy())
        self.current_step = (self.current_step + 1) % self.n_samples
        next_state = self.X_data[self.current_step].astype(np.float32)
        return next_state, reward, False, False, {}

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        return grid

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size; self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 500
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, s,a,r,ns,d):
        self.memory.append((s,a,r,ns,d))

    def act(self, state, training=True):
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q = self.model.predict(state.reshape(1,-1), verbose=0)
        return int(np.argmax(q[0]))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])

        targets = self.model.predict(states, verbose=0)
        next_q = self.target_model.predict(next_states, verbose=0)
        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q[i])
        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = MaintenanceEnvironment(X_train_scaled, y_train)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)

episodes = 200
max_steps = 100
update_target_freq = 10

episode_rewards = []
episode_epsilons = []

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.remember(state, action, reward, next_state, done)
        agent.replay()
        state = next_state
        total_reward += reward
        if done:
            break
    if episode % update_target_freq == 0:
        agent.update_target_model()
    episode_rewards.append(total_reward)
    episode_epsilons.append(agent.epsilon)
    if (episode + 1) % 20 == 0:
        print(f"Episode {episode+1}/{episodes} - Avg Reward (last20): {np.mean(episode_rewards[-20:]):.2f} - Epsilon: {agent.epsilon:.3f}")

# DQN değerlendirme (kısa)
env_test = MaintenanceEnvironment(X_test_scaled, y_test)
dqn_predictions = []
for s in X_test_scaled:
    dqn_predictions.append(agent.act(s, training=False))
dqn_predictions = np.array(dqn_predictions)
dqn_acc = accuracy_score(y_test, dqn_predictions)
dqn_f1 = f1_score(y_test, dqn_predictions)
print(f"DQN - Acc: {dqn_acc:.4f}, F1: {dqn_f1:.4f}")

# --- MODELLERİ VE VERİYİ KAYDET ---
print("Modeller ve veriler kaydediliyor...")
joblib.dump(rf_model, os.path.join(MODELS_DIR, 'rf_model.joblib'))
joblib.dump(gb_model, os.path.join(MODELS_DIR, 'gb_model.joblib'))
joblib.dump(svm_model, os.path.join(MODELS_DIR, 'svm_model.joblib'))
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))
dl_model.save(os.path.join(MODELS_DIR, 'dl_model'))
agent.model.save_weights(os.path.join(MODELS_DIR, 'dqn_weights.h5'))

np.savez_compressed(os.path.join(DATA_DIR, 'test_data.npz'),
                    X_test_scaled=X_test_scaled, y_test=y_test)

np.savez_compressed(os.path.join(MODELS_DIR, 'dqn_meta.npz'),
                    episode_rewards=np.array(episode_rewards),
                    episode_epsilons=np.array(episode_epsilons))

print("Kaydetme tamamlandı. visualize.py ile yükleyip görselleştirebilirsiniz.")