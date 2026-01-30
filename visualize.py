import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import imageio

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random

import gymnasium as gym
from gymnasium import spaces

# --- Yollar ---
BASE_DIR = '/Users/almilaltintas/Desktop/claude'
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# --- DQN agent ve ortam (aynı mimari) ---
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
        self.agent_pos = [0,0]; self.trajectory = []

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
        self.epsilon = 0.01
        self.learning_rate = 0.001
        self.batch_size = 64
        self.model = self._build_model()

    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=self.state_size),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state, training=False):
        q = self.model.predict(state.reshape(1,-1), verbose=0)
        return int(np.argmax(q[0]))

# --- Yükleme ---
print("Modeller ve veri yükleniyor...")
rf = joblib.load(os.path.join(MODELS_DIR, 'rf_model.joblib'))
gb = joblib.load(os.path.join(MODELS_DIR, 'gb_model.joblib'))
svm = joblib.load(os.path.join(MODELS_DIR, 'svm_model.joblib'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.joblib'))
dl = keras.models.load_model(os.path.join(MODELS_DIR, 'dl_model'))

npz = np.load(os.path.join(DATA_DIR, 'test_data.npz'))
X_test_scaled = npz['X_test_scaled']
y_test = npz['y_test']

# DQN yükle
state_size = X_test_scaled.shape[1]
action_size = 2
dqn_agent = DQNAgent(state_size, action_size)
dqn_agent.model.load_weights(os.path.join(MODELS_DIR, 'dqn_weights.h5'))

# --- Tahminler ve metrikler ---
print("Tahminler hesaplanıyor...")
preds = {
    'Random Forest': rf.predict(X_test_scaled),
    'Gradient Boosting': gb.predict(X_test_scaled),
    'SVM': svm.predict(X_test_scaled),
    'Deep Learning': (dl.predict(X_test_scaled) > 0.5).astype(int).flatten()
}

dqn_preds = np.array([dqn_agent.act(s, training=False) for s in X_test_scaled])
preds['DQN (RL)'] = dqn_preds

for name, p in preds.items():
    print(f"{name}: Acc={accuracy_score(y_test,p):.4f}, F1={f1_score(y_test,p):.4f}")

# --- Animasyon (DQN ajanı) ---
print("Grid animasyonu oluşturuluyor...")
env_visual = MaintenanceEnvironment(X_test_scaled[:100], y_test[:100], grid_size=6)
state, _ = env_visual.reset()
frames = []
for step in range(100):
    action = dqn_agent.act(state, training=False)
    next_state, reward, terminated, truncated, _ = env_visual.step(action)

    fig, ax = plt.subplots(figsize=(6,6))
    grid = env_visual.render()
    ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    if len(env_visual.trajectory) > 1:
        traj = np.array(env_visual.trajectory)
        ax.plot(traj[:,1], traj[:,0], 'b-', linewidth=2, alpha=0.6)
        ax.plot(traj[-1,1], traj[-1,0], 'ro', markersize=8)
    for i in range(7):
        ax.axhline(i-0.5, color='black', linewidth=0.5); ax.axvline(i-0.5, color='black', linewidth=0.5)
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_title(f'Adım {step+1} - Action: {"Bakım" if action==1 else "Pasif"} - R:{reward:.1f}')
    plt.tight_layout()
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(img)
    plt.close(fig)
    state = next_state
    if (step+1) % 20 == 0:
        print(f"Frame {step+1}/100 oluşturuldu")

out_gif = os.path.join(BASE_DIR, 'dqn_agent_animation.gif')
print("Animasyon kaydediliyor...")
imageio.mimsave(out_gif, frames, fps=5, loop=0)
print("✓ Animasyon kaydedildi:", out_gif)

# --- Karşılaştırma grafikleri ---
print("Grafikler oluşturuluyor...")
model_names = list(preds.keys())
accuracies = [accuracy_score(y_test, preds[m]) for m in model_names]
f1s = [f1_score(y_test, preds[m]) for m in model_names]

fig, axes = plt.subplots(1,2,figsize=(14,6))
colors = ['#3498db','#2ecc71','#e74c3c','#f39c12','#9b59b6']
axes[0].bar(model_names, accuracies, color=colors); axes[0].set_title('Accuracy')
axes[1].bar(model_names, f1s, color=colors); axes[1].set_title('F1-Score')
plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'model_comparison.png'), dpi=300, bbox_inches='tight'); plt.close()
print("✓ model_comparison.png kaydedildi")

# Confusion matrices
fig, axes = plt.subplots(2,3,figsize=(18,12)); axes = axes.ravel()
for idx, name in enumerate(model_names):
    cm = confusion_matrix(y_test, preds[name])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(name)
axes[5].axis('off')
plt.tight_layout(); plt.savefig(os.path.join(BASE_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight'); plt.close()
print("✓ confusion_matrices.png kaydedildi")

# Detaylı metrik heatmap
metrics = []
for name in model_names:
    precision = precision_score(y_test, preds[name])
    recall = recall_score(y_test, preds[name])
    metrics.append([accuracy_score(y_test, preds[name]), f1_score(y_test, preds[name]), precision, recall])

metrics_df = np.array(metrics)
import pandas as pd
metrics_df = pd.DataFrame(metrics_df, columns=['Accuracy','F1','Precision','Recall'], index=model_names)
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax)
plt.tight_layout(); plt.savefig(os.path.join(BASE_DIR, 'detailed_metrics.png'), dpi=300, bbox_inches='tight'); plt.close()
print("✓ detailed_metrics.png kaydedildi")

print("Tüm görselleştirme tamamlandı.")