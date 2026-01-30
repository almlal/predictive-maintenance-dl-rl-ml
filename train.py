"""
Kestirimci Bakım Projesi
Derin Pekiştirmeli Öğrenme, Derin Öğrenme ve Makine Öğrenmesi Karşılaştırması
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Gym için gerekli kütüphaneler
import gymnasium as gym
from gymnasium import spaces
import imageio
from collections import deque
import random

# Veriyi yükle ve hazırla
print("=" * 60)
print("VERİ YÜKLEME VE HAZIRLIK")
print("=" * 60)

df = pd.read_csv('/Users/almilaltintas/Desktop/claude/ai4i2020.csv')
print(f"\nVeri boyutu: {df.shape}")
print(f"\nİlk 5 satır:\n{df.head()}")
print(f"\nHata dağılımı:\n{df['Machine failure'].value_counts()}")
print(f"\nHata oranı: {df['Machine failure'].mean():.4f}")

# Özellik mühendisliği
df_features = df.copy()

# Type sütununu encode et
le = LabelEncoder()
df_features['Type_encoded'] = le.fit_transform(df_features['Type'])

# Kullanılacak özellikler
feature_columns = ['Air temperature [K]', 'Process temperature [K]', 
                   'Rotational speed [rpm]', 'Torque [Nm]', 
                   'Tool wear [min]', 'Type_encoded']

X = df_features[feature_columns].values
y = df_features['Machine failure'].values

# Veriyi böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizasyon
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nEğitim seti boyutu: {X_train_scaled.shape}")
print(f"Test seti boyutu: {X_test_scaled.shape}")

# ============================================================================
# 1. MAKİNE ÖĞRENMESİ MODELLERİ
# ============================================================================
print("\n" + "=" * 60)
print("MAKİNE ÖĞRENMESİ MODELLERİ")
print("=" * 60)

ml_results = {}

# Random Forest
print("\n[1/3] Random Forest eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
print(f"Random Forest - Accuracy: {rf_accuracy:.4f}, F1-Score: {rf_f1:.4f}")
ml_results['Random Forest'] = {'accuracy': rf_accuracy, 'f1': rf_f1, 'predictions': rf_pred}

# Gradient Boosting
print("\n[2/3] Gradient Boosting eğitiliyor...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train_scaled, y_train)
gb_pred = gb_model.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)
print(f"Gradient Boosting - Accuracy: {gb_accuracy:.4f}, F1-Score: {gb_f1:.4f}")
ml_results['Gradient Boosting'] = {'accuracy': gb_accuracy, 'f1': gb_f1, 'predictions': gb_pred}

# SVM
print("\n[3/3] SVM eğitiliyor...")
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)
svm_pred = svm_model.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_f1 = f1_score(y_test, svm_pred)
print(f"SVM - Accuracy: {svm_accuracy:.4f}, F1-Score: {svm_f1:.4f}")
ml_results['SVM'] = {'accuracy': svm_accuracy, 'f1': svm_f1, 'predictions': svm_pred}

# ============================================================================
# 2. DERİN ÖĞRENME MODELİ
# ============================================================================
print("\n" + "=" * 60)
print("DERİN ÖĞRENME MODELİ")
print("=" * 60)

# Sınıf ağırlıkları (dengesiz veri için)
class_weights = {0: 1.0, 1: len(y_train) / (2 * np.sum(y_train))}
print(f"\nSınıf ağırlıkları: {class_weights}")

# DL model mimarisi
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

dl_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nModel Mimarisi:")
dl_model.summary()

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model eğitimi
print("\nDerin öğrenme modeli eğitiliyor...")
history = dl_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=64,
    class_weight=class_weights,
    callbacks=[early_stopping],
    verbose=1
)

# DL değerlendirme
dl_pred_proba = dl_model.predict(X_test_scaled)
dl_pred = (dl_pred_proba > 0.5).astype(int).flatten()
dl_accuracy = accuracy_score(y_test, dl_pred)
dl_f1 = f1_score(y_test, dl_pred)
print(f"\nDerin Öğrenme - Accuracy: {dl_accuracy:.4f}, F1-Score: {dl_f1:.4f}")

# ============================================================================
# 3. DERİN PEKİŞTİRMELİ ÖĞRENME (DQN)
# ============================================================================
print("\n" + "=" * 60)
print("DERİN PEKİŞTİRMELİ ÖĞRENME - DQN")
print("=" * 60)

class MaintenanceEnvironment(gym.Env):
    """
    Kestirimci Bakım için Özel Ortam
    
    State Space: [Air temp, Process temp, Rotation, Torque, Tool wear, Type]
    Action Space: 0 = Bakım Yapma, 1 = Bakım Yap
    
    Reward:
    - Doğru bakım kararı (gerçekten arıza varsa bakım yap): +10
    - Yanlış alarm (arıza yokken bakım yap): -5
    - Kaçırılan arıza (arıza varken bakım yapma): -20
    - Doğru karar (arıza yokken bakım yapma): +1
    """
    
    def __init__(self, X_data, y_data, grid_size=6):
        super(MaintenanceEnvironment, self).__init__()
        
        self.X_data = X_data
        self.y_data = y_data
        self.n_samples = len(X_data)
        self.current_step = 0
        self.grid_size = grid_size
        
        # Action space: 0 = Bakım Yapma, 1 = Bakım Yap
        self.action_space = spaces.Discrete(2)
        
        # Observation space: 6 özellik
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(6,), dtype=np.float32
        )
        
        # Grid pozisyonu (görselleştirme için)
        self.agent_pos = [0, 0]
        self.trajectory = []
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = np.random.randint(0, self.n_samples)
        self.agent_pos = [0, 0]
        self.trajectory = [self.agent_pos.copy()]
        return self.X_data[self.current_step].astype(np.float32), {}
    
    def step(self, action):
        # Gerçek hata durumu
        true_failure = self.y_data[self.current_step]
        
        # Reward hesaplama
        if action == 1 and true_failure == 1:  # Doğru bakım
            reward = 10.0
        elif action == 1 and true_failure == 0:  # Yanlış alarm
            reward = -5.0
        elif action == 0 and true_failure == 1:  # Kaçırılan arıza
            reward = -20.0
        else:  # Doğru karar (bakım gereksiz)
            reward = 1.0
        
        # Grid pozisyonunu güncelle (görselleştirme için)
        if action == 1:  # Bakım yapıldı
            self.agent_pos[0] = min(self.agent_pos[0] + 1, self.grid_size - 1)
        else:  # Bakım yapılmadı
            self.agent_pos[1] = min(self.agent_pos[1] + 1, self.grid_size - 1)
        
        self.trajectory.append(self.agent_pos.copy())
        
        # Sonraki state
        self.current_step = (self.current_step + 1) % self.n_samples
        next_state = self.X_data[self.current_step].astype(np.float32)
        
        # Episode bitmedi (sürekli öğrenme)
        terminated = False
        truncated = False
        
        return next_state, reward, terminated, truncated, {}
    
    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.agent_pos[0], self.agent_pos[1]] = 1
        return grid

class DQNAgent:
    """Deep Q-Network Agent"""
    
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.train_start = 500
        
        # Ana model ve hedef model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """Q-Network mimarisi"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_dim=self.state_size),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(
            loss='mse',
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
        )
        return model
    
    def update_target_model(self):
        """Hedef modeli güncelle"""
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """Experience'i hafızaya ekle"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """Epsilon-greedy action selection"""
        if training and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Experience replay ile öğrenme"""
        if len(self.memory) < self.train_start:
            return
        
        # Mini-batch sampling
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        
        states = np.array([t[0] for t in minibatch])
        actions = np.array([t[1] for t in minibatch])
        rewards = np.array([t[2] for t in minibatch])
        next_states = np.array([t[3] for t in minibatch])
        dones = np.array([t[4] for t in minibatch])
        
        # Q-değerlerini hesapla
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.target_model.predict(next_states, verbose=0)
        
        for i in range(len(minibatch)):
            if dones[i]:
                targets[i][actions[i]] = rewards[i]
            else:
                targets[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Model eğitimi
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# DQN ortamını oluştur
env = MaintenanceEnvironment(X_train_scaled, y_train)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(f"\nOrtam bilgileri:")
print(f"State boyutu: {state_size}")
print(f"Action sayısı: {action_size}")

# DQN agent oluştur
agent = DQNAgent(state_size, action_size)

# DQN Eğitimi
episodes = 200
max_steps = 100
update_target_freq = 10

episode_rewards = []
episode_epsilons = []

print(f"\nDQN eğitimi başlıyor ({episodes} episode, her biri {max_steps} adım)...")

for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # Action seç
        action = agent.act(state)
        
        # Adım at
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Hafızaya ekle
        agent.remember(state, action, reward, next_state, done)
        
        # Öğren
        agent.replay()
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Hedef modeli güncelle
    if episode % update_target_freq == 0:
        agent.update_target_model()
    
    episode_rewards.append(total_reward)
    episode_epsilons.append(agent.epsilon)
    
    if (episode + 1) % 20 == 0:
        avg_reward = np.mean(episode_rewards[-20:])
        print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

print("\nDQN eğitimi tamamlandı!")

# DQN Değerlendirme
print("\nDQN modeli test ediliyor...")
env_test = MaintenanceEnvironment(X_test_scaled, y_test)
dqn_predictions = []
test_states = X_test_scaled

for state in test_states:
    action = agent.act(state, training=False)
    dqn_predictions.append(action)

dqn_predictions = np.array(dqn_predictions)
dqn_accuracy = accuracy_score(y_test, dqn_predictions)
dqn_f1 = f1_score(y_test, dqn_predictions)
print(f"DQN - Accuracy: {dqn_accuracy:.4f}, F1-Score: {dqn_f1:.4f}")

# ============================================================================
# 4. GÖRSELLEŞTİRME ve ANİMASYON
# ============================================================================
print("\n" + "=" * 60)
print("GÖRSELLEŞTİRME VE ANİMASYON")
print("=" * 60)

# 6x6 Grid üzerinde ajanın hareketlerini görselleştir
print("\nGrid animasyonu oluşturuluyor...")

# Test ortamında bir episode çalıştır
env_visual = MaintenanceEnvironment(X_test_scaled[:100], y_test[:100], grid_size=6)
state, _ = env_visual.reset()

frames = []
for step in range(100):
    action = agent.act(state, training=False)
    next_state, reward, terminated, truncated, _ = env_visual.step(action)
    
    # Frame oluştur
    fig, ax = plt.subplots(figsize=(8, 8))
    grid = env_visual.render()
    
    # Grid'i göster
    im = ax.imshow(grid, cmap='RdYlGn', vmin=0, vmax=1, interpolation='nearest')
    
    # Trajectory çiz
    if len(env_visual.trajectory) > 1:
        traj = np.array(env_visual.trajectory)
        ax.plot(traj[:, 1], traj[:, 0], 'b-', linewidth=2, alpha=0.5, label='Yörünge')
        ax.plot(traj[-1, 1], traj[-1, 0], 'ro', markersize=15, label='Ajan')
    
    # Grid çizgileri
    for i in range(7):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
        ax.axvline(i - 0.5, color='black', linewidth=0.5)
    
    # Eksen etiketleri
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(['Bölge ' + str(i) for i in range(6)], fontsize=10)
    ax.set_yticklabels(['Bölge ' + str(i) for i in range(6)], fontsize=10)
    
    # Başlık ve bilgi
    action_text = "Bakım Yap" if action == 1 else "Bakım Yapma"
    reward_color = 'green' if reward > 0 else 'red'
    
    ax.set_title(
        f'DQN Ajan Hareketi - Adım: {step+1}\n'
        f'Action: {action_text}, Reward: {reward:.1f}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Legend
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('Bakım Yapma Yönü →', fontsize=12)
    ax.set_ylabel('Bakım Yap Yönü ↑', fontsize=12)
    
    plt.tight_layout()
    
    # Frame'i kaydet
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(image)
    plt.close(fig)
    
    state = next_state
    
    if (step + 1) % 20 == 0:
        print(f"Frame {step + 1}/100 oluşturuldu")

# GIF olarak kaydet
print("\nAnimasyon kaydediliyor...")
imageio.mimsave('/Users/almilaltintas/Desktop/claude/dqn_agent_animation.gif', frames, fps=5, loop=0)
print("✓ Animasyon kaydedildi: dqn_agent_animation.gif")

# ============================================================================
# 5. KARŞILAŞTIRMA GRAFİKLERİ
# ============================================================================
print("\n" + "=" * 60)
print("KARŞILAŞTIRMA GRAFİKLERİ")
print("=" * 60)

# Model sonuçlarını topla
models = {
    'Random Forest': ml_results['Random Forest'],
    'Gradient Boosting': ml_results['Gradient Boosting'],
    'SVM': ml_results['SVM'],
    'Deep Learning': {'accuracy': dl_accuracy, 'f1': dl_f1, 'predictions': dl_pred},
    'DQN (RL)': {'accuracy': dqn_accuracy, 'f1': dqn_f1, 'predictions': dqn_predictions}
}

# Grafik 1: Accuracy ve F1-Score Karşılaştırması
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

model_names = list(models.keys())
accuracies = [models[m]['accuracy'] for m in model_names]
f1_scores = [models[m]['f1'] for m in model_names]

colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

# Accuracy
axes[0].bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black')
axes[0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
axes[0].set_title('Model Accuracy Karşılaştırması', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.9, 1.0])
axes[0].grid(axis='y', alpha=0.3)
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# F1-Score
axes[1].bar(model_names, f1_scores, color=colors, alpha=0.8, edgecolor='black')
axes[1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
axes[1].set_title('Model F1-Score Karşılaştırması', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, max(f1_scores) * 1.2])
axes[1].grid(axis='y', alpha=0.3)
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/almilaltintas/Desktop/claude/model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Karşılaştırma grafiği kaydedildi: model_comparison.png")
plt.close()

# Grafik 2: Confusion Matrix Karşılaştırması
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (model_name, results) in enumerate(models.items()):
    cm = confusion_matrix(y_test, results['predictions'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                cbar_kws={'label': 'Sayı'}, annot_kws={'size': 14, 'weight': 'bold'})
    axes[idx].set_title(f'{model_name}\nAccuracy: {results["accuracy"]:.4f}', 
                        fontsize=12, fontweight='bold')
    axes[idx].set_ylabel('Gerçek', fontsize=11)
    axes[idx].set_xlabel('Tahmin', fontsize=11)
    axes[idx].set_xticklabels(['Normal', 'Arıza'])
    axes[idx].set_yticklabels(['Normal', 'Arıza'])

# Son subplot'u boş bırak
axes[5].axis('off')

plt.suptitle('Confusion Matrix Karşılaştırması - Tüm Modeller', 
             fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('/Users/almilaltintas/Desktop/claude/confusion_matrices.png', dpi=300, bbox_inches='tight')
print("✓ Confusion matrix grafiği kaydedildi: confusion_matrices.png")
plt.close()

# Grafik 3: DQN Eğitim Süreci
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Episode Rewards
axes[0].plot(episode_rewards, color='#3498db', linewidth=2, alpha=0.7, label='Episode Reward')
axes[0].plot(np.convolve(episode_rewards, np.ones(20)/20, mode='valid'), 
             color='#e74c3c', linewidth=2, label='Hareketli Ortalama (20)')
axes[0].set_xlabel('Episode', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Toplam Reward', fontsize=12, fontweight='bold')
axes[0].set_title('DQN Eğitim Süreci - Reward', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# Epsilon Decay
axes[1].plot(episode_epsilons, color='#2ecc71', linewidth=2)
axes[1].set_xlabel('Episode', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Epsilon (Keşif Oranı)', fontsize=12, fontweight='bold')
axes[1].set_title('DQN Epsilon Decay', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/almilaltintas/Desktop/claude/dqn_training_process.png', dpi=300, bbox_inches='tight')
print("✓ DQN eğitim süreci grafiği kaydedildi: dqn_training_process.png")
plt.close()

# Grafik 4: Detaylı Metrik Karşılaştırması
fig, ax = plt.subplots(figsize=(12, 8))

metrics_data = []
for model_name, results in models.items():
    precision = precision_score(y_test, results['predictions'])
    recall = recall_score(y_test, results['predictions'])
    metrics_data.append({
        'Model': model_name,
        'Accuracy': results['accuracy'],
        'F1-Score': results['f1'],
        'Precision': precision,
        'Recall': recall
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df = metrics_df.set_index('Model')

# Heatmap
sns.heatmap(metrics_df, annot=True, fmt='.4f', cmap='YlGnBu', 
            cbar_kws={'label': 'Skor'}, linewidths=0.5, linecolor='gray',
            ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
ax.set_title('Tüm Modeller - Detaylı Metrik Karşılaştırması', 
             fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('/Users/almilaltintas/Desktop/claude/detailed_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Detaylı metrik grafiği kaydedildi: detailed_metrics.png")
plt.close()

# ============================================================================
# 6. ÖZET RAPOR
# ============================================================================
print("\n" + "=" * 60)
print("ÖZET RAPOR")
print("=" * 60)

print("\n📊 MODEL PERFORMANS SIRLAMASI (F1-Score'a göre):")
sorted_models = sorted(models.items(), key=lambda x: x[1]['f1'], reverse=True)
for rank, (model_name, results) in enumerate(sorted_models, 1):
    print(f"{rank}. {model_name:20s} - Accuracy: {results['accuracy']:.4f}, F1-Score: {results['f1']:.4f}")

print("\n📁 OLUŞTURULAN DOSYALAR:")
print("  • dqn_agent_animation.gif - DQN ajanının 6x6 grid üzerindeki hareketi")
print("  • model_comparison.png - Accuracy ve F1-Score karşılaştırması")
print("  • confusion_matrices.png - Tüm modellerin confusion matrix'leri")
print("  • dqn_training_process.png - DQN eğitim süreci ve epsilon decay")
print("  • detailed_metrics.png - Detaylı metrik karşılaştırması")

print("\n" + "=" * 60)
print("PROJE TAMAMLANDI! ✓")
print("=" * 60)