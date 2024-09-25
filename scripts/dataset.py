import numpy as np
import matplotlib.pyplot as plt

# Caminho para o arquivo de histórico de perdas
history_path = 'outputs/hist_it040000.npy'  # Altere o caminho conforme necessário

# Carregar o histórico de perdas
loss_history = np.load(history_path)

# Plotar o histórico de perdas
plt.figure(figsize=(10, 6))
plt.plot(loss_history, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.grid(True)
plt.show()
