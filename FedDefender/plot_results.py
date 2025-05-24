import os
import glob
import json
import matplotlib.pyplot as plt

# Liste de tes datasets
datasets = ['mnist', 'fmnist', 'cifar10']

# Dictionnaire pour stocker les histories
histories = {}

for ds in datasets:
    # On suppose que tu as un fichier `history.json` dans ./results/<dataset>/
    pattern = os.path.join('results', ds, '*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f" Aucun history JSON trouvé pour {ds} avec le pattern {pattern}")
        continue
    
    # Si plusieurs fichiers, tu peux en choisir un, ici on prend le dernier
    path = files[-1]
    with open(path, 'r') as f:
        history = json.load(f)
    
    # On s’attend à ce que history soit un dict avec 
    # history['round'], history['accuracy'], history['loss']
    histories[ds] = history

# --- Plot Accuracy ---
plt.figure(figsize=(8,5))
for ds, h in histories.items():
    plt.plot(h['round'], h['accuracy'], marker='o', label=ds.upper())
plt.title('Accuracy vs. Rounds')
plt.xlabel('Round')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_comparison.png')
print("Enregistré → accuracy_comparison.png")

# --- Plot Loss ---
plt.figure(figsize=(8,5))
for ds, h in histories.items():
    plt.plot(h['round'], h['loss'], marker='o', label=ds.upper())
plt.title('Loss vs. Rounds')
plt.xlabel('Round')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_comparison.png')
print(" Enregistré → loss_comparison.png")

plt.show()
