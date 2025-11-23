import math
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Configuração do modelo
# -----------------------------
FEATURE_SCALES = {
    "idade":        (18, 90, 100),
    "tumor_cm":     (0.1, 10.0, 10),
    "grau":         (1, 3, 3),
    "linfonodos":   (0, 20, 20),
    "ER":           (0, 1, 1),
    "PR":           (0, 1, 1),
    "HER2":         (0, 1, 1),
    "vascular":     (0, 1, 1),
    "invasao":      (0, 1, 1),
    "comorb":       (0, 1, 1),
}

DEFAULT_WEIGHTS = {
    "idade":      0.02,
    "tumor_cm":   0.28,
    "grau":       0.32,
    "linfonodos": 0.45,
    "ER":        -0.30,
    "PR":        -0.18,
    "HER2":       0.22,
    "vascular":   0.15,
    "invasao":    0.27,
    "comorb":     0.10,
}
BIAS = -0.22
THRESHOLDS = {"alto": 0.75, "medio": 0.50}

# -----------------------------
# Funções auxiliares
# -----------------------------
def _clamp(x, lo, hi): return max(lo, min(hi, x))
def _normalize(x, scale): return _clamp(x, scale[0], scale[1]) / scale[2]
def _logistic(z): return 1.0 / (1.0 + math.exp(-z))

def classificar_risco(p):
    if p >= THRESHOLDS["alto"]: return "Alto risco 🔴"
    elif p >= THRESHOLDS["medio"]: return "Risco médio 🟠"
    else: return "Baixo risco 🟢"

def prever_metastase(idade, tumor_cm, grau, linfonodos,
                     ER, PR, HER2, vascular, invasao, comorb):
    feats = {
        "idade":      _normalize(idade, FEATURE_SCALES["idade"]),
        "tumor_cm":   _normalize(tumor_cm, FEATURE_SCALES["tumor_cm"]),
        "grau":       _normalize(grau, FEATURE_SCALES["grau"]),
        "linfonodos": _normalize(linfonodos, FEATURE_SCALES["linfonodos"]),
        "ER": float(ER), "PR": float(PR), "HER2": float(HER2),
        "vascular": float(vascular), "invasao": float(invasao), "comorb": float(comorb),
    }
    z = BIAS + sum(feats[k] * DEFAULT_WEIGHTS[k] for k in feats)
    p = _logistic(z)
    return p, classificar_risco(p), feats

# -----------------------------
# Interface Tkinter com gráficos
# -----------------------------
def run_gui():
    root = tk.Tk()
    root.title("Painel Diagnóstico de Metástase")

    labels = [
        "Idade", "Tamanho do tumor (cm)", "Grau (1-3)", "Linfonodos positivos",
        "ER (0/1)", "PR (0/1)", "HER2 (0/1)", "Vascularização (0/1)",
        "Invasão linfovascular (0/1)", "Comorbidades (0/1)"
    ]
    entries = []
    defaults = [55, 2.0, 2, 0, 1, 1, 0, 0, 0, 0]

    for i, label in enumerate(labels):
        tk.Label(root, text=label).grid(row=i, column=0, padx=5, pady=5, sticky="w")
        entry = tk.Entry(root)
        entry.grid(row=i, column=1, padx=5, pady=5)
        entry.insert(0, str(defaults[i]))
        entries.append(entry)

    # Área dos gráficos
    fig = plt.Figure(figsize=(10,3))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=len(labels)+1, column=0, columnspan=2)

    def calcular():
        try:
            vals = [float(e.get()) for e in entries]
            prob, risco, feats = prever_metastase(*vals)

            # Mostra apenas a categoria de risco (sem porcentagem)
            messagebox.showinfo("Resultado", f"Categoria de risco: {risco}")

            # Atualiza gráficos
            fig.clear()

            # Barra da probabilidade (sem texto de porcentagem)
            ax1 = fig.add_subplot(131)
            ax1.bar(["Risco"], [prob],
                    color="red" if prob>=0.75 else "orange" if prob>=0.5 else "green")
            ax1.set_ylim(0,1)
            ax1.set_title(risco)

            # Perfil clínico
            ax2 = fig.add_subplot(132)
            var_names = ["Idade","Tumor","Grau","Linfonodos"]
            var_values = [feats["idade"], feats["tumor_cm"], feats["grau"], feats["linfonodos"]]
            ax2.bar(var_names, var_values, color="skyblue")
            ax2.set_title("Perfil clínico")
            ax2.set_ylabel("Normalizado")

            # Radar chart biomarcadores
            ax3 = fig.add_subplot(133, polar=True)
            biom_names = ["ER","PR","HER2","Vasc","Inv","Comorb"]
            biom_values = [feats["ER"], feats["PR"], feats["HER2"],
                           feats["vascular"], feats["invasao"], feats["comorb"]]
            angles = np.linspace(0, 2*np.pi, len(biom_names), endpoint=False).tolist()
            biom_values += biom_values[:1]
            angles += angles[:1]
            ax3.plot(angles, biom_values, "o-", linewidth=2)
            ax3.fill(angles, biom_values, alpha=0.25)
            ax3.set_xticks(angles[:-1])
            ax3.set_xticklabels(biom_names)
            ax3.set_title("Biomarcadores")

            canvas.draw()

        except Exception as e:
            messagebox.showerror("Erro", str(e))

    tk.Button(root, text="Calcular", command=calcular).grid(
        row=len(labels), column=0, columnspan=2, pady=10
    )

    root.mainloop()

if __name__ == "__main__":
    run_gui()