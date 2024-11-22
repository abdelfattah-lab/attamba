import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve, N, lambdify

# Define symbols
F, E, D, L, P = symbols('F E D L P', real=True, positive=True)

# Constants
L_val = 4096  # Transformer Model Sequence Length
D_val = 32    # Attamba Model State Dimension

# IsoFLOP Equation: 3F^2 + 2LF = E^2 + 6ED + 21D + 2LE/P
isoFLOP = Eq(3*F**2 + 2*L*F, E**2 + 6*E*D + 21*D + 2*L*E/P)

# IsoKV Equation: F = E/P + D/L
isoKV = Eq(F, E/P + D/L)
plt.figure(figsize=(10, 6))
# Substitute constants into equations
variables = {L: L_val, D: D_val}
isoFLOP_sub = isoFLOP.subs(variables)
isoKV_sub = isoKV.subs(variables)

# Lambdify equations for numerical evaluation
isoFLOP_func = lambdify((E, P), solve(isoFLOP_sub, F)[0])
isoKV_func = lambdify((E, P), solve(isoKV_sub, F)[0])

# Values of P and E/F ratios for fixed E
P_values = [2, 4, 8, 16, 32, 64, 128]
E_val_fixed = 512  # Fix E to a single dimension
E_F_ratios_isoFLOP_fixed_E = [E_val_fixed / isoFLOP_func(E_val_fixed, P_val) for P_val in P_values]
E_F_ratios_isoKV_fixed_E = [E_val_fixed / isoKV_func(E_val_fixed, P_val) for P_val in P_values]

# Plot E/F vs. P for isoFLOP and isoKV
plt.plot(P_values, E_F_ratios_isoFLOP_fixed_E, label="Iso-FLOP", marker="o", markersize=12)
plt.plot(P_values, E_F_ratios_isoKV_fixed_E, label="Iso-KV-Cache", marker="x", linestyle="--", markersize=12)
# Add IsoParam horizontal line at 1
plt.axhline(y=1, color='r', linestyle='-', label="Iso-Param")
plt.xlabel("P (Chunk Size)", fontsize=24)
plt.ylabel("Model Dimension Ratio", fontsize=24)
plt.title("Model Dimension Ratio vs. Chunk Size (P)", fontsize=24)
# plt.yscale('log')
# plt.xticks(P_values)  # Ensure all P values are shown
plt.legend(fontsize=24)
plt.tight_layout()
plt.grid()

plt.savefig("P_vs_msize.pdf")

# Generate LaTeX table format
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
latex_table += "P (Chunk Size) & E/F (IsoFLOP) & E/F (IsoKV) \\\\\n\\hline\n"

for i, P_val in enumerate(P_values):
    latex_table += f"{P_val} & {N(E_F_ratios_isoFLOP_fixed_E[i]):.4f} & {N(E_F_ratios_isoKV_fixed_E[i]):.4f} \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\caption{E/F Ratios for IsoFLOP and IsoKV at Fixed Model Dimension (E=512)}\n\\end{table}"

# Print LaTeX table for easy copying
print("\nLaTeX Table:")
print(latex_table)

# Generate LaTeX table format
latex_table = "\\begin{table}[h]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
latex_table += "P (Chunk Size) & F (IsoFLOP) & F (IsoKV) \\\\\n\\hline\n"

for i, P_val in enumerate(P_values):
    latex_table += f"{P_val} & {E_val_fixed / N(E_F_ratios_isoFLOP_fixed_E[i]):.4f} & {E_val_fixed / N(E_F_ratios_isoKV_fixed_E[i]):.4f} \\\\\n"

latex_table += "\\hline\n\\end{tabular}\n\\caption{F for IsoFLOP and IsoKV at Fixed Model Dimension (E=512)}\n\\end{table}"

# Print LaTeX table for easy copying
print("\nLaTeX Table:")
print(latex_table)
