
plt.figure(figsize=(15, 6))

# Subplot 1: Variance par composante
plt.subplot(1, 2, 1)
n_components = len(pca_full.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), pca_full.explained_variance_ratio_, 'bo-', markersize=4)
plt.xlabel('Composante')
plt.ylabel('Variance expliquée')
plt.title(f'Variance par composante (Total: {n_components} features)')
plt.grid(True, alpha=0.3)

# Adapter l'axe x pour avoir des ticks réguliers
if n_components > 20:
    # Pour beaucoup de features, afficher moins de ticks
    step = max(1, n_components // 10)
    plt.xticks(range(1, n_components + 1, step))
else:
    # Pour peu de features, afficher tous les ticks
    plt.xticks(range(1, n_components + 1))

# Subplot 2: Variance cumulée
plt.subplot(1, 2, 2)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, n_components + 1), cumvar, 'ro-', markersize=4)
plt.xlabel('Composante')
plt.ylabel('Variance cumulée')
plt.title('Variance cumulée')
plt.axhline(y=0.95, color='k', linestyle='--', label='95%')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90%')
plt.axhline(y=0.85, color='green', linestyle='--', label='85%')

# Adapter l'axe x de la même manière
if n_components > 20:
    step = max(1, n_components // 10)
    plt.xticks(range(1, n_components + 1, step))
else:
    plt.xticks(range(1, n_components + 1))

plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.05)

plt.tight_layout()
plt.show()

# Informations supplémentaires
print(f"Nombre total de features: {n_components}")
print(f"Nombre de composantes pour 85% de variance: {np.argmax(cumvar >= 0.85) + 1}")
print(f"Nombre de composantes pour 90% de variance: {np.argmax(cumvar >= 0.90) + 1}")
print(f"Nombre de composantes pour 95% de variance: {np.argmax(cumvar >= 0.95) + 1}")


# Top 10 des composantes qui expliquent le plus de variance
print(f"\nTop 10 des composantes avec le plus de variance:")
top_components = np.argsort(pca_full.explained_variance_ratio_)[::-1][:10]
for i, comp_idx in enumerate(top_components):
    print(f"  Composante {comp_idx + 1}: {pca_full.explained_variance_ratio_[comp_idx]:.3%}")

# Graphique supplémentaire: Focus sur les premières composantes importantes
plt.figure(figsize=(12, 5))

# Subplot 1: Premières 20 composantes
plt.subplot(1, 2, 1)
n_show = min(20, n_components)
plt.bar(range(1, n_show + 1), pca_full.explained_variance_ratio_[:n_show])
plt.xlabel('Composante')
plt.ylabel('Variance expliquée')
plt.title(f'Variance des {n_show} premières composantes')
plt.xticks(range(1, n_show + 1))
plt.grid(True, alpha=0.3, axis='y')

# Subplot 2: Variance cumulée des premières composantes
plt.subplot(1, 2, 2)
cumvar_first = np.cumsum(pca_full.explained_variance_ratio_[:n_show])
plt.plot(range(1, n_show + 1), cumvar_first, 'go-', markersize=6)
plt.xlabel('Composante')
plt.ylabel('Variance cumulée')
plt.title(f'Variance cumulée - {n_show} premières composantes')
plt.axhline(y=0.95, color='k', linestyle='--', label='95%')
plt.axhline(y=0.90, color='orange', linestyle='--', label='90%')
plt.axhline(y=0.85, color='green', linestyle='--', label='85%')
plt.xticks(range(1, n_show + 1))
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Création d'un DataFrame avec les composantes et leurs importances
components_df = pd.DataFrame(
    pca_full.components_,
    columns=Features,
    index=[f'PC{i+1}' for i in range(len(pca_full.components_))]
)

# Trier les features par importance pour chaque composante principale
for i in range(len(pca_full.components_)):
    pc_importance = pd.DataFrame({
        'Feature': Features,
        'Importance': abs(pca_full.components_[i])
    }).sort_values('Importance', ascending=False)
    
    print(f"\nComposante principale {i+1} ({pca_full.explained_variance_ratio_[i]:.2%} de variance expliquée)")
    print(pc_importance.head(5))  # Affiche les 5 features les plus importantes