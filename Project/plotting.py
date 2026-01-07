#plotting

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
plt.rcParams['font.size'] = 16

def plot_risk_vs_iterations(df, metric='test_risk', title='Risk Convergence', baseline = None, save_path= None):
    """
    Plotta l'andamento del rischio al variare di m per diverse strategie.
    
    :param df: Pandas DataFrame contenente le colonne ['m', 'strategy', metric]
    :param metric: Nome della colonna da plottare (es. 'test_risk', 'train_risk', 'time')
    """
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid") # Stile pulito per paper scientifici

    # Seaborn fa la magia:
    # - Raggruppa per 'm' e 'strategy'
    # - Calcola la media (linea centrale)
    # - Calcola l'intervallo di confidenza al 95% (banda colorata trasparente)
    sns.lineplot(
        data=df,
        x='tot_iterations',
        y=metric,
        hue='strategy',    # Colora linee diverse per ogni strategia
        style='strategy',  # Usa tratteggi/marker diversi per accessibilità
        markers=True,
        dashes=False,      # Linee solide per tutti (più leggibile con i marker)
        err_style='band',  # 'band' per l'ombra, 'bars' per le barre di errore classiche
        linewidth=2.5,
        markersize=8
    )

    if baseline is not None:
        plt.axhline(
            y=baseline,       # L'altezza della linea (valore scalare)
            color='black',    # Colore neutro per distinguerla
            linestyle='--',   # Tratteggiata per indicare che è un riferimento
            linewidth=2,
            label='Full Model' # Etichetta per la legenda
        )

    # Scala Logaritmica (quasi obbligatoria per m e per il rischio)
    plt.xscale('log')
    plt.yscale('log')

    # Etichette e Titoli
    plt.xlabel(r'Number of iterations ($T$)', fontsize=14)
    plt.xticks(sorted(df['tot_iterations'].unique()),sorted(df['tot_iterations'].unique()))
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.title(title, fontsize=16)

    
    # Gestione legenda
    plt.legend(title='Strategy', fontsize=12, title_fontsize=12)
    
    # Miglioramenti estetici griglia
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_convergence_vs_iterations.pdf")
    plt.show()

def plot_risk_vs_number_of_nystrom_points(df, metric='test_risk', title='Risk Convergence', baseline = None, save_path= None):
    """
    Plotta l'andamento del rischio al variare di m per diverse strategie.
    
    :param df: Pandas DataFrame contenente le colonne ['m', 'strategy', metric]
    :param metric: Nome della colonna da plottare (es. 'test_risk', 'train_risk', 'time')
    """
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid") # Stile pulito per paper scientifici

    # Seaborn fa la magia:
    # - Raggruppa per 'm' e 'strategy'
    # - Calcola la media (linea centrale)
    # - Calcola l'intervallo di confidenza al 95% (banda colorata trasparente)
    sns.lineplot(
        data=df,
        x='number_of_nystrom_points',
        y=metric,
        hue='strategy',    # Colora linee diverse per ogni strategia
        style='strategy',  # Usa tratteggi/marker diversi per accessibilità
        markers=True,
        dashes=False,      # Linee solide per tutti (più leggibile con i marker)
        err_style='band',  # 'band' per l'ombra, 'bars' per le barre di errore classiche
        linewidth=2.5,
        markersize=8
    )

    if baseline is not None:
        plt.axhline(
            y=baseline,       # L'altezza della linea (valore scalare)
            color='black',    # Colore neutro per distinguerla
            linestyle='--',   # Tratteggiata per indicare che è un riferimento
            linewidth=2,
            label='Full Model' # Etichetta per la legenda
        )

    # Scala Logaritmica (quasi obbligatoria per m e per il rischio)
    plt.xscale('log')
    plt.yscale('log')

    # Etichette e Titoli
    plt.xlabel(r'Number of Nystrom points ($m$)', fontsize=14)
    plt.xticks(sorted(df['number_of_nystrom_points'].unique()),sorted(df['number_of_nystrom_points'].unique()))
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
    plt.title(title, fontsize=16)

    
    # Gestione legenda
    plt.legend(title='Strategy', fontsize=12, title_fontsize=12)
    
    # Miglioramenti estetici griglia
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}_convergence_vs_number_of_nystrom_points.pdf")
    plt.show()


def plot_1D_fullKRR_vs_Nystrom(
    X_train, y_train,           # I dati di training (N,)
    X_nystrom, y_nystrom,       # I punti selezionati (m,)
    X_grid,                     # Griglia per le linee (N_eval,)
    y_true_grid,                # La curva vera (Ground Truth)
    y_full_grid,                # La curva del modello Full
    y_approx_grid,              # La curva del modello Nystrom
    strategy_name,              # Stringa per titoli e legende
    scores=None,                # (Opzionale) Leverage Scores per RLS
    save_path=None              # (Opzionale) Path completo dove salvare
):

    
    # 1. Configurazione Subplots
    # Se abbiamo gli 'scores', facciamo 2 grafici, altrimenti 1
    number_of_axes = 2 if scores is not None else 1
    
    fig, ax = plt.subplots(1, number_of_axes, figsize=(8 * number_of_axes, 6), squeeze=False)
    # Nota: con squeeze=False, ax è sempre una matrice 2D -> ax[0, 0], ax[0, 1]

    # --- PLOT PRINCIPALE (ax[0, 0]) ---
    ax_main = ax[0, 0]
    
    # Dati originali (Arancione)
    ax_main.scatter(X_train, y_train, color='orange', alpha=0.6, label=f'Data (n={len(X_train)})', marker='x')
    
    # Punti Nyström (Bianchi col bordo nero)
    ax_main.scatter(X_nystrom, y_nystrom, color='white', edgecolors='black', s=50, 
                    label=f'Nyström points (m={len(X_nystrom)})', zorder=5)

    # Linee (Modelli)
    if y_true_grid is not None:
        ax_main.plot(X_grid, y_true_grid, label='True model', color='black', linewidth=1.5)
    
    if y_full_grid is not None:
        ax_main.plot(X_grid, y_full_grid, label='Full model', color='red', linestyle='--', alpha=0.8)
        
    ax_main.plot(X_grid, y_approx_grid, label=f'Nyström {strategy_name}', color='green', linestyle='-.', linewidth=2)

    # Cosmetica
    ax_main.set_xlabel(r"$x$")
    ax_main.set_ylabel(r"$y$")
    ax_main.set_title(f"Fit Comparison: {strategy_name}")
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)

    # --- PLOT SCORES (ax[0, 1]) --- (Solo se strategy è RLS o simile)
    if scores is not None and number_of_axes > 1:
        ax_score = ax[0, 1]
        
        # Ordiniamo X per fare un plot pulito della linea degli score
        sort_idx = np.argsort(X_train.flatten())
        X_sorted = X_train[sort_idx]
        scores_sorted = scores[sort_idx]
        
        ax_score.plot(X_sorted, scores_sorted, color='purple', marker='.', linestyle='-', alpha=0.7)
        ax_score.fill_between(X_sorted.flatten(), 0, scores_sorted.flatten(), color='purple', alpha=0.2)
        
        ax_score.set_title(f"{strategy_name} Sampling Scores")
        ax_score.set_xlabel(r"$x$")
        ax_score.set_ylabel("Leverage Score")
        ax_score.grid(True, alpha=0.3)

    # Salvataggio
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        #print(f"Plot salvato in: {save_path}")
    
    plt.show()