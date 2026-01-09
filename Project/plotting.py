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

def plot_single_strategy_cost_function(df, target_strategy='Blended_MP', metric_1='train_risk', metric_2='regularization_cost', metric_3='cost_function', title=None, baseline=None, save_path=None):
    """
    Plotta 3 metriche presenti nel dataframe per una singola strategia specifica.
    Usa colori diversi per distinguere le metriche.
    """
    
    # 1. FILTRO: Teniamo solo la strategia richiesta
    subset = df[df['strategy'] == target_strategy].copy()
    
    if subset.empty:
        print(f"Attenzione: Nessun dato trovato per la strategia '{target_strategy}'")
        return

    # 2. PREPARAZIONE LABELS: Rendiamo i nomi leggibili per la legenda
    label_1 = metric_1.replace('_', ' ').title()
    label_2 = metric_2.replace('_', ' ').title()
    label_3 = metric_3.replace('_', ' ').title()
    
    # Mappa per rinominare le colonne originali con i label puliti
    rename_map = {
        metric_1: label_1, 
        metric_2: label_2, 
        metric_3: label_3
    }
    
    # Rinominiamo le colonne nel subset
    subset_renamed = subset.rename(columns=rename_map)

    # 3. MELT: Trasformiamo il DF da 'largo' a 'lungo' per includere tutte e 3 le metriche
    df_long = subset_renamed.melt(
        id_vars=['number_of_nystrom_points'], 
        # Qui passiamo le 3 colonne (che ora hanno i nomi puliti)
        value_vars=[label_1, label_2, label_3],
        var_name='Metric Type',
        value_name='Value'
    )

    # 4. PLOTTING
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")

    sns.lineplot(
        data=df_long,
        x='number_of_nystrom_points',
        y='Value',
        hue='Metric Type',   # Un colore diverso per ogni metrica (1, 2 e 3)
        style='Metric Type', # Tratteggi diversi
        markers=True,
        dashes=True,
        err_style='band',
        linewidth=2.5,
        markersize=9
    )

    # 5. BASELINE (Opzionale)
    if baseline is not None:
        plt.axhline(
            y=baseline,
            color='gray',
            linestyle='--',
            linewidth=2,
            label='Baseline', # Puoi personalizzare es: f'Baseline ({label_1})'
            alpha=0.7
        )

    # 6. SCALA E FORMATTAZIONE
    plt.xscale('log')
    plt.yscale('log')

    if title is None:
        title = f'Cost Function Analysis for {target_strategy}'
    
    plt.title(title, fontsize=16)
    plt.xlabel(r'Number of Nystrom points ($m$)', fontsize=14)
    plt.ylabel('Value (Log Scale)', fontsize=14)

    # Gestione Ticks asse X (mostra solo i valori di m presenti nel dataset)
    if not subset['number_of_nystrom_points'].empty:
        unique_points = sorted(subset['number_of_nystrom_points'].unique())
        plt.xticks(unique_points, unique_points)
    
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    # Legenda
    plt.legend(title='Metrics', fontsize=12, title_fontsize=12)
    
    plt.tight_layout()
    
    # Salvataggio
    if save_path:
        # Pulisco il nome del file per evitare caratteri strani
        clean_strategy = target_strategy.replace(" ", "_")
        plt.savefig(f"{save_path}_{clean_strategy}_cost_function.pdf", bbox_inches='tight')
        
    plt.show()

def plot_single_strategy_cost_function_separate_plots(df, target_strategy='Blended_MP', metric_1='train_risk', metric_2='regularization_cost', metric_3='cost_function', title=None, baseline=None, save_path=None):
    """
    Plotta 3 grafici distinti affiancati e annota il numero di atomi su ogni punto.
    """
    
    # 1. FILTRO
    subset = df[df['strategy'] == target_strategy].copy()
    
    if subset.empty:
        print(f"Attenzione: Nessun dato trovato per la strategia '{target_strategy}'")
        return

    # 2. SETUP FIGURA
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharex=True)
    sns.set_style("whitegrid")

    metrics = [metric_1, metric_2, metric_3]
    
    titles = [
        metric_1.replace('_', ' ').title(),
        metric_2.replace('_', ' ').title(),
        metric_3.replace('_', ' ').title()
    ]

    # 3. CICLO DI PLOTTING
    for ax, metric, plot_title in zip(axes, metrics, titles):
        
        # Plot della linea con intervalli di confidenza
        sns.lineplot(
            data=subset,
            x='tot_iterations',
            y=metric,
            color='tab:blue', 
            marker='o',
            markersize=8,
            linewidth=2.5,
            err_style='band',
            ax=ax
        )

        # ---------------------------------------------------------
        # ### SEZIONE AGGIUNTA: Annotazione Number of Atoms ###
        # ---------------------------------------------------------
        # Raggruppiamo per 'tot_iterations' per trovare la coordinata Y media (dove sta il pallino)
        # e il valore di 'number_of_nystrom_points' corrispondente.
        annot_data = subset.groupby('tot_iterations')[[metric, 'number_of_nystrom_points']].mean().reset_index()

        for _, row in annot_data.iterrows():
            x_coord = row['tot_iterations']
            y_coord = row[metric]
            # Assumiamo che gli atomi siano interi, li forziamo a int per togliere i decimali
            n_atoms = int(row['number_of_nystrom_points']) 
            
            ax.text(
                x_coord, 
                y_coord, 
                f" {n_atoms}",          # Il testo da scrivere (uno spazio prima per staccarlo)
                fontsize=10,            # Grandezza testo
                color='black',          # Colore testo
                verticalalignment='bottom', # Posizione rispetto al punto (sopra)
                horizontalalignment='left', # Posizione rispetto al punto (a destra)
                fontweight='bold'       # Opzionale: grassetto per renderlo visibile
            )
        # ---------------------------------------------------------

        # Formattazione
        ax.set_title(plot_title, fontsize=16)
        ax.set_xlabel(r'Number of iterations ($T$)', fontsize=14)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=14)
        
        # Scale Logaritmiche
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Ticks asse X
        unique_points = sorted(subset['tot_iterations'].unique())
        ax.set_xticks(unique_points)
        ax.set_xticklabels(unique_points)
        
        ax.grid(True, which="both", ls="-", alpha=0.5)

    # 4. BASELINE
    if baseline is not None:
        axes[2].axhline(
            y=baseline,
            color='black',
            linestyle='--',
            linewidth=2,
            label='Baseline'
        )
        axes[2].legend(fontsize=12)

    # 5. TITOLO E SALVATAGGIO
    if title is None:
        title = f'Analysis for {target_strategy}'
    
    plt.suptitle(title, fontsize=20, y=1.02)
    plt.tight_layout()
    
    if save_path:
        clean_strategy = target_strategy.replace(" ", "_")
        plt.savefig(f"{save_path}_{clean_strategy}_annotated.pdf", bbox_inches='tight')
    
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