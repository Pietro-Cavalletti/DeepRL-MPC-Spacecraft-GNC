import os
import glob
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from datetime import datetime
import argparse

class TensorBoardExporter:
    """
    Classe per esportare tutti i dati da TensorBoard in formato CSV
    """
    
    def __init__(self, tensorboard_dir="./tensorboard/"):
        self.tensorboard_dir = r"C:\Users\caval\OneDrive - Politecnico di Milano\Thesis\Code\RL on QR_T\tensorboard\RecurrentPPO_1"
        self.data = {}
        
    def find_event_files(self):
        """Trova tutti i file di eventi TensorBoard"""
        pattern = os.path.join(self.tensorboard_dir, "**/events.out.tfevents.*")
        event_files = glob.glob(pattern, recursive=True)
        
        if not event_files:
            print(f"❌ Nessun file di eventi trovato in: {self.tensorboard_dir}")
            return []
            
        print(f"📁 Trovati {len(event_files)} file di eventi:")
        for file in event_files:
            rel_path = os.path.relpath(file, self.tensorboard_dir)
            print(f"   📄 {rel_path}")
            
        return event_files
    
    def extract_scalars(self, event_files):
        """Estrae tutti i dati scalari dai file di eventi"""
        all_scalars = {}
        
        for file_path in event_files:
            print(f"🔍 Processando: {os.path.basename(file_path)}")
            
            try:
                # Carica il file di eventi
                ea = EventAccumulator(file_path)
                ea.Reload()
                
                # Ottieni tutte le metriche scalari
                scalar_keys = ea.scalars.Keys()
                print(f"   📊 Metriche trovate: {len(scalar_keys)}")
                
                for key in scalar_keys:
                    scalar_events = ea.scalars.Items(key)
                    
                    if scalar_events:
                        # Estrai dati
                        steps = [event.step for event in scalar_events]
                        values = [event.value for event in scalar_events]
                        wall_times = [event.wall_time for event in scalar_events]
                        
                        # Aggiungi ai dati globali
                        if key not in all_scalars:
                            all_scalars[key] = {'steps': [], 'values': [], 'wall_times': []}
                        
                        all_scalars[key]['steps'].extend(steps)
                        all_scalars[key]['values'].extend(values)
                        all_scalars[key]['wall_times'].extend(wall_times)
                        
                        print(f"     ✅ {key}: {len(scalar_events)} punti dati")
                        
            except Exception as e:
                print(f"   ❌ Errore nel processare {file_path}: {e}")
                
        return all_scalars
    
    def create_dataframe(self, scalars_data):
        """Crea un DataFrame pandas dai dati scalari"""
        print("🔄 Creando DataFrame...")
        
        # Trova tutti gli step unici
        all_steps = set()
        for metric_data in scalars_data.values():
            all_steps.update(metric_data['steps'])
        
        all_steps = sorted(list(all_steps))
        print(f"📈 Steps totali: {len(all_steps)} (da {min(all_steps)} a {max(all_steps)})")
        
        # Crea il DataFrame con tutti gli step
        df = pd.DataFrame(index=all_steps)
        df.index.name = 'step'
        
        # Aggiungi ogni metrica
        for metric_name, metric_data in scalars_data.items():
            # Crea una Series per questa metrica
            metric_series = pd.Series(
                data=metric_data['values'],
                index=metric_data['steps'],
                name=metric_name
            )
            
            # Rimuovi duplicati mantenendo l'ultimo valore
            metric_series = metric_series[~metric_series.index.duplicated(keep='last')]
            
            # Aggiungi al DataFrame
            df[metric_name] = metric_series
        
        print(f"📊 DataFrame creato: {df.shape[0]} righe, {df.shape[1]} colonne")
        return df
    
    def export_all_formats(self, df, base_filename="tensorboard_export", output_dir=r"C:\Users\caval\OneDrive - Politecnico di Milano\Thesis\Code\RL on QR_T\tensorboard"):
        """Esporta i dati in diversi formati in una directory specifica"""
        
        # Path hardcoded dove salvare i file
        output_dir = r"C:\Users\caval\OneDrive - Politecnico di Milano\Thesis\Code\RL on QR_T\tensorboard"
        os.makedirs(output_dir, exist_ok=True)
        base_path = os.path.join(output_dir, base_filename)
        
        # OPZIONE 1: Senza timestamp (sovrascrive sempre)
        csv_file = f"{base_path}.csv"
        main_csv = f"{base_path}_main_metrics.csv"
        excel_file = f"{base_path}.xlsx"

        
        # 1. CSV completo
        df.to_csv(csv_file)
        print(f"📁 CSV completo salvato: {csv_file}")
        
        # 2. CSV solo metriche principali (senza NaN)
        main_metrics = [col for col in df.columns if any(keyword in col.lower() for keyword in 
                    ['reward', 'loss', 'length', 'success', 'episode', 'eval'])]
        
        if main_metrics:
            main_df = df[main_metrics].dropna(how='all')
            main_df.to_csv(main_csv)
            print(f"📁 CSV metriche principali salvato: {main_csv}")
        
        # 3. Excel con fogli separati per categoria
        with pd.ExcelWriter(excel_file) as writer:
            # Foglio completo
            df.to_excel(writer, sheet_name='All_Data')
            
            # Fogli per categoria
            categories = {
                'Rewards': [col for col in df.columns if 'reward' in col.lower()],
                'Losses': [col for col in df.columns if 'loss' in col.lower()],
                'Policy': [col for col in df.columns if any(x in col.lower() for x in ['policy', 'entropy', 'kl'])],
                'Episodes': [col for col in df.columns if any(x in col.lower() for x in ['episode', 'length'])],
                'Evaluation': [col for col in df.columns if 'eval' in col.lower()],
                'Actions': [col for col in df.columns if 'action' in col.lower()],
                'Training': [col for col in df.columns if 'train' in col.lower()]
            }
            
            for category, columns in categories.items():
                if columns:
                    cat_df = df[columns].dropna(how='all')
                    if not cat_df.empty:
                        cat_df.to_excel(writer, sheet_name=category)
        
        print(f"📁 Excel salvato: {excel_file}")
        
        return csv_file, excel_file
    
    def print_summary(self, df):
        """Stampa un riassunto dei dati"""
        print("\n" + "="*60)
        print("📊 RIASSUNTO DATI TENSORBOARD")
        print("="*60)
        
        print(f"📈 Range steps: {df.index.min():,} - {df.index.max():,}")
        print(f"📊 Totale data points: {len(df):,}")
        print(f"📋 Metriche totali: {len(df.columns)}")
        
        print("\n🏷️  METRICHE PER CATEGORIA:")
        categories = {
            'Rewards': [col for col in df.columns if 'reward' in col.lower()],
            'Losses': [col for col in df.columns if 'loss' in col.lower()],
            'Policy': [col for col in df.columns if any(x in col.lower() for x in ['policy', 'entropy', 'kl'])],
            'Episodes': [col for col in df.columns if any(x in col.lower() for x in ['episode', 'length'])],
            'Evaluation': [col for col in df.columns if 'eval' in col.lower()],
            'Actions': [col for col in df.columns if 'action' in col.lower()],
            'Training': [col for col in df.columns if 'train' in col.lower()]
        }
        
        for category, columns in categories.items():
            if columns:
                print(f"   {category}: {len(columns)} metriche")
                for col in columns[:3]:  # Mostra solo le prime 3
                    print(f"     • {col}")
                if len(columns) > 3:
                    print(f"     ... e altre {len(columns) - 3}")
        
        # Metriche non categorizzate
        categorized = set()
        for columns in categories.values():
            categorized.update(columns)
        
        uncategorized = [col for col in df.columns if col not in categorized]
        if uncategorized:
            print(f"   Altri: {len(uncategorized)} metriche")
        
        print("\n📊 STATISTICHE CHIAVE:")
        key_metrics = ['rollout/ep_rew_mean', 'eval/mean_reward', 'train/loss', 'train/policy_loss']
        for metric in key_metrics:
            if metric in df.columns:
                series = df[metric].dropna()
                if not series.empty:
                    print(f"   {metric}:")
                    print(f"     📊 Punti dati: {len(series)}")
                    print(f"     📈 Range: {series.min():.2f} - {series.max():.2f}")
                    print(f"     📊 Ultimo valore: {series.iloc[-1]:.2f}")

    def export_tensorboard_data(self, output_prefix="tensorboard_export", output_dir=r"C:\Users\caval\OneDrive - Politecnico di Milano\Thesis\Code\RL on QR_T\tensorboard"):
        """Metodo principale per esportare tutti i dati"""
        print("🚀 Avvio esportazione dati TensorBoard...")
        
        # Trova i file
        event_files = self.find_event_files()
        if not event_files:
            return None
        
        # Estrai i dati
        scalars_data = self.extract_scalars(event_files)
        if not scalars_data:
            print("❌ Nessun dato scalare trovato")
            return None
        
        # Crea DataFrame
        df = self.create_dataframe(scalars_data)
        
        # Stampa riassunto
        self.print_summary(df)
        
        # Esporta in diversi formati nella directory specificata
        csv_file, excel_file = self.export_all_formats(df, output_prefix, output_dir)
        
        print(f"\n✅ Esportazione completata!")
        print(f"📁 File CSV: {csv_file}")
        print(f"📁 File Excel: {excel_file}")
        
        return df, csv_file, excel_file

  

def main():
    """Funzione principale con argomenti da linea di comando"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Esporta dati TensorBoard in CSV/Excel')
    parser.add_argument('--tensorboard_dir', '-t', default='./tensorboard/', 
                    help='Directory dei log TensorBoard (default: ./tensorboard/)')
    parser.add_argument('--output_prefix', '-o', default='tensorboard_export',
                    help='Prefisso per i file di output (default: tensorboard_export)')
    parser.add_argument('--export_dir', '-e', default=None,
                    help='Directory assoluta dove salvare i file esportati')
    
    args = parser.parse_args()
    
    # Crea l'exporter ed esegui
    exporter = TensorBoardExporter(args.tensorboard_dir)
    result = exporter.export_tensorboard_data(args.output_prefix, args.export_dir)
    
    if result:
        df, csv_file, excel_file = result
        print(f"\n🎉 Esportazione riuscita! DataFrame shape: {df.shape}")
    else:
        print("❌ Esportazione fallita")

if __name__ == "__main__":
    main()



import pandas as pd
import matplotlib.pyplot as plt

# === Load data ===
csv_path =r"C:\Users\caval\OneDrive - Politecnico di Milano\Thesis\Code\RL on QR_T\tensorboard\tensorboard_export.csv"
df = pd.read_csv(csv_path, sep="\t" if "\t" in open(csv_path).readline() else ",")

# Convert step to thousands
df['step_k'] = df['step'] / 1000

# # Predefined color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']



# === FIGURE: Real Actions, Training Metrics, Indici Rollout e Eval ===
plt.figure(figsize=(12, 4))
plt.suptitle("Action statistics", fontsize=16)

# === 1. Qp ===
plt.subplot(1, 2, 1)
for label, color in zip(['max', 'mean', 'min'], colors[:3]):
    col = f'episode/real_actions_Qp_{label}'
    plt.plot(df['step_k'], df[col], color=color, linewidth=1, alpha=0.6)
    plt.scatter(df['step_k'], df[col], label=f'Qp {label.capitalize()}', color=color, s=10)

plt.fill_between(df['step_k'],
                 df['episode/real_actions_Qp_mean'] - df['episode/real_actions_Qp_std'],
                 df['episode/real_actions_Qp_mean'] + df['episode/real_actions_Qp_std'],
                 color=colors[1], alpha=0.2, label='Qp ±1 Std')
plt.title('Qp Values Over Time')
plt.xlabel('Training Steps (k)')
plt.ylabel('Qp Value')

plt.legend()
plt.grid(True)


# === 3. R ===
plt.subplot(1, 2, 2)
for label, color in zip(['max', 'mean', 'min'], colors[:3]):
    col = f'episode/real_actions_R_{label}'
    plt.plot(df['step_k'], df[col], color=color, linewidth=1, alpha=0.6)
    plt.scatter(df['step_k'], df[col], label=f'R {label.capitalize()}', color=color, s=10)

plt.fill_between(df['step_k'],
                 df['episode/real_actions_R_mean'] - df['episode/real_actions_R_std'],
                 df['episode/real_actions_R_mean'] + df['episode/real_actions_R_std'],
                 color=colors[1], alpha=0.2, label='R ±1 Std')
plt.title('R Values Over Time')
plt.xlabel('Training Steps (k)')
plt.ylabel('R Value')

plt.legend()
plt.grid(True)


# # === 4. Learning Rate ===
# plt.subplot(2, 3, 4)
# if 'train/learning_rate' in df.columns:
#     plt.plot(df['step_k'], df['train/learning_rate'], color=colors[1], linewidth=1.5)
# plt.yscale("log")
# plt.title('Learning Rate')
# plt.xlabel('Training Steps (k)')
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)



plt.tight_layout(rect=[0, 0.02, 1, 0.97])
##################################################################


# === FIGURE 2: Episode Performance ===
plt.figure(figsize=(12, 6))
plt.suptitle("Figure 2 - Episode and Eval Performance", fontsize=14)

# plt.subplot(2, 2, 1)
raw = df[['step_k', 'rollout/ep_rew_mean']].dropna()
ma = raw['rollout/ep_rew_mean'].rolling(15).mean()

plt.plot(raw['step_k'], raw['rollout/ep_rew_mean'], color=colors[0], alpha=0.3, linewidth=1.0, label='Raw')
plt.plot(raw['step_k'], ma, color=colors[0], linewidth=2.0, label='Moving Avg')
plt.title('Mean Rollout Reward')
plt.xlabel('Training Steps (k)')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)


# plt.subplot(2, 2, 2)
# subset = df[['step_k', 'eval/mean_reward']].dropna()
# ma = subset['eval/mean_reward'].rolling(5).mean()
# plt.plot(subset['step_k'], subset['eval/mean_reward'], color=colors[1], alpha=0.3, linewidth=1.0, label='Raw')
# plt.plot(subset['step_k'], ma, color=colors[1], linewidth=2.0, label='Moving Avg')
# plt.scatter(subset['step_k'], subset['eval/mean_reward'], color=colors[1], s=15, alpha=0.5)
# plt.title('Mean Evaluation Reward')
# plt.xlabel('Training Steps (k)')
# plt.ylabel('Reward')
# plt.legend()
# plt.grid(True)


# # === 4. FPS  ===
# plt.subplot(2, 2, 3)
# if 'time/fps' in df.columns:
#     plt.plot(df['step_k'], df['time/fps'], color=colors[1], linewidth=1.5)
# plt.title('FPS')
# plt.xlabel('Training Steps (k)')
# plt.grid(True)


# # === 6. Indici Eval ===
# plt.subplot(2, 2, 4)
# eval_indices = [col for col in df.columns if col.startswith('eval/episode_perf/')]
# if eval_indices:
#     for i, col in enumerate(eval_indices):
#         plt.plot(df['step_k'], df[col], label=col.replace('eval/', ''), linewidth=1.2)
# plt.title('Evaluation Indices')
# plt.xlabel('Training Steps (k)')
# plt.ylabel('Value')
# plt.legend(fontsize=8)
# plt.grid(True)



plt.tight_layout(rect=[0, 0.02, 1, 0.97])


# === FIGURE 3: Optimization Indicators ===
plt.figure(figsize=(14, 7))
plt.suptitle("Figure 3 - Optimization Indicators", fontsize=14)

plt.subplot(2, 3, 1)
plt.plot(df['step_k'], df['train/approx_kl'], label='Approx KL', color=colors[0], linewidth=2)
plt.title('Approximate KL Divergence')
plt.xlabel('Training Steps (k)')
plt.ylabel('KL')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(df['step_k'], df['train/loss'], label='Total Loss', color=colors[1], linewidth=2)
plt.plot(df['step_k'], df['train/value_loss'], label='Value Loss', color=colors[2], linewidth=2)
plt.title('Loss and Value Loss')
plt.xlabel('Training Steps (k)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(df['step_k'], df['train/policy_gradient_loss'], label='Policy Gradient Loss', color=colors[3], linewidth=2)
plt.title('Policy Gradient Loss')
plt.xlabel('Training Steps (k)')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(df['step_k'], df['train/std'], label='Policy Std', color=colors[0], linewidth=2)
plt.title('Policy Std Dev')
plt.xlabel('Training Steps (k)')
plt.ylabel('Std')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(df['step_k'], df['train/entropy_loss'], label='Entropy Loss', color=colors[2], linewidth=2)
plt.title('Policy Entropy')
plt.xlabel('Training Steps (k)')
plt.ylabel('Entropy')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(df['step_k'], df['train/explained_variance'], label='Explained Variance', color=colors[3], linewidth=2)
plt.title('Explained Variance')
plt.xlabel('Training Steps (k)')
plt.ylabel('Variance')
plt.legend()
plt.grid(True)


plt.tight_layout(rect=[0, 0.02, 1, 0.97])
plt.show()



