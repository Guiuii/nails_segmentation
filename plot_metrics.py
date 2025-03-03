import os
import matplotlib.pyplot as plt
import yaml

def plot_metrics(metrics):
    ''' Визуализация графиков зависимости Loss, F1-score от кол-ва эпох '''
    
    # Загрузка конфига
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
        
    epochs_num = config["training"]["num_epochs"]
    
    plt.figure(figsize=(10,10))
    
    # metrics =  {'Loss': full_loss,'F1-score': full_f1}
    
    for name, val in metrics.items():
        plt.clf()
        
        plt.plot(range(1, epochs_num+1), val, color = config['visualization']['color'])
        plt.xlabel('Epochs') 
        plt.ylabel(name)  
        plt.title(name) 
        if name=="F1-score":
            plt.ylim([0, 1])

        # Определяем путь для сохранения
        save_path = os.path.join(
        config['visualization']['plots_dir'],
        f'{name}_per_epoch.png')

        plt.savefig(save_path, bbox_inches="tight")
    