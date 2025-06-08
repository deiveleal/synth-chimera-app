import gc
import torch
import numpy as np

def clear_gpu_memory():
    """
    Limpa completamente a memória da GPU antes de iniciar a aplicação.
    """
    # Libera referências Python
    gc.collect()
    
    # Limpa cache CUDA se disponível
    if torch.cuda.is_available():
        # Libera todas as alocações CUDA
        torch.cuda.empty_cache()
        
        # Força sincronização de todas as streams
        torch.cuda.synchronize()
        
        # Reinicia estatísticas de alocação (opcional)
        torch.cuda.reset_peak_memory_stats()
        
        # Em alguns casos extremos, pode ser necessário reiniciar o dispositivo
        # Isso é raramente necessário e só deve ser usado se tudo mais falhar
        # torch.cuda.device(torch.cuda.current_device()).reset()
        
        # Informa estatísticas de memória
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        
        print(f"GPU Memory: Alocado: {allocated:.2f} MB | "
              f"Máximo alocado: {max_allocated:.2f} MB | "
              f"Reservado: {reserved:.2f} MB")
        
        return True
    else:
        print("CUDA não disponível - executando apenas em CPU.")
        return False