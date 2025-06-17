import gc
import torch


def clear_gpu_memory():
    """
    Limpa completamente a memória da GPU antes de iniciar a aplicação.
    """
    gc.collect()
    
    # Limpa cache CUDA se disponível
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

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