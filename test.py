import torch
print(torch.cuda.is_available())  # True가 나와야 함
print(torch.cuda.device_count())  # 0보다 커야 함
print(torch.cuda.get_device_name(0))  # GPU 이름이 출력되어야 함