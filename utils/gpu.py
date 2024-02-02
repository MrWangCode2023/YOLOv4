import torch


def select_device(id):
    """
    总结：这段代码定义了一个函数 `select_device`，用于选择设备（CPU或CUDA），并打印相关信息。
    输入：`id`，表示设备的ID，如果为-1，则强制使用CPU。
    功能：根据输入的设备ID，判断是否使用CUDA，然后选择相应的设备（CPU或CUDA）。同时，打印出关于选择设备的信息，包括使用的是CPU还是CUDA，以及CUDA设备的相关信息。
    输出：返回选定的设备对象。
    """
    # 判断是否强制使用CPU
    force_cpu = False
    if id == -1:
        force_cpu = True
    # 检查是否支持CUDA
    cuda = False if force_cpu else torch.cuda.is_available()
    # 根据CUDA的可用性选择设备
    device = torch.device("cuda:{}".format(id) if cuda else "cpu")

    # 打印设备信息
    if not cuda:
        print("Using CPU")
    if cuda:
        c = 1024 ** 2  # B to MB（进率）
        # 获取当前系统上可用的CUDA设备数量（显卡数量）
        ng = torch.cuda.device_count()
        # 获取指定索引（i）处GPU的属性。它返回一个_CudaDeviceProperties对象，包含有关该GPU的各种属性信息，如设备名称、总内存大小等。
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        print(
            "Using CUDA device0 _CudaDeviceProperties(name='%s', total_memory=%dMB)"
            % (x[0].name, x[0].total_memory / c)
        )
        if ng > 0:
            # 打印其他CUDA设备信息
            for i in range(1, ng):
                print(
                    "           device%g _CudaDeviceProperties(name='%s', total_memory=%dMB)"
                    % (i, x[i].name, x[i].total_memory / c)
                )

    return device


if __name__ == "__main__":
    _ = select_device(0)
