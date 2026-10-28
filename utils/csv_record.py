import csv
import copy

# ========================== #
#   定义 CSV 文件列名结构     #
# ========================== #

# 训练过程中记录的字段
train_fileHeader = [
    "local_model", "round", "epoch", "internal_epoch",
    "average_loss", "accuracy", "correct_data", "total_data"
]

# 测试记录字段
test_fileHeader = ["model", "epoch", "average_loss", "accuracy", "correct_data", "total_data"]

# 用于存放训练结果（每一行对应 train_fileHeader）
train_result = []

# 普通测试结果列表
test_result = []

# 中毒测试结果列表（整体 poisoned test）
posiontest_result = []

# ====================== #
#   触发器测试相关字段     #
# ====================== #

# 触发器测试 CSV 表头
triggertest_fileHeader = [
    "model", "trigger_name", "trigger_value",
    "epoch", "average_loss", "accuracy", "correct_data","total_data"
]

# 用于存储单独触发器攻击效果
poisontriggertest_result = []

# ========== 一些额外结果存储（权重, scale） ========== #

posion_test_result = []          # unused in this script
posion_posiontest_result = []    # unused in this script

weight_result = []               # 用来保存模型参数权重结果（例如 aggregation 过程）
scale_result = []                # 保存一些 scale 数值（可能在 model scaling 使用）
scale_temp_one_row = []          # 临时存储 scale 单行，在 epoch 结束时会加入 scale_result


# ============================================================ #
#   保存所有结果到 CSV 文件（在每一轮 or 最终都可能被调用）      #
# ============================================================ #
def save_result_csv(epoch, is_posion, folder_path):
    """
    将训练/测试/中毒测试/触发器测试等结果保存为 CSV 文件。
    :param epoch: 当前 epoch（仅用于日志，无实际写入）
    :param is_posion: 是否保存中毒结果（有些情况下不需要）
    :param folder_path: 保存 CSV 文件的目标文件夹
    """

    # ------------------------------ #
    #  保存训练结果 train_result.csv  #
    # ------------------------------ #
    train_csvFile = open(f'{folder_path}/train_result.csv', "w")
    train_writer = csv.writer(train_csvFile)
    train_writer.writerow(train_fileHeader)  # 写表头
    train_writer.writerows(train_result)     # 写所有训练数据
    train_csvFile.close()

    # ------------------------------ #
    #  保存普通测试结果 test_result.csv
    # ------------------------------ #
    test_csvFile = open(f'{folder_path}/test_result.csv', "w")
    test_writer = csv.writer(test_csvFile)
    test_writer.writerow(test_fileHeader)
    test_writer.writerows(test_result)
    test_csvFile.close()

    # ------------------------------ #
    #   保存权重结果 weight_result.csv
    #   （如果有记录）
    # ------------------------------ #
    if len(weight_result) > 0:
        weight_csvFile = open(f'{folder_path}/weight_result.csv', "w")
        weight_writer = csv.writer(weight_csvFile)
        weight_writer.writerows(weight_result)
        weight_csvFile.close()

    # ------------------------------ #
    #   保存 scale 结果 scale_result.csv
    # ------------------------------ #
    if len(scale_temp_one_row) > 0:
        _csvFile = open(f'{folder_path}/scale_result.csv', "w")
        _writer = csv.writer(_csvFile)

        # 将当前行写入 scale_result（深拷贝避免指针引用问题）
        scale_result.append(copy.deepcopy(scale_temp_one_row))
        scale_temp_one_row.clear()  # 清空临时缓存

        _writer.writerows(scale_result)
        _csvFile.close()

    # ===================================================== #
    #   如果本轮包含攻击行为，则保存中毒相关的两个文件       #
    # ===================================================== #
    if is_posion:
        # --------------------------- #
        #  保存中毒测试结果 posiontest_result.csv
        # --------------------------- #
        test_csvFile = open(f'{folder_path}/posiontest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(test_fileHeader)
        test_writer.writerows(posiontest_result)
        test_csvFile.close()

        # --------------------------- #
        #  保存触发器测试 poisontriggertest_result.csv
        # --------------------------- #
        test_csvFile = open(f'{folder_path}/poisontriggertest_result.csv', "w")
        test_writer = csv.writer(test_csvFile)
        test_writer.writerow(triggertest_fileHeader)
        test_writer.writerows(poisontriggertest_result)
        test_csvFile.close()


# =========================================== #
#  添加权重记录（如 FedAvg 每轮的聚合权重）     #
# =========================================== #
def add_weight_result(name, weight, alpha):
    """
    添加权重记录（一般用于记录 fedavg 权重 或 attacker scale 等）
    :param name: 权重名称（如攻击者 id 或 layer 名）
    :param weight: 权重值
    :param alpha: 额外参数（如平滑系数等）
    """
    weight_result.append(name)
    weight_result.append(weight)
    weight_result.append(alpha)
