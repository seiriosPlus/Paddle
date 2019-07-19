本地计算AUC逻辑：

    1. 用户在做infer的时候，把predict和label的值通过fetch_target取出来
    2. predict是个[[pre0,pre1], [pre0,pre1]]格式的二维数组， label是 [[lab0],[lab0]]格式的二维数组
    3. 定义一个独立的目录，用来存放infer输出的值
    4. 将每行predict/label的值，按照 "pre0 pre1 lab0"的形式逐行写入到文件中
    4. 使用auc.py脚本， 执行 python -u auc.py auclog目录
