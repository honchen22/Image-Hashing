##  开发环境

*   已在 Windows8 / MacOS 上进行过测试

*   Python 3.7.2

*   opencv-python 4.1.2

    pip3 install opencv-python --user

*   numpy 1.71.4

    pip3 install numpy --user

*   matplotlib 3.1.2

    pip3 install matplotlib --user

*   sklearn 0.22

    pip3 install sklearn --user

##  运行说明

*   首先将终端/命令行切换到当前目录, `ls` 之后可以得到 `run.py`, `modules`, `readme.md`

*   运行 python3 run.py [option]

    如果不加 option，则可以得到具体的 Usage

*   option 说明

    ```
    1: get_image_hash
    2: get_cc
    3: get_fig2
    4: get_table1
    5: get_fig3
    6: get_table2
    7: get_fig4
    8: get_table3
    ```

##  关于数据集

*   `\modules\benchmark\standard` 文件夹中包含了基准数据集以及经过亮度变换、对比度调整等处理之后的图片

*   另外两个数据集尺寸较大，不便于打包上传，故以连接形式给出数据集来源

    [UCID Dataset v2](http://jasoncantarella.com/downloads/ucid.v2.tar.gz)

    [Copydays Dataset](http://pascal.inrialpes.fr/data/holidays/copydays_original.tar.gz)
