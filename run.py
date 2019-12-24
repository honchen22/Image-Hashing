import sys, json
from modules.benchmark.process_standard_image import BenchmarkHash2JSONParser
from modules.benchmark_cc.get_cc import CorrelationCoefficient
from modules.figure2.plot_figure2 import Figure2
from modules.figure3.plot_figure3 import Figure3
from modules.figure4.plot_figure4 import Figure4
from modules.table1.plot_table1 import Table1
from modules.table2.plot_table2 import Table2
from modules.table3.plot_table3 import Table3

if __name__ == '__main__':
    arg_list = sys.argv
    if len(arg_list) == 1:
        print('Usage:')
        print('1: get_image_hash')
        print('2: get_cc')
        print('3: get_fig2')
        print('4: get_table1')
        print('5: get_fig3')
        print('6: get_table2')
        print('7: get_fig4')
        print('8: get_table3')
    else:
        arg = int(arg_list[1])
        if arg == 1:
            print('get_image_hash')
            parser = BenchmarkHash2JSONParser()
            parser.run()
        elif arg == 2:
            print('get_cc')
            cc = CorrelationCoefficient()
            cc.run()
        elif arg == 3:
            print('get_fig2')
            f2 = Figure2()
            f2.run()
        elif arg == 4:
            print('get_table1')
            t1 = Table1()
            t1.run()
        elif arg == 5:
            print('get_fig3')
            f3 = Figure3()
            option = input('divide contrast or not? (default no)')
            f3.run(option)
        elif arg == 6:
            print('get_table2')
            t2 = Table2()
            t2.run()
        elif arg == 7:
            print('get_figure4')
            f4 = Figure4()
            f4.run()
        elif arg == 8:
            print('get_table3')
            t3 = Table3()
            t3.run()