#导入库
import matplotlib.pyplot as plt
import numpy as np
import palettable
from palettable.tableau import TableauMedium_10


all_file_list = ['col_loop','row_loop','cache_11','cache_12','cache_21','cache_22','cache_31','cache_32','cache_33','cache_41','simd','thread1']
test_set = [0,1,2,3,4,5,6]
test_list = [[0,1],[1,2,3],[1,2,3,4,5],[1,2,3,4,5,6,7,8],[1,3,5,7,9],[1,3,5,7,9,10],[1,3,5,7,9,10,11]]
top_list = [0.3,0.5,0.5,0.7,1.0,4.0,6]
test_name_list = ['row_col','cache1','cache2','cache3','cache4','simd','thread1']
base_path = "./logs/"

def main():

    for i in test_set:
        fig=plt.figure(dpi=200)
        test = test_list[i]
        for j in range(len(test)):
            name = all_file_list[test[j]]
            with open(base_path+name+".log","r") as fp:
                x = []
                y = []
                lines=fp.readlines()
                for line in lines:
                    dim, num, time = line.split(' ')
                    dim = int(dim)
                    num = int(num)
                    time = float(time)
                    gf = (dim*dim*dim*num) /(10**9*time)
                    y.append(gf)
                    x.append(dim)
                plt.plot(x, y, label=name, ls=':', color=TableauMedium_10.mpl_colors[j], marker='*')
        plt.title("Optimize gemm")
        plt.rcParams.update({'font.size':6})
        plt.legend(loc="best") 
        plt.xlabel("dim of matrix", loc="center")
        plt.ylabel("GFLOPS/sec", loc="center")
        plt.ylim(top=top_list[i])
        fig.savefig(f"./images/{test_name_list[i]}.jpg")


if __name__ == "__main__":
    main()
