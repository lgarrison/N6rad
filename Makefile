TARGETS := n6rad
NVCC := nvcc

n6rad: n6rad.cu
	$(NVCC) -O3 -g -arch=sm_75 -Xcompiler "-fopenmp -Wall" --ptxas-options=-v -o $@ $<
    
ptx: n6rad.cu
	$(NVCC) -ptx -O3 -g -arch=sm_75 -Xcompiler "-fopenmp -Wall" --ptxas-options=-v -o $(<:.cu=.ptx) $<

clean:
	$(RM) $(TARGETS)
distclean: clean

.PHONY: clean distclean
